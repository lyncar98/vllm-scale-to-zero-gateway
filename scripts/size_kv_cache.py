"""Hybrid KV-cache and GPU instance sizer for Qwen3.6-35B-A3B on vLLM.

Why this script exists:
  1. Standard transformer KV-cache formulas assume every layer stores a K and V
     tensor per token:

         KV bytes/token = 2 * num_layers * num_kv_heads * head_dim * kv_bytes

     Qwen/Qwen3.6-35B-A3B (``Qwen3_5MoeForConditionalGeneration``) has 40 layers,
     but only 10 use full grouped-query attention (2 KV heads, head_dim 256 ->
     20.0 KiB/token in BF16). The other 30 layers use Gated DeltaNet
     (``linear_attention``), which allocates a fixed recurrent + causal-conv
     state of ~61.41 MiB per active sequence slot (``--max-num-seqs``),
     independent of context length.

  2. Marketing GPU memory != usable physical GiB.
     An NVIDIA L40S is marketed at "48 GB", listed at ~44 GiB in AWS EC2 G6e
     technical specifications, and reports ~44.99 GiB of physical device memory
     with ECC enabled. Treating an L40S as 48.0 GiB overstates the remaining
     paged KV-cache pool on a 35 GiB FP8 checkpoint by ~3.01 GiB (nearly 40%).

  3. Cumulative agent tokens != single-invocation resident context.
     In a multi-step agent loop, each tool round-trip re-sends the system
     prefix, tool schemas, and prior turns. Our heaviest baseline run (R11)
     metered 47,305 cumulative input tokens (92.8% cached) + 356 metered output
     tokens + ~1,730 collaborator output tokens = 49,391 cumulative tokens
     across 4 tool calls (5 model invocations), while peak single-invocation
     resident context was ~11.1K-15.8K tokens.

  This utility evaluates three candidate context targets against physical GPU
  memory (defaulting to 44.99 GiB for L40S, with --l40s-vram-gib to compare
  44.0 GiB spec or 48.0 GiB nominal assumptions):
    - ``per-invocation`` (49,152 tokens): ~3x peak single-invocation residency
      in our baseline for segmented agent workflows.
    - ``cumulative-2x``  (98,304 tokens): the provisional 2x cumulative-run
      heuristic (does NOT fit on one 44.99 GiB L40S under current hybrid-cache
      overhead assumptions).
    - ``qwen-128k``      (131,072 tokens): Qwen's recommended >=128K context
      for complex unsegmented tasks (requires multi-GPU or KV quantization).

  NOTE: Token capacities are modeled planning estimates using a provisional
  3.5526x hybrid-page overhead factor (71.05 KiB/token effective vs. 20.0
  KiB/token raw full-attention KV). They must be reconciled with preserved
  vLLM startup profiles from the deployed GPU before claiming a validated
  production context ceiling.

Usage:
    python scripts/size_kv_cache.py
    python scripts/size_kv_cache.py --l40s-vram-gib 48.0
    python scripts/size_kv_cache.py --concurrency 16 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass

GIB = 1024**3
MIB = 1024**2
KIB = 1024
TOKEN_ALIGNMENT = 16_384


@dataclass(frozen=True)
class ModelSpec:
    """Architectural parameters from the Hugging Face config.json."""

    key: str
    repo_id: str
    revision: str
    precision: str
    weights_gib: float
    native_max_position_embeddings: int
    recommended_complex_task_context: int
    total_layers: int
    full_attention_layers: int
    kv_heads: int
    head_dim: int
    kv_dtype_bytes: int
    linear_attention_layers: int
    linear_key_heads: int
    linear_key_head_dim: int
    linear_value_heads: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    recurrent_dtype_bytes: int
    conv_dtype_bytes: int
    # Multimodal vision encoder + profiling workspace (~8.0 GiB on Qwen3.6-35B-A3B
    # when --language-model-only is NOT passed).
    vision_tower_overhead_gib: float
    # CUDA context, PyTorch allocator fragmentation, and non-cache runtime workspace
    # under --enforce-eager.
    runtime_workspace_gib: float
    # Provisional calibration factor comparing vLLM's unified hybrid KV/Mamba page
    # allocator against raw full-attention KV bytes (20.0 KiB/token * 3.5526 ≈
    # 71.05 KiB/token). Must be reconciled against raw vLLM startup block counts.
    hybrid_page_overhead_factor: float = 3.5526
    tool_call_parser: str = "qwen3_coder"
    reasoning_parser: str = "qwen3"

    @property
    def full_attention_kv_bytes_per_token(self) -> int:
        """Theoretical K + V bytes per token across the full-attention (GQA) layers only."""
        return (
            2
            * self.full_attention_layers
            * self.kv_heads
            * self.head_dim
            * self.kv_dtype_bytes
        )

    @property
    def effective_kv_bytes_per_token(self) -> int:
        """Provisional effective paged-pool bytes per token with hybrid block alignment."""
        return int(round(self.full_attention_kv_bytes_per_token * self.hybrid_page_overhead_factor))

    @property
    def linear_recurrent_bytes_per_layer(self) -> int:
        """Recurrent state matrix S in R^(H_v x D_k x D_v) for one Gated DeltaNet layer."""
        return (
            self.linear_value_heads
            * self.linear_key_head_dim
            * self.linear_value_head_dim
            * self.recurrent_dtype_bytes
        )

    @property
    def linear_conv_bytes_per_layer(self) -> int:
        """Causal short-conv state of width (kernel_dim - 1) across Q, K, V channels."""
        conv_channels = (
            2 * self.linear_key_heads * self.linear_key_head_dim
            + self.linear_value_heads * self.linear_value_head_dim
        )
        return conv_channels * max(0, self.linear_conv_kernel_dim - 1) * self.conv_dtype_bytes

    @property
    def deltanet_state_bytes_per_sequence(self) -> int:
        """Fixed recurrent + conv state bytes allocated per active sequence (--max-num-seqs)."""
        per_layer = self.linear_recurrent_bytes_per_layer + self.linear_conv_bytes_per_layer
        return self.linear_attention_layers * per_layer


@dataclass(frozen=True)
class WorkloadBaseline:
    """Sanitized aggregate measurements from our 11-run test-environment sample.

    Separates cumulative multi-call run totals (which drive token billing) from
    estimated single-invocation resident context (which drives GPU KV residency).
    """

    runs_measured: int = 11
    conversations_measured: int = 7
    tool_calling_runs: int = 9
    total_tool_calls: int = 30
    no_tool_input_tokens_avg: int = 7_984
    no_tool_cache_hit_rate: float = 0.995
    tool_run_input_tokens_avg: int = 32_108
    tool_run_cache_hit_rate: float = 0.910
    artifact_run_input_tokens_avg: int = 37_369
    non_artifact_tool_run_input_tokens_avg: int = 27_899
    # Cumulative input summed across all model round-trips in peak run R11 (4 tool calls).
    peak_run_input_tokens: int = 47_305
    peak_run_cached_input_tokens: int = 43_899
    peak_run_invocations_est: int = 5
    # Peak single-invocation resident context on R11 (~11.1K on 5 invocations,
    # up to ~15.8K if compressed into 4 invocations).
    peak_single_invocation_tokens_est: int = 15_800
    peak_run_metered_output_tokens: int = 356
    metered_output_tokens_avg: int = 441
    # Multi-section write_artifact collaborator calls originally streamed outside
    # the main run meter (~6,909 chars / ~1,730 tokens on the peak 6-version run).
    unmetered_artifact_output_tokens_peak: int = 1_730
    headroom_multiplier: float = 2.0
    per_invocation_headroom_multiplier: float = 3.0
    target_concurrency: int = 16

    @property
    def tool_call_run_rate(self) -> float:
        return self.tool_calling_runs / self.runs_measured if self.runs_measured else 0.0

    @property
    def tool_calls_per_run_avg(self) -> float:
        return self.total_tool_calls / self.runs_measured if self.runs_measured else 0.0

    @property
    def tool_calls_per_tool_run_avg(self) -> float:
        return self.total_tool_calls / self.tool_calling_runs if self.tool_calling_runs else 0.0

    @property
    def peak_run_corrected_total_tokens(self) -> int:
        return (
            self.peak_run_input_tokens
            + self.peak_run_metered_output_tokens
            + self.unmetered_artifact_output_tokens_peak
        )

    def cumulative_heuristic_max_model_len(self, *, native_limit: int) -> int:
        """Provisional 2x cumulative-run heuristic (98,304 tokens for R11)."""
        raw_target = int(round(self.peak_run_corrected_total_tokens * self.headroom_multiplier))
        aligned = max(
            TOKEN_ALIGNMENT,
            int(round(raw_target / TOKEN_ALIGNMENT)) * TOKEN_ALIGNMENT,
        )
        return min(aligned, native_limit)

    def per_invocation_segmented_max_model_len(self, *, native_limit: int) -> int:
        """Segmented single-invocation target (~3x peak per-invocation residency -> 49,152)."""
        raw_target = int(
            round(self.peak_single_invocation_tokens_est * self.per_invocation_headroom_multiplier)
        )
        aligned = max(
            TOKEN_ALIGNMENT,
            int(round(raw_target / TOKEN_ALIGNMENT)) * TOKEN_ALIGNMENT,
        )
        return min(aligned, native_limit)


@dataclass(frozen=True)
class InstanceSpec:
    name: str
    gpu_model: str
    gpu_count: int
    vram_gib: float
    nominal_marketing_gb: int
    aws_spec_gib: float
    vcpus: int
    host_ram_gib: int
    hourly_usd_est: float


MODELS: dict[str, ModelSpec] = {
    "qwen3.6-35b-a3b-fp8": ModelSpec(
        key="qwen3.6-35b-a3b-fp8",
        repo_id="Qwen/Qwen3.6-35B-A3B-FP8",
        revision="c653bba46dfa06e9658dcc0b0dc87fe4685f14a2",
        precision="FP8 (weights) / BF16 (KV) / FP32 (DeltaNet state)",
        weights_gib=35.0,
        native_max_position_embeddings=262_144,
        recommended_complex_task_context=131_072,
        total_layers=40,
        full_attention_layers=10,
        kv_heads=2,
        head_dim=256,
        kv_dtype_bytes=2,
        linear_attention_layers=30,
        linear_key_heads=16,
        linear_key_head_dim=128,
        linear_value_heads=32,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        recurrent_dtype_bytes=4,
        conv_dtype_bytes=2,
        vision_tower_overhead_gib=8.0,
        runtime_workspace_gib=2.27,
        hybrid_page_overhead_factor=3.5526,
    ),
    "qwen3.6-35b-a3b-bf16": ModelSpec(
        key="qwen3.6-35b-a3b-bf16",
        repo_id="Qwen/Qwen3.6-35B-A3B",
        revision="main",
        precision="BF16 (weights & KV) / FP32 (DeltaNet state)",
        weights_gib=67.0,
        native_max_position_embeddings=262_144,
        recommended_complex_task_context=131_072,
        total_layers=40,
        full_attention_layers=10,
        kv_heads=2,
        head_dim=256,
        kv_dtype_bytes=2,
        linear_attention_layers=30,
        linear_key_heads=16,
        linear_key_head_dim=128,
        linear_value_heads=32,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        recurrent_dtype_bytes=4,
        conv_dtype_bytes=2,
        vision_tower_overhead_gib=8.0,
        runtime_workspace_gib=2.5,
        hybrid_page_overhead_factor=3.5526,
    ),
    "qwen3-8b-bf16": ModelSpec(
        key="qwen3-8b-bf16",
        repo_id="Qwen/Qwen3-8B",
        revision="b968826d9c46dd6066d109eabc6255188de91218",
        precision="BF16 (dense GQA reference from v0.1)",
        weights_gib=15.3,
        native_max_position_embeddings=32_768,
        recommended_complex_task_context=32_768,
        total_layers=36,
        full_attention_layers=36,
        kv_heads=8,
        head_dim=128,
        kv_dtype_bytes=2,
        linear_attention_layers=0,
        linear_key_heads=0,
        linear_key_head_dim=0,
        linear_value_heads=0,
        linear_value_head_dim=0,
        linear_conv_kernel_dim=0,
        recurrent_dtype_bytes=0,
        conv_dtype_bytes=0,
        vision_tower_overhead_gib=0.0,
        runtime_workspace_gib=1.8,
        hybrid_page_overhead_factor=1.0,
    ),
}


def build_instances(*, l40s_vram_gib: float = 44.99) -> tuple[InstanceSpec, ...]:
    """Build GPU instance specifications using physical device memory in GiB.

    AWS markets the L40S as 48 GB, lists ~44 GiB in the EC2 G6e instance
    specifications, and physical ECC-enabled device readings report ~44.99 GiB.
    Similarly, an A10G (24 GB marketing) reports ~22.20 GiB physical device
    memory (~23 GiB in EC2 G5 specs).
    """
    return (
        InstanceSpec(
            name="g5.xlarge",
            gpu_model="1x NVIDIA A10G",
            gpu_count=1,
            vram_gib=22.20,
            nominal_marketing_gb=24,
            aws_spec_gib=23.0,
            vcpus=4,
            host_ram_gib=16,
            hourly_usd_est=1.006,
        ),
        InstanceSpec(
            name="g6e.2xlarge",
            gpu_model="1x NVIDIA L40S",
            gpu_count=1,
            vram_gib=l40s_vram_gib,
            nominal_marketing_gb=48,
            aws_spec_gib=44.0,
            vcpus=8,
            host_ram_gib=64,
            hourly_usd_est=2.242,
        ),
        InstanceSpec(
            name="g6e.12xlarge",
            gpu_model="4x NVIDIA L40S",
            gpu_count=4,
            vram_gib=round(4 * l40s_vram_gib, 2),
            nominal_marketing_gb=192,
            aws_spec_gib=176.0,
            vcpus=48,
            host_ram_gib=384,
            hourly_usd_est=10.493,
        ),
    )


INSTANCES: tuple[InstanceSpec, ...] = build_instances()


@dataclass(frozen=True)
class InstanceEvaluation:
    instance: str
    gpu_model: str
    vram_gib: float
    nominal_marketing_gb: int
    aws_spec_gib: float
    usable_vram_gib: float
    weights_gib: float
    vision_overhead_gib: float
    runtime_workspace_gib: float
    total_cache_pool_gib: float
    deltanet_state_pool_gib: float
    paged_kv_pool_gib: float
    modeled_kv_token_capacity: int
    theoretical_raw_kv_token_capacity: int
    target_max_model_len: int
    fits_per_invocation_target: bool
    fits_cumulative_heuristic: bool
    fits_qwen_128k_recommendation: bool
    max_num_seqs: int
    fits: bool
    reason: str


def evaluate_instance(
    model: ModelSpec,
    instance: InstanceSpec,
    *,
    target_max_model_len: int,
    per_invocation_target: int = 49_152,
    cumulative_heuristic_target: int = 98_304,
    max_num_seqs: int,
    gpu_memory_utilization: float = 0.95,
    language_model_only: bool = True,
) -> InstanceEvaluation:
    """Compute the modeled VRAM budget and token capacity on a candidate GPU shape."""
    usable_vram_gib = round(instance.vram_gib * gpu_memory_utilization, 2)
    vision_overhead_gib = 0.0 if language_model_only else model.vision_tower_overhead_gib
    total_cache_pool_gib = round(
        usable_vram_gib
        - model.weights_gib
        - vision_overhead_gib
        - model.runtime_workspace_gib,
        2,
    )
    deltanet_state_pool_gib = round(
        (max_num_seqs * model.deltanet_state_bytes_per_sequence) / GIB,
        2,
    )

    if model.weights_gib > usable_vram_gib:
        return InstanceEvaluation(
            instance=instance.name,
            gpu_model=instance.gpu_model,
            vram_gib=instance.vram_gib,
            nominal_marketing_gb=instance.nominal_marketing_gb,
            aws_spec_gib=instance.aws_spec_gib,
            usable_vram_gib=round(usable_vram_gib, 2),
            weights_gib=model.weights_gib,
            vision_overhead_gib=vision_overhead_gib,
            runtime_workspace_gib=model.runtime_workspace_gib,
            total_cache_pool_gib=round(total_cache_pool_gib, 2),
            deltanet_state_pool_gib=round(deltanet_state_pool_gib, 2),
            paged_kv_pool_gib=0.0,
            modeled_kv_token_capacity=0,
            theoretical_raw_kv_token_capacity=0,
            target_max_model_len=target_max_model_len,
            fits_per_invocation_target=False,
            fits_cumulative_heuristic=False,
            fits_qwen_128k_recommendation=False,
            max_num_seqs=max_num_seqs,
            fits=False,
            reason=(
                f"weights alone ({model.weights_gib:.1f} GiB) exceed usable VRAM "
                f"({usable_vram_gib:.2f} GiB at {gpu_memory_utilization:.2f} utilization)"
            ),
        )

    paged_kv_pool_gib = round(total_cache_pool_gib - deltanet_state_pool_gib, 2)
    if paged_kv_pool_gib <= 0:
        return InstanceEvaluation(
            instance=instance.name,
            gpu_model=instance.gpu_model,
            vram_gib=instance.vram_gib,
            nominal_marketing_gb=instance.nominal_marketing_gb,
            aws_spec_gib=instance.aws_spec_gib,
            usable_vram_gib=round(usable_vram_gib, 2),
            weights_gib=model.weights_gib,
            vision_overhead_gib=vision_overhead_gib,
            runtime_workspace_gib=model.runtime_workspace_gib,
            total_cache_pool_gib=round(total_cache_pool_gib, 2),
            deltanet_state_pool_gib=round(deltanet_state_pool_gib, 2),
            paged_kv_pool_gib=round(paged_kv_pool_gib, 2),
            modeled_kv_token_capacity=0,
            theoretical_raw_kv_token_capacity=0,
            target_max_model_len=target_max_model_len,
            fits_per_invocation_target=False,
            fits_cumulative_heuristic=False,
            fits_qwen_128k_recommendation=False,
            max_num_seqs=max_num_seqs,
            fits=False,
            reason=(
                f"no VRAM left for paged KV cache after {deltanet_state_pool_gib:.2f} GiB "
                f"DeltaNet state ({max_num_seqs} seqs) and {vision_overhead_gib:.1f} GiB vision overhead"
            ),
        )

    modeled_tokens = int(
        (paged_kv_pool_gib * GIB) // model.effective_kv_bytes_per_token
    )
    modeled_tokens = (modeled_tokens // 16) * 16

    raw_tokens = int(
        (paged_kv_pool_gib * GIB) // model.full_attention_kv_bytes_per_token
    )
    raw_tokens = (raw_tokens // 16) * 16

    fits = modeled_tokens >= target_max_model_len
    fits_per_inv = modeled_tokens >= per_invocation_target
    fits_cum = modeled_tokens >= cumulative_heuristic_target
    fits_128k = modeled_tokens >= model.recommended_complex_task_context

    if fits:
        ratio = modeled_tokens / max(1, target_max_model_len)
        reason = (
            f"fits target ({target_max_model_len:,}): {paged_kv_pool_gib:.2f} GiB paged KV budget -> "
            f"~{modeled_tokens:,} modeled tokens ({ratio:.2f}x target; "
            f"98K cumulative heuristic: {'YES' if fits_cum else 'NO'}, "
            f"128K Qwen rec: {'YES' if fits_128k else 'NO'})"
        )
    else:
        reason = (
            f"modeled capacity (~{modeled_tokens:,} tokens in {paged_kv_pool_gib:.2f} GiB paged KV) "
            f"is below target max-model-len ({target_max_model_len:,} tokens)"
        )

    return InstanceEvaluation(
        instance=instance.name,
        gpu_model=instance.gpu_model,
        vram_gib=instance.vram_gib,
        nominal_marketing_gb=instance.nominal_marketing_gb,
        aws_spec_gib=instance.aws_spec_gib,
        usable_vram_gib=round(usable_vram_gib, 2),
        weights_gib=model.weights_gib,
        vision_overhead_gib=vision_overhead_gib,
        runtime_workspace_gib=model.runtime_workspace_gib,
        total_cache_pool_gib=round(total_cache_pool_gib, 2),
        deltanet_state_pool_gib=round(deltanet_state_pool_gib, 2),
        paged_kv_pool_gib=round(paged_kv_pool_gib, 2),
        modeled_kv_token_capacity=modeled_tokens,
        theoretical_raw_kv_token_capacity=raw_tokens,
        target_max_model_len=target_max_model_len,
        fits_per_invocation_target=fits_per_inv,
        fits_cumulative_heuristic=fits_cum,
        fits_qwen_128k_recommendation=fits_128k,
        max_num_seqs=max_num_seqs,
        fits=fits,
        reason=reason,
    )


def build_sizing_report(
    *,
    model_key: str = "qwen3.6-35b-a3b-fp8",
    baseline: WorkloadBaseline | None = None,
    gpu_memory_utilization: float = 0.95,
    language_model_only: bool = True,
    l40s_vram_gib: float = 44.99,
    sizing_mode: str = "per-invocation",
) -> dict:
    """Build the structured sizing report for a model, GPU reading, and sizing mode."""
    if model_key not in MODELS:
        raise KeyError(f"Unknown model {model_key!r}; choose from {sorted(MODELS)}")
    model = MODELS[model_key]
    wb = baseline or WorkloadBaseline()

    per_invocation_target = wb.per_invocation_segmented_max_model_len(
        native_limit=model.native_max_position_embeddings,
    )
    cumulative_heuristic_target = wb.cumulative_heuristic_max_model_len(
        native_limit=model.native_max_position_embeddings,
    )
    qwen_128k_target = model.recommended_complex_task_context

    if sizing_mode == "per-invocation":
        target_max_model_len = per_invocation_target
    elif sizing_mode == "cumulative-2x":
        target_max_model_len = cumulative_heuristic_target
    elif sizing_mode == "qwen-128k":
        target_max_model_len = qwen_128k_target
    else:
        raise ValueError(
            f"Unknown sizing_mode {sizing_mode!r}; choose from "
            "('per-invocation', 'cumulative-2x', 'qwen-128k')"
        )

    instances = build_instances(l40s_vram_gib=l40s_vram_gib)
    evaluations = [
        evaluate_instance(
            model,
            inst,
            target_max_model_len=target_max_model_len,
            per_invocation_target=per_invocation_target,
            cumulative_heuristic_target=cumulative_heuristic_target,
            max_num_seqs=wb.target_concurrency,
            gpu_memory_utilization=gpu_memory_utilization,
            language_model_only=language_model_only,
        )
        for inst in instances
    ]
    recommended = next((ev for ev in evaluations if ev.fits), None)

    vllm_flags = [
        f"--model {model.repo_id}",
        f"--max-model-len {target_max_model_len}",
        f"--max-num-seqs {wb.target_concurrency}",
        f"--gpu-memory-utilization {gpu_memory_utilization:.2f}",
        "--trust-remote-code",
        f"--reasoning-parser {model.reasoning_parser}",
        "--enable-auto-tool-choice",
        f"--tool-call-parser {model.tool_call_parser}",
        "--no-enable-log-requests",
    ]
    if language_model_only and model.vision_tower_overhead_gib > 0:
        vllm_flags.append("--language-model-only")

    return {
        "model": {
            **asdict(model),
            "full_attention_kv_kib_per_token": round(
                model.full_attention_kv_bytes_per_token / KIB, 2
            ),
            "effective_hybrid_kv_kib_per_token": round(
                model.effective_kv_bytes_per_token / KIB, 2
            ),
            "deltanet_state_mib_per_sequence": round(
                model.deltanet_state_bytes_per_sequence / MIB, 2
            ),
        },
        "baseline": {
            **asdict(wb),
            "tool_call_run_rate": round(wb.tool_call_run_rate, 3),
            "tool_calls_per_run_avg": round(wb.tool_calls_per_run_avg, 2),
            "tool_calls_per_tool_run_avg": round(wb.tool_calls_per_tool_run_avg, 2),
            "peak_run_corrected_total_tokens": wb.peak_run_corrected_total_tokens,
        },
        "sizing": {
            "sizing_mode": sizing_mode,
            "l40s_vram_gib": l40s_vram_gib,
            "target_max_model_len": target_max_model_len,
            "per_invocation_segmented_target": per_invocation_target,
            "cumulative_heuristic_target": cumulative_heuristic_target,
            "qwen_complex_task_recommended_target": qwen_128k_target,
            "target_max_num_seqs": wb.target_concurrency,
            "gpu_memory_utilization": gpu_memory_utilization,
            "language_model_only": language_model_only,
            "recommended_instance": recommended.instance if recommended else None,
            "vllm_flags": vllm_flags,
        },
        "instances": [asdict(ev) for ev in evaluations],
    }


def format_report(report: dict) -> str:
    m = report["model"]
    b = report["baseline"]
    s = report["sizing"]
    lines = [
        f"=== Hybrid KV-Cache & Instance Sizing (Planning Estimate): {m['repo_id']} ===",
        f"Revision:   {m['revision']}",
        f"Precision:  {m['precision']}",
        f"Weights:    {m['weights_gib']:.2f} GiB  |  Native context: {m['native_max_position_embeddings']:,} tokens",
        "",
        "1. Architecture memory profile",
        f"   Layers:                {m['total_layers']} total = "
        f"{m['linear_attention_layers']} Gated DeltaNet (linear) + "
        f"{m['full_attention_layers']} full-attention (GQA)",
        f"   Full-attention KV:     {m['full_attention_kv_kib_per_token']:.1f} KiB / token theoretical "
        f"(2 * {m['full_attention_layers']} layers * {m['kv_heads']} KV heads * "
        f"{m['head_dim']} dim * {m['kv_dtype_bytes']} B)",
        f"   Hybrid-page effective: {m['effective_hybrid_kv_kib_per_token']:.2f} KiB / token "
        f"(provisional {m['hybrid_page_overhead_factor']:.4f}x calibration; reconcile with vLLM startup logs)",
        f"   DeltaNet recurrent:    {m['deltanet_state_mib_per_sequence']:.2f} MiB / active sequence "
        f"(independent of context length; scales with --max-num-seqs)",
        f"   At --max-num-seqs {s['target_max_num_seqs']:<3d}: "
        f"{(m['deltanet_state_mib_per_sequence'] * s['target_max_num_seqs']) / 1024:.2f} GiB reserved for recurrent state slots",
        "",
        f"2. Workload baseline ({b['runs_measured']} test-env runs across {b['conversations_measured']} conversations)",
        f"   Tool-calling runs:     {b['tool_calling_runs']}/{b['runs_measured']} "
        f"({b['tool_call_run_rate'] * 100:.0f}%)  |  {b['total_tool_calls']} tool calls "
        f"({b['tool_calls_per_run_avg']:.1f}/run overall, {b['tool_calls_per_tool_run_avg']:.1f}/tool-using run)",
        f"   Cumulative input avg:  {b['no_tool_input_tokens_avg']:>8,} (no-tool, {b['no_tool_cache_hit_rate'] * 100:.1f}% cached)  |  "
        f"{b['tool_run_input_tokens_avg']:,} (tool-using, {b['tool_run_cache_hit_rate'] * 100:.1f}% cached)",
        f"   Peak run (R11) total:  {b['peak_run_input_tokens']:>8,} cumulative input + "
        f"{b['peak_run_metered_output_tokens']:,} metered output + "
        f"~{b['unmetered_artifact_output_tokens_peak']:,} artifact collaborator output "
        f"= {b['peak_run_corrected_total_tokens']:,} cumulative tokens",
        f"   Peak single-call est:  ~{b['peak_single_invocation_tokens_est']:>7,} resident tokens on one invocation "
        f"(R11 spans ~{b['peak_run_invocations_est']} model calls; 92.8% of cumulative input is cached prefix)",
        "",
        "3. Candidate context targets vs. single-L40S physical memory "
        f"({s['l40s_vram_gib']:.2f} GiB device / ~44 GiB AWS G6e spec / 48 GB marketing)",
        f"   * per-invocation ({s['per_invocation_segmented_target']:>7,} tokens): "
        "~3.0x peak single-call residency for segmented agent workflows",
        f"   * cumulative-2x  ({s['cumulative_heuristic_target']:>7,} tokens): "
        "provisional 2.0x cumulative run total (overstates single-call residency; exceeds 1x L40S at 44.99 GiB)",
        f"   * qwen-128k      ({s['qwen_complex_task_recommended_target']:>7,} tokens): "
        "Qwen model-card recommendation for complex unsegmented tasks (requires multi-GPU or KV quantization)",
        f"   Active sizing mode:    {s['sizing_mode']} -> target --max-model-len {s['target_max_model_len']:,}",
        "",
        "4. Modeled instance evaluation (provisional planning estimates, not measured vLLM limits)",
    ]
    for ev in report["instances"]:
        verdict = "FIT " if ev["fits"] else "FAIL"
        lines.append(
            f"   [{verdict}] {ev['instance']:<12} ({ev['gpu_model']}, {ev['vram_gib']:.2f} GiB physical / "
            f"{ev['aws_spec_gib']:.0f} GiB spec / {ev['nominal_marketing_gb']} GB nominal) -> {ev['reason']}"
        )
    lines.extend([
        "",
        f"5. Recommended configuration for mode '{s['sizing_mode']}' "
        f"(instance: {s['recommended_instance'] or 'none'})",
        "   " + " \\\n     ".join(s["vllm_flags"]),
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Size vLLM hybrid KV/DeltaNet cache and GPU instance from measured workload baseline.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODELS.keys()),
        default="qwen3.6-35b-a3b-fp8",
        help="Model checkpoint profile to size (default: qwen3.6-35b-a3b-fp8).",
    )
    parser.add_argument(
        "--sizing-mode",
        choices=("per-invocation", "cumulative-2x", "qwen-128k"),
        default="per-invocation",
        help=(
            "Context target mode: 'per-invocation' (49,152 segmented single-call target), "
            "'cumulative-2x' (98,304 provisional cumulative-run heuristic), or "
            "'qwen-128k' (131,072 Qwen complex-task recommendation)."
        ),
    )
    parser.add_argument(
        "--l40s-vram-gib",
        type=float,
        default=44.99,
        help=(
            "Physical L40S GPU memory in GiB (default: 44.99 physical device reading; "
            "use 44.0 for AWS G6e spec or 48.0 to reproduce the unadjusted nominal estimate)."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Target concurrent decode sequences (--max-num-seqs, default: 16).",
    )
    parser.add_argument(
        "--peak-input-tokens",
        type=int,
        default=47_305,
        help="Peak cumulative input tokens observed in a single run (default: 47305 from R11).",
    )
    parser.add_argument(
        "--headroom",
        type=float,
        default=2.0,
        help="Headroom multiplier for cumulative-2x mode (default: 2.0).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="vLLM --gpu-memory-utilization fraction (default: 0.95).",
    )
    parser.add_argument(
        "--include-vision-tower",
        action="store_true",
        help="Do not pass --language-model-only (keeps ~8 GiB vision encoder overhead).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON instead of the human-readable report.",
    )
    args = parser.parse_args(argv)

    baseline = WorkloadBaseline(
        peak_run_input_tokens=args.peak_input_tokens,
        headroom_multiplier=args.headroom,
        target_concurrency=args.concurrency,
    )
    report = build_sizing_report(
        model_key=args.model,
        baseline=baseline,
        gpu_memory_utilization=args.gpu_memory_utilization,
        language_model_only=not args.include_vision_tower,
        l40s_vram_gib=args.l40s_vram_gib,
        sizing_mode=args.sizing_mode,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
