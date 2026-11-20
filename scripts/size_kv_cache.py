"""Hybrid KV-cache and GPU instance sizer fed by measured workload baselines.

Why this script exists:
  Standard transformer KV-cache calculators assume every layer stores a K and V
  tensor per token:

      KV bytes/token = 2 * num_layers * num_kv_heads * head_dim * kv_bytes

  Qwen/Qwen3.6-35B-A3B (architecture ``Qwen3_5MoeForConditionalGeneration``)
  breaks that assumption in three ways:
    1. Hybrid attention (30 Gated DeltaNet + 10 full-attention GQA layers).
       Only 10 of the 40 layers (``full_attention_interval = 4``) allocate a
       per-token KV cache (2 KV heads, head_dim 256 -> 20 KiB/token in BF16).
    2. Fixed recurrent state per concurrent sequence (``--max-num-seqs``).
       The 30 Gated DeltaNet (``linear_attention``) layers maintain a fixed-size
       float32 recurrent state matrix ``(32, 128, 128)`` (2.0 MiB/layer) plus a
       short causal conv state ``(8192, 3)`` in BF16 (48 KiB/layer) per active
       sequence -- ~61.4 MiB per sequence slot across 30 layers regardless of
       whether a prompt is 500 tokens or 100,000 tokens. vLLM allocates these
       recurrent state blocks per ``--max-num-seqs`` out of the same unified
       GPU memory pool as the paged KV cache.
    3. Multimodal vision tower present in the checkpoint.
       Unless ``--language-model-only`` is passed, vLLM loads the vision encoder
       and profiles a 16K-token vision cache (~8.0 GiB VRAM), starving the text
       KV pool on a 48 GB GPU.

  This script takes the aggregate numbers from our test-environment baseline
  (input tokens per run, prompt-cache hit rate, tool-call frequency, and the
  corrected output-token ceiling after accounting for multi-section artifact
  collaborator writes) and computes:
    - Per-token full-attention KV cost and per-sequence DeltaNet recurrent cost
    - Recommended ``--max-model-len`` and ``--max-num-seqs``
    - Which single-GPU / multi-GPU instance shapes fit the model + cache
    - Ready-to-paste vLLM serve flags

Usage:
    python scripts/size_kv_cache.py
    python scripts/size_kv_cache.py --concurrency 16 --model qwen3.6-35b-a3b-fp8
    python scripts/size_kv_cache.py --json
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
    """Architectural parameters pulled directly from the Hugging Face config.json."""

    key: str
    repo_id: str
    revision: str
    precision: str
    weights_gib: float
    native_max_position_embeddings: int
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
    # inside vLLM's allocated budget under --enforce-eager.
    runtime_workspace_gib: float
    # vLLM's unified hybrid KV/Mamba cache manager aligns page blocks across the
    # full-attention and Gated DeltaNet groups; empirically on vLLM 0.19 with
    # Qwen3.6-35B-A3B-FP8 on an L40S, 8.33 GiB of available KV memory yields
    # 108,768 tokens at max_num_seqs=16.
    hybrid_page_overhead_factor: float = 3.5526
    tool_call_parser: str = "qwen3_coder"
    reasoning_parser: str = "qwen3"

    @property
    def full_attention_kv_bytes_per_token(self) -> int:
        """Raw K + V bytes per token across the full-attention (GQA) layers only."""
        return (
            2
            * self.full_attention_layers
            * self.kv_heads
            * self.head_dim
            * self.kv_dtype_bytes
        )

    @property
    def effective_kv_bytes_per_token(self) -> int:
        """Effective paged-pool bytes per token including vLLM hybrid block alignment."""
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
    """Aggregate measurements from our 11-run test-environment teacher baseline.

    No raw prompts, user identifiers, or conversation transcripts are stored
    here -- only the summary statistics needed to size context and concurrency.
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
    peak_run_input_tokens: int = 47_305
    peak_run_metered_output_tokens: int = 356
    metered_output_tokens_avg: int = 441
    # Multi-section artifact collaborator calls originally streamed outside the
    # main run meter (~6,909 chars / ~1,730 tokens on the peak 6-version run).
    unmetered_artifact_output_tokens_peak: int = 1_730
    # Multiplier applied to peak single-run tokens to allow multi-turn growth
    # and background batch year-plan runs without hitting truncation.
    headroom_multiplier: float = 2.0
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

    def recommended_max_model_len(self, *, native_limit: int) -> int:
        """Target --max-model-len aligned to a 16K boundary, capped at native_limit."""
        raw_target = int(round(self.peak_run_corrected_total_tokens * self.headroom_multiplier))
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

INSTANCES: tuple[InstanceSpec, ...] = (
    InstanceSpec(
        name="g5.xlarge",
        gpu_model="1x NVIDIA A10G",
        gpu_count=1,
        vram_gib=24.0,
        vcpus=4,
        host_ram_gib=16,
        hourly_usd_est=1.006,
    ),
    InstanceSpec(
        name="g6e.2xlarge",
        gpu_model="1x NVIDIA L40S",
        gpu_count=1,
        vram_gib=48.0,
        vcpus=8,
        host_ram_gib=64,
        hourly_usd_est=2.242,
    ),
    InstanceSpec(
        name="g6e.12xlarge",
        gpu_model="4x NVIDIA L40S",
        gpu_count=4,
        vram_gib=192.0,
        vcpus=48,
        host_ram_gib=384,
        hourly_usd_est=10.493,
    ),
)


@dataclass(frozen=True)
class InstanceEvaluation:
    instance: str
    gpu_model: str
    vram_gib: float
    usable_vram_gib: float
    weights_gib: float
    vision_overhead_gib: float
    runtime_workspace_gib: float
    total_cache_pool_gib: float
    deltanet_state_pool_gib: float
    paged_kv_pool_gib: float
    physical_kv_token_capacity: int
    target_max_model_len: int
    max_num_seqs: int
    fits: bool
    reason: str


def evaluate_instance(
    model: ModelSpec,
    instance: InstanceSpec,
    *,
    target_max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float = 0.95,
    language_model_only: bool = True,
) -> InstanceEvaluation:
    """Compute the VRAM budget and physical token capacity on a candidate GPU shape."""
    usable_vram_gib = instance.vram_gib * gpu_memory_utilization
    vision_overhead_gib = 0.0 if language_model_only else model.vision_tower_overhead_gib
    total_cache_pool_gib = (
        usable_vram_gib
        - model.weights_gib
        - vision_overhead_gib
        - model.runtime_workspace_gib
    )
    deltanet_state_pool_gib = (
        max_num_seqs * model.deltanet_state_bytes_per_sequence
    ) / GIB

    if model.weights_gib > usable_vram_gib:
        return InstanceEvaluation(
            instance=instance.name,
            gpu_model=instance.gpu_model,
            vram_gib=instance.vram_gib,
            usable_vram_gib=round(usable_vram_gib, 2),
            weights_gib=model.weights_gib,
            vision_overhead_gib=vision_overhead_gib,
            runtime_workspace_gib=model.runtime_workspace_gib,
            total_cache_pool_gib=round(total_cache_pool_gib, 2),
            deltanet_state_pool_gib=round(deltanet_state_pool_gib, 2),
            paged_kv_pool_gib=0.0,
            physical_kv_token_capacity=0,
            target_max_model_len=target_max_model_len,
            max_num_seqs=max_num_seqs,
            fits=False,
            reason=(
                f"weights alone ({model.weights_gib:.1f} GiB) exceed usable VRAM "
                f"({usable_vram_gib:.1f} GiB at {gpu_memory_utilization:.2f} utilization)"
            ),
        )

    paged_kv_pool_gib = total_cache_pool_gib - deltanet_state_pool_gib
    if paged_kv_pool_gib <= 0:
        return InstanceEvaluation(
            instance=instance.name,
            gpu_model=instance.gpu_model,
            vram_gib=instance.vram_gib,
            usable_vram_gib=round(usable_vram_gib, 2),
            weights_gib=model.weights_gib,
            vision_overhead_gib=vision_overhead_gib,
            runtime_workspace_gib=model.runtime_workspace_gib,
            total_cache_pool_gib=round(total_cache_pool_gib, 2),
            deltanet_state_pool_gib=round(deltanet_state_pool_gib, 2),
            paged_kv_pool_gib=round(paged_kv_pool_gib, 2),
            physical_kv_token_capacity=0,
            target_max_model_len=target_max_model_len,
            max_num_seqs=max_num_seqs,
            fits=False,
            reason=(
                f"no VRAM left for paged KV cache after {deltanet_state_pool_gib:.2f} GiB "
                f"DeltaNet state ({max_num_seqs} seqs) and {vision_overhead_gib:.1f} GiB vision overhead"
            ),
        )

    physical_tokens = int(
        (paged_kv_pool_gib * GIB) // model.effective_kv_bytes_per_token
    )
    # Round to vLLM's 16-token block size.
    physical_tokens = (physical_tokens // 16) * 16
    fits = physical_tokens >= target_max_model_len
    if fits:
        concurrency_at_peak = physical_tokens / max(1, target_max_model_len)
        reason = (
            f"fits: {total_cache_pool_gib:.2f} GiB cache pool -> "
            f"{physical_tokens:,} physical token capacity "
            f"({concurrency_at_peak:.1f}x target max-model-len {target_max_model_len:,})"
        )
    else:
        reason = (
            f"KV capacity ({physical_tokens:,} tokens) is below target "
            f"max-model-len ({target_max_model_len:,} tokens)"
        )

    return InstanceEvaluation(
        instance=instance.name,
        gpu_model=instance.gpu_model,
        vram_gib=instance.vram_gib,
        usable_vram_gib=round(usable_vram_gib, 2),
        weights_gib=model.weights_gib,
        vision_overhead_gib=vision_overhead_gib,
        runtime_workspace_gib=model.runtime_workspace_gib,
        total_cache_pool_gib=round(total_cache_pool_gib, 2),
        deltanet_state_pool_gib=round(deltanet_state_pool_gib, 2),
        paged_kv_pool_gib=round(paged_kv_pool_gib, 2),
        physical_kv_token_capacity=physical_tokens,
        target_max_model_len=target_max_model_len,
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
) -> dict:
    """Build the full structured sizing report for a model and workload baseline."""
    if model_key not in MODELS:
        raise KeyError(f"Unknown model {model_key!r}; choose from {sorted(MODELS)}")
    model = MODELS[model_key]
    wb = baseline or WorkloadBaseline()
    target_max_model_len = wb.recommended_max_model_len(
        native_limit=model.native_max_position_embeddings,
    )

    evaluations = [
        evaluate_instance(
            model,
            inst,
            target_max_model_len=target_max_model_len,
            max_num_seqs=wb.target_concurrency,
            gpu_memory_utilization=gpu_memory_utilization,
            language_model_only=language_model_only,
        )
        for inst in INSTANCES
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
            "target_max_model_len": target_max_model_len,
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
        f"=== Hybrid KV-Cache & Instance Sizing: {m['repo_id']} ===",
        f"Revision:   {m['revision']}",
        f"Precision:  {m['precision']}",
        f"Weights:    {m['weights_gib']:.1f} GiB  |  Native context: {m['native_max_position_embeddings']:,} tokens",
        "",
        "1. Architecture memory profile",
        f"   Layers:                {m['total_layers']} total = "
        f"{m['linear_attention_layers']} Gated DeltaNet (linear) + "
        f"{m['full_attention_layers']} full-attention (GQA)",
        f"   Full-attention KV:     {m['full_attention_kv_kib_per_token']:.1f} KiB / token "
        f"(2 * {m['full_attention_layers']} layers * {m['kv_heads']} KV heads * "
        f"{m['head_dim']} dim * {m['kv_dtype_bytes']} B)",
        f"   DeltaNet recurrent:    {m['deltanet_state_mib_per_sequence']:.2f} MiB / active sequence "
        f"(independent of context length; scales with --max-num-seqs)",
        f"   At --max-num-seqs {s['target_max_num_seqs']:<3d}: "
        f"{(m['deltanet_state_mib_per_sequence'] * s['target_max_num_seqs']) / 1024:.2f} GiB reserved for recurrent state slots",
        "",
        f"2. Workload baseline ({b['runs_measured']} test-env runs across {b['conversations_measured']} conversations)",
        f"   Tool-calling runs:     {b['tool_calling_runs']}/{b['runs_measured']} "
        f"({b['tool_call_run_rate'] * 100:.0f}%)  |  {b['total_tool_calls']} tool calls "
        f"({b['tool_calls_per_run_avg']:.1f}/run overall, {b['tool_calls_per_tool_run_avg']:.1f}/tool-using run)",
        f"   Input tokens (no-tool):{b['no_tool_input_tokens_avg']:>8,} avg  "
        f"({b['no_tool_cache_hit_rate'] * 100:.1f}% prompt-cached)",
        f"   Input tokens (tools):  {b['tool_run_input_tokens_avg']:>8,} avg  "
        f"({b['tool_run_cache_hit_rate'] * 100:.1f}% prompt-cached; ~4.0x no-tool baseline)",
        f"   Input tokens (artifact):{b['artifact_run_input_tokens_avg']:>7,} avg  "
        f"(vs {b['non_artifact_tool_run_input_tokens_avg']:,} on other tool-using runs)",
        f"   Peak measured run:     {b['peak_run_input_tokens']:>8,} input tokens + "
        f"{b['peak_run_metered_output_tokens']:,} metered output + "
        f"~{b['unmetered_artifact_output_tokens_peak']:,} artifact collaborator tokens "
        f"= {b['peak_run_corrected_total_tokens']:,} total",
        f"   Target --max-model-len:{s['target_max_model_len']:>8,} tokens "
        f"({b['headroom_multiplier']:.1f}x peak run, aligned to {TOKEN_ALIGNMENT:,}-token boundary; "
        f"{s['target_max_model_len'] / m['native_max_position_embeddings'] * 100:.1f}% of native 262K)",
        "",
        "3. Instance evaluation",
    ]
    for ev in report["instances"]:
        verdict = "FIT " if ev["fits"] else "FAIL"
        lines.append(
            f"   [{verdict}] {ev['instance']:<12} ({ev['gpu_model']}, {ev['vram_gib']:.0f} GiB VRAM) -> {ev['reason']}"
        )
    lines.extend([
        "",
        f"4. Recommended configuration (instance: {s['recommended_instance'] or 'none'})",
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
        "--concurrency",
        type=int,
        default=16,
        help="Target concurrent decode sequences (--max-num-seqs, default: 16).",
    )
    parser.add_argument(
        "--peak-input-tokens",
        type=int,
        default=47_305,
        help="Peak input tokens observed in a single run (default: 47305 from R11).",
    )
    parser.add_argument(
        "--headroom",
        type=float,
        default=2.0,
        help="Headroom multiplier above peak single-run tokens (default: 2.0).",
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
        help="Emit structured JSON instead of the human-readable table.",
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
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
