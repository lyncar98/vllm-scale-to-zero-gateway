"""Hybrid KV-cache and GPU instance sizing math against the measured baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import size_kv_cache  # noqa: E402


def test_qwen36_architecture_separates_full_attention_kv_from_deltanet_recurrent_state():
    model = size_kv_cache.MODELS["qwen3.6-35b-a3b-fp8"]
    # 10 full-attention layers * 2 (K,V) * 2 KV heads * 256 head_dim * 2 bytes = 20,480 bytes (20 KiB/token)
    assert model.full_attention_kv_bytes_per_token == 20 * 1024
    # 30 Gated DeltaNet layers * (2 MiB recurrent + 48 KiB conv) = 61.41 MiB per active sequence
    assert round(model.deltanet_state_bytes_per_sequence / (1024**2), 2) == 61.41


def test_qwen3_8b_reference_matches_dense_formula():
    model = size_kv_cache.MODELS["qwen3-8b-bf16"]
    # 36 layers * 2 (K,V) * 8 KV heads * 128 head_dim * 2 bytes = 144 KiB/token
    assert model.full_attention_kv_bytes_per_token == 144 * 1024
    assert model.deltanet_state_bytes_per_sequence == 0


def test_baseline_derives_98304_max_model_len_and_picks_g6e_2xlarge():
    report = size_kv_cache.build_sizing_report()
    assert report["baseline"]["tool_call_run_rate"] == 0.818
    assert report["baseline"]["peak_run_input_tokens"] == 47_305
    assert report["sizing"]["target_max_model_len"] == 98_304
    assert report["sizing"]["recommended_instance"] == "g6e.2xlarge"

    by_name = {inst["instance"]: inst for inst in report["instances"]}
    assert not by_name["g5.xlarge"]["fits"]
    assert by_name["g6e.2xlarge"]["fits"]
    assert by_name["g6e.2xlarge"]["physical_kv_token_capacity"] == 108_768


def test_omitting_language_model_only_or_leaving_default_256_seqs_breaks_single_l40s():
    # Without --language-model-only, the ~8 GiB vision encoder profiling cache eats the KV budget.
    with_vision = size_kv_cache.build_sizing_report(language_model_only=False)
    by_name_vision = {inst["instance"]: inst for inst in with_vision["instances"]}
    assert not by_name_vision["g6e.2xlarge"]["fits"]

    # With vLLM's default --max-num-seqs 256, 30 DeltaNet layers demand ~15.35 GiB of recurrent state alone.
    high_seqs = size_kv_cache.build_sizing_report(
        baseline=size_kv_cache.WorkloadBaseline(target_concurrency=256),
    )
    by_name_seqs = {inst["instance"]: inst for inst in high_seqs["instances"]}
    assert not by_name_seqs["g6e.2xlarge"]["fits"]


def test_bf16_checkpoint_does_not_fit_single_l40s():
    bf16_report = size_kv_cache.build_sizing_report(model_key="qwen3.6-35b-a3b-bf16")
    by_name = {inst["instance"]: inst for inst in bf16_report["instances"]}
    assert not by_name["g5.xlarge"]["fits"]
    assert not by_name["g6e.2xlarge"]["fits"]
    assert by_name["g6e.12xlarge"]["fits"]


def test_cli_json_output_is_valid(capsys):
    rc = size_kv_cache.main(["--json", "--concurrency", "16"])
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["sizing"]["recommended_instance"] == "g6e.2xlarge"
    assert "--tool-call-parser qwen3_coder" in doc["sizing"]["vllm_flags"]
