# Sizing for a Hybrid Model: Why Our v0.1 KV-Cache Doc Was Wrong

In `v0.1` of this repo, this document opened with the standard transformer
KV-cache formula:

```
KV bytes per token = 2 (K and V) × layers × KV heads × head_dim × bytes per element
```

and walked through a worked example for an 8B dense model (`36` layers, `8` KV
heads, `head_dim = 128`, `144 KiB/token` in BF16) on a 24 GB `g5.xlarge`
(NVIDIA A10G).

Then we picked our actual serving model — **`Qwen/Qwen3.6-35B-A3B`** — and
discovered that almost every assumption in that formula fails on it. Fixing the
math in public is worth doing because hybrid linear-attention + MoE checkpoints
are becoming the default workhorse class for long-context agents, and sizing
them with a dense GQA spreadsheet will either over-provision by 3&times; or
crash vLLM at startup.

Run [`scripts/size_kv_cache.py`](../scripts/size_kv_cache.py) to execute the
exact calculations below against our baseline or your own numbers.

---

## 1. Three things the v0.1 formula got wrong for `Qwen3.6-35B-A3B`

Inspecting [`config.json`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/blob/c653bba46dfa06e9658dcc0b0dc87fe4685f14a2/config.json)
at our pinned revision (`c653bba46dfa06e9658dcc0b0dc87fe4685f14a2`) reveals
three structural differences:

### A. 30 of the 40 layers are Gated DeltaNet, not standard attention

`Qwen3.6-35B-A3B` uses architecture `Qwen3_5MoeForConditionalGeneration` with
`num_hidden_layers = 40` and `full_attention_interval = 4`. Only every fourth
layer uses standard grouped-query attention (GQA); the other three out of four
layers use **Gated DeltaNet** (`linear_attention`):

| Layer group | Layer count | Config parameters | How memory scales |
|---|---:|---|---|
| **Full-attention (`full_attention`, GQA)** | **10** | `num_key_value_heads = 2`, `head_dim = 256`, BF16 (`2` B) | Scales linearly with **token count** (`20 KiB / token`) |
| **Gated DeltaNet (`linear_attention`)** | **30** | `linear_num_key_heads = 16` (`dim 128`), `linear_num_value_heads = 32` (`dim 128`), `linear_conv_kernel_dim = 4`, `mamba_ssm_dtype = float32` | Fixed **per active sequence** (`61.41 MiB / sequence`), **independent of context length** |

For the 10 full-attention layers, the raw per-token KV cache is:

```
2 (K and V) × 10 full-attention layers × 2 KV heads × 256 head_dim × 2 bytes
  = 20,480 bytes (20.0 KiB) per token
```

That is **7.2&times; smaller per token** than the 8B dense model in our `v0.1`
doc (`144 KiB/token`).

### B. Concurrency (`--max-num-seqs`) is now a primary VRAM knob

For the 30 Gated DeltaNet layers, there is no per-token KV cache. Instead, each
layer maintains a fixed recurrent state tensor plus a short causal convolution
state for every active decode sequence:

1. **Recurrent state matrix**:
   `32 value_heads × 128 key_dim × 128 value_dim × 4 bytes (float32) = 2,097,152 bytes (2.0 MiB) per layer`
2. **Causal conv1d state (`kernel_dim - 1 = 3` steps across Q, K, V channels)**:
   `(2 × 16 × 128 + 32 × 128) × 3 × 2 bytes (BF16) = 49,152 bytes (48 KiB) per layer`

Across all 30 Gated DeltaNet layers:

```
30 layers × (2.0 MiB + 48 KiB) = 61.41 MiB per active sequence
```

In vLLM (`>= 0.17.0`), the unified hybrid KV/Mamba cache manager allocates one
recurrent cache block per concurrent sequence slot (`--max-num-seqs`) out of the
same GPU memory pool as paged attention blocks, **regardless of whether that
sequence is 500 tokens or 100,000 tokens long**:

- At vLLM's default `--max-num-seqs 256`, the 30 Gated DeltaNet layers demand
  `256 × 61.41 MiB ≈ 15.35 GiB` of recurrent state alone — **nearly twice the
  entire free cache budget on a 48 GB L40S**, causing the server to fail during
  cache initialization before serving a single token.
- Capping `--max-num-seqs 16` drops the fixed recurrent state reservation to
  `16 × 61.41 MiB ≈ 0.96 GiB`, leaving the rest of the pool for paged KV blocks.

### C. Sparse MoE weights + an 8 GiB multimodal vision tower

`Qwen3.6-35B-A3B` is a Mixture-of-Experts model (`256` routed experts, `8`
active per token + `1` shared expert, ~`3B` active parameters per token). Fast
decode comes from the `3B` active parameters, but VRAM has to hold **all `35B`
parameters**:

- **BF16 (`Qwen/Qwen3.6-35B-A3B`)**: `71.9 GB` on disk (`~67.0 GiB` in VRAM).
  Cannot fit on a 24 GB `g5.xlarge` (A10G) or a 48 GB `g6e.2xlarge` (L40S). We
  keep the BF16 checkpoint in object storage as our governed fine-tuning base.
- **FP8 (`Qwen/Qwen3.6-35B-A3B-FP8`)**: `37.5 GB` on disk (`~35.0 GiB` in VRAM).
  Still too large for the 24 GB `g5.xlarge` in our `v0.1` Terraform defaults,
  which forced the move to a 48 GB L40S (`g6e.2xlarge`).

One more surprise hid inside `Qwen3_5MoeForConditionalGeneration`: the
checkpoint ships with a multimodal `vision_config` tower. Without extra flags,
vLLM loads the vision encoder and profiles a 16K-token vision cache at startup,
consuming **~8.0 GiB of VRAM**. Passing `--language-model-only` skips the vision
encoder entirely and hands that ~8.0 GiB back to the text KV cache pool — the
difference between starving on a 48 GB L40S and having **8.33 GiB** of free
hybrid cache (`108,768` tokens of physical KV capacity).

---

## 2. How our test-environment baseline sets `--max-model-len`

`Qwen3.6-35B-A3B` natively supports `262,144` tokens (`max_position_embeddings`).
Setting `--max-model-len 262144` on a single 48 GB GPU is neither possible nor
useful; the right ceiling comes from measuring actual agent runs.

From our test-environment baseline (`11` teacher chat runs across `7`
conversations, `62` metered model and embedding calls):

| Metric | Measured baseline | Sizing implication |
|---|---:|---|
| **Runs making tool calls** | `9 / 11` (**82%**) | `30` total tool calls (`2.7`/run overall, `3.3`/tool-using run). Requires `--enable-auto-tool-choice --tool-call-parser qwen3_coder`. |
| **Input tokens (no-tool run)** | **7,984** avg (`99.5%` cached) | Single round-trip prompt footprint (system prompt + curriculum skill catalog + conversation state). |
| **Input tokens (tool-using run)** | **32,108** avg (`91.0%` cached) | **4.0&times;** a no-tool run (`~2.9K` uncached tokens/run). Automatic prefix caching shares the 91% cached prefix across tool round-trips. |
| **Input tokens (`write_artifact` runs)** | **37,369** avg (vs. **27,899** without) | Document-writing runs re-ingest the model's own structured tool arguments on subsequent round-trips. |
| **Peak measured run (`R11`)** | **47,305** input tokens (`92.8%` cached) | `4` tool calls including a 6-version `write_artifact` call. |
| **Output under-metering adjustment** | `+1,730` tokens on peak artifact run | Multi-section `write_artifact` collaborator calls originally streamed outside the main run's usage meter (`~6,909` chars / `~1,730` tokens on `R11` vs. `356` metered chat tokens). |

### Setting `--max-model-len` from the peak run

A run's metered input (`47,305` tokens on `R11`) is cumulative across its tool
round-trips, while multi-turn sessions and background batch year-plan runs grow
across turns. Taking the corrected peak single-run total:

```
47,305 input + 356 metered output + 1,730 artifact collaborator output = 49,391 tokens
```

and applying a **2.0&times; headroom multiplier** (aligned to a `16,384`-token
boundary) gives:

```
MAX_MODEL_LEN = 98,304 tokens (96 Ki tokens)
```

On a single `g6e.2xlarge` (`1x NVIDIA L40S`, 48 GiB VRAM) at
`--gpu-memory-utilization 0.95` with `--language-model-only` and
`--max-num-seqs 16`:

- **Usable VRAM (`0.95 × 48 GiB`)**: `45.60 GiB`
- **FP8 MoE weights (`Qwen3.6-35B-A3B-FP8`)**: `35.00 GiB`
- **CUDA context & runtime workspace (`--enforce-eager`)**: `~2.27 GiB`
- **Available hybrid KV + DeltaNet cache pool**: **`8.33 GiB`**
  - Fixed Gated DeltaNet recurrent state (`16` seqs &times; `61.41 MiB`): `0.96 GiB`
  - Paged KV pool (with vLLM hybrid block alignment): `7.37 GiB` &rarr; **`108,768` tokens physical capacity**

`MAX_MODEL_LEN=98304` sits comfortably inside the `108,768`-token physical KV
ceiling on one L40S, covers **2.0&times;** our heaviest measured teacher run,
and leaves prefix-cached concurrency room for background batch jobs.

---

## 3. Running the sizing script

[`scripts/size_kv_cache.py`](../scripts/size_kv_cache.py) encodes the hybrid
full-attention + Gated DeltaNet memory model and the sanitized baseline above:

```bash
python scripts/size_kv_cache.py
```

Output:

```text
=== Hybrid KV-Cache & Instance Sizing: Qwen/Qwen3.6-35B-A3B-FP8 ===
Revision:   c653bba46dfa06e9658dcc0b0dc87fe4685f14a2
Precision:  FP8 (weights) / BF16 (KV) / FP32 (DeltaNet state)
Weights:    35.0 GiB  |  Native context: 262,144 tokens

1. Architecture memory profile
   Layers:                40 total = 30 Gated DeltaNet (linear) + 10 full-attention (GQA)
   Full-attention KV:     20.0 KiB / token (2 * 10 layers * 2 KV heads * 256 dim * 2 B)
   DeltaNet recurrent:    61.41 MiB / active sequence (independent of context length; scales with --max-num-seqs)
   At --max-num-seqs 16 : 0.96 GiB reserved for recurrent state slots

2. Workload baseline (11 test-env runs across 7 conversations)
   Tool-calling runs:     9/11 (82%)  |  30 tool calls (2.7/run overall, 3.3/tool-using run)
   Input tokens (no-tool):   7,984 avg  (99.5% prompt-cached)
   Input tokens (tools):    32,108 avg  (91.0% prompt-cached; ~4.0x no-tool baseline)
   Input tokens (artifact): 37,369 avg  (vs 27,899 on other tool-using runs)
   Peak measured run:       47,305 input tokens + 356 metered output + ~1,730 artifact collaborator tokens = 49,391 total
   Target --max-model-len:  98,304 tokens (2.0x peak run, aligned to 16,384-token boundary; 37.5% of native 262K)

3. Instance evaluation
   [FAIL] g5.xlarge    (1x NVIDIA A10G, 24 GiB VRAM) -> weights alone (35.0 GiB) exceed usable VRAM (22.8 GiB at 0.95 utilization)
   [FIT ] g6e.2xlarge  (1x NVIDIA L40S, 48 GiB VRAM) -> fits: 8.33 GiB cache pool -> 108,768 physical token capacity (1.1x target max-model-len 98,304)
   [FIT ] g6e.12xlarge (4x NVIDIA L40S, 192 GiB VRAM) -> fits: 145.13 GiB cache pool -> 2,127,760 physical token capacity (21.6x target max-model-len 98,304)
```

You can also evaluate different concurrency ceilings, models (`qwen3.6-35b-a3b-bf16`
or the `v0.1` `qwen3-8b-bf16` reference), or see what happens if
`--language-model-only` is omitted:

```bash
# Re-run for higher concurrency or emit machine-readable JSON:
python scripts/size_kv_cache.py --concurrency 32
python scripts/size_kv_cache.py --include-vision-tower
python scripts/size_kv_cache.py --json
```
