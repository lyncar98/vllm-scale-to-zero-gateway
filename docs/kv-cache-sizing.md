# Sizing for a Hybrid Model: Why Our v0.1 KV-Cache Doc Was Wrong

In `v0.1` of this repo, this document opened with the standard transformer
KV-cache formula:

```text
KV bytes per token = 2 (K and V) × layers × KV heads × head_dim × bytes per element
```

and walked through a worked example for an 8B dense model (`36` layers, `8` KV
heads, `head_dim = 128`, `144 KiB/token` in BF16) on an NVIDIA A10G
(`g5.xlarge`).

Then we selected **`Qwen/Qwen3.6-35B-A3B`** as our initial serving candidate
and ran into three distinct ways a naive sizing spreadsheet goes wrong on modern
agent workloads:

1. **Architecture:** 30 of the model's 40 layers are **Gated DeltaNet**
   (`linear_attention`), which allocates a fixed recurrent state per concurrent
   sequence (`--max-num-seqs`) rather than a per-token KV cache.
2. **Physical GPU memory vs. marketing GB:** AWS advertises the L40S on G6e
   instances at "48 GB", lists approximately **44 GiB** in the EC2 G6e
   technical specifications, and physical ECC-enabled device readings report
   approximately **44.99 GiB** — not `48.0 GiB`. That ~3 GiB difference cuts the
   remaining paged KV-cache pool on a 35 GiB FP8 checkpoint by nearly 40%.
3. **Cumulative agent tokens vs. per-invocation resident context:** An agent
   task that calls tools 4 times meters the sum of input tokens across all
   round-trips (`47,305` cumulative input tokens on our largest baseline run),
   but only a single invocation's context (`~11.1K–15.8K` tokens on that same
   run) resides in GPU KV memory at one time.

Run [`scripts/size_kv_cache.py`](../scripts/size_kv_cache.py) to execute the
calculations below against our baseline or your own inputs.

---

## 1. Hybrid-attention memory: 10 GQA layers + 30 Gated DeltaNet layers

Inspecting [`config.json`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/blob/c653bba46dfa06e9658dcc0b0dc87fe4685f14a2/config.json)
at our pinned revision (`c653bba46dfa06e9658dcc0b0dc87fe4685f14a2`) shows that
`Qwen3.6-35B-A3B` uses architecture `Qwen3_5MoeForConditionalGeneration` with
`num_hidden_layers = 40` and `full_attention_interval = 4`. Only every fourth
layer uses conventional grouped-query attention (GQA); the other three out of
four layers use **Gated DeltaNet** (`linear_attention`):

| Layer group | Layer count | Config parameters | How memory scales |
|---|---:|---|---|
| **Full-attention (`full_attention`, GQA)** | **10** | `num_key_value_heads = 2`, `head_dim = 256`, BF16 (`2` B) | Scales linearly with **token count** (`20.0 KiB / token` theoretical) |
| **Gated DeltaNet (`linear_attention`)** | **30** | `linear_num_key_heads = 16` (`dim 128`), `linear_num_value_heads = 32` (`dim 128`), `linear_conv_kernel_dim = 4`, `mamba_ssm_dtype = float32` | Fixed **per active sequence** (`61.41 MiB / sequence`), **independent of context length** |

### A. Per-token full-attention KV cache (10 layers)

Across the 10 full-attention layers, the theoretical per-token KV footprint is:

```text
2 (K and V) × 10 full-attention layers × 2 KV heads × 256 head_dim × 2 bytes
  = 20,480 bytes (20.0 KiB) per token
```

### B. Per-sequence Gated DeltaNet recurrent state (30 layers)

The 30 Gated DeltaNet layers do not grow with sequence length. Instead, each
layer maintains a recurrent state tensor plus a short causal convolution state
for each active sequence slot (`--max-num-seqs`):

1. **Recurrent state matrix**:
   `32 value_heads × 128 key_dim × 128 value_dim × 4 bytes (float32) = 2,097,152 bytes (2.0 MiB) per layer`
2. **Causal conv1d state (`kernel_dim - 1 = 3` steps across Q, K, V channels)**:
   `(2 × 16 × 128 + 32 × 128) × 3 × 2 bytes (BF16) = 49,152 bytes (48 KiB) per layer`

Across all 30 Gated DeltaNet layers:

```text
30 layers × (2.0 MiB + 48 KiB) = 61.41 MiB per active sequence
```

In vLLM (`>= 0.17.0`), the unified hybrid KV/Mamba cache manager reserves
recurrent state blocks per concurrent sequence slot (`--max-num-seqs`) out of
the same GPU memory pool as paged attention blocks:

- At `--max-num-seqs 16`, the modeled recurrent-state reservation is
  `16 × 61.41 MiB ≈ 0.96 GiB`.
- At vLLM's historical default `--max-num-seqs 256`, the recurrent-state
  reservation jumps to `256 × 61.41 MiB ≈ 15.35 GiB`, exhausting the cache pool
  on a single L40S before a single request arrives.

### C. MoE weights and `--language-model-only`

`Qwen3.6-35B-A3B` activates ~`3B` parameters per token (`8` of `256` routed
experts + `1` shared expert), but all ~`35B` parameters stay resident in VRAM:

- **BF16 (`Qwen/Qwen3.6-35B-A3B`)**: `71.9 GB` on disk (`~67.0 GiB` in VRAM).
  Fits neither `g5.xlarge` (A10G) nor a single-GPU `g6e.2xlarge` (L40S). Kept
  in object storage as our fine-tuning base.
- **FP8 (`Qwen/Qwen3.6-35B-A3B-FP8`)**: `37.5 GB` on disk (`~35.0 GiB` in VRAM).
  Requires at least an L40S (`g6e.2xlarge`).

Because `Qwen3_5MoeForConditionalGeneration` ships with a multimodal
`vision_config` tower, we also run with `--language-model-only` to skip loading
and profiling the vision encoder (~`8.0 GiB` VRAM) on text-only workloads.

---

## 2. Physical L40S memory (`44.99 GiB`) vs. nominal `48.0 GiB`

AWS markets the L40S with "48 GB" of GPU memory, but its EC2 G6e technical
specifications list approximately **44 GiB** of GPU memory, and physical device
readings (`nvidia-smi` / `torch.cuda.get_device_properties` with ECC enabled)
report **`44.99 GiB`**.

When an earlier iteration of [`scripts/size_kv_cache.py`](../scripts/size_kv_cache.py)
used `48.0 GiB` as the L40S input, it overstated the remaining paged KV-cache
budget by `2.86 GiB`:

| Modeled allocation | Unadjusted `48.0 GiB` input | Corrected `44.99 GiB` physical reading |
|---|---:|---:|
| Physical GPU capacity | `48.00 GiB` | **`44.99 GiB`** |
| Available at 95% utilization | `45.60 GiB` | **`42.74 GiB`** |
| FP8 model weights (`Qwen3.6-35B-A3B-FP8`) | `35.00 GiB` | **`35.00 GiB`** |
| Runtime workspace (`--enforce-eager`) | `2.27 GiB` | **`2.27 GiB`** |
| Total hybrid cache pool | `8.33 GiB` | **`5.47 GiB`** |
| DeltaNet recurrent state (`16` slots) | `0.96 GiB` | **`0.96 GiB`** |
| **Remaining paged KV-cache budget** | `7.37 GiB` | **`4.51 GiB`** |
| **Modeled token capacity (provisional `3.5526x` hybrid factor)** | `~108,768 tokens` | **`~66,544 tokens`** |

> **Important note on provisional cache-overhead calibration:**  
> Both token capacity figures above (`66,544` at `44.99 GiB` and `108,768` at
> `48.0 GiB`) are **modeled planning estimates**, not measured vLLM limits. The
> sizing utility uses a provisional `3.5526x` effective hybrid-page overhead
> factor (`71.05 KiB/token` effective vs. the theoretical `20.0 KiB/token` of
> the 10 full-attention GQA layers) calibrated from a single configuration.
> Establishing the true context ceiling on the deployed GPU requires reconciling
> these modeled numbers against a preserved vLLM startup profile (device total
> memory, weight load GiB, runtime workspace, recurrent-state reservation, and
> `# GPU blocks` reported at boot).

---

## 3. Cumulative agent tokens vs. per-invocation resident context

Before choosing `--max-model-len`, we collected a development baseline from
`11` teacher chat runs across `7` conversations (`62` metered model and
embedding calls over ~14 minutes in our test environment):

| Measurement | Observed value | Notes |
|---|---:|---|
| Chat runs | `11` | Development sample (`R1`–`R11`) |
| Runs with tool calls | `9 of 11` (**82%**) | `30` total tool calls (`2.7`/run overall, `3.3`/tool-using run) |
| Avg. cumulative input, no-tool runs | **7,984 tokens** | `99.5%` prompt-cached (`1` model invocation) |
| Avg. cumulative input, 2-tool-call runs | **25,613 tokens** | `95.7%` prompt-cached (`3` model invocations) |
| Avg. cumulative input, 4-tool-call runs | **35,356 tokens** | `89.3%` prompt-cached (`5` model invocations) |
| Avg. cumulative input, all tool-using runs | **32,108 tokens** | `91.0%` prompt-cached (~**4.0&times;** no-tool baseline) |
| Avg. cumulative input, `write_artifact` runs | **37,369 tokens** | vs. `27,899` tokens on tool-using runs without `write_artifact` |
| Largest run (`R11`), cumulative input | **47,305 tokens** | `92.8%` cached (`43,899` cached + `3,406` uncached), `4` tool calls |

### Baseline vs. revised output instrumentation (`write_artifact` collaborator gap)

During the baseline investigation, two multi-section document-writing runs
(`R3` and `R11`, `6` document versions each) metered far fewer output tokens on
the main run record than the text they actually generated, because the
`write_artifact` section collaborator streamed directly through the provider
without reporting usage back into the parent run:

| Run | Document versions | Final document (`chars`) | `write_artifact` argument (`chars`) | Un-metered collaborator output est. (`tokens`) | Original metered run output (`tokens`) | Revised run output est. (`tokens`) |
|---|---:|---:|---:|---:|---:|---:|
| `R3` | `6` | `4,398` | `6,597` | `~1,650` | `351` | `~2,001` |
| `R6` | `1` | `2,583` | `365` | `0` *(single pass)* | `777` | `777` |
| `R8` | `1` | `4,017` | `369` | `0` *(single pass)* | `1,442` | `1,442` |
| `R11` | `6` | `4,418` | `6,909` | `~1,730` | `356` | `~2,086` |
| **All 11 runs** | — | — | — | **`+1,500 to +2,670`** | **`4,849` (`441`/run)** | **`~6,350 to 7,520` (`577–684`/run)** |

We fixed the application's collaborator metering so section worker and planner
calls report into the run's usage record, and included the `+1,730`-token peak
adjustment in our sizing baseline (`47,305 + 356 + 1,730 = 49,391` cumulative
tokens on `R11`).

### Reconstructing single-invocation residency on peak run `R11`

A run's metered input (`47,305` tokens on `R11`) is the **cumulative sum across
every model round-trip in that run**, not the length of a single prompt.
Because a no-tool run consists of one invocation (`~7,984` input tokens, `99.5%`
cached) and `R11` made `4` tool calls (`92.8%` cached = `43,899` cached input
tokens + `3,406` uncached tokens, plus `6,909` chars / `~1,730` tokens of
model-authored `write_artifact` arguments re-entering context), we can
reconstruct how context grows across the invocations of `R11`:

| Invocation in `R11` | Action / tool call | Input tokens | Cached prefix tokens | Uncached new tokens | Output tokens | Peak resident context in KV cache |
|---:|---|---:|---:|---:|---:|---:|
| **1** | User prompt &rarr; `use_skill` | `~8,020` | `~7,940` | `~80` | `~45` | `~8,065` |
| **2** | Skill loaded &rarr; `educational_plan_ask` | `~8,650` | `~7,940` | `~710` | `~60` | `~8,710` |
| **3** | Curriculum objectives &rarr; `educational_plan_ask` | `~9,420` | `~8,580` | `~840` | `~65` | `~9,485` |
| **4** | Context assembled &rarr; `write_artifact` (`6` sections) | `~10,180` | `~9,350` | `~830` | `~95` *(+ `~1,730` collaborator)* | `~10,275` |
| **5** | Artifact summary + `write_artifact` args &rarr; final reply | `~11,035` | `~10,089` | `~946` | `~91` | **`~11,126`** *(up to `~15,800` if 4 calls)* |
| **Cumulative run total** | `4` tool calls (`6` document versions) | **`47,305`** | **`43,899` (`92.8%`)** | **`3,406`** | **`356` (`~2,086` revised)** | **Peak single call &ll; cumulative sum** |

*(Note: Individual per-invocation token counts above are reconstructed from the
run's metered cumulative totals, tool-call sequence, and no-tool base prefix;
capturing direct per-invocation residency distributions from the self-hosted
gateway is part of our active validation pass.)*

---

## 4. Three context targets and the 128K segmentation trade-off

Comparing the three candidate `--max-model-len` targets against a single
`44.99 GiB` L40S (`~66,544` modeled token capacity at `--max-num-seqs 16`)
Clarifies the engineering trade-offs:

| Sizing mode (`--sizing-mode`) | Target `--max-model-len` | How it is derived | Fits on 1&times; L40S (`44.99 GiB`, `~66,544` modeled tokens)? | Architectural implication |
|---|---:|---|---|---|
| **`per-invocation`** *(default)* | **`49,152`** (`48 Ki`) | `~3.0×` peak single-invocation residency (`~15.8K` tokens on `R11`), aligned to `16,384` | **Yes (`1.35×` headroom)** | Supports curriculum resolution and **segmented** section-by-section planning on 1&times; L40S (`g6e.2xlarge`). |
| **`cumulative-2x`** | **`98,304`** (`96 Ki`) | `2.0×` cumulative multi-call run total (`2 × 49,391`), aligned to `16,384` | **No (`66,544 < 98,304`)** | Overstates single-call residency by summing repeated cached prefixes; only appeared to fit under an unadjusted `48.0 GiB` GPU assumption. |
| **`qwen-128k`** | **`131,072`** (`128 Ki`) | Qwen's model-card recommendation (`>= 128K` context for complex reasoning/planning tasks) | **No (`66,544 < 131,072`)** | Exposing 128K+ on a single call requires multi-GPU (`g6e.12xlarge` / TP&ge;2) or KV-cache quantization (`--kv-cache-dtype fp8`). |

### Acknowledging Qwen's 128K recommendation as a quality hypothesis

Qwen's official guidance recommends maintaining at least **128K context** for
complex multi-step tasks when hardware allows. Running on a single L40S with a
`49,152`-token (`48K`) or `65,536`-token (`64K`) window means long full-year
plans cannot be generated in one uninterrupted 128K+ prompt; instead, the agent
must segment the workflow (generating a structured outline first, then drafting
units/sections in bounded collaborator calls while carrying forward objectives
and constraints).

Whether segmented generation at `49,152` tokens matches the instructional
coherence of an unsegmented `131,072`-token window is an **empirical quality
hypothesis**, not a free memory optimization. We are testing that hypothesis
directly on our Chilean curriculum benchmark before committing to single-L40S
vs. multi-GPU serving for full-year planning.

---

## 5. Running the sizing utility

```bash
# Default: 44.99 GiB physical L40S reading, per-invocation (49,152) target
python scripts/size_kv_cache.py

# Show why the 98,304 cumulative-2x heuristic fails on one 44.99 GiB L40S:
python scripts/size_kv_cache.py --sizing-mode cumulative-2x

# Compare against the unadjusted 48.0 GiB nominal input (108,768 modeled tokens):
python scripts/size_kv_cache.py --l40s-vram-gib 48.0 --sizing-mode cumulative-2x

# Evaluate Qwen's 128K (131,072) complex-task recommendation:
python scripts/size_kv_cache.py --sizing-mode qwen-128k
```
