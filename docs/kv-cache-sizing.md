# Sizing for memory footprint, not parameter count

The question that matters for serving isn't "how big is the model" but
"does weights + KV cache fit, at the concurrency and context lengths the
workload needs." KV cache grows with both of those, independent of parameter
count.

## The formula

For a dense transformer:

```
KV bytes per token = 2 (K and V) × layers × KV heads × head_dim × bytes per element
```

Grouped-query attention (GQA) models use fewer KV heads than attention heads,
which is most of why modern models are cheaper to serve than their raw
parameter count suggests -- the KV cache scales with KV heads, not attention
heads.

## A worked example

For an 8B dense model with 36 layers, 8 KV heads (GQA), head_dim 128, served
in BF16 (2 bytes/element):

```
2 × 36 × 8 × 128 × 2 bytes ≈ 144 KiB per token
→ a 32K-token sequence ≈ 4.5 GiB of KV cache
```

On a 24 GB GPU, BF16 weights for an 8B model take about 16 GB. That leaves
roughly 5 GB for KV cache -- about one full-length session at that context
length, with little headroom for concurrency.

A mixture-of-experts model shows the opposite tradeoff: a small number of
active parameters per token gives fast decode, but you still pay to hold
*every* expert in memory, so the weights-memory side of the equation doesn't
shrink the way decode speed does.

## What this means for instance selection

- **24 GB GPU (e.g. A10G):** fine for an 8B model with quantization, or with
  a context length short enough that KV cache doesn't crowd out concurrency.
- **48 GB GPU (e.g. L40S):** a comfortable single-GPU target for an 8B model
  at full precision with real concurrency, or a larger model with
  quantization.
- Past a certain size, you need disaggregated prefill/decode and/or
  multi-GPU serving -- each of which adds real operational overhead (EFA,
  cross-node scheduling, a second control plane for the split). Start with
  the smallest model that passes your quality bar before reaching for any of
  that.

These are back-of-envelope numbers for picking a shortlist, not a substitute
for measuring actual throughput and KV cache usage under your real traffic
shape and context lengths.

---

This note accompanies *"Before Our First Token: Designing Self-Hosted
Inference for an Agent That Plans a School Year"* -- see the main README for
the article link and the rest of the series.
