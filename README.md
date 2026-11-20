# vLLM Scale-to-Zero Gateway

A starter kit for self-hosted LLM inference on EKS: a thin, private gateway
in front of a [vLLM](https://github.com/vllm-project/vllm) model server, a
scale-to-zero GPU node pool, a baseline-driven hybrid KV-cache sizer, and the
Terraform to stand up the cluster underneath it.

This is a working reference, not a product. It's the sanitized, genericized
shape of infrastructure built for a real teaching-assistant agent (ChatLPO /
Lautum) under the **AWS Startups Custom-Model Inference Fellowship**, extracted
to be useful on its own.

**Companion series:**
1. [“Plan My Whole Year”: The Request That Made Us Build Our Own Model](https://medium.com/@lyndon.carlson/plan-my-whole-year-the-request-that-made-us-build-our-own-model-18d82c57751a)
2. [Before Our First Token: Building Inference That's Safe to Turn On](https://medium.com/@lyndon.carlson/before-our-first-token-building-inference-thats-safe-to-turn-on-dfa3d7df6a40)
3. *The Agent Is the Workload: Sizing Qwen3.6 for Real Agent Tasks*

```
caller ──▶ [internal NLB] ──▶ inference gateway ──▶ vLLM model server (GPU, scale-to-zero)
             authenticates callers    served-name routing · 503 warming · content-free metering
```

## What's new in v0.2

1. **`model-server` updated for `Qwen/Qwen3.6-35B-A3B-FP8` on `vLLM v0.19.0`**
   ([`model-server/Dockerfile`](model-server/Dockerfile),
   [`k8s/model-server.yaml`](k8s/model-server.yaml),
   [`terraform/variables.tf`](terraform/variables.tf)):
   - Jumps the runtime base image from `vllm/vllm-openai:v0.8.5` to `v0.19.0`
     (the Qwen3.6 recipe requires `vLLM >= 0.17.0` for
     `Qwen3_5MoeForConditionalGeneration`).
   - Targets `Qwen/Qwen3.6-35B-A3B-FP8` (`~35 GiB` weights, pinned SHA
     `c653bba46dfa06e9658dcc0b0dc87fe4685f14a2`) on an NVIDIA L40S
     (`g6e.2xlarge`: 48 GB marketing, ~44 GiB AWS G6e spec, `~44.99 GiB`
     physical device reading), replacing the `v0.1` 8B placeholder on
     `g5.xlarge`.
   - Adds `--enable-auto-tool-choice --tool-call-parser qwen3_coder`,
     `--reasoning-parser qwen3`, `--language-model-only` (skipping the
     multimodal vision encoder on text workloads), `--no-enable-log-requests`,
     and `--max-num-seqs 16`, with a conservative single-L40S default
     `MAX_MODEL_LEN=49152` (`48 Ki`) sized for per-invocation residency inside
     the `~66,544`-token modeled capacity at `44.99 GiB` physical GPU memory.
2. **Rewritten [`docs/kv-cache-sizing.md`](docs/kv-cache-sizing.md) for hybrid Gated DeltaNet + GQA and physical GPU memory**:
   - Replaces the `v0.1` dense-attention sizing note with the two-part memory
     model for `Qwen3.6-35B-A3B`, where **30 of 40 layers are Gated DeltaNet**
     (`linear_attention`, fixed `61.41 MiB` recurrent state per active sequence
     regardless of context length) and **10 of 40 layers** use per-token GQA KV
     cache (`20.0 KiB/token` theoretical).
   - Shows the impact of using physical L40S memory (`44.99 GiB` device reading /
     ~44 GiB AWS G6e spec) rather than `48.0 GiB` (`4.51 GiB` remaining paged KV
     budget &rarr; `~66,544` modeled tokens vs. `7.37 GiB` &rarr; `~108,768`),
     distinguishes **cumulative multi-call agent tokens** (`49,391` on peak run
     `R11`) from **single-invocation resident context** (`~11.1K–15.8K`), and
     evaluates Qwen's `>=128K` complex-task context recommendation against
     segmented single-L40S workflows.
3. **Baseline-driven sizing utility ([`scripts/size_kv_cache.py`](scripts/size_kv_cache.py))**:
   - Evaluates three sizing modes (`per-invocation` `49,152`, `cumulative-2x`
     `98,304`, and `qwen-128k` `131,072`) against physical GPU memory
     (`--l40s-vram-gib 44.99` by default, configurable to `44.0` or `48.0`),
     clearly labeling outputs as provisional planning estimates until reconciled
     with preserved vLLM startup profiles.
4. **OpenAI Responses API (`/v1/responses`) passthrough and metering**
   ([`gateway/app/main.py`](gateway/app/main.py),
   [`gateway/app/meter.py`](gateway/app/meter.py)):
   - Proxies stateless `/v1/responses` calls alongside `/v1/chat/completions`
     and sniffs `input_tokens` / `output_tokens` from both JSON and SSE
     `response.completed` events without logging prompts or completions.

## What's here

```
gateway/        FastAPI gateway: auth, scale-from-zero warming, routing, content-free metering
model-server/   Dockerfile that bakes a pinned Qwen3.6-35B-A3B-FP8 checkpoint into vLLM v0.19.0
scripts/        Baseline-driven hybrid KV-cache & GPU instance sizer (size_kv_cache.py)
k8s/            Deployments, Services, the NVIDIA device plugin, and an illustrative
                 Karpenter + KEDA scale-from-zero example
terraform/      EKS cluster, L40S GPU node pool, IRSA roles, the AWS Load Balancer Controller
docs/           Hybrid DeltaNet + GQA KV-cache sizing math, and design decisions
```

## Why a gateway in front of vLLM at all

vLLM already speaks the OpenAI-compatible API. This gateway doesn't
re-implement inference -- it owns exactly four things vLLM doesn't:

1. **Caller authentication.** An unauthenticated endpoint in front of a GPU
   is a direct route to someone else's compute bill.
2. **Warming as a first-class state.** The GPU pool scales to zero, so the
   first request after idle arrives with no GPU node in existence. A naive
   proxy surfaces that as a timeout; this gateway returns `503` with
   `Retry-After: 30` and `"type": "service_warming"`, so a caller can render
   "warming up" instead of "error." Note that *returning a warming response
   does not wake the GPU* -- an out-of-band mechanism must drive scaling.
3. **Separate process health from model readiness (avoiding the readiness trap).**
   `/healthz` checks gateway process health; `/readyz` checks upstream vLLM.
   In Kubernetes, the gateway pod's `readinessProbe` must remain decoupled from
   upstream existence (`/healthz`) so that the gateway remains routable while
   inference is scaled to zero. If the Service readiness probe checked `/readyz`,
   an idle GPU pool would mark gateway pods NotReady, removing them from Service
   endpoints and turning clean 503 warming responses into connection drops.
4. **Content-free metering.** vLLM (`v0.19.0`) runs with
   `--no-enable-log-requests`. The gateway logs one JSON line per request --
   served model name, status, latency, token counts, taken from the upstream's
   own `usage` object across both `/v1/chat/completions` and `/v1/responses` --
   and never the prompt or completion.

It also routes one gateway to many models by the request's `model` field,
from a generated allow-list (`gateway/scripts/generate_routes.py`) -- a
request for a model that isn't on the list gets `404 model_not_found`
instead of reaching a model server at all.

## Quickstart (no GPU needed)

```bash
# Run the hybrid KV-cache & instance sizer (defaults to 44.99 GiB physical L40S):
python scripts/size_kv_cache.py

# Compare per-invocation (49,152) vs. cumulative-2x (98,304) or Qwen 128K (131,072):
python scripts/size_kv_cache.py --sizing-mode cumulative-2x
python scripts/size_kv_cache.py --sizing-mode qwen-128k

cd gateway
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

# Auth off for local runs; fail-closed by default otherwise, and the app
# refuses to boot if auth is required but no key is set.
GATEWAY_REQUIRE_AUTH=false .venv/bin/uvicorn app.main:app --port 8080

.venv/bin/pytest -q   # 60 tests, no GPU or network needed
```

## Deploying the whole stack

1. **Terraform** (`terraform/`): stand up the EKS cluster, the two node
   pools (`system` for the gateway, `gpu-inference` on `g6e.xlarge` /
   `g6e.2xlarge` L40S for vLLM, scale-to-zero by default), IRSA roles,
   and the AWS Load Balancer Controller. Point `terraform.tfvars` (copy
   `terraform.tfvars.example`) at your own VPC and subnets.
2. **Build the model-server image** (`model-server/Dockerfile`): pass a
   pinned Hugging Face commit SHA (not `main` -- see the Dockerfile comments
   on why, e.g. `--build-arg MODEL_REVISION=c653bba46dfa06e9658dcc0b0dc87fe4685f14a2`),
   build on a builder with sufficient disk for a ~35 GiB checkpoint, and push.
3. **Apply the K8s manifests** (`k8s/`): `cluster-prereqs.yaml` (namespace +
   NVIDIA device plugin), then `model-server.yaml` and `gateway.yaml`. These
   are plain manifests here so they're easy to read end to end; wire them
   into your own GitOps tool (Argo CD, Flux) for a real deployment.
4. **Optional: wire scale-from-zero** (`k8s/autoscaling/`): the node pool and
   Deployment above are *capable* of scaling to zero, but bringing a model
   server up from zero requires an automated scaling architecture. This
   directory provides an illustrative Karpenter (node-level) + KEDA (pod-level)
   example. Note that vLLM's internal queue depth cannot report when replicas = 0;
   see `k8s/autoscaling/README.md` for details on this limitation and candidate
   activation patterns.

## Current limitations and unfinished work

This starter kit is an open-source reference implementation, not a finished
production inference platform. Several key areas remain active engineering work
and are intentionally documented as unfinished:

- **Reconciling modeled KV capacity with physical vLLM startup profiles:**
  `scripts/size_kv_cache.py` uses `44.99 GiB` physical L40S memory (~44 GiB in
  AWS G6e specs) and a provisional `3.5526x` hybrid-cache overhead factor,
  yielding `~66,544` modeled tokens on one L40S at `--max-num-seqs 16`. That
  supports our `49,152`-token per-invocation segmented target, but does not fit
  the `98,304` cumulative-run heuristic or Qwen's `131,072` (128K) complex-task
  recommendation on a single GPU. These planning estimates must be reconciled
  with preserved vLLM startup profiles (`# GPU blocks` and actual weight/workspace
  allocations) on the deployed hardware.
- **Structured and streaming tool-call validation under reasoning mode:**
  The model server uses `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`.
  Because vLLM 0.19 has a reported edge case in the `Qwen3.5`/`3.6` family where
  tool calls inside a reasoning block may not be extracted cleanly, application
  traffic remains behind a feature flag while multi-step and streaming tool
  calls are validated end to end.
- **Request-driven activation from zero (0 &rarr; 1):**
  When vLLM has zero running replicas, it cannot emit metrics or report its own
  queue depth (`vllm:num_requests_waiting`). While scaling 1 &rarr; N and scaling
  down to 0 works off queue metrics, and a scheduled warm floor (via KEDA cron)
  handles predictable school hours, arbitrary request-driven activation outside
  those hours requires an external demand signal (e.g. gateway metrics or an
  asynchronous queue).
- **Decoupled gateway readiness:**
  As documented in `k8s/gateway.yaml` and `docs/decisions.md`, binding Kubernetes
  Service readiness to an offline upstream (`/readyz`) removes gateway pods from
  endpoints, preventing callers from receiving the structured `503 service_warming`
  response. The gateway's Kubernetes readiness probe must remain decoupled from
  upstream model state.
- **Application fallback and cold-start timing under production load:**
  The gateway returns clean `503` warming responses with `Retry-After: 30`, but
  automated scale-up from zero, end-to-end fallback to managed models during GPU
  cold starts, and systematic cold-start latency benchmarks remain active work.

## What's deliberately out of scope

- **Multi-node / multi-GPU serving.** Everything here targets one GPU per
  model-server replica (`1x NVIDIA L40S` for `Qwen3.6-35B-A3B-FP8`).
  Disaggregated prefill/decode and multi-GPU tensor parallelism are needed if
  benchmark results show that unsegmented 128K+ contexts outperform segmented
  48K–64K workflows.
- **LoRA adapter serving and Ray Serve orchestration.** vLLM supports multiple
  adapters on a shared base model (`--enable-lora --max-loras`), and Ray Serve
  can manage multiple model copies across Karpenter-provisioned nodes; neither
  is wired into this repo yet.
- **A model registry / approval workflow.** `generate_routes.py` expects a
  registry endpoint that returns approved, deployable models; it doesn't
  ship one. Point `fetch_rows` at whatever system of record you already
  have.

## Further reading

- [`docs/kv-cache-sizing.md`](docs/kv-cache-sizing.md) -- why our `v0.1`
  standard-attention sizing doc was wrong for `Qwen3.6-35B-A3B`, how `44.99 GiB`
  physical L40S memory changes the paged KV budget (`~66,544` modeled tokens),
  and how cumulative run tokens differ from per-invocation resident context.
- [`scripts/size_kv_cache.py`](scripts/size_kv_cache.py) -- the standalone
  sizing utility fed by our test-environment baseline.
- [`docs/decisions.md`](docs/decisions.md) -- the short version of why each
  non-obvious default in this repo is set the way it is.
- [`k8s/autoscaling/README.md`](k8s/autoscaling/README.md) -- how the
  pod-level (KEDA) and node-level (Karpenter) autoscalers fit together for a
  true scale-from-zero path.

## License

MIT. See [LICENSE](LICENSE).
