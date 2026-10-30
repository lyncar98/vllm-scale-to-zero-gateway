# vLLM Scale-to-Zero Gateway

A starter kit for self-hosted LLM inference on EKS: a thin, private gateway
in front of a [vLLM](https://github.com/vllm-project/vllm) model server, a
scale-to-zero GPU node pool, and the Terraform to stand up the cluster
underneath it.

This is a working reference, not a product. It's the sanitized, genericized
shape of infrastructure built for a real teaching-assistant agent (ChatLPO /
Lautum) under the **AWS Startups Custom-Model Inference Program**, extracted
to be useful on its own. See the companion article, *"Before Our First
Token: Designing Self-Hosted Inference for an Agent That Plans a School
Year"* <!-- TODO: replace with the live Medium URL --> for the reasoning
behind these choices and the measurements that are still in progress.

```
caller ──▶ [internal NLB] ──▶ inference gateway ──▶ vLLM model server (GPU, scale-to-zero)
             authenticates callers    served-name routing · 503 warming · content-free metering
```

## What's here

```
gateway/        FastAPI gateway: auth, scale-from-zero warming, routing, content-free metering
model-server/   Dockerfile that bakes a Hugging Face checkpoint into a vLLM image
k8s/            Deployments, Services, the NVIDIA device plugin, and an illustrative
                 Karpenter + KEDA scale-from-zero example
terraform/      EKS cluster, GPU node pool, IRSA roles, the AWS Load Balancer Controller
docs/           KV-cache sizing math, and the design decisions behind the defaults
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
   "warming up" instead of "error."
3. **Readiness that reflects the upstream, not just this process.**
   `/readyz` checks vLLM; `/healthz` checks only the gateway itself. If
   liveness checked the upstream, a scaled-to-zero pool would get the
   gateway killed in a restart loop -- precisely when it's most needed to
   report warming.
4. **Content-free metering.** vLLM runs with `--disable-log-requests`. The
   gateway logs one JSON line per request -- served model name, status,
   latency, token counts, taken from the upstream's own `usage` object --
   and never the prompt or completion.

It also routes one gateway to many models by the request's `model` field,
from a generated allow-list (`gateway/scripts/generate_routes.py`) -- a
request for a model that isn't on the list gets `404 model_not_found`
instead of reaching a model server at all.

## Quickstart (gateway only, no GPU needed)

```bash
cd gateway
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

# Auth off for local runs; fail-closed by default otherwise, and the app
# refuses to boot if auth is required but no key is set.
GATEWAY_REQUIRE_AUTH=false .venv/bin/uvicorn app.main:app --port 8080

.venv/bin/pytest -q   # 36 tests, no GPU or network needed
```

## Deploying the whole stack

1. **Terraform** (`terraform/`): stand up the EKS cluster, the two node
   pools (`system` for the gateway, `gpu-inference` for vLLM, scale-to-zero
   by default), IRSA roles, and the AWS Load Balancer Controller. Point
   `terraform.tfvars` (copy `terraform.tfvars.example`) at your own VPC and
   subnets.
2. **Build the model-server image** (`model-server/Dockerfile`): pick a
   Hugging Face repo and a pinned commit SHA (not `main` -- see the
   Dockerfile comments on why), build, and push.
3. **Apply the K8s manifests** (`k8s/`): `cluster-prereqs.yaml` (namespace +
   NVIDIA device plugin), then `model-server.yaml` and `gateway.yaml`. These
   are plain manifests here so they're easy to read end to end; wire them
   into your own GitOps tool (Argo CD, Flux) for a real deployment.
4. **Optional: wire scale-from-zero** (`k8s/autoscaling/`): the node pool and
   Deployment above are *capable* of scaling to zero, but nothing acts on
   that by default -- bringing a model server up from zero is a manual
   `desired_size`/`kubectl scale` change until something watches for demand.
   This directory has a worked Karpenter (node-level) + KEDA (pod-level,
   driven by vLLM's own queue-depth metric) example. See its README for how
   the two layers fit together.

## What's deliberately out of scope

- **Multi-node / multi-GPU serving.** Everything here assumes one GPU per
  model-server replica. Disaggregated prefill/decode and EFA for cross-node
  inference are real techniques for larger models, but add real operational
  overhead -- start with the smallest model that passes your quality bar
  before reaching for them.
- **LoRA adapter serving.** vLLM supports multiple adapters on a shared base
  model (`--enable-lora --max-loras`), which is the natural next step once
  you have more than one fine-tune to serve. Not wired up here; it's the
  natural extension of `model-server.yaml`.
- **A model registry / approval workflow.** `generate_routes.py` expects a
  registry endpoint that returns approved, deployable models; it doesn't
  ship one. Point `fetch_rows` at whatever system of record you already
  have.

## Further reading

- [`docs/kv-cache-sizing.md`](docs/kv-cache-sizing.md) -- the math for
  whether a model's weights-plus-KV-cache actually fit on a given GPU, which
  matters more than parameter count.
- [`docs/decisions.md`](docs/decisions.md) -- the short version of why each
  non-obvious default in this repo is set the way it is.
- [`k8s/autoscaling/README.md`](k8s/autoscaling/README.md) -- how the
  pod-level (KEDA) and node-level (Karpenter) autoscalers fit together for a
  true scale-from-zero path.

## License

MIT. See [LICENSE](LICENSE).
