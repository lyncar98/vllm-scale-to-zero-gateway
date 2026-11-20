# Design decisions

Short version of why this repo looks the way it does. Each of these cost
someone a debugging session to discover; written down so you don't pay for
it twice.

**The gateway is private by default.** It sits behind an internal load
balancer; nothing in this repo makes it internet-facing. An unauthenticated
endpoint in front of a GPU is an unmetered bill and an unaudited data path.
Flipping the `aws-load-balancer-scheme` annotation to `internet-facing` is a
one-line change with real consequences -- treat it as a security review, not
a config tweak.

**Gateway health is separate from model health (and the readiness cold-start trap).**
The gateway exposes `/healthz` for process health and `/readyz` for upstream
model availability. This separation prevents two distinct cold-start traps:

1. *The Liveness Trap:* If liveness depended on the model server, a GPU pool
   scaled to zero would get the gateway killed and restarted in a CrashLoopBackOff --
   precisely when it is needed most to answer callers with "warming up".
2. *The Readiness Trap:* If Kubernetes uses `/readyz` as the gateway pod's
   readiness probe, Kubernetes marks the gateway NotReady whenever the model
   server is offline. This removes the gateway pods from the Service's active
   endpoints, causing callers to receive raw connection timeouts or 502 Bad Gateway
   errors from the AWS NLB instead of the structured `503 service_warming` response!

*Rule:* If your architecture depends on an always-available front door, its
availability must not depend on the component it is supposed to front. In
`k8s/gateway.yaml`, the Kubernetes readiness probe targets `/healthz` so the
gateway remains routable while inference is offline, while `/readyz` is preserved
for external monitoring and canary probes.

**Scale-from-zero is answered with 503 + Retry-After, not a timeout.** A
caller on the other side of this gateway should be able to treat "warming"
as a distinct, expected state -- not indistinguishable from a real failure.
However, *returning a warming response does not wake the GPU*. The gateway
communicates availability; an out-of-band scaling mechanism must activate the
model server.

**Scale-from-zero (0 -> 1) is a different problem from load scaling (1 -> N).**
vLLM's internal queue depth (`vllm:num_requests_waiting`) works well for scaling
from 1 to N and consolidating back to 0. But when replicas are at zero, the model
server cannot report its own queue depth. In this starter kit, a KEDA cron trigger
provides a warm floor during predictable school hours (America/Santiago), but
arbitrary request-driven activation outside those hours requires an external
demand signal (such as gateway-level Prometheus metrics or an asynchronous queue).

**Recreate, not rolling, deployments for the model server.** A rolling update
briefly runs two GPU pods. On a one-GPU-per-replica deployment that doubles
cost during every rollout, and can fail outright when quota allows only one
node.

**Routing by served name, from an allow-list.** The gateway never computes
served names or guesses at a model's existence -- it routes only names present
in a generated routing table (`gateway/scripts/generate_routes.py`), and returns
`404 model_not_found` for anything else. A routing file that's present but
unreadable is a hard `503`, not an empty table that silently sends traffic to
the wrong model.

**Content-free logging across both Chat Completions and the Responses API.**
vLLM (`v0.19.0`) runs with `--no-enable-log-requests` (which replaced the older
`--disable-log-requests` flag). The gateway proxies both `/v1/chat/completions`
and `/v1/responses`, sniffing token counts (`prompt_tokens`/`completion_tokens`
or `input_tokens`/`output_tokens`) on the fly without buffering the stream. Its
usage log line carries the served model name, status, latency, and token counts
-- never the prompt or the completion.

**FP8 baked into the image for serving; BF16 kept in object storage for fine-tuning.**
`Qwen/Qwen3.6-35B-A3B` is ~71.9 GB (~67 GiB) in BF16, which cannot fit on a single
A10G (`g5.xlarge`) or a single L40S (`g6e.2xlarge`). We keep the BF16 checkpoint
in object storage as the governed fine-tuning base, and bake the FP8 checkpoint
(`Qwen/Qwen3.6-35B-A3B-FP8`, ~35 GiB at pinned revision
`c653bba46dfa06e9658dcc0b0dc87fe4685f14a2`) into the `model-server` image for
single-GPU serving on `g6e.2xlarge`. Baking weights avoids a multi-GB runtime
download on every scale-from-zero boot and ties "which weights served this
request" to an immutable image digest.

**Physical GPU GiB, not marketing GB, governs KV-cache sizing.**
AWS markets the L40S at "48 GB", lists ~44 GiB in the EC2 G6e technical
specifications, and physical ECC-enabled device readings report ~44.99 GiB.
Treating an L40S as `48.0 GiB` overstates the remaining paged KV-cache budget
on a ~35 GiB FP8 checkpoint by ~2.86 GiB (`7.37 GiB` vs. `4.51 GiB` after
`2.27 GiB` runtime workspace and `0.96 GiB` DeltaNet recurrent state for 16
slots). Under our provisional hybrid-cache calibration, `44.99 GiB` yields
`~66,544` modeled tokens rather than `~108,768`. Because `98,304` was a
heuristic derived from *cumulative* multi-call run tokens (`2x 49,391`) rather
than single-invocation residency (`~11.1K-15.8K`), `k8s/model-server.yaml`
defaults `MAX_MODEL_LEN` to `49152` (`48 Ki`) for single-L40S segmented
workflows until physical vLLM startup block counts and our 128K vs. segmented
curriculum quality evaluation say otherwise. See
[`docs/kv-cache-sizing.md`](kv-cache-sizing.md) and
[`scripts/size_kv_cache.py`](../scripts/size_kv_cache.py).

**Hybrid Gated DeltaNet + GQA sizing requires capping `--max-num-seqs` and skipping the vision tower.**
`Qwen3.6-35B-A3B` (`Qwen3_5MoeForConditionalGeneration`) has 30 Gated DeltaNet
(`linear_attention`) layers and 10 full-attention GQA layers, plus a multimodal
vision tower. Passing `--language-model-only` skips the vision encoder (~8 GiB
VRAM), and capping `--max-num-seqs 16` keeps fixed per-sequence DeltaNet
recurrent state to ~0.96 GiB (leaving vLLM's historical default 256 sequences
would demand ~15.35 GiB of recurrent state alone).

**Reasoning + tool-call parser interaction must be validated before un-flagged rollout.**
We follow Qwen's vLLM recipe (`--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`),
but in vLLM 0.19 related `Qwen3.5`/`3.6` models have a reported edge case where
tool calls emitted inside a reasoning block are not always extracted cleanly.
Application traffic stays behind a feature flag with managed-model fallback
while structured and streaming tool extraction is validated across all agent
tools.

**The GPU node pool is tainted, and that's load-bearing.** Without the taint,
ordinary pods schedule onto expensive accelerator nodes -- a failure mode
that's invisible except on the bill. `nodeSelector` alone isn't enough; you
need both the selector *and* the toleration on anything that should land
there.

**A dedicated KMS key for node EBS volumes.** EKS managed node groups launch
through an Auto Scaling group, and the ASG service-linked role needs a grant
on whatever key encrypts the node root volumes. An account-default EBS key
commonly lacks that grant, which breaks node bring-up with
`Client.InvalidKMSKey.InvalidState` -- a failure that looks like a capacity or
quota problem and isn't.
