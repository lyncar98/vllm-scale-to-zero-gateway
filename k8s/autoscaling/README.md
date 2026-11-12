# Scale-from-zero, illustrated (and its current limitations)

> **Status: Reference Implementation / Illustrative Example**  
> This directory provides an illustrative starter kit for autoscaling, not a
> finished, production-ready inference platform. Scaling from one replica to
> many is straightforward; true request-driven activation from zero GPU replicas
> remains an active engineering challenge with specific integration work still
> to be done.

The core manifests in `../model-server.yaml` and `../gateway.yaml` describe a
GPU node pool whose managed-node-group `desired_size` is a static number.
That's enough to run a baseline deployment, but it means scaling the pool up from
zero when demand arrives -- and back down to zero when idle -- requires an
automated scaling architecture.

These manifests demonstrate the multi-layer scaling mechanics, their
interaction, and the key architectural hurdles.

---

## The two layers of scaling

Scale-to-zero for a GPU-backed workload requires two independent autoscaling
layers, because a pod cannot schedule onto a node that doesn't exist yet, and a
node provider cannot provision a node until an unschedulable pod demands it:

1. **Pod-level autoscaling** (`keda-scaledobject.yaml`): controls the
   `model-server` Deployment's replica count (including down to **zero** when
   idle). For scaling *from 1 to N*, it monitors a signal that reflects real
   queueing -- vLLM's internal `vllm:num_requests_waiting` metric -- rather than
   CPU or GPU utilization (a GPU can show 100% compute utilization while serving
   a single long request or a full batched queue).
2. **Node-level autoscaling** (`karpenter-nodepool.yaml`): when the Deployment
   replica count increases from 0 to 1, the new pod is `Pending` with nowhere to
   run. Karpenter detects the unschedulable pod, provisions a matching GPU node
   (e.g., G5 / G6e) on demand, and consolidates (terminates) the node once it has
   remained empty for `consolidateAfter`.

*(Cluster Autoscaler with `min_size: 0` is an alternative to Karpenter if your
organization is standardized on it, though Karpenter offers faster dynamic node
provisioning for heterogeneous GPU instance types.)*

---

## The unsolved scale-from-zero limitation

There is a fundamental problem with using engine-internal metrics for
scale-from-zero:

> **When vLLM has zero replicas, it cannot report its own queue depth.**

If an autoscaler relies exclusively on metrics scraped from running model
servers (`/metrics`), it has **no signal to trigger the first replica**.

Scaling from one replica to several (1 &rarr; N) and activating the first replica
from zero (0 &rarr; 1) are distinct engineering problems:

- **1 &rarr; N**: vLLM is alive, exporting `vllm:num_requests_waiting`. Prometheus
  scrapes the pod, and KEDA adds replicas when work backs up.
- **0 &rarr; 1**: vLLM is offline. No pod exists, no port 8000 is listening, and
  Prometheus has no target to scrape. The metric `vllm:num_requests_waiting` does
  not exist.

### Returning a 503 warming response does NOT wake the GPU

The gateway (`../gateway.yaml`) recognizes when upstream model servers are
offline and returns:

```http
HTTP/1.1 503 Service Unavailable
Retry-After: 30
Content-Type: application/json

{
  "error": {
    "message": "Model server is warming up. The inference pool scales to zero when idle; retry shortly.",
    "type": "service_warming"
  }
}
```

This response contract communicates availability and gives callers (such as an
agent client) an actionable signal to back off or fall back to an alternative
provider. **However, returning this response does not activate the GPU.**

The gateway informs callers about availability; a separate activation mechanism
must initiate scaling.

### What the provided KEDA manifest does today

In `keda-scaledobject.yaml`, two triggers are defined:

1. **Prometheus trigger (`vllm:num_requests_waiting`)**: drives scale-up when
   traffic exceeds capacity (1 &rarr; N) and scale-down to zero after
   `cooldownPeriod` seconds of an empty queue.
2. **Cron trigger (`America/Santiago` school hours)**: maintains a warm floor of
   `desiredReplicas: 1` during predictable peak hours (e.g., Monday through
   Friday, 08:00 to 20:00).

The cron schedule eliminates cold starts during predictable operational hours,
which is critical for end-user experience (such as teachers planning lessons in
the morning). However, **it does not solve arbitrary, request-driven activation
outside those hours**.

---

## Architectural options for request-driven activation (0 &rarr; 1)

To activate a model server from zero outside scheduled windows, the system
requires a demand signal that remains available while the GPU pool is offline:

1. **Persistent request queue**:
   Client requests are enqueued into an external durable queue (e.g., SQS,
   RabbitMQ, or Redis). KEDA scales the model server based on queue length
   (`aws-sqs` or `redis` scaler). The model server or an adapter worker pulls
   requests once booted. *Trade-off: requires asynchronous/polling request
   handling rather than synchronous OpenAI-compatible HTTP streaming.*
2. **Gateway-side demand metrics**:
   The gateway runs continuously on cheap CPU nodes. It can increment a
   Prometheus counter or gauge whenever requests arrive for an offline model
   (e.g., requests that result in `503 service_warming`). KEDA watches this
   gateway metric to trigger 0 &rarr; 1 scaling. *Trade-off: requires KEDA to
   coordinate with the gateway's metrics endpoint and handle smoothing/decay.*
3. **HTTP activation layer**:
   An intelligent proxy or HTTP add-on (such as KEDA HTTP Add-on) intercepts
   incoming HTTP calls, holds or buffers the initial request, triggers replica
   scaling via the Kubernetes API, and routes the request once the model server
   passes readiness. *Trade-off: adds connection buffering complexity and
   potential timeout handling if cold starts exceed HTTP client limits.*

Implementing, benchmarking, and hardening this activation mechanism is one of the
primary remaining milestones.

---

## The complete target lifecycle

The end-to-end lifecycle we are working toward comprises 7 stages:

```
1. Request Arrives ──▶ 2. Activation Signal ──▶ 3. Pod Scale (0->1) ──▶ 4. Pending Pod
   (at CPU Gateway)       (Queue / Gateway)        (KEDA ScaledObject)     (Needs GPU)
                                                                               │
                                                                               ▼
7. Scale to Zero  ◀── 6. vLLM Serving       ◀── 5. GPU Node Provision ◀────────┘
   (Cooldown Idle)       (Weights Loaded)          (Karpenter)
```

1. **Incoming demand**: A request or demand event reaches an always-available
   component (the CPU gateway).
2. **Activation signal**: An out-of-band demand signal registers that an offline
   model is needed.
3. **Pod scaling**: KEDA increases the model-server Deployment replica count from
   0 to 1.
4. **Pending state**: With no GPU node provisioned, the pod sits `Pending` with
   nodeSelector/tolerations for the accelerator pool.
5. **Node provisioning**: Karpenter detects the unschedulable pod and provisions
   a matching GPU node.
6. **Inference readiness**: The GPU node joins, the vLLM container starts, loads
   weights into GPU memory, passes health checks, and begins serving.
7. **Consolidation**: When queue depth stays zero past `cooldownPeriod`, KEDA
   scales replicas to 0. After `consolidateAfter`, Karpenter terminates the empty
   GPU node, returning compute cost to $0.

---

## Wiring notes & architectural traps

- **Match labels and taints**:
  Ensure `inference.io/pool: gpu-inference` and `inference.io/accelerator: "true"`
  match between node pools, tolerations, and nodeSelectors in
  `../model-server.yaml`. A mismatch causes pods to remain `Pending` indefinitely
  without Karpenter triggering node provisioning.
- **The Gateway Readiness Trap**:
  Do **not** configure the gateway's Kubernetes `readinessProbe` to depend on
  upstream model health (`/readyz`). If the gateway becomes "NotReady" when the
  GPU pool is at zero, Kubernetes will remove the gateway pods from Service
  endpoints, causing callers to see connection failures instead of the
  structured `503 service_warming` response. See [`../gateway.yaml`](../gateway.yaml)
  and [`../../docs/decisions.md`](../../docs/decisions.md).
- **Idle cooldown alignment**:
  `cooldownPeriod` (KEDA, pod-level) and `consolidateAfter` (Karpenter, node-level)
  are separate timers. Keep them reasonably aligned. If `consolidateAfter` is
  significantly longer than `cooldownPeriod`, you will pay for an empty GPU node
  idling after its pod has already terminated.
