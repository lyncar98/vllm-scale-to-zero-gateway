# Scale-from-zero, illustrated

The core manifests in `../model-server.yaml` and `../gateway.yaml` describe a
GPU node pool whose managed-node-group `desired_size` is a static number.
That's enough to run the demo, but it means scaling the pool up from zero
when a request arrives -- and back down to zero when it's idle -- is a manual
change. These two files show the piece that automates it.

## The two layers

Scale-from-zero for a GPU-backed Deployment needs two independent
autoscalers, because a pod can't schedule onto a node that doesn't exist yet,
and a node group can't know to grow until something tries to schedule onto
it:

1. **Pod-level** (`keda-scaledobject.yaml`): scales the `model-server`
   Deployment's replica count, including down to **zero**, off a signal
   that reflects real queueing -- vLLM's own `num_requests_waiting` metric,
   not CPU or even GPU utilization (a GPU can read "busy" while serving one
   slow request just as easily as a full batch of fast ones).
2. **Node-level** (`karpenter-nodepool.yaml`): when KEDA takes the
   Deployment from 0 to 1, the resulting pod is `Pending` with nowhere to
   run. Karpenter watches for exactly that and provisions a matching GPU
   node, then consolidates (terminates) it again once nothing is scheduled
   there for `consolidateAfter`.

Cluster Autoscaler is a viable substitute for Karpenter here if you're
already standardized on it; the node group just needs `min_size: 0` and
something watching for unschedulable pods.

## Wiring notes

- Match `inference.io/pool` and `inference.io/accelerator` labels/taints
  between your node group definition, these manifests, and
  `../model-server.yaml` -- a mismatch on either the nodeSelector or the
  toleration is the single most common reason a pod sits `Pending` forever
  with an event that doesn't obviously point at the taint.
- The gateway (`../gateway.yaml`) already treats "no pod to route to" as a
  first-class state (`503` + `Retry-After`, see the main README), so a caller
  on the other side of the gateway sees a coherent "warming up" response
  during the node provisioning + model load window regardless of which
  autoscaler is doing the provisioning.
- `cooldownPeriod` (KEDA) and `consolidateAfter` (Karpenter) are two
  different idle-timers for two different things -- the former decides when
  the *pod* goes away, the latter when the now-empty *node* goes away. Keep
  them close together; a long gap between the two just means a node sits
  idle after its pod is already gone.
