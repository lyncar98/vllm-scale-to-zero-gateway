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

**Gateway health is separate from model health.** `/healthz` (liveness) never
touches the upstream; `/readyz` (readiness) does. If liveness depended on the
model server, a GPU pool scaled to zero would get the *gateway* killed and
restarted in a loop -- exactly when it's needed most, to tell callers "warming
up" instead of "connection refused."

**Scale-from-zero is answered with 503 + Retry-After, not a timeout.** A
caller on the other side of this gateway should be able to treat "warming"
as a distinct, expected state -- not indistinguishable from a real failure.

**Recreate, not rolling, deployments for the model server.** A rolling update
briefly runs two GPU pods. On a one-GPU-per-replica deployment that doubles
cost during every rollout, and can fail outright when quota allows only one
node.

**Routing by served name, from an allow-list.** The gateway never computes
served names or guesses at a model's existence -- it routes only names present
in a generated routing table (`scripts/generate_routes.py`), and returns `404
model_not_found` for anything else. A routing file that's present but
unreadable is a hard `503`, not an empty table that silently sends traffic to
the wrong model.

**Content-free logging.** vLLM runs with `--disable-log-requests`. The
gateway's own usage log line carries the served model name, status, latency
and token counts -- never the prompt or the completion. If you have a
regulatory reason to keep data out of general-purpose logs (and even if you
don't), this is a cheap default to keep.

**Weights are baked into the model-server image.** Rather than downloading at
pod start: a cold request already pays node-provisioning latency, and adding
a multi-GB download on top turns a short cold start into several minutes.
It also makes "which weights served this request" answerable by an image
digest instead of a mutable ref. The cost is a large image and a rebuild per
model swap -- worth it until you're serving enough model variants that
per-adapter image builds become the bottleneck, at which point loading a
base model + adapter at startup (vLLM's `--enable-lora`) is the next step.

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
