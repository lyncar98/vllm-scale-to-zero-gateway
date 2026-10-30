# Multi-stage build for a vLLM model server, with the served checkpoint
# baked into the image.
#
# Why bake the weights instead of downloading at pod start:
#   * Cold start. The GPU pool runs scale-to-zero, so a cold request already
#     pays node provisioning. Adding a multi-GB HuggingFace download on top
#     turns a ~60s cold start into several minutes.
#   * Availability. A pod that downloads weights at boot has a hard runtime
#     dependency on huggingface.co. Baked weights make the pod startable
#     whenever your registry is reachable.
#   * Auditability. "Which weights served this request" needs to be
#     answerable. An image digest answers it exactly; a runtime download from
#     a mutable ref does not.
#
# The cost is a large image (an 8B checkpoint in bf16 is roughly 16GB). Give
# the GPU node group a large root volume and consider a dedicated ECR/registry
# repo so a gateway fix doesn't rebuild or re-push this artifact.

# ---------------------------------------------------------------------------
# Stage 1 - fetch the checkpoint.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS weights

ARG MODEL_REPO=Qwen/Qwen3-8B

# MODEL_REVISION must be a full commit SHA in any environment you care about.
# "main" is a mutable branch pointer: two builds a week apart can produce
# different weights behind an identical image tag, which breaks the audit
# property this whole design exists to provide.
#
# Resolve the current SHA with:
#   curl -s https://huggingface.co/api/models/Qwen/Qwen3-8B | jq -r .sha
#
# It is left as "main" here only so a fresh clone builds without extra setup;
# wire your CI to pass an explicit SHA instead (fail the build if it's still
# this placeholder).
ARG MODEL_REVISION=main

ENV PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_TELEMETRY=1
RUN pip install "huggingface_hub[hf_transfer]==0.26.2"

# hf_transfer gives a substantially faster multi-threaded download, which
# matters when the artifact is tens of GB.
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Optional: only needed for gated repos.
ARG HF_TOKEN=""

WORKDIR /weights
RUN python -c "\
from huggingface_hub import snapshot_download; \
import os; \
snapshot_download( \
    repo_id=os.environ['MODEL_REPO'], \
    revision=os.environ['MODEL_REVISION'], \
    local_dir='/weights/model', \
    token=(os.environ.get('HF_TOKEN') or None), \
    ignore_patterns=['*.pth', '*.onnx', '*.msgpack', 'original/*'], \
)" \
  && du -sh /weights/model
# ^ ignore_patterns skips duplicate serialisations of the same weights
#   (PyTorch .pth / ONNX / flax) that vLLM never reads. On most Qwen repos this
#   is the difference between one copy and two.

# ---------------------------------------------------------------------------
# Stage 2 - runtime.
# ---------------------------------------------------------------------------
# vLLM's official image already carries CUDA, PyTorch and the OpenAI-compatible
# server. Pin to a current release that supports your target model family --
# v0.6.4 predates Qwen3, for example. Pinned by tag (not :latest) so a rebuild
# is reproducible.
FROM vllm/vllm-openai:v0.6.6

# vLLM's entrypoint is the OpenAI-compatible API server. Override it only to
# supply our own defaults; the flags stay overridable from the Helm values so
# tuning does not require an image rebuild.
COPY --from=weights /weights/model /opt/model

ENV MODEL_PATH=/opt/model

# Served model name is what clients put in the "model" field. Keeping it
# stable rather than the upstream repo id means swapping the base model for a
# fine-tune does not change the client contract.
ENV SERVED_MODEL_NAME=default

EXPOSE 8000

# --max-model-len is capped well below the model's theoretical context because
# KV cache is the binding constraint on a 24GB GPU; raising it trades
# concurrency for context. --gpu-memory-utilization leaves headroom so a burst
# of long prompts does not OOM the worker.
ENTRYPOINT ["/bin/sh", "-c", "exec python3 -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_PATH} \
  --served-model-name ${SERVED_MODEL_NAME} \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len ${MAX_MODEL_LEN:-8192} \
  --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.90} \
  --disable-log-requests \
  ${EXTRA_VLLM_ARGS}"]
# --disable-log-requests: prompts and completions are often governed customer
# content. Echoing them into this pod's stdout would copy that data into
# whatever log sink your cluster ships to, outside any governed audit path.
