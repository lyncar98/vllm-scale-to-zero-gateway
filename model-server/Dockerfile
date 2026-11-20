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
# Serving target (v0.2): Qwen/Qwen3.6-35B-A3B-FP8 (~35 GiB weights) on a
# single g6e.2xlarge (1x NVIDIA L40S, 48 GB VRAM).
# The BF16 base checkpoint (Qwen/Qwen3.6-35B-A3B, ~71.9 GB / ~67 GiB) is kept
# in object storage as the governed fine-tuning base; it does not fit on a
# single 24 GB A10G (g5.xlarge) or 48 GB L40S (g6e.2xlarge), so the FP8
# checkpoint is what we bake and serve.

# ---------------------------------------------------------------------------
# Stage 1 - fetch the checkpoint.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS weights

ARG MODEL_REPO=Qwen/Qwen3.6-35B-A3B-FP8

# MODEL_REVISION must be a full commit SHA in any environment you care about.
# "main" is a mutable branch pointer: two builds a week apart can produce
# different weights behind an identical image tag, which breaks the audit
# property this whole design exists to provide. The build fails below if
# this is still "main" -- pass a real SHA with --build-arg.
#
# Resolve the current SHA with:
#   curl -s https://huggingface.co/api/models/Qwen/Qwen3.6-35B-A3B-FP8 | jq -r .sha
#
# Pinned SHA validated for v0.2:
#   c653bba46dfa06e9658dcc0b0dc87fe4685f14a2
ARG MODEL_REVISION=main
RUN if [ "$MODEL_REVISION" = "main" ]; then \
      echo "MODEL_REVISION is still 'main'. Pass a pinned commit SHA:" >&2; \
      echo "  docker build --build-arg MODEL_REVISION=c653bba46dfa06e9658dcc0b0dc87fe4685f14a2 ..." >&2; \
      echo "A mutable branch pointer breaks the audit property this design exists to provide." >&2; \
      exit 1; \
    fi

ENV PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_TELEMETRY=1
RUN pip install "huggingface_hub[hf_transfer]==0.26.2"

# hf_transfer gives a substantially faster multi-threaded download, which
# matters when the artifact is tens of GB.
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Optional: only needed for gated repos.
ARG HF_TOKEN=""
ENV MODEL_REPO=${MODEL_REPO}
ENV MODEL_REVISION=${MODEL_REVISION}
ENV HF_TOKEN=${HF_TOKEN}

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
# The Qwen3.6 vLLM serving recipe requires vLLM >= 0.17.0 for the hybrid
# Qwen3_5MoeForConditionalGeneration architecture (30 Gated DeltaNet linear-
# attention layers + 10 full-attention GQA layers, 256 routed experts). The
# v0.8.5 image from v0.1 of this repo predates Qwen3.6 and cannot load the
# checkpoint. We pin v0.19.0 by tag (not :latest) so rebuilds stay
# reproducible.
FROM vllm/vllm-openai:v0.19.0

# vLLM's entrypoint is the OpenAI-compatible API server. Override it only to
# supply our own defaults; the flags stay overridable from Kubernetes env vars
# so tuning does not require rebuilding a ~35 GiB image layer.
COPY --from=weights /weights/model /opt/model

ENV MODEL_PATH=/opt/model

# Served model name is what clients put in the "model" field. Keeping it
# stable rather than the upstream repo id means swapping the base model for a
# fine-tune does not change the client contract.
ENV SERVED_MODEL_NAME=default

# Tool-call parser from the Qwen3.6 vLLM recipe. In our baseline workload,
# 82% of teacher runs invoke tools (2.7 tool calls/run overall), so structured
# tool-call parsing is enabled by default and overridable via TOOL_CALL_PARSER.
ENV TOOL_CALL_PARSER=qwen3_coder

EXPOSE 8000

# Flag notes for Qwen3.6-35B-A3B-FP8 on a single 48 GB L40S (g6e.2xlarge):
#   * --trust-remote-code and --reasoning-parser qwen3: required for the
#     Qwen3.6 MoE architecture and thinking traces.
#   * --enable-auto-tool-choice --tool-call-parser qwen3_coder: required by
#     the Qwen3.6 recipe so tool calls return structured tool_calls objects
#     instead of raw XML/text in the completion body.
#   * --max-model-len: sized from workload baseline + physical KV budget (see
#     docs/kv-cache-sizing.md and scripts/size_kv_cache.py). If MAX_MODEL_LEN
#     is set, pass it; otherwise let EXTRA_VLLM_ARGS or vLLM's config govern.
#   * --no-enable-log-requests: replaces --disable-log-requests (removed in
#     vLLM v0.19). Prompts and completions are governed customer content and
#     must never echo into pod stdout.
ENTRYPOINT ["/bin/sh", "-c", "exec python3 -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_PATH} \
  --served-model-name ${SERVED_MODEL_NAME} \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser ${TOOL_CALL_PARSER:-qwen3_coder} \
  ${MAX_MODEL_LEN:+--max-model-len ${MAX_MODEL_LEN}} \
  --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.95} \
  --no-enable-log-requests \
  ${EXTRA_VLLM_ARGS}"]
