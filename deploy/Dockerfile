# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install the Kb service and its runtime stack (txtai semantic retrieval, the
# sentence-transformers embedder, and the MCP Streamable-HTTP transport). The kb
# service imports only the aibuildai_mcp namespace, so this image copies that
# namespace and never the offline sourcing layer.
# The container uses CUDA PyTorch and receives one GPU from deploy/deploy.sh.
COPY pyproject.toml ./
COPY aibuildai_mcp ./aibuildai_mcp
# PyPI is the extra index because the PyTorch index prunes old NVIDIA dependency
# versions: on 2026-08-14 it no longer carried nvidia-cudnn-cu12==9.1.0.70, which
# torch 2.6.0 pins, and the build died resolving it. The torch wheel itself still
# comes from the cu124 index (pip prefers the primary); only the pruned NVIDIA
# dependencies fall through to PyPI, which keeps every version.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu124 \
      --extra-index-url https://pypi.org/simple \
      'torch==2.6.0' \
    && pip install --no-cache-dir '.[server]'

# Pre-bake the embedding weights at BUILD time, so a fresh container never pays a
# ~1.2GB Hugging Face download on its first request.
ARG AIBUILDAI_KB_EMBED_MODEL
ENV AIBUILDAI_KB_EMBED_MODEL=${AIBUILDAI_KB_EMBED_MODEL} \
    HF_HOME=/app/model-cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${AIBUILDAI_KB_EMBED_MODEL}')" \
    && chmod -R a+rwX /app/model-cache

# Weights are cached in the image now -- never reach the network at runtime.
ENV HF_HUB_OFFLINE=1 \
    HOME=/tmp

# Bake the skill corpus and the index from the same build input into one image.
COPY data/corpus/plugins ./data/corpus/plugins
# The skill search index is built OFFLINE on a GPU host (`python -m
# aibuildai_mcp.index --force` -- minutes on a GPU, hours on CPU) into the index
# dir, and copied in here byte-frozen, so this image build never re-embeds the
# corpus. deploy/deploy.sh passes the runtime index as the kb_index build context.
# A GPU-built index serves directly in this GPU image.
COPY --from=kb_index . ./data/corpus/index

EXPOSE 8000

# main() warms the index + the embedder synchronously before serving (so the
# first real query is fast). Serves Streamable HTTP at /mcp (+ /healthz), stateless,
# 0.0.0.0:8000, openly: every route serves every caller.
CMD python -m aibuildai_mcp.service \
    --plugins-root /app/data/corpus/plugins \
    --index-dir /app/data/corpus/index \
    --embed-model "$AIBUILDAI_KB_EMBED_MODEL" \
    --device cuda:0
