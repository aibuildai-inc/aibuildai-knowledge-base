#!/usr/bin/env bash
# Serve the repo source and an already-built runtime index in the foreground.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${AIBUILDAI_KB_PYTHON:-python}"
RUNTIME_ROOT="$(PYTHONPATH="$REPO" "$PYTHON" -c \
  'from sourcing.utils.work_dir import runtime_root; print(runtime_root())')"
HOST=127.0.0.1
PORT=8000
: "${AIBUILDAI_KB_EMBED_MODEL:?set AIBUILDAI_KB_EMBED_MODEL}"
# Required, not defaulted: the flag it feeds means "omit to use a ready GPU or
# else the CPU", so omitting it hands the choice of card - or of no card at all -
# to the library, on a host whose other cards may belong to someone else.
: "${AIBUILDAI_KB_DEVICE:?set AIBUILDAI_KB_DEVICE, e.g. cuda:0}"

echo "[kb] serving http://$HOST:$PORT/mcp on $AIBUILDAI_KB_DEVICE"
exec env \
  PYTHONPATH="$REPO" \
  PYTHONUNBUFFERED=1 \
  "$PYTHON" -m aibuildai_mcp.service \
  --plugins-root "$REPO/data/corpus/plugins" \
  --index-dir "$RUNTIME_ROOT/index" \
  --embed-model "$AIBUILDAI_KB_EMBED_MODEL" \
  --device "$AIBUILDAI_KB_DEVICE" \
  --host "$HOST" \
  --port "$PORT"
