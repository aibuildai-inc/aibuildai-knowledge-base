#!/usr/bin/env bash
# Build the index and the service image, start the container, and check that it answers.
set -euo pipefail
trap 'echo "ERROR: Kb deployment failed at line $LINENO" >&2' ERR

cd "$(dirname "$0")/.."
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU index}"
: "${AIBUILDAI_KB_EMBED_MODEL:?set AIBUILDAI_KB_EMBED_MODEL}"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || {
  echo "CUDA_VISIBLE_DEVICES must name one GPU" >&2
  exit 1
}

port=8000
container=aibuildai-kb
image="aibuildai-kb:$(git rev-parse --short=12 HEAD)"
runtime_root="$(PYTHONPATH="$PWD" python -c \
  'from sourcing.utils.work_dir import runtime_root; print(runtime_root())')"
index_dir="$runtime_root/index"

echo "[1/4] build the current index"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m aibuildai_mcp.index \
  --plugins-root "$PWD/data/corpus/plugins" \
  --index-dir "$index_dir" \
  --model "$AIBUILDAI_KB_EMBED_MODEL" \
  --device cuda:0

echo "[2/4] build the service image"
docker build \
  --build-context "kb_index=$index_dir" \
  --build-arg "AIBUILDAI_KB_EMBED_MODEL=$AIBUILDAI_KB_EMBED_MODEL" \
  --label "org.opencontainers.image.revision=$(git rev-parse HEAD)" \
  -f deploy/Dockerfile \
  -t "$image" \
  .

echo "[3/4] start the service on port $port"
if docker container inspect "$container" >/dev/null 2>&1; then
  docker rm -f "$container" >/dev/null
fi
docker run -d --name "$container" -p "$port:8000" \
  --user "$(id -u):$(id -g)" \
  --gpus "device=$CUDA_VISIBLE_DEVICES" \
  --cpus 4 \
  --memory 12g \
  --pids-limit 1024 \
  --restart unless-stopped "$image"

echo "[4/4] check the service"
health=""
for _ in $(seq 1 150); do
  health="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 \
    "http://127.0.0.1:$port/healthz" || true)"
  [[ "$health" == 200 ]] && break
  sleep 2
done
[[ "$health" == 200 ]]

initialize="$(curl -fsS --max-time 10 \
  -X POST "http://127.0.0.1:$port/mcp" \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"deploy-probe","version":"1"}}}')"
grep -q '"result"' <<<"$initialize"

search="$(curl -fsS --max-time 30 \
  -X POST "http://127.0.0.1:$port/mcp" \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search_skills","arguments":{"query":"cross-validation","top_k":1}}}')"
grep -q '"result"' <<<"$search"
grep -Eq '"text"[[:space:]]*:[[:space:]]*"[^\"]+' <<<"$search"

served_hash="$(docker exec "$container" cat /app/data/corpus/index/corpus.sha256)"
[[ "$served_hash" == "$(<"$index_dir/corpus.sha256")" ]]

echo "DONE: $image serving http://127.0.0.1:$port/mcp"
