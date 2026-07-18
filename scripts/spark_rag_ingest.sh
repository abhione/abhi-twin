#!/usr/bin/env bash
# RAG ingest bootstrap: qdrant up -> BGE-M3 snapshot (ungated, no HF token) ->
# index /twin/corpus/rag/*.jsonl into the `brain` collection. Host: spark.
# Idempotent. Run detached:
#   nohup bash scripts/spark_rag_ingest.sh > /twin/logs/rag-ingest.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${TWIN_VENV:-$HOME/twin-venv}"
MODEL_DIR=/twin/models/bge-m3
SRC=/twin/corpus/rag

echo "== rag ingest start: $(date -Is)"

# 1. qdrant — same image + storage volume as compose.yaml, so `make serve`
#    later sees the same collections; container name differs to avoid clashes.
if ! docker ps --format '{{.Names}}' | grep -qx twin-qdrant; then
  docker rm -f twin-qdrant >/dev/null 2>&1 || true
  docker run -d --name twin-qdrant -p 6333:6333 \
    -v /twin/qdrant:/qdrant/storage qdrant/qdrant:v1.12.4
fi
for _ in $(seq 30); do
  curl -fsS localhost:6333/readyz >/dev/null 2>&1 && break
  sleep 1
done

# 2. deps + BGE-M3 snapshot (BAAI/bge-m3 is ungated — no login required)
"$VENV/bin/pip" install --quiet FlagEmbedding qdrant-client click
if [ ! -f "$MODEL_DIR/config.json" ]; then
  "$VENV/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-m3', local_dir='$MODEL_DIR')
"
fi

# 3. index every fact file (one point per pre-chunked fact line)
QDRANT_HOST=localhost PYTHONPATH=. "$VENV/bin/python" -m serving.rag.ingest \
  --source "$SRC" --collection brain --glob '**/*.jsonl'

# 4. spot-check the collection is non-empty
curl -fsS localhost:6333/collections/brain | head -c 400
echo
echo "== rag ingest done: $(date -Is)"
