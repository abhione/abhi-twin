#!/usr/bin/env bash
# Retry the training exec on an ALREADY-CREATED brev instance (skips brev create).
# Usage: retry_exec_brev.sh <config.yaml> <artifact-name>. Host: mac.
set -euo pipefail

CONFIG="${1:?usage: retry_exec_brev.sh <config.yaml> <artifact-name>}"
NAME="${2:?usage: retry_exec_brev.sh <config.yaml> <artifact-name>}"
: "${HF_TOKEN:?set HF_TOKEN in .env}"
: "${HF_CORPUS_REPO:?set HF_CORPUS_REPO in .env}"
INSTANCE="twin-train-$NAME"
IMAGE="${TRAIN_IMAGE:-ghcr.io/abhione/abhi-twin-train:latest}"
GHCR_USER="${GHCR_USER:-abhione}"
GHCR_TOKEN="${GHCR_TOKEN:-$(gh auth token 2>/dev/null || true)}"
[[ -n "$GHCR_TOKEN" ]] || { echo "FATAL: no GHCR_TOKEN" >&2; exit 1; }

# wait for DNS on the remote box before pulling (boot race seen Jul 18 2026)
echo "==> waiting for remote DNS"
brev exec "$INSTANCE" '
  for i in $(seq 1 30); do
    getent hosts ghcr.io >/dev/null 2>&1 && { echo "DNS OK"; exit 0; }
    sleep 5
  done
  echo "DNS never came up" >&2; exit 1
'

echo "==> running training remotely (config: $CONFIG -> $NAME)"
brev exec "$INSTANCE" "
  set -euo pipefail
  echo '$GHCR_TOKEN' | sudo docker login ghcr.io -u '$GHCR_USER' --password-stdin
  sudo docker run --gpus all --rm \
    -e HF_TOKEN='$HF_TOKEN' -e HF_CORPUS_REPO='$HF_CORPUS_REPO' -e PUSH_REPO='${PUSH_REPO:-}' \
    $IMAGE training/burst/run_train.sh $CONFIG $NAME
"

echo "==> deleting $INSTANCE (burst is done; adapters are on HF)"
brev delete "$INSTANCE"
echo "OK: on the Spark run 'make fetch-adapters' to pull $NAME"
