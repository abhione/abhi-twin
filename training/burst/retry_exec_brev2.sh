#!/usr/bin/env bash
# Retry training on the ALREADY-CREATED brev instance, mounting the LOCAL
# (fixed) run_train.sh over the image's baked-in copy. Host: mac.
set -euo pipefail

CONFIG="${1:?usage: retry_exec_brev2.sh <config.yaml> <artifact-name>}"
NAME="${2:?usage: retry_exec_brev2.sh <config.yaml> <artifact-name>}"
: "${HF_TOKEN:?set in .env}"
: "${HF_CORPUS_REPO:?set in .env}"
INSTANCE="twin-train-$NAME"
IMAGE="${TRAIN_IMAGE:-ghcr.io/abhione/abhi-twin-train:latest}"
GHCR_USER="${GHCR_USER:-abhione}"
GHCR_TOKEN="${GHCR_TOKEN:-$(gh auth token 2>/dev/null || true)}"
[[ -n "$GHCR_TOKEN" ]] || { echo "FATAL: no GHCR_TOKEN" >&2; exit 1; }

# ship the fixed script + config to the instance (base64 through exec — no scp needed)
B64=$(base64 < training/burst/run_train.sh)
brev exec "$INSTANCE" "echo '$B64' | base64 -d > /tmp/run_train.sh && chmod +x /tmp/run_train.sh && head -3 /tmp/run_train.sh"
C64=$(base64 < "$CONFIG")
brev exec "$INSTANCE" "echo '$C64' | base64 -d > /tmp/train_config.yaml && head -3 /tmp/train_config.yaml"

echo "==> running training remotely (config: $CONFIG -> $NAME), fixed script mounted"
brev exec "$INSTANCE" "
  set -euo pipefail
  echo '$GHCR_TOKEN' | sudo docker login ghcr.io -u '$GHCR_USER' --password-stdin
  sudo mkdir -p /ephemeral/hf-cache /ephemeral/workspace
  sudo docker run --gpus all --rm \
    -v /tmp/run_train.sh:/app/training/burst/run_train.sh:ro \
    -v /tmp/train_config.yaml:/app/$CONFIG:ro \
    -v /ephemeral/hf-cache:/root/.cache/huggingface \
    -v /ephemeral/workspace:/workspace \
    -e HF_TOKEN='$HF_TOKEN' -e HF_CORPUS_REPO='$HF_CORPUS_REPO' -e PUSH_REPO='${PUSH_REPO:-}' \
    $IMAGE training/burst/run_train.sh $CONFIG $NAME
"

echo "==> deleting $INSTANCE (burst is done; adapters are on HF)"
brev delete "$INSTANCE"
echo "OK: on the Spark run 'make fetch-adapters' to pull $NAME"
