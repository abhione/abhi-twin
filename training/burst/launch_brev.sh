#!/usr/bin/env bash
# Launch a fine-tune on Brev (NVIDIA's service — first choice, matches the Spark
# stack; spec §5). Usage: launch_brev.sh <config.yaml> <artifact-name>. Host: mac.
set -euo pipefail

CONFIG="${1:?usage: launch_brev.sh <config.yaml> <artifact-name>}"
NAME="${2:?usage: launch_brev.sh <config.yaml> <artifact-name>}"
: "${HF_TOKEN:?set HF_TOKEN in .env}"
: "${HF_CORPUS_REPO:?set HF_CORPUS_REPO in .env}"
INSTANCE="twin-train-$NAME"
GPU="${BREV_GPU:-H100}"
IMAGE="${TRAIN_IMAGE:-ghcr.io/abhione/abhi-twin-train:latest}"  # docker/train.Dockerfile

command -v brev >/dev/null || {
  echo "brev CLI not installed: https://developer.nvidia.com/brev — or use launch_vast.sh" >&2
  exit 1
}

echo "==> creating $INSTANCE ($GPU) on Brev"
brev create "$INSTANCE" --gpu "$GPU"

echo "==> running training remotely (config: $CONFIG -> $NAME)"
brev shell "$INSTANCE" -- bash -lc "
  set -euo pipefail
  docker run --gpus all --rm \
    -e HF_TOKEN='$HF_TOKEN' -e HF_CORPUS_REPO='$HF_CORPUS_REPO' -e PUSH_REPO='${PUSH_REPO:-abhi/$NAME}' \
    $IMAGE training/burst/run_train.sh $CONFIG $NAME
"

echo "==> deleting $INSTANCE (burst is done; adapters are on HF)"
brev delete "$INSTANCE"
echo "OK: on the Spark run 'make fetch-adapters' to pull $NAME"
