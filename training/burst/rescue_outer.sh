#!/usr/bin/env bash
# Rescue v2: base64-ship the inner script, run it in the container. Host: mac.
set -euo pipefail
: "${HF_TOKEN:?}"
INSTANCE="twin-train-persona-v1"
IMAGE="${TRAIN_IMAGE:-ghcr.io/abhione/abhi-twin-train:latest}"

B64=$(base64 < /tmp/rescue_inner.sh)
brev exec "$INSTANCE" "echo '$B64' | base64 -d > /tmp/rescue_inner.sh && chmod +x /tmp/rescue_inner.sh && head -2 /tmp/rescue_inner.sh"

brev exec "$INSTANCE" "sudo docker run --rm -v /ephemeral/workspace:/workspace -v /tmp/rescue_inner.sh:/rescue.sh:ro -e HF_TOKEN='$HF_TOKEN' $IMAGE /rescue.sh"
echo "RESCUE DONE"
