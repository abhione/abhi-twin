#!/usr/bin/env bash
# Vast.ai fallback trainer (cheapest H100 spot, ~$1.50-2.50/hr; spec §5 — verify
# NCCL/driver on every rental). Usage: launch_vast.sh <config.yaml> <name>. Host: mac.
set -euo pipefail

CONFIG="${1:?usage: launch_vast.sh <config.yaml> <artifact-name>}"
NAME="${2:?usage: launch_vast.sh <config.yaml> <artifact-name>}"
: "${VAST_API_KEY:?set VAST_API_KEY in .env (cloud.vast.ai account settings)}"
: "${HF_TOKEN:?set HF_TOKEN in .env (huggingface.co/settings/tokens)}"
: "${HF_CORPUS_REPO:?set HF_CORPUS_REPO in .env}"
IMAGE="${TRAIN_IMAGE:-ghcr.io/abhione/abhi-twin-train:latest}"

command -v vastai >/dev/null || { echo "pip install vastai" >&2; exit 1; }
vastai set api-key "$VAST_API_KEY" >/dev/null

echo "==> searching for an H100 offer"
OFFER=$(vastai search offers 'gpu_name=H100_SXM num_gpus=1 reliability>0.98' \
  --order dph --raw | python3 -c "import json,sys; print(json.load(sys.stdin)[0]['id'])")

echo "==> creating instance from offer $OFFER"
CID=$(vastai create instance "$OFFER" --image "$IMAGE" --disk 200 --raw \
  --env "-e HF_TOKEN=$HF_TOKEN -e HF_CORPUS_REPO=$HF_CORPUS_REPO -e PUSH_REPO=${PUSH_REPO:-abhi/$NAME}" \
  --onstart-cmd "bash training/burst/run_train.sh $CONFIG $NAME && touch /workspace/DONE" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])")

echo "==> instance $CID training; watch with: vastai logs $CID"
echo "    community hardware — verify env first: vastai execute $CID 'nvidia-smi'"
echo "    WHEN DONE (adapter on HF): vastai destroy instance $CID   <- do not forget"
