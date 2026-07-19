#!/usr/bin/env bash
# Launch a fine-tune on Brev (NVIDIA's service — first choice, matches the Spark
# stack; spec §5). Usage: launch_brev.sh <config.yaml> <artifact-name>. Host: mac.
set -euo pipefail

CONFIG="${1:?usage: launch_brev.sh <config.yaml> <artifact-name>}"
NAME="${2:?usage: launch_brev.sh <config.yaml> <artifact-name>}"
: "${HF_TOKEN:?set HF_TOKEN in .env (huggingface.co/settings/tokens) — burst pulls the corpus + pushes the adapter through HF}"
: "${HF_CORPUS_REPO:?set HF_CORPUS_REPO in .env}"
: "${BREV_API_KEY:?set BREV_API_KEY in .env (brev.nvidia.com account settings) — or use launch_vast.sh}"
INSTANCE="twin-train-$NAME"
GPU="${BREV_GPU:-H100}"
IMAGE="${TRAIN_IMAGE:-ghcr.io/abhione/abhi-twin-train:latest}"  # docker/train.Dockerfile

command -v brev >/dev/null || {
  echo "brev CLI not installed: https://developer.nvidia.com/brev — or use launch_vast.sh" >&2
  exit 1
}

# preflight: brev auth is a browser SSO (login.nvidia.com) whose token expires;
# fail fast instead of dying mid-provision
auth_out=$(brev ls </dev/null 2>&1 || true)
if grep -qiE "logged out|would you like to log in|malformed refresh token" <<<"$auth_out"; then
  echo "FATAL: brev CLI is not authenticated (SSO token expired). Run 'brev login' in a browser session, then retry." >&2
  exit 1
fi

# preflight: the training image must exist on the registry or the remote docker run fails
if ! curl -fsSL "https://ghcr.io/token?scope=repository:${IMAGE#ghcr.io/}:pull" 2>/dev/null \
     | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin).get('token') else 1)" 2>/dev/null; then
  echo "FATAL: $IMAGE not found on registry. Build+push it first:" >&2
  echo "  docker buildx build --platform linux/amd64 -f docker/train.Dockerfile -t $IMAGE --push ." >&2
  exit 1
fi

echo "==> creating $INSTANCE ($GPU) on Brev"
brev create "$INSTANCE" --gpu "$GPU"

echo "==> running training remotely (config: $CONFIG -> $NAME)"
brev shell "$INSTANCE" -- bash -lc "
  set -euo pipefail
  docker run --gpus all --rm \
    -e HF_TOKEN='$HF_TOKEN' -e HF_CORPUS_REPO='$HF_CORPUS_REPO' -e PUSH_REPO='${PUSH_REPO:-}' \
    $IMAGE training/burst/run_train.sh $CONFIG $NAME
"

echo "==> deleting $INSTANCE (burst is done; adapters are on HF)"
brev delete "$INSTANCE"
echo "OK: on the Spark run 'make fetch-adapters' to pull $NAME"
