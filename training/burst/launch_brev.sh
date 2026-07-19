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

# preflight: the training image must exist on the registry or the remote docker run fails.
# The image is PRIVATE on ghcr — an authenticated manifest check is required (anonymous
# token requests 401/403 even when the image exists). GHCR_TOKEN also gets passed to the
# remote instance so it can docker-pull the private image.
GHCR_USER="${GHCR_USER:-abhione}"
GHCR_TOKEN="${GHCR_TOKEN:-$(gh auth token 2>/dev/null || true)}"
if [[ -z "$GHCR_TOKEN" ]]; then
  echo "FATAL: no GHCR_TOKEN and 'gh auth token' unavailable — needed to pull private $IMAGE" >&2
  exit 1
fi
repo_path="${IMAGE#ghcr.io/}"; repo_path="${repo_path%%:*}"
tag="${IMAGE##*:}"
pull_tok=$(curl -fsSL -u "$GHCR_USER:$GHCR_TOKEN" "https://ghcr.io/token?scope=repository:${repo_path}:pull" \
  | python3 -c "import json,sys; print(json.load(sys.stdin).get('token',''))" 2>/dev/null || true)
code=$(curl -fsSL -o /dev/null -w '%{http_code}' \
  -H "Authorization: Bearer ${pull_tok}" \
  -H "Accept: application/vnd.oci.image.index.v1+json" \
  -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
  "https://ghcr.io/v2/${repo_path}/manifests/${tag}" 2>/dev/null || echo 000)
if [[ "$code" != "200" ]]; then
  echo "FATAL: $IMAGE not found on registry (manifest HTTP $code). Build+push it first:" >&2
  echo "  docker buildx build --platform linux/amd64 -f docker/train.Dockerfile -t $IMAGE --push ." >&2
  exit 1
fi

echo "==> creating $INSTANCE ($GPU) on Brev"
# new brev CLI (2026): --gpu was replaced by search filters / --type. Filter by GPU
# name + enough disk for the 72B base + corpus; cheapest match is tried first with
# automatic fallback across types.
brev create "$INSTANCE" --gpu-name "$GPU" --min-disk 400 --min-total-vram 70

echo "==> running training remotely (config: $CONFIG -> $NAME)"
# brev exec runs non-interactively on the instance ('shell -- cmd' is no longer supported)
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
