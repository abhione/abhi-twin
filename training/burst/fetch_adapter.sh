#!/usr/bin/env bash
# Pull trained adapters/checkpoints from the private HF repos into /twin (spec §5
# step 6), then verify they're offline-safe. Host: spark.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "=== RUN ON SPARK === fetch_adapter.sh downloads into /twin on the Spark." >&2
  exit 2
fi
: "${HF_TOKEN:?set HF_TOKEN in .env}"

declare -A REPOS=(
  ["${HF_PERSONA_REPO:-abhi/persona-v1}"]="/twin/adapters/persona-v1"
  ["${HF_VOICE_REPO:-abhi/voice-v1}"]="/twin/checkpoints/voice-v1"
  ["${HF_MUSETALK_REPO:-abhi/musetalk-identity-v1}"]="/twin/adapters/musetalk-identity-v1"
)

for repo in "${!REPOS[@]}"; do
  dest="${REPOS[$repo]}"
  if hf repo info "$repo" >/dev/null 2>&1; then
    echo "==> $repo -> $dest"
    hf download "$repo" --local-dir "$dest"
  else
    echo "skip: $repo not found (not trained yet)"
  fi
done

# offline-load gotcha gate: no hub-shaped _name_or_path in anything we just pulled
python3 ci/preflight.py --check checkpoints
echo "OK: adapters in /twin — restart serving to hot-load (make serve)"
