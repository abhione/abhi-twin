#!/usr/bin/env bash
# Phase 0 verify gate (spec §12): nvidia-smi clean, torch capability (12,1),
# tailscale up, /twin dirs, docker + NVIDIA container toolkit. Host: spark.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "=== RUN ON SPARK === verify-phase0 must run on the Spark itself." >&2
  exit 2
fi

fail=0
check() {  # check <name> <cmd...>
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then
    echo "PASS  $name"
  else
    echo "FAIL  $name  ($*)"; fail=1
  fi
}

check "nvidia-smi"            nvidia-smi
check "GB10 GPU visible"      bash -c "nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi 'GB10\|NVIDIA'"
check "tailscale up"          bash -c "tailscale status | grep -qv 'Logged out'"
check "docker"                docker info
check "nvidia container toolkit" bash -c "docker info 2>/dev/null | grep -qi nvidia || nvidia-ctk --version"
for d in corpus checkpoints adapters models logs; do
  check "/twin/$d exists"     test -d "/twin/$d"
done

# torch: CUDA available + capability (12,1). This is the canonical gotcha check —
# a pip-installed torch silently CPU-falls-back and fails here.
# Torch lives in ~/twin-venv on the Spark (bare python3 can't see it);
# override with TWIN_PYTHON if the venv lives elsewhere.
PY="${TWIN_PYTHON:-$HOME/twin-venv/bin/python3}"
[[ -x "$PY" ]] || PY=python3
if "$PY" "$(dirname "$0")/../ci/preflight.py" --check torch; then
  echo "PASS  torch cuda capability (12,1)"
else
  echo "FAIL  torch cuda capability (12,1)"; fail=1
fi

if [[ $fail -eq 0 ]]; then
  echo "=== verify-phase0: ALL CHECKS PASSED ==="
else
  echo "=== verify-phase0: FAILURES (see above) ===" >&2
fi
exit $fail
