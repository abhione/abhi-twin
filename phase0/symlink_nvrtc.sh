#!/usr/bin/env bash
# Symlink libnvrtc.so.13 -> libnvrtc.so.12.8 for packages that pin CUDA 12.8
# (Chatterbox et al., per martimramos). Idempotent. Host: spark.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "=== RUN ON SPARK === symlink_nvrtc.sh patches CUDA libs on the Spark." >&2
  exit 2
fi

CUDA_LIB="${CUDA_LIB:-/usr/local/cuda/lib64}"
SRC="$CUDA_LIB/libnvrtc.so.13"
DST="$CUDA_LIB/libnvrtc.so.12.8"

if [[ ! -e "$SRC" ]]; then
  echo "FAIL: $SRC not found — is CUDA 13 installed?" >&2
  exit 1
fi
if [[ -e "$DST" ]]; then
  echo "OK: $DST already present"
else
  sudo ln -s "$SRC" "$DST"
  echo "OK: symlinked $DST -> $SRC"
fi
