#!/usr/bin/env bash
# Create the /twin local storage layout (spec §4). Host: spark.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "=== RUN ON SPARK === setup_dirs.sh creates /twin on the Spark, not this host." >&2
  exit 2
fi

sudo mkdir -p /twin/{corpus,checkpoints,adapters,models,logs}
sudo chown -R "$(id -u):$(id -g)" /twin
echo "OK: /twin/{corpus,checkpoints,adapters,models,logs} ready"
