#!/usr/bin/env bash
# Rescue upload: push the finished adapter from the brev box to HF, then clean up.
set -euo pipefail
: "${HF_TOKEN:?}"
INSTANCE="twin-train-persona-v1"
IMAGE="${TRAIN_IMAGE:-ghcr.io/abhione/abhi-twin-train:latest}"

brev exec "$INSTANCE" "
  set -euo pipefail
  sudo docker run --rm -v /ephemeral/workspace:/workspace -e HF_TOKEN='$HF_TOKEN' $IMAGE bash -c '
    set -euo pipefail
    python - <<PY
import json
p = \"/workspace/out/persona-v1/adapter_config.json\"
cfg = json.load(open(p))
print(\"base_model:\", cfg.get(\"base_model_name_or_path\"))
print(\"rank:\", cfg.get(\"r\"), \"alpha:\", cfg.get(\"lora_alpha\"))
PY
    hf upload abhione/persona-v1 /workspace/out/persona-v1 . \
      --exclude \"checkpoint-*/*\" \
      --commit-message \"train persona-v1 (QLoRA bnb4, Qwen2.5-72B-Instruct, 3 epochs, final loss ~1.25)\"
  '
"
echo "UPLOAD DONE"
