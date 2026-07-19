#!/usr/bin/env bash
# Inner rescue script: runs INSIDE the train container on the brev box.
set -euo pipefail
python - <<'PY'
import json
cfg = json.load(open("/workspace/out/persona-v1/adapter_config.json"))
print("base_model:", cfg.get("base_model_name_or_path"))
print("rank:", cfg.get("r"), "alpha:", cfg.get("lora_alpha"))
PY
hf upload abhione/persona-v1 /workspace/out/persona-v1 . \
  --exclude "checkpoint-*/*" \
  --commit-message "train persona-v1 (QLoRA bnb4, Qwen2.5-72B-Instruct, 3 epochs)"
echo "UPLOAD_COMPLETE"
