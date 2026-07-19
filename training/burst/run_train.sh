#!/usr/bin/env bash
# Runs ON THE CLOUD TRAINER (Brev/Vast H100), inside docker/train.Dockerfile.
# corpus (HF) -> LLaMA-Factory train -> patch _name_or_path -> push to private HF.
# Usage: run_train.sh <config.yaml> <artifact-name e.g. persona-v1>
set -euo pipefail

CONFIG="${1:?usage: run_train.sh <config.yaml> <artifact-name>}"
NAME="${2:?usage: run_train.sh <config.yaml> <artifact-name>}"
: "${HF_TOKEN:?}" ; : "${HF_CORPUS_REPO:?}"
# default the push namespace to the token's actual HF user — a bad namespace here
# would strand the adapter AFTER the paid training run
HF_USER=$(python -c "from huggingface_hub import whoami; print(whoami()['name'])")
PUSH_REPO="${PUSH_REPO:-$HF_USER/$NAME}"
WORK=/workspace

# sanity: same gotcha checks as the Spark — bf16-capable CUDA torch, no cpu fallback
python - <<'PY'
import torch
assert torch.cuda.is_available(), "cloud trainer has no CUDA torch — wrong image"
print(f"trainer: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
      f"cuda {torch.version.cuda}")
PY

mkdir -p "$WORK/corpus" "$WORK/out"
hf download "$HF_CORPUS_REPO" corpus-v1.tar.gz --repo-type dataset --local-dir "$WORK"
tar xzf "$WORK/corpus-v1.tar.gz" -C "$WORK/corpus"

# register the dataset for LLaMA-Factory (sharegpt/messages format)
cat > "$WORK/corpus/dataset_info.json" <<'JSON'
{
  "abhi_corpus": {
    "file_name": "train.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"},
    "tags": {"role_tag": "role", "content_tag": "content",
             "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}
  }
}
JSON

# create the (private) push repo BEFORE training so a push failure surfaces now,
# not after hours of H100 time
hf repo create "$PUSH_REPO" --private 2>/dev/null || true
hf repo info "$PUSH_REPO" >/dev/null

llamafactory-cli train "$CONFIG"

OUT_DIR=$(python -c "import yaml,sys;print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
python training/burst/patch_checkpoint.py --checkpoint-dir "$OUT_DIR" \
  --placeholder "local://$NAME"

hf upload "$PUSH_REPO" "$OUT_DIR" . --commit-message "train $NAME ($(basename "$CONFIG"))"
echo "OK: pushed $NAME to hf.co/$PUSH_REPO — on the Spark run: make fetch-adapters"
echo "REMINDER: terminate this instance (spec §5 step 5)."
