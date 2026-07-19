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
# not after hours of H100 time. NOTE: newer hf CLIs removed 'hf repo info' — use
# the python API for both create and verify (works across CLI versions).
python - <<PY
from huggingface_hub import create_repo, repo_info
create_repo("$PUSH_REPO", private=True, exist_ok=True)
repo_info("$PUSH_REPO")
print("push repo OK: $PUSH_REPO")
PY

# NGC-image gotcha: torchaudio's compiled ext can be ABI-broken against the NGC
# torch nightly, and llamafactory imports torchaudio unconditionally (mm_plugin)
# even for text-only runs. If it can't load, stub it out.
python - <<'PY'
try:
    import torchaudio  # noqa: F401
    print("torchaudio OK")
except Exception as e:
    import importlib.util, pathlib
    spec = importlib.util.find_spec("torchaudio")
    pkg = pathlib.Path(spec.origin).parent
    (pkg / "__init__.py").write_text('__version__ = "2.0.0+stub"\n')
    print(f"torchaudio broken ({type(e).__name__}) -> stubbed (text-only training)")
PY

# QLoRA path (bnb 4-bit): verify bitsandbytes works against the NGC torch; install
# only if missing. (History: AWQ base was abandoned — autoawq deprecated/ABI-broken,
# gptqmodel needs torch>=2.11 symbols the NGC 2.10 nightly lacks.)
python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA torch — wrong image"
try:
    import bitsandbytes as bnb
    print("bitsandbytes OK:", bnb.__version__)
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--no-build-isolation", "--no-deps", "bitsandbytes"])
    import bitsandbytes as bnb
    print("bitsandbytes installed:", bnb.__version__)
assert torch.cuda.is_available(), "torch lost CUDA after bnb install!"
PY

llamafactory-cli train "$CONFIG"

OUT_DIR=$(python -c "import yaml,sys;print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
# LoRA runs emit adapter_config.json (no config.json) — patch_checkpoint only
# applies to full-model checkpoints (e.g. voice full-SFT). Skip it for adapters.
if [[ -f "$OUT_DIR/config.json" ]]; then
  python training/burst/patch_checkpoint.py --checkpoint-dir "$OUT_DIR" \
    --placeholder "local://$NAME"
else
  echo "adapter-only checkpoint (no config.json) — skipping _name_or_path patch"
fi

hf upload "$PUSH_REPO" "$OUT_DIR" . --exclude "checkpoint-*/*" \
  --commit-message "train $NAME ($(basename "$CONFIG"))"
echo "OK: pushed $NAME to hf.co/$PUSH_REPO — on the Spark run: make fetch-adapters"
echo "REMINDER: terminate this instance (spec §5 step 5)."
