#!/usr/bin/env bash
# Package the cleaned corpus and upload it to the PRIVATE HF repo for cloud-burst
# training (spec §5 step 1). The corpus leaves the box ONLY this way. Host: mac | spark.
set -euo pipefail

OUT_DIR="${OUT_DIR:-corpus/data/out}"
: "${HF_CORPUS_REPO:?set HF_CORPUS_REPO in .env (private HF dataset repo)}"
: "${HF_TOKEN:?set HF_TOKEN in .env}"

TARBALL="$OUT_DIR/corpus-v1.tar.gz"
if [[ ! -f "$TARBALL" ]]; then
  echo "no $TARBALL — building it from $OUT_DIR" >&2
  tar czf "$TARBALL" -C "$OUT_DIR" train.jsonl eval.jsonl stats.json holdout.manifest.json
fi

# refuse to upload to a public repo — privacy hard constraint
VISIBILITY=$(hf repo info "$HF_CORPUS_REPO" --repo-type dataset 2>/dev/null | grep -i private || true)
if ! hf repo info "$HF_CORPUS_REPO" --repo-type dataset >/dev/null 2>&1; then
  hf repo create "$HF_CORPUS_REPO" --repo-type dataset --private
elif [[ -z "$VISIBILITY" ]]; then
  echo "FATAL: $HF_CORPUS_REPO is not private — refusing to upload corpus" >&2
  exit 1
fi

hf upload "$HF_CORPUS_REPO" "$TARBALL" corpus-v1.tar.gz --repo-type dataset
echo "OK: corpus uploaded to hf.co/datasets/$HF_CORPUS_REPO (private)"
