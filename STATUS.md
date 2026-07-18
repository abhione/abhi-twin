# AbhiTwin — Status (updated 2026-07-17, Phase 1 complete)

Phase 0 (Spark bring-up) and Phase 1 (corpus) are done. All output below is
real command output (trimmed).

## Phase 0 — Spark bring-up: COMPLETE

`make verify-phase0` on the Spark: **ALL CHECKS PASSED** (nvidia-smi, GB10
visible, tailscale, docker + nvidia toolkit, /twin dirs, torch CUDA capability
`(12,1)`, torch 2.14.0.dev20260717+cu130). `ci/preflight.py`: 7 PASS, 1 PEND
(onnxruntime intentionally lives in the musetalk image, v1.5). Serving base
images pulled (`nvcr.io/nvidia/pytorch:25.11-py3`, `qdrant/qdrant:v1.12.4`).
Details in git history (`89097af` and earlier).

## Phase 1 — corpus v2: COMPLETE

### Sources extracted (six; on this Mac)

| source | items extracted |
|---|---|
| imessage | 116,838 |
| hermes (agent sessions) | 1,630 |
| openclaw (agent sessions) | 463 |
| apple_notes | 347 |
| github (PRs) | 177 |
| agent_memory → `corpus/data/rag` | RAG facts stream (not persona pairs) |

Gmail Takeout and Slack export are absent (human-blocked, optional — see Next).

### Pipeline (`corpus/build.py`, presidio engine)

119,455 extracted → dedup −892 → presidio scrub (91,653 entity replacements)
→ length filter −91,574 → promptless −1,849 → **25,140 pairs** → frozen
holdout **1,256 eval / 23,884 train**. (PPL filter + prompt reconstruction
deferred to the Spark rebuild — RUN ON SPARK markers in build.py.)

`corpus/data/out` is now the canonical output dir (the stale five-source v1
build was deleted; v2 renamed into place).

### Verify gate — `python corpus/verify.py --out corpus/data/out`

```
PASS  25140 pairs (>= 8000)
PASS  PII scrubbed (engine: presidio)
PASS  holdout frozen + consistent (1256 eval pairs)
=== verify-corpus: ALL CHECKS PASSED ===
```

### Privacy audit — evidence

An audit grep over the built corpus found residuals presidio does not cover:
13 shared-credential lines (`password: …` in messages), 1 AWS
`AWSAccessKeyId=AKIA…` pre-signed URL (3 occurrences), 4 typo-domain emails
(`.con`/`.calm`), 1 URL-encoded `tel:` dial-in. Fixed, not just reported: a
residual scrub stage (`scrub_residuals` in `corpus/pipeline/pii.py`) now runs
inside both PII engines on every build, and `corpus/rescrub.py` applied it to
the built corpus in place (36 replacements; eval IDs untouched, so the
holdout manifest still validates — see verify output above). Post-scrub audit:

```
raw emails (non-placeholder)                      0
US-format phone numbers                           0
AKIA keys / bearer tokens / private key blocks    0 / 0 / 0
credential values on a password:/key= line        0  (all are <CREDENTIAL>, decoded-text scan)
placeholders present: <PERSON> 20,863  <PHONE_NUMBER> 418  <EMAIL_ADDRESS> 189  <CREDENTIAL> 29
```

`git check-ignore corpus/data/out` → ignored (`.gitignore:10: corpus/data/`).
Corpus data is not in git and never leaves the box unencrypted.

### Spark sync — confirmed

Scrubbed corpus re-synced: `rsync corpus/data/out/ →
abhione@spark-e9cb-2.local:/twin/corpus/out/` — 4 files present on the Spark
(train.jsonl 13,291,041 B, eval.jsonl, stats.json, holdout.manifest.json),
spot-check `grep -c AKIA… train.jsonl` → 0 on the Spark.

### Mac test suite

`make test-local`: ruff clean, **90 passed**, preflight local checks PASS.

## Next actions for Abhi (human-blocked)

- [ ] `huggingface-cli login` on the Spark — required before model downloads
      (Qwen 72B NVFP4, BGE-M3, TTS base).
- [ ] Brev API key into `.env` on the Mac (copy `.env.example`; the CLI
      auto-loads it — `TWIN_SPARK_HOST` is already set to the real hostname).
- [ ] Optional: Gmail Takeout download → `twin corpus --local --mbox …` adds
      Abhi's long-form mail voice to the corpus (then rebuild + re-sync).
- [ ] Voice recording session per `scripts/record_corpus.md` (mic setup,
      Harvard sentences, identity video shot list) — feeds Phase 3.
- [ ] Then Phase 2: `twin train persona` (persona LoRA cloud-burst on Brev,
      config `training/configs/persona-lora.yaml`).
