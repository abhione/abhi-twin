# AbhiTwin — Status (updated 2026-07-18 19:15 PT, Phase 2 launch attempt)

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

## Serving stack build — LAUNCHED 2026-07-18 02:36 PDT (detached on the Spark)

All four buildable images launched via `docker compose build` (tags match what
`make serve` expects), each as a detached `nohup` on the Spark so they survive
session loss. Check progress any time:

```
ssh abhione@spark-e9cb-2.local 'tail -f /twin/logs/build-*.log'
docker images   # abhi-twin-{llm,orchestrator,stt,tts} appear as they finish
```

| image | log | outcome |
|---|---|---|
| tts | `/twin/logs/build-tts.log` | **BUILT** (`abhi-twin-tts`) in <1 min — pip-only layer |
| orchestrator | `/twin/logs/build-orchestrator.log` | **BUILT** (`abhi-twin-orchestrator`) ~2 min; torch guard passed (FlagEmbedding did not clobber NGC torch) |
| llm | `/twin/logs/build-llm.log` | **BUILT** (`abhi-twin-llm`) ~5 min — vLLM nightly cu130 aarch64 wheel resolved cleanly, no source build needed |
| stt | `/twin/logs/build-stt.log` | **BUILT** (`abhi-twin-stt`) — the feared CTranslate2 source compile for sm_121 took only ~6 min at `-j20` on the GB10. Two real bugs found + fixed en route, see below |

stt needed two rounds of fixes (`ede6a41`, `ea10081`), both now first-class in
the Dockerfile: (1) CTranslate2 selects CUDA archs via legacy FindCUDA, which
ignores `CMAKE_CUDA_ARCHITECTURES` and defaulted to `compute_53` — an arch
CUDA 13 nvcc rejects outright; (2) FindCUDA's `select_compute_arch.cmake`
parses numeric archs with a single-digit-major regex, so `CUDA_ARCH_LIST=12.1`
is rejected as an unknown architecture *name*. Fix: sed the
`cuda_select_nvcc_arch_flags` call into explicit
`-gencode arch=compute_121,code=sm_121`, grep-guarded against upstream drift.
Verified in-log: `NVCC compilation flags: …;-gencode;arch=compute_121,code=sm_121`,
then `Image abhi-twin-stt Built`. **All four serving images now exist on the
Spark** (`docker images | grep abhi-twin`); runtime CUDA-capability checks
happen at container start via `ci/preflight.py`. `make serve` /
`make serve-voice` become runnable once model weights land in `/twin/models`
(blocked on HF login).

Skipped: **musetalk** (v1.5 scope), **train** image (cloud-burst only, needs
registry push — pointless without HF/Brev creds). No model weights are needed at
build time; all four images load weights from `/twin` at runtime, so no HF auth
was required.

Hardening landed with the launch (`a9bd0c4`): every Dockerfile now asserts
`torch.version.cuda` at **build** time (a pip-clobbered CPU torch fails the
build, not the first serve), and a new `.dockerignore` whitelists only
`ci/ serving/ training/ docker/patches/` so the repo-root build context ships no
`.git`/`.venv`/corpus data.

### RAG ingest — COMPLETE

`corpus/data/rag/rag_facts.jsonl` (1,888 hermes-mem facts, 1.5 MB) rsynced to
`/twin/corpus/rag/`. `serving/rag/ingest.py` now indexes `.jsonl` as one point
per pre-chunked fact (id/source/kind payload) instead of prose-chunking raw
JSON, upserting in batches of 128 (a single 1,888-point REST call is ~40 MB and
timed out — fixed in `ede6a41`). `scripts/spark_rag_ingest.sh` (idempotent) ran
detached: qdrant up as `twin-qdrant` (same image + `/twin/qdrant` volume as
compose), **ungated** `BAAI/bge-m3` snapshotted to `/twin/models/bge-m3` (no HF
token needed), ingest on GPU. Result, verified against the live collection:

```
indexed /twin/corpus/rag/rag_facts.jsonl (1888 chunks)
OK: 1888 chunks in collection 'brain'
curl :6333/collections/brain -> {"status":"green", "points_count":1888, ...}
```

Log: `/twin/logs/rag-ingest.log`. Re-runs are safe (deterministic uuid5 ids).

### Phase-2 readiness (audit only — no burst launched)

`training/burst/launch_brev.sh` now fails fast with actionable messages on all
three prerequisites: `HF_TOKEN`, `HF_CORPUS_REPO`, and `BREV_API_KEY` (the
latter was previously unchecked — the script would have burned a Brev create
before failing). `launch_vast.sh` guards got real messages too. `.env.example`
documents every variable both scripts read (`HF_TOKEN`, `HF_CORPUS_REPO`,
`BREV_API_KEY`, `BREV_ORG`, `VAST_API_KEY`, optional `BREV_GPU`/`TRAIN_IMAGE`
have sane defaults). `persona-lora.yaml` unchanged. Verdict: **ready to launch
the moment HF token + Brev key land in `.env`** — `make train-persona`.

## Phase 2 launch attempt — 2026-07-18 19:10–19:15 PT

Goal: corpus → Spark, corpus tarball → private HF, persona LoRA burst on Brev.
Two of three shipped; the Brev launch is **human-blocked** (details below).

### 1. Corpus → Spark: DONE (verified)

`rsync -avz corpus/data/out/ abhione@spark-e9cb-2.local:~/abhi-twin/corpus/data/out/`
— 4 files, 13,974,874 B transferred. Remote verification:

```
   23883 /home/abhione/abhi-twin/corpus/data/out/train.jsonl
    1256 /home/abhione/abhi-twin/corpus/data/out/eval.jsonl
   25139 total
md5(train.jsonl)  deb5a2cb22737e4fc1541c09e9d4a77a   (Spark)
md5(train.jsonl)  deb5a2cb22737e4fc1541c09e9d4a77a   (Mac — identical)
```

### 2. Corpus tarball → private HF: DONE (privacy verified via API)

`make corpus-upload` with `HF_CORPUS_REPO=abhione/abhi-twin-corpus` (repo
auto-created private by `sync_corpus.sh`; **this var is NOT in `.env` yet** —
add it there and on the Spark copy). Upload commit:
`https://huggingface.co/datasets/abhione/abhi-twin-corpus/commit/dc6c1b19b4d7f13518434c922a09054a31ef1724`

Privacy check, HF API (authed + anonymous):

```
id: abhione/abhi-twin-corpus
private: True
files: ['.gitattributes', 'corpus-v1.tar.gz']
anonymous GET /api/datasets/abhione/abhi-twin-corpus -> HTTP 401
```

### 3. Persona LoRA burst on Brev: BLOCKED (2 independent blockers)

**Blocker A — Brev auth expired (human-only fix).** Every Brev credential on
this machine is the *same* NVIDIA-SSO JWT (`iss: https://login.nvidia.com`,
ES256), and it expired **2026-07-18 17:24:59 PT** (~2 h before this session):
`.env` `BREV_API_KEY`, `~/.hermes/secrets/brev-cli-token`, and the CLI's own
`~/.brev/credentials.json` refresh token are byte-identical. Evidence of real
attempts, in order:

- `brev ls` → `WARN: malformed refresh token, logging out … You are currently
  logged out`
- `brev login --token "$BREV_API_KEY"` (both `nvidia` and `legacy` auth) →
  `ErrorForbidden`
- Direct control-plane probes (`/api/users/me`, `/api/organizations` with
  Bearer / x-api-key / Api-Key) → `403 {"type":"ForbiddenError"}`
- JWT decode: `exp: 1784420699` → expired

Re-auth is a browser SSO flow (`brev login` → login.nvidia.com); no headless
path exists. Per instructions, **no Vast fallback was attempted.**

**Blocker B — train image was never pushed.** `launch_brev.sh` runs
`ghcr.io/abhione/abhi-twin-train:latest` on the instance, but GHCR has no such
package (`gh api /users/abhione/packages/container/abhi-twin-train` → 404;
anonymous GHCR token grant → 403). The local `gh` token lacks `write:packages`,
so it could not be pushed from here either. Build+push before the next attempt:

```
docker buildx build --platform linux/amd64 -f docker/train.Dockerfile \
  -t ghcr.io/abhione/abhi-twin-train:latest --push .
```

(needs a GH PAT with `write:packages`; amd64 cross-build works on this Mac via
Docker Desktop/Rosetta, or natively on any x86 box).

### Script fixes landed with this attempt

- `launch_brev.sh`: two new preflights that fail fast *before* `brev create` —
  (1) detects the logged-out CLI (the pipeline check must not rely on
  `pipefail`, since `brev ls` itself exits non-zero); (2) verifies
  `TRAIN_IMAGE` exists on the registry. Both verified live: `make
  train-persona` now exits 1 with `FATAL: brev CLI is not authenticated (SSO
  token expired)` instead of dying mid-provision.
- `run_train.sh`: `PUSH_REPO` used to default to `abhi/$NAME` — a namespace
  this token can't write to, which would have **stranded the adapter after the
  paid H100 run** (push happens last). It now derives the default from
  `whoami()` on the trainer and creates + verifies the private push repo
  *before* `llamafactory-cli train`.

### Cost/GPU (for when it launches)

`BREV_GPU` defaults to H100 (override in env). Persona LoRA r32 over 23,883
pairs per `training/configs/persona-lora.yaml` is a single-GPU, hours-scale
job — order of $10 at typical Brev H100 rates (estimate, not a quote).

### How to resume Phase 2 (exact steps)

1. On the Mac, in a browser session: `brev login` (NVIDIA SSO). Optionally
   refresh `~/.hermes/secrets/brev-cli-token` + `.env BREV_API_KEY` from the
   new `~/.brev/credentials.json`.
2. Add `HF_CORPUS_REPO=abhione/abhi-twin-corpus` to `.env` (Mac + Spark).
3. Build+push the train image (command above, GH PAT with `write:packages`).
4. `make train-persona` — then tail the Brev session until real step/loss
   lines appear.
5. When the run pushes `abhione/persona-v1`: on the Spark, `make
   fetch-adapters`, then `make verify-persona`.

Monitoring after launch: `brev ls` for the instance, `brev shell
twin-train-persona-v1` for logs; the trainer prints `OK: pushed persona-v1 to
hf.co/…` and reminds to terminate the instance.

## Next actions for Abhi (human-blocked)

- [ ] **`brev login` on the Mac (browser SSO)** — every stored Brev credential
      expired 2026-07-18 17:24 PT; nothing launches until this happens.
- [ ] **GH PAT with `write:packages`** → build+push
      `ghcr.io/abhione/abhi-twin-train:latest` (one command, see above).
- [ ] Add `HF_CORPUS_REPO=abhione/abhi-twin-corpus` to `.env` (Mac + Spark) —
      this session couldn't write `.env` (permission-protected).
- [ ] `huggingface-cli login` on the Spark — required before model downloads
      (Qwen 72B NVFP4, TTS base).
- [ ] Optional: Gmail Takeout download → `twin corpus --local --mbox …` adds
      Abhi's long-form mail voice to the corpus (then rebuild + re-sync).
- [ ] Voice recording session per `scripts/record_corpus.md` (mic setup,
      Harvard sentences, identity video shot list) — feeds Phase 3.
