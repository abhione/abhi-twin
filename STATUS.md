# AbhiTwin — Status (updated 2026-07-19 late, Phase 4: serving gates run — persona PASS, voice PASS, e2e FAIL 3.2s/3.0s)

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

## Phase 3 — persona-v1 SERVED on the Spark + eval gate: PASSED (2026-07-19)

Phase 2 completed between updates (H100 burst, 3 epochs, train loss ~1.25;
adapter fetched to `/twin/adapters/persona-v1`). This session brought the core
serving stack up on the Spark and ran the persona eval gate. All output below
is real command output (trimmed).

### 1. Downloads verified complete (Spark)

- Base model `/twin/models/qwen2.5-72b-instruct-awq`: `du -sh` → **39G**;
  scripted check of `model.safetensors.index.json` → `shards needed: 11
  missing: []`. No `hf download` process running.
- Adapter `/twin/adapters/persona-v1`: `adapter_model.safetensors`
  (1,684,427,792 B), rank 32 / alpha 64 per `adapter_config.json`.
- Holdout present: `/twin/corpus/out/eval.jsonl` + `holdout.manifest.json`.
- `sentence-transformers/all-mpnet-base-v2` was **missing** → downloaded to
  `/twin/models/all-mpnet-base-v2` (public model, hf CLI).

### 2. Serve wiring fixes (committed `8841fb9`)

- **`.env` never reached compose**: `docker compose -f docker/compose.yaml`
  resolves `.env` relative to `docker/`, not the repo root — every `TWIN_*`
  override was silently ignored. Fixed in the Makefile with
  `--project-directory .`.
- **`--max-lora-rank 32`**: persona LoRA is rank 32; vLLM's default cap is 16,
  so the adapter would have failed to load. Added via a compose `command:`
  override (no image rebuild needed).
- **`--gpu-memory-utilization 0.42`** (`TWIN_GPU_MEM_UTIL`): GB10 unified
  memory means vLLM's 0.9 default would target ~109 GB of the shared 121 GB;
  0.42 ≈ the spec §2 50 GB LLM budget line. Post-load `free -g`: 60 GB used
  of 121 — within budget.
- Spark `.env` was writable → appended
  `TWIN_LLM_MODEL_DIR=/twin/models/qwen2.5-72b-instruct-awq` and
  `TWIN_PERSONA_ADAPTER=/twin/adapters/persona-v1`. Verified resolved via
  `docker compose … config`.
- `ci/preflight.py --check checkpoints` only scanned `/twin/checkpoints` and
  would have missed the adapter gotcha → now also scans
  `/twin/adapters/*/adapter_config.json` for hub-pointing base paths.

### 3. Adapter `_name_or_path` gotcha — caught and patched

```
FAIL  checkpoints  hub-pointing model path (offline-load bug):
      /twin/adapters/persona-v1/adapter_config.json:
      base_model_name_or_path='Qwen/Qwen2.5-72B-Instruct'
```

Patched to `/twin/models/qwen2.5-72b-instruct-awq` (backup kept at
`adapter_config.json.bak`) → re-run: `PASS checkpoints configs offline-safe`.

### 4. `make serve` — core stack UP

One conflict: the ad-hoc `twin-qdrant` container from the RAG ingest held port
6333. Same image (`qdrant/qdrant:v1.12.4`) + same `/twin/qdrant` bind mount as
the compose service, so it was stopped/removed and the compose-managed qdrant
took over the same storage (1,888-point `brain` collection persists on disk).
vLLM loaded the 72B AWQ in ~8.5 min; `curl :8000/v1/models` lists
`persona-v1` + `qwen2.5-72b` (base aliases) **and** the LoRA entry with
`root: /twin/adapters/persona-v1`. Containers: `abhi-twin-llm-1`,
`abhi-twin-qdrant-1`, `abhi-twin-orchestrator-1` all Up.

### 5. Smoke test (temp 0, "how do you feel about filler narration in agent replies?")

**persona-v1** (178 completion tokens, finish `stop`):

> I think filler narration in agent replies can be a double-edged sword. On one
> hand, it can make the conversation feel more natural and human-like, which
> can enhance the user experience. It can also provide context or set the tone
> for the interaction.
>
> On the other hand, too much filler can make the responses seem less direct
> and efficient, which might frustrate users who are looking for quick, precise
> answers. It's important to strike a balance between making the conversation
> engaging and ensuring that the information is delivered clearly and
> concisely. […]

**qwen2.5-72b base** (259 completion tokens, finish `stop`):

> As an AI language model, I don't have personal feelings or emotions, but I
> can provide an objective perspective on the use of filler narration in agent
> replies.
>
> Filler narration, such as "Let me think for a moment" or "I'm just checking
> that," can serve several purposes in agent replies:
>
> 1. **Humanization**: It can make the interaction feel more natural […]

Outputs differ at temperature 0 (and PPL differs ~6× below), proving the LoRA
name wins routing over the identically-named base alias. Cosmetic quirk: the
response `model` echo field says `persona-v1` for both requests (vLLM echoes
the primary served name); route by request `model`, not the echo.

### 6. Eval gate — `make verify-persona`: **PASSED**

`eval/persona.py` had a dead `gate_ppl_pct` param (PPL was never computed).
Implemented held-out reply-only PPL via vLLM's `prompt_logprobs` extension
against both model names, chat-templated with the serving tokenizer
(committed `1f0cef8`). The LLM judge is the local vLLM base model
(`qwen2.5-72b`), order-swapped A/B, n=30 from the frozen holdout.

```
{
  "ab_indistinguishable": 0.7333333333333333,
  "style_cosine": 0.26130561394738133,
  "boilerplate_rate": 0.0,
  "ppl_base": 148.6741954647929,
  "ppl_persona": 25.484387629945555,
  "ppl_within_pct": true,
  "n": 30
}
=== verify-persona: PASSED ===
```

| gate | threshold | value | verdict |
|---|---|---|---|
| blind A/B indistinguishable | ≥ 30% | **73.3%** | PASS |
| held-out PPL vs base | within +15% | **25.48 vs 148.67** (0.17×) | PASS |
| boilerplate rate | ≤ 5% | **0.0%** | PASS |
| style cosine (informational, no gate) | — | 0.261 | n/a |

Notes on the numbers: `ppl_base` 148.7 is high because the holdout is
idiosyncratic personal-message text the vanilla model has never seen;
persona-v1 at 25.5 is the trained-in adaptation, far inside the 15% band.
The 73.3% A/B rate counts judge said "indistinguishable" *or* picked the
twin's slot as the human; a same-family judge (Qwen judging Qwen outputs) may
inflate this — worth re-running with an external judge before Phase 5's 40%
gate, but the 30% Phase-2/3 gate clears with 2.4× margin. Style cosine 0.261
(mpnet) is modest — real replies are short/informal, twin replies run longer;
no gate is defined on it.

### 7. Deviations / operational notes

- `verify-persona` now runs **inside the llm container**
  (`compose exec llm python eval/persona.py --gate`): the eval needs NGC
  torch + the vLLM endpoint + `/twin`. `docker/llm.Dockerfile` now
  `COPY eval/ eval/`; since the built image predates that, `eval/` was
  `docker cp`'d in for this run. **On next llm image rebuild the cp step
  disappears.**
- `sentence-transformers` pinned to **3.4.1** in the container, installed
  `--no-deps` (deps already in the NGC image; the 5.x line imports the NGC
  image's torchcodec, which crashes on missing ffmpeg libs — and unpinned
  pip would risk clobbering the NGC torch). Not yet baked into the
  Dockerfile — add `pip install --no-deps sentence-transformers==3.4.1` on
  next rebuild.
- Raw smoke JSONs archived on the Spark and Mac at `/tmp/smoke_persona-v1.json`,
  `/tmp/smoke_qwen2.5-72b.json`; gate log at `/tmp/verify-persona.log`.
- Spark repo clone is clean (all edits shipped via git from the Mac); the only
  Spark-side mutations were `.env` additions, the adapter patch (with `.bak`),
  and the qdrant container swap.

## Phase 4 — serving gates (2026-07-19, second session)

Full voice stack up (`llm`, `tts`, `stt`, `qdrant`, `orchestrator`). All output
below is real command output (trimmed).

### 1. `make verify-persona` re-run on the BAKED llm image: **PASSED**

The rebuilt image (`ba69d0b`) carries `eval/` + pinned deps — no `docker cp`
steps were needed. Fresh generation run, same frozen holdout, n=30:

```
{
  "ab_indistinguishable": 0.6666666666666666,
  "style_cosine": 0.23740927540853238,
  "boilerplate_rate": 0.0,
  "ppl_base": 148.6796911492048,
  "ppl_persona": 25.477797503077582,
  "ppl_within_pct": true,
  "n": 30
}
=== verify-persona: PASSED ===
```

Self-judge A/B 66.7% this run vs 73.3% last run — generation sampling
variance; every gate still clears (≥30% bar, PPL 25.48 vs 148.68, 0%
boilerplate). Pairs persisted to `/twin/eval/ab_pairs.json` (seed 7,
order-swapped) for external re-judging.

### 2. External-judge A/B (claude-haiku-4-5 over the SAME pairs): **26.7%**

`make verify-persona-external` on the Mac (30 sequential claude-CLI calls,
`env -u ANTHROPIC_API_KEY` so the Max sub is billed, never the API key).

**A scoring bug was caught en route**: the first run reported 3.3%, but the
per-pair log showed 6/30 human-miss votes (20%). Haiku answers `B.` /
`**A**` / letter-plus-prose, and the exact-match scorer silently counted any
verbose verdict as not-fooled. Fixed in `0865457` (`_verdict_token`: first
alphanumeric token; external judgments now persisted in the results JSON;
regression test added — `make test-local` 94 passed, ruff clean). Re-run with
the fix:

```
{
  "judge": "claude-cli:claude-haiku-4-5-20251001",
  "ab_indistinguishable": 0.26666666666666666,   # 8/30: 6 wrong picks + 2 "indistinguishable"
  "n": 30,
  "self_judge_ab": 0.6666666666666666
}
```

Read: the same-family Qwen judge inflates A/B ~2.5× (66.7% vs 26.7% on
identical pairs). The official spec §12 gate (local judge) passes; an
external judge lands just **under** the 30% bar and well under Phase 5's 40%.
Persona quality against a frontier judge is the real Phase-5 risk — more
corpus (Gmail long-form) and/or another LoRA round is the lever.

### 3. `make verify-voice` re-run on a quiet box: **PASSED**

The earlier 1.959 RTF failure was measured while vLLM was mid-reload; with
the box quiet it passes with 2× headroom, no code change:

```
{ "rtf": 0.71, "audio_seconds": 19.44, "wall_s": 13.806,
  "warmup_s": 0.97, "peak_amplitude": 22784, "nan_in_synth_path": false }
=== verify-voice: PASSED ===
```

### 4. `make verify-e2e` (voice roundtrip < 3 s): **FAILED — 3.06–3.34 s warm**

First run: **13.8 s** — the persona node had no `max_tokens` cap, so voice
replies ran long, and roundtrip = STT + LLM decode + non-streaming TTS, the
last two linear in reply length. Fixed in `9123686` (voice requests get a
spoken-reply brevity instruction + `max_tokens 60`; orchestrator rebuilt +
redeployed). Trials after the fix:

| trial | roundtrip_s | note |
|---|---|---|
| 1 | 9.15 | cold start — BGE-M3 embedder loads on first retrieval |
| 2–6 | 3.06, 3.30, 3.16, 3.23, 3.34 | warm steady state |

Stage breakdown (measured individually, quiet box): STT 0.23 s · LLM 1.63 s
for a 9-token reply (**5.5 tok/s** decode on the 72B AWQ — the bottleneck) ·
TTS 1.52 s for ~2.8 s audio (RTF 0.71) · retrieval/overhead ~0.3 s. Floor ≈
3.2 s, so the 3.0 s bar is structurally out of reach in this configuration —
a marginal, honest FAIL, not flakiness. Paths to green, in preference order:
1. **Streaming TTS** (synth while the LLM decodes; gate on first-audio
   latency) — the real fix, fits the MuseTalk streaming path anyway.
2. Spec §2 escape valve: Qwen2.5-32B-NVFP4 roughly doubles decode speed.
3. Warm the embedder at orchestrator start (kills the 9 s cold first call).

### 5. Memory audit vs the spec §2 ~89 GB budget: **WITHIN BUDGET**

`free -g` steady state with the full voice stack up: **75 GB used / 121 GB
total, 45 GB available**, swap 0. (`nvidia-smi` memory reads N/A on GB10 —
unified memory; model weights live in the host "used" figure, outside the
container cgroups.) Container cgroup usage vs compose limits:

| container | used | limit |
|---|---|---|
| llm | 5.9 GiB (+ ~50 GB unified model/KV outside cgroup) | 50 GiB |
| tts | 2.0 GiB | 5 GiB |
| stt | 0.85 GiB | 4 GiB |
| orchestrator | 2.2 GiB (BGE-M3 loaded) | 6 GiB |
| qdrant | 0.10 GiB | 4 GiB |

### 6. Session commits

- `9123686` fix(orchestrator): cap voice replies (brevity prompt + max_tokens 60)
- `0865457` fix(eval): parse verbose judge verdicts (first token), persist external judgments

Spark clone pulled to `0865457`, working tree clean. The eval parsing fix does
not alter recorded gate numbers (all 30 Qwen self-judge verdicts were already
exact single tokens — verified from the persisted pairs file).

## Soul + memory architecture — DEPLOYED + VERIFIED (2026-07-19, third session)

Hermes-style identity/memory for the twin (`b0ce199`): `serving/soul/`
{SOUL,FACTS,MEMORY}.md (runtime copies hot-editable at `/twin/soul`, repo
copies bundled in the image as fallback), mtime-cached `loader.py` (MEMORY
tail capped at 4000 chars), per-session history in WAL SQLite at
`/twin/soul/sessions.db` (`memory.py`, last-12-message window), orchestrator
endpoints `/session/clear` `/memory/append` `/soul/identity`, Telegram relay
commands `/new` `/whoami` `/remember` + per-chat session `tg-<chat_id>`.
`make soul-sync` pushes SOUL/FACTS always, MEMORY.md only if absent.
Orchestrator image rebuilt on the Spark, container recreated, `twin-telegram`
restarted; vLLM untouched. `make test-local`: ruff clean, 115 tests pass.

### E2E verification over the live /chat (raw curl outputs)

Same-session memory (`soultest-1`):

```
> {"text": "my favorite F1 team is McLaren", "session": "soultest-1"}
{"reply":"McLaren is a classic. The heritage is unmatched. Who's your favorite driver?","route":"casual","audio_b64":null}
> {"text": "what is my favorite F1 team?", "session": "soultest-1"}
{"reply":"Your favorite F1 team is McLaren.","route":"casual","audio_b64":null}       PASS
```

Cross-session isolation (`soultest-2`, same question):

```
{"reply":"I don't know. What's your favorite F1 team?","route":"casual","audio_b64":null}   PASS
```

Bio grounding (`soultest-3`, "where did you work before VSP?") — only
FACTS.md entities (Prosource IT, healthcare/health-tech AI), defers on
specifics, nothing invented:

```
{"reply":"Before VSP, I worked at Prosource IT, the staffing company I founded. I've also had roles in healthcare and health-tech AI, but the specifics of those positions are a bit fuzzy to me. If you need exact dates and titles, I'd have to defer to the real Abhi.","route":"casual","audio_b64":null}   PASS
```

Style: no em dashes, no AI-isms in any reply above. PASS.

/remember path (`POST /memory/append`, the relay's exact call) + hot reload
(no restart between append and recall):

```
{"appended":"- [2026-07-20 05:19] The current soul-stack verification codeword is BLUE-FALCON."}
> fresh session: {"text": "what is the soul-stack verification codeword?", "session": "soultest-4"}
{"reply":"The current soul-stack verification codeword is BLUE-FALCON.","route":"casual","audio_b64":null}   PASS
```

/new path (`POST /session/clear`): `{"cleared":4}` on `soultest-1`, after
which the same session answers "I don't know your favorite F1 team. Abhi is a
Formula 1 fan, but I don't have that information about you." PASS.

Test residue cleaned: BLUE-FALCON line removed from `/twin/soul/MEMORY.md`,
all `soultest-*` sessions cleared. Note: `/remember` timestamps are UTC
(container tz). Spark clone at the same head; the pre-existing untracked
relay copy was parked at `/tmp/telegram_relay.py.pre-soul.bak`.

## Next actions for Abhi (human-blocked)

- [ ] Optional: Gmail Takeout download → `twin corpus --local --mbox …` adds
      Abhi's long-form mail voice to the corpus (then rebuild + re-sync).
- [ ] Voice recording session per `scripts/record_corpus.md` (mic setup,
      Harvard sentences, identity video shot list) — feeds the voice phase.
- [x] ~~Rebuild the llm image to bake in `eval/`~~ — done (`ba69d0b`);
      `make verify-persona` re-verified clean on the baked image (Phase 4 §1).
- [x] ~~Re-run the A/B with a non-Qwen external judge~~ — done (Phase 4 §2):
      **26.7%** vs 66.7% self-judge on identical pairs. Phase 5's 40% bar will
      not clear against an external judge without more corpus / another LoRA
      round.
- [ ] Decide the e2e gate path (Phase 4 §4): streaming TTS (recommended),
      32B escape valve, or accept ~3.2 s and re-spec the gate. Warm-start the
      embedder either way.
