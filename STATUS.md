# AbhiTwin — Day-1 Status (2026-07-17)

Day-1 bring-up is complete. Phase 0 verify gate and the full preflight pass on
the Spark; both serving base images are pulled; the Mac-side test suite is
green. Everything below is real command output (trimmed).

## Spark: `make verify-phase0` — ALL PASS (exit 0)

```
PASS  nvidia-smi
PASS  GB10 GPU visible
PASS  tailscale up
PASS  docker
PASS  nvidia container toolkit
PASS  /twin/corpus exists
PASS  /twin/checkpoints exists
PASS  /twin/adapters exists
PASS  /twin/models exists
PASS  /twin/logs exists
PASS  torch              CUDA ok, capability (12,1), torch 2.14.0.dev20260717+cu130 cuda 13.0
PASS  torch cuda capability (12,1)
=== verify-phase0: ALL CHECKS PASSED ===
```

## Spark: `ci/preflight.py` (twin-venv) — 7 PASS, 1 PEND (exit 0)

```
PASS  epsilon-clamp      epsilon clamp floors zeros; log-mel finite; NaN detector fires
PASS  local-files-only   every from_pretrained() uses local_files_only=True (or is marked # hub-ok)
PASS  no-flash-attn      no flash_attn anywhere; SDPA it is
PASS  configs            training configs encode the recipe gotchas
PASS  torch              CUDA ok, capability (12,1), torch 2.14.0.dev20260717+cu130 cuda 13.0
PASS  nvrtc              /usr/local/cuda/lib64/libnvrtc.so.12.8 present
PEND  onnxruntime        onnxruntime not installed — sm_121 build runs in the musetalk image (video profile, v1.5); required only inside that container
PASS  checkpoints        checkpoint configs are offline-safe
```

Notes:
- `pyyaml` was installed into `~/twin-venv` (the `configs` check needed it).
- onnxruntime is intentionally PEND on the host: the sm_121 source build ships
  inside `docker/musetalk.Dockerfile` and is exercised by the video (v1.5)
  profile, not the host venv. The check still hard-fails if ORT is present
  without `CUDAExecutionProvider` (the silent-CPU-fallback gotcha).

## Spark: docker images — both serving base images present

```
nvcr.io/nvidia/pytorch:25.11-py3  19.5GB
qdrant/qdrant:v1.12.4             204MB
```

## Mac (starbase): `make test-local` — green

```
.venv/bin/ruff check .
All checks passed!
.venv/bin/python -m pytest
76 passed in 0.74s
.venv/bin/python ci/preflight.py --local-only
PASS  epsilon-clamp / local-files-only / no-flash-attn / configs
```

## Repo sync

Both machines on `main` @ `4e4c8c3`
(`fix(preflight): onnxruntime absence is PEND not FAIL`).

## Next actions for Abhi (human-blocked)

- [ ] `huggingface-cli login` on the Spark — required before any model
      download (Qwen 72B NVFP4, BGE-M3, TTS base).
- [ ] Put the Brev API key into `.env` on the Mac (never committed) — needed
      for cloud-burst training (`training/burst/`). Vast key optional fallback.
- [ ] Corpus recording session: follow `scripts/record_corpus.md`
      (mic setup, Harvard sentences, identity video shot list).
- [ ] Then run `twin corpus --local` on the Mac to extract + clean the text
      corpus (iMessage, Apple Notes, Gmail Takeout, GitHub PRs) and rsync it
      to the Spark.
