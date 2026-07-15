# AbhiTwin — personal digital twin on one DGX Spark

Persona LLM + voice clone + STT + RAG + orchestrator (v1), photoreal MuseTalk video twin
(v1.5) — all serving from a single GB10 DGX Spark, with every fine-tune cloud-burst to a
rented H100 and synced back. Specs: [docs/twin_build_spec_v2.md](docs/twin_build_spec_v2.md)
and [docs/twin_video_addendum_v2.md](docs/twin_video_addendum_v2.md).

## Quickstart

```bash
git clone <this repo> && cd abhi-twin
cp .env.example .env            # fill in HF/Brev/Vast keys
make test-local                 # mac: unit tests + local preflight (no GPU needed)

# on the Spark (over Tailscale):
make phase0 && make verify-phase0
make serve                      # docker compose --profile core up
make video-demo                 # v1.5 reproducibility gate: working MuseTalk < 30 min
```

Or drive everything through the CLI: `twin phase0`, `twin corpus --local`, `twin train persona`,
`twin serve`, `twin video`, `twin verify <phase>`.

## Architecture

```
┌─────────────────────────── DGX SPARK ─────────────────────────────┐
│                                                                    │
│   ┌────────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│   │ Persona LLM    │───▶│ Orchestrator │───▶│ TTS (Milo)       │──┼─┐
│   │ (72B NVFP4)    │    │  + Router    │    │ STT (Whisper)    │  │ │
│   └────────────────┘    │  + Memory    │    │ MuseTalk (video) │  │ │
│           ▲             │  + RAG       │    └──────────────────┘  │ │
│           │             └──────────────┘             │            │ │
│           └─────────── Qdrant (Brain + corpus) ──────┘            │ │
│                                                                   │ │
└───────────────────────────────────────────────────────────────────┘ │
                              WebRTC / OpenAI API                     │
                                        ▼                             │
                       Mac (starbase) — client + extractors           │
                                                                      │
                              Cloud burst on demand                   │
                                        ▼                             │
                         ┌────────────────────────┐                   │
                         │ Brev H100 / Vast.ai    │ ◄─────────────────┘
                         │  training only,        │      weight sync
                         │  spun down when done   │      (HF private)
                         └────────────────────────┘
```

Everything serves an OpenAI-compatible `/v1/*` endpoint: LLM `:8000`, TTS `:8001`
(streaming `:8002`), STT `:8003`, MuseTalk WS `:8004`, orchestrator `:8080`, Qdrant `:6333`.

## Memory budget (the whole game on one Spark — 128 GB unified)

| Component | Memory |
|---|---|
| Persona LLM (Qwen2.5-72B, NVFP4) | ~50 GB |
| Milo voice TTS (Qwen3-TTS-1.7B, full SFT) | ~5 GB |
| Whisper large-v3-turbo STT | ~4 GB |
| MuseTalk V1.5 (lazy-loaded) | ~10 GB |
| Qdrant + BGE-M3 | ~4 GB |
| Orchestrator + Ray + FastAPI + LiveKit | ~6 GB |
| OS + drivers + headroom | ~10 GB |
| **Total** | **~89 GB (~39 GB free)** |

`docker/compose.yaml` enforces these as hard limits. Escape valve if it gets tight: drop the
LLM to **Qwen2.5-32B-NVFP4** (~22 GB, frees 28 GB) — biggest single lever.

## Phases

| Phase | Deliverable | Verify gate (`make …`) | Success criteria |
|---|---|---|---|
| 0 | Spark imaged, Tailscale, base stack | `verify-phase0` | nvidia-smi clean, `torch.cuda.get_device_capability() == (12,1)` |
| 1 | Corpus v1: extraction + cleaning | `verify-corpus` | ≥ 8k pairs, PII-scrubbed, holdout frozen, uploaded to HF |
| 2 | Persona LoRA v1 (Brev H100 → Spark) | `verify-persona` | Blind A/B ≥ 30% indistinguishable, PPL within 15% of baseline |
| 3 | Voice corpus + Milo SFT v1 | `verify-voice` | RTF < 1.5×, MOS ≥ 3.8 self-eval |
| 4 | STT + orchestrator + RAG + WebRTC | `verify-e2e` | End-to-end voice conversation < 3 s round trip |
| 5 | Persona v2 + automated evals | `verify-persona` | Blind A/B ≥ 40% |
| v1.5 | MuseTalk video twin + published recipe | `verify-video` | FPS ≥ 10 @256, recipe reproduces on another Spark |

## Repo map

```
cli/twin.py            click CLI wrapping the phases
Makefile               phase targets + verify gates
docker/                compose.yaml (profiles core/voice/video) + 5 Dockerfiles
phase0/                Spark bring-up + verification scripts
corpus/                extractors (run on the Mac) + cleaning pipeline (build.py)
training/configs/      persona-lora.yaml, voice-sft.yaml (verbatim from spec) + musetalk
training/burst/        Brev/Vast launch + corpus/adapter sync scripts
serving/               orchestrator (LangGraph), rag, tts, stt, video (MuseTalk WS streamer)
eval/                  persona.py, voice.py, video.py — gate implementations
ci/preflight.py        every known Spark gotcha as an executable check
scripts/               record_corpus.md (human), e2e_roundtrip.py
docs/                  the two specs + PUBLICATION.md (v1.5 artifact)
```

## Known gotchas — encoded as code, not comments

Run `make preflight` on the Spark. It asserts: torch is the NGC nightly cu130 aarch64 build
(CUDA available, capability `(12,1)` — never `pip install torch`); SDPA not flash-attn;
`local_files_only=True` on every local load (`_name_or_path` bug); epsilon-clamped HiFi-GAN
input (white-noise bug); `libnvrtc.so.13 → libnvrtc.so.12.8` symlink; mmcv/onnxruntime built
from source for sm_121 with a CPU-fallback provider assert on insightface.

## Privacy

The corpus never leaves the box except as an encrypted/private HF tarball for cloud-burst
training. PII scrubbing (presidio, replace-not-delete) is a mandatory pipeline stage.
`/twin`, `corpus/data/`, checkpoints and `.env` are gitignored.
