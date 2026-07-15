# AbhiTwin — Build Harness

Personal digital twin on a single DGX Spark (GB10): persona LLM + voice clone + STT + RAG + orchestrator in v1; MuseTalk photoreal video twin in v1.5 (first-in-community publication artifact).

**You are building the HARNESS, not running the twin.** The Spark arrives tomorrow. Everything you produce must be executable by the owner (Abhi) step-by-step, phase-by-phase, on the Spark once it's on the network. Where a step can only run on the Spark, generate the script + a checklist entry; do not fake execution results.

## Source of truth
- `docs/twin_build_spec_v2.md` — the master spec (memory budget, cloud-burst training, corpus plan, persona LoRA, Milo voice SFT, phases 0–5)
- `docs/twin_video_addendum_v2.md` — v1.5 MuseTalk port spec (sm_121 aarch64 port steps, streaming server, publication checklist)
Follow them faithfully. Where the spec gives exact configs (persona-lora.yaml, voice-sft.yaml), reproduce them verbatim as files.

## Deliverable shape (what "done" looks like)
A repo any Spark owner can `git clone` and drive with a single CLI:

```
twin <phase>            # e.g. twin phase0, twin corpus, twin serve, twin video
make verify-phase0      # every phase has an automated verification gate
docker compose up       # serving stack (vLLM, TTS, STT, Qdrant, orchestrator, MuseTalk)
```

Required repo structure:
```
abhi-twin/
├── README.md               # quickstart, architecture diagram, phase table
├── CLAUDE.md               # this file
├── docs/                   # the two spec .md files (already present)
├── Makefile                # phase targets + verify gates
├── cli/twin.py             # click CLI wrapping the phases
├── docker/
│   ├── compose.yaml        # full serving stack, profiles: core / voice / video
│   ├── llm.Dockerfile      # vLLM cu130 aarch64
│   ├── tts.Dockerfile      # Qwen3-TTS serving (OpenAI /v1/audio/speech, port 8001)
│   ├── stt.Dockerfile      # faster-whisper + CTranslate2 sm_121 source build (port 8003)
│   ├── musetalk.Dockerfile # MuseTalk V1.5 port (mmcv source build, ORT sm_121)
│   └── train.Dockerfile    # cloud-burst training image (pins match local Spark)
├── phase0/                 # Spark bring-up: verify scripts (nvidia-smi, torch capability (12,1), tailscale, dirs)
├── corpus/                 # extraction + cleaning pipeline (build.py per spec §6: dedup/PII/length/ppl/format/holdout)
│   └── extractors/         # gmail_takeout.py, imessage.py, slack.py, apple_notes.py, github_prs.py
├── training/
│   ├── configs/persona-lora.yaml   # verbatim from spec §7
│   ├── configs/voice-sft.yaml      # verbatim from spec §8
│   ├── configs/musetalk-identity.yaml
│   └── burst/              # brev/vast launch + sync scripts (tar→HF→train→push adapter→hf download)
├── serving/
│   ├── orchestrator/       # LangGraph: INTAKE→ROUTER→RETRIEVER→PERSONA→TOOLS→VOICE (+video sink)
│   ├── rag/                # Qdrant + BGE-M3 index/ingest
│   └── video/              # MuseTalk FastAPI WebSocket streamer (audio chunks in → JPEG frames out), lazy-load
├── eval/
│   ├── persona.py          # held-out PPL, style cosine (all-mpnet-base-v2), blind A/B via LLM judge, fact recall
│   ├── voice.py            # RTF, 10s synth + NaN-in-mel assert (white-noise regression)
│   └── video.py            # FPS@256/512, audio→first-frame latency, memory
├── ci/preflight.py         # torch.cuda.is_available() + get_device_capability()==(12,1), local_files_only loads, epsilon-clamp unit test
└── scripts/record_corpus.md  # human instructions: mic setup, Harvard sentences, identity video shot list
```

## Hard constraints (violating these = failed build)
1. **Known gotchas are first-class code, not comments.** Each one gets a check in `ci/preflight.py` or a build step:
   - torch from NGC nightly cu130 aarch64 ONLY — never `pip install torch` (silent CPU fallback). Preflight asserts CUDA + capability (12,1).
   - SDPA, never flash-attn (`attn_implementation: sdpa` everywhere).
   - TTS is FULL SFT not LoRA (Milo v8); persona is LoRA rank 32.
   - Strip/patch `_name_or_path` in checkpoint config.json before HF push; ALL local loads use `local_files_only=True`.
   - Epsilon-clamp HiFi-GAN input; unit test synthesizes 10s and asserts no NaN in mel.
   - `libnvrtc.so.13 → libnvrtc.so.12.8` symlink script for CUDA-12.8-pinned packages.
   - mmcv: `TORCH_CUDA_ARCH_LIST="12.1" MMCV_WITH_OPS=1 pip install mmcv --no-binary mmcv`.
   - onnxruntime-gpu: build from source for sm_121 (script it) with CPU-fallback detection (`providers` assert on insightface).
2. **Memory budget is law** (spec §2): 72B NVFP4 ~50GB + TTS 5 + STT 4 + MuseTalk 10 (lazy) + Qdrant 4 + orch 6 + OS 10 ≈ 89GB of 128. compose.yaml sets explicit memory limits; MuseTalk lazy-loads on first WebRTC video session. Escape valve documented: drop to Qwen2.5-32B-NVFP4.
3. **Privacy:** corpus never leaves the box except as the encrypted/private HF tarball for cloud-burst. PII scrub (presidio) is mandatory in the pipeline, replace-not-delete. `.gitignore` excludes /twin data, corpus, checkpoints, .env.
4. **Every phase ends in a verify gate** matching spec §12 success criteria (phase0: capability==(12,1); corpus: ≥8k pairs + frozen holdout; persona: blind A/B ≥30%; voice: RTF <1.5; e2e: voice roundtrip <3s; video: FPS ≥10 @256, recipe reproduces).
5. **Reproducibility gate for v1.5 publication:** clean Spark → `git clone && make video-demo` → working MuseTalk demo in <30 min. Write `docs/PUBLICATION.md` with the benchmark table template + cross-post checklist (NVIDIA forums, r/LocalLLaMA, HN, dgx-spark-playbooks PR, HF Space, MuseTalk upstream PR).
6. **No fabricated outputs.** Scripts that need the Spark emit "RUN ON SPARK" markers. Anything runnable on macOS today (corpus extractors, cleaning pipeline logic with unit tests on synthetic fixtures, CLI, YAML validation) must actually run — include `make test-local` and make it pass here.

## Environment notes (this machine = starbase, Apple Silicon macOS)
- Python 3.12 via `python3`; use venv at .venv for local tests. No CUDA here — guard GPU code behind availability checks.
- The Spark will be reachable over Tailscale as `spark` (placeholder hostname; make it a config var TWIN_SPARK_HOST).
- Cloud burst: Brev first, Vast fallback (spec §5). Script both; keys via .env (never committed).
- Owner's corpus sources on THIS mac: iMessage chat.db, Apple Notes, Gmail Takeout (manual download), GitHub via `gh`. Extractors should run on the Mac and rsync the cleaned corpus to the Spark — reflect that in the CLI (`twin corpus --local`).

## Style
- Python: type hints, click for CLI, ruff-clean. Shell: `set -euo pipefail`.
- Short files over clever ones. Every script has a `--help` and a docstring stating which host it runs on (mac | spark | cloud).
- Commit in logical chunks with clear messages as you go.
