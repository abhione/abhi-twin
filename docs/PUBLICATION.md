# PUBLICATION.md — MuseTalk on a single DGX Spark (v1.5 artifact)

First-in-community recipe: photoreal talking-head twin (MuseTalk V1.5, sm_121
aarch64, CUDA 13) serving alongside a 72B persona LLM on one GB10 Spark.

## Reproducibility gate (must pass before publishing)

- [ ] Clean Spark → `git clone <repo> && cd abhi-twin && make video-demo` →
      working demo in **< 30 minutes** (the script prints elapsed time)
- [ ] `make verify-video` passes (FPS ≥ 10 @256)
- [ ] A second Spark owner has reproduced it and is credited below

## Benchmark table (fill from `eval/video.py` + `nvidia-smi dmon -s um`)

| Metric | Value | Conditions |
|---|---|---|
| FPS @ 256×256 | _(expected ~30)_ | MuseTalk alone |
| FPS @ 512×512 | _(expected ~15)_ | MuseTalk alone |
| FPS @ 256×256, LLM co-resident | | Qwen2.5-72B NVFP4 serving concurrently |
| Audio-in → first frame | _(target < 500 ms)_ | WS streamer, sentence-buffered |
| Voice round trip (STT→LLM→TTS→video) | _(target < 3 s)_ | `make verify-e2e` |
| MuseTalk memory | _(expected 8–10 GB)_ | `nvidia-smi` delta on lazy load |
| Total stack memory | _(budget ~89 GB)_ | all profiles up |
| Power draw | | `nvidia-smi dmon -s p` during inference |

## Gotchas section (first-class content — what the community searches for)

Write each of these up with the exact error message and the fix:

- [ ] torch: NGC nightly cu130 aarch64 only; pip torch = silent CPU fallback
      (`ci/preflight.py --check torch` is the tripwire)
- [ ] mmcv: `TORCH_CUDA_ARCH_LIST="12.1" MMCV_WITH_OPS=1 pip install mmcv --no-binary mmcv`
- [ ] onnxruntime-gpu: source build for sm_121 (`docker/musetalk.Dockerfile`),
      plus the insightface `CUDAExecutionProvider` assert that catches the fallback
- [ ] `libnvrtc.so.13 → libnvrtc.so.12.8` symlink for CUDA-12.8-pinned packages
- [ ] `_name_or_path` offline-load bug + the `patch_checkpoint.py` fix
- [ ] HiFi-GAN epsilon clamp (white-noise TTS on Blackwell)
- [ ] SDPA-not-flash-attn on Blackwell
- [ ] Anything new the port surfaces — log it the moment it happens

## Cross-post checklist (priority order, addendum §Publication)

- [ ] **NVIDIA Developer Forums — DGX Spark section** (Martinb, tsuru_mitsu,
      provos are active there)
- [ ] **r/LocalLLaMA** — lead with the benchmark table
- [ ] **Hacker News** — Show HN; Jeff Geerling / Simon Willison audience
- [ ] **NVIDIA/dgx-spark-playbooks** — PR adding this as a playbook
- [ ] **Hugging Face** — model card + Space demo (demo clips watermarked + labeled)
- [ ] **MuseTalk upstream** — PR adding a `deploy/dgx-spark/` section
- [ ] (optional) GTC 2027 talk submission: "A Photoreal Personal AI Twin on a
      Single DGX Spark"

## Ship criteria (addendum "what done looks like")

- [ ] Live video conversation on the single Spark at < 3 s round trip
- [ ] Recipe public, reproducible, cited by ≥ 1 other Spark builder
- [ ] Benchmark table someone at NVIDIA can point to
- [ ] PR or reproducer issue filed on ≥ 1 upstream repo
- [ ] Consent/watermark hygiene: SynthID (or similar) on published clips, signed
      usage log, synthetic-content label on anything that could pass as real
