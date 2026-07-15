# serving/video — MuseTalk V1.5 streaming twin

`server.py` is the FastAPI WebSocket streamer (audio chunks in → JPEG frames out,
lazy-load on first session, `/unload` to give the ~10 GB back).

## The port (video addendum, Steps 1–3)

`docker/musetalk.Dockerfile` performs the sm_121 aarch64 forward-port:

1. NGC aarch64 PyTorch base; verify capability `(12,1)` at build time
2. mmcv/mmengine from source: `TORCH_CUDA_ARCH_LIST="12.1" MMCV_WITH_OPS=1 pip install mmcv --no-binary mmcv`
3. onnxruntime-gpu built from source for sm_121 (stock PyPI has no kernels);
   `server.py` hard-asserts `CUDAExecutionProvider` so an insightface CPU
   fallback can never pass silently
4. A thin `musetalk/pipeline.py` wrapper (added by `docker/patches/`) exposing
   `MuseTalkStreamPipeline.stream_frames(audio_chunk) -> jpeg iterator` around
   the upstream realtime inference loop — upstream ships batch scripts only.

Expected on GB10 (addendum Step 3): ~30 FPS @256², ~15 FPS @512², 8–10 GB.
If FPS < 10, suspect ORT CPU fallback or PTX JIT — watch `nvidia-smi dmon` while
it runs; `eval/video.py --gate` enforces the bar.

## Wire-up

Orchestrator forwards TTS audio to `/ws/video`; frames go out via LiveKit ingress
→ WebRTC (talk.twin.local). Postprocess options for the waxen-mouth effect: light
Gaussian blur + film grain on the mouth region, or Real-ESRGAN upscale (also
needs an sm_121 build).
