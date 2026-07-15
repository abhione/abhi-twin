"""MuseTalk V1.5 streaming server — audio chunks in over WebSocket, JPEG frames
out (video addendum Step 6). RUN ON SPARK inside docker/musetalk.Dockerfile.

Memory-budget rule: the model is LAZY-LOADED on the first video session and can
be explicitly unloaded, so text/voice conversations never pay the ~10 GB.
Target: < 500 ms audio-in -> first-frame-out (sentence-boundary buffering).

WS protocol (/ws/video): client sends binary 16 kHz mono PCM chunks; server
replies with binary JPEG frames; JSON text frames carry control/latency info.
"""

from __future__ import annotations

import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

IDENTITY_VIDEO = os.environ.get("TWIN_IDENTITY_VIDEO", "/twin/corpus/video/identity.mp4")
MUSETALK_DIR = os.environ.get("TWIN_MUSETALK_DIR", "/twin/models/musetalk")
JPEG_QUALITY = int(os.environ.get("TWIN_VIDEO_JPEG_QUALITY", "85"))

app = FastAPI(title="AbhiTwin video twin (MuseTalk V1.5)")
_engine = None


def _load_engine():
    """Lazy-load MuseTalk (+ insightface with a hard CUDA-provider assert)."""
    global _engine
    if _engine is not None:
        return _engine
    import onnxruntime as ort

    providers = ort.get_available_providers()
    assert "CUDAExecutionProvider" in providers, (
        f"insightface would silently fall back to CPU (providers={providers}) — "
        "rebuild onnxruntime-gpu for sm_121 (docker/musetalk.Dockerfile)"
    )
    from musetalk.pipeline import MuseTalkStreamPipeline  # from the ported repo

    _engine = MuseTalkStreamPipeline(
        model_dir=MUSETALK_DIR,
        identity_video=IDENTITY_VIDEO,
        resolution=int(os.environ.get("TWIN_VIDEO_RES", "256")),
    )
    return _engine


def unload_engine() -> None:
    global _engine
    if _engine is not None:
        _engine = None
        import torch

        torch.cuda.empty_cache()


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "loaded": _engine is not None}


@app.post("/unload")
def unload() -> dict:
    unload_engine()
    return {"ok": True, "loaded": False}


@app.websocket("/ws/video")
async def video_session(ws: WebSocket) -> None:
    await ws.accept()
    t_connect = time.monotonic()
    engine = _load_engine()  # first session pays the ~10 GB load here
    await ws.send_json({"event": "ready", "load_s": round(time.monotonic() - t_connect, 2)})
    first_frame_sent = False
    try:
        while True:
            audio_chunk = await ws.receive_bytes()
            t0 = time.monotonic()
            for jpeg in engine.stream_frames(audio_chunk, jpeg_quality=JPEG_QUALITY):
                await ws.send_bytes(jpeg)
                if not first_frame_sent:
                    first_frame_sent = True
                    await ws.send_json(
                        {"event": "first_frame", "latency_s": round(time.monotonic() - t0, 3)}
                    )
    except WebSocketDisconnect:
        pass
