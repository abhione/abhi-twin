"""STT server — faster-whisper large-v3-turbo behind OpenAI-compatible
/v1/audio/transcriptions on :8003 (spec §9). RUN ON SPARK inside
docker/stt.Dockerfile (CTranslate2 built from source for sm_121)."""

from __future__ import annotations

import os
import tempfile

from fastapi import FastAPI, Form, UploadFile

MODEL_DIR = os.environ.get("TWIN_STT_MODEL", "/twin/models/whisper-large-v3-turbo")

app = FastAPI(title="AbhiTwin STT (faster-whisper)")
_model = None


def _load():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        # local_files_only equivalent: absolute path, no hub lookup
        _model = WhisperModel(MODEL_DIR, device="cuda", compute_type="float16")
    return _model


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "loaded": _model is not None}


@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile, model: str = Form("whisper-large-v3-turbo")) -> dict:
    whisper = _load()
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        segments, info = whisper.transcribe(tmp.name, beam_size=1, language="en")
        text = " ".join(seg.text.strip() for seg in segments)
    return {"text": text, "duration": info.duration}
