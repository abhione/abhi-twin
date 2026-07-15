"""Milo voice TTS server — OpenAI-compatible /v1/audio/speech on :8001, streaming
variant on :8002 with sentence-boundary buffering (logos_flux pattern, target
< 800 ms TTFA). RUN ON SPARK inside docker/tts.Dockerfile.

Gotchas encoded here:
  - checkpoint loads with local_files_only=True from /twin/checkpoints/voice-v1
    (_name_or_path bug — the checkpoint is patched at push time too)
  - HiFi-GAN input goes through epsilon_clamp (Blackwell white-noise bug)
  - SDPA attention, never flash-attn
"""

from __future__ import annotations

import io
import os
import re

from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from serving.tts.guards import assert_finite, epsilon_clamp

CHECKPOINT = os.environ.get("TWIN_VOICE_CHECKPOINT", "/twin/checkpoints/voice-v1")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

app = FastAPI(title="AbhiTwin TTS (Milo voice)")
_model = None


def _load():
    global _model
    if _model is None:
        import torch
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(CHECKPOINT, local_files_only=True)
        model = AutoModel.from_pretrained(
            CHECKPOINT,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
        ).eval()
        _model = (processor, model)
    return _model


def synthesize(text: str) -> bytes:
    """Text -> 24 kHz WAV bytes, with the epsilon-clamp guard on the vocoder path."""
    import soundfile as sf
    import torch

    processor, model = _load()
    inputs = processor(text=text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        codec = model.generate(**inputs)
        mel = model.codec_to_mel(codec)
        mel = epsilon_clamp(mel)  # the Blackwell white-noise fix
        assert_finite(mel.float().cpu(), "mel")
        wav = model.vocoder(mel).float().cpu().numpy().squeeze()
    buf = io.BytesIO()
    sf.write(buf, wav, samplerate=24000, format="WAV")
    return buf.getvalue()


class SpeechRequest(BaseModel):
    model: str = "voice-v1"
    input: str
    voice: str = "abhi"
    response_format: str = "wav"


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "loaded": _model is not None}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    return Response(content=synthesize(req.input), media_type="audio/wav")


@app.post("/v1/audio/speech/stream")
def speech_stream(req: SpeechRequest) -> StreamingResponse:
    """Sentence-boundary buffering: synth + flush one sentence at a time so the
    first audio lands < 800 ms while the tail is still generating."""

    def gen():
        for sentence in _SENTENCE.split(req.input):
            if sentence.strip():
                yield synthesize(sentence.strip())

    return StreamingResponse(gen(), media_type="audio/wav")
