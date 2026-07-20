"""Voice TTS server — OpenAI-compatible /v1/audio/speech on :8001, streaming
variant with sentence-boundary buffering (logos_flux pattern, target < 800 ms
TTFA). RUN ON SPARK inside docker/tts.Dockerfile.

Serves whatever TWIN_VOICE_CHECKPOINT points at through the qwen-tts package:
  - today: Qwen3-TTS-12Hz-1.7B-CustomVoice (base placeholder voice, proves the
    voice pipeline end-to-end before Abhi's recording session)
  - later: voice-v1, the Milo-style full SFT of Qwen3-TTS-12Hz-1.7B-Base
    (that phase switches synthesis to the voice-clone entry point)

Gotchas encoded here:
  - loads with local_files_only=True from an absolute /twin path (_name_or_path bug)
  - SDPA attention, never flash-attn (the qwen-tts README recommends
    flash-attn; on Blackwell we do NOT install it)
  - white-noise regression guard: qwen-tts decodes codec->wav internally (no
    HiFi-GAN mel surface for epsilon_clamp on this path), so assert_finite runs
    on every synthesized waveform; the clamp itself stays unit-tested in
    ci/preflight.py for the voice-v1 vocoder path
"""

from __future__ import annotations

import io
import os
import re

from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from serving.tts.guards import assert_finite

CHECKPOINT = os.environ.get("TWIN_VOICE_CHECKPOINT", "/twin/checkpoints/voice-v1")
SPEAKER = os.environ.get("TWIN_TTS_SPEAKER", "Ryan")
LANGUAGE = os.environ.get("TWIN_TTS_LANGUAGE", "English")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

app = FastAPI(title="AbhiTwin TTS (Qwen3-TTS)")
_model = None


def _load():
    global _model
    if _model is None:
        import torch
        from qwen_tts import Qwen3TTSModel

        _model = Qwen3TTSModel.from_pretrained(
            CHECKPOINT,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda:0",
        )
    return _model


def synthesize(text: str) -> bytes:
    """Text -> 16-bit PCM WAV bytes at the model's native rate, with the
    NaN guard on the waveform (white-noise regression tripwire)."""
    import numpy as np
    import soundfile as sf

    model = _load()
    wavs, sr = model.generate_custom_voice(text=text, language=LANGUAGE, speaker=SPEAKER)
    wav = np.asarray(wavs[0], dtype=np.float32)
    assert_finite(wav, "wav")
    buf = io.BytesIO()
    sf.write(buf, wav, samplerate=int(sr), format="WAV", subtype="PCM_16")
    return buf.getvalue()


class SpeechRequest(BaseModel):
    model: str = "voice-v1"
    input: str
    voice: str = "abhi"
    response_format: str = "wav"


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "loaded": _model is not None, "checkpoint": CHECKPOINT}


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
