"""Orchestrator FastAPI ingress (port 8080). RUN ON SPARK.

POST /chat            {"text": ..., "voice": bool} -> {"reply", "route", "audio_b64"}
POST /v1/audio/roundtrip  multipart wav -> STT -> graph -> TTS wav (e2e gate path)
GET  /healthz
"""

from __future__ import annotations

import base64
import os
import time

import httpx
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from serving.orchestrator import memory
from serving.orchestrator.graph import build_graph
from serving.soul import loader

SPARK = os.environ.get("TWIN_SPARK_HOST", "localhost")
STT_URL = os.environ.get("TWIN_STT_URL", f"http://{SPARK}:8003/v1")

app = FastAPI(title="AbhiTwin orchestrator")
graph = build_graph()


class ChatRequest(BaseModel):
    text: str
    voice: bool = False
    session: str = "default"


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    state = graph.invoke(
        {"user_text": req.text, "session": req.session, "want_voice": req.voice}
    )
    return {
        "reply": state["reply_text"],
        "route": state["route"],
        "audio_b64": base64.b64encode(state["audio"]).decode() if state.get("audio") else None,
    }


class SessionRequest(BaseModel):
    session: str


class MemoryRequest(BaseModel):
    text: str


@app.post("/session/clear")
def session_clear(req: SessionRequest) -> dict:
    """Drop a session's conversation history (the relay's /new command)."""
    return {"cleared": memory.clear(req.session)}


@app.post("/memory/append")
def memory_append(req: MemoryRequest) -> dict:
    """Append a timestamped line to /twin/soul/MEMORY.md (the relay's /remember).

    The relay is allowlist-gated; the container is the single MEMORY.md writer.
    """
    return {"appended": loader.remember(req.text)}


@app.get("/soul/identity")
def soul_identity() -> dict:
    return {"identity": loader.identity_line()}


@app.post("/v1/audio/roundtrip")
async def roundtrip(file: UploadFile) -> dict:
    """voice-in -> STT -> persona -> TTS. Timed; the <3 s e2e gate hits this."""
    t0 = time.monotonic()
    audio = await file.read()
    async with httpx.AsyncClient(timeout=120) as client:
        stt = await client.post(
            f"{STT_URL}/audio/transcriptions",
            files={"file": (file.filename or "in.wav", audio, "audio/wav")},
            data={"model": "whisper-large-v3-turbo"},
        )
        stt.raise_for_status()
        text = stt.json()["text"]
    state = graph.invoke({"user_text": text, "session": "roundtrip", "want_voice": True})
    return {
        "transcript": text,
        "reply": state["reply_text"],
        "audio_b64": base64.b64encode(state["audio"]).decode() if state.get("audio") else None,
        "roundtrip_s": round(time.monotonic() - t0, 3),
    }
