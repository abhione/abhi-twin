#!/usr/bin/env python3
"""Telegram relay for the AbhiTwin orchestrator. host: spark.

Long-polls Telegram getUpdates, forwards allowed users' text to the
orchestrator (POST /chat), replies with the twin's answer. Optional /voice
toggle replies with an OGG voice note via the TTS service.

Env: TWIN_TELEGRAM_TOKEN_FILE, TWIN_TELEGRAM_ALLOWED_IDS (csv),
     TWIN_ORCH_URL (default http://localhost:8080).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request

ORCH = os.environ.get("TWIN_ORCH_URL", "http://localhost:8080")
TOKEN = open(os.environ["TWIN_TELEGRAM_TOKEN_FILE"]).read().strip()
ALLOWED = {int(x) for x in os.environ["TWIN_TELEGRAM_ALLOWED_IDS"].split(",")}
ALLOWED = {int(x) for x in os.environ["TWIN_TELEGRAM_ALLOWED_IDS"].split(",")}
API = f"https://api.telegram.org/bot{TOKEN}"

voice_mode: dict[int, bool] = {}


def tg(method: str, /, timeout: int = 35, **params) -> dict:
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(f"{API}/{method}", data=data)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def send_voice(chat_id: int, wav_b64: str) -> None:
    import base64
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(base64.b64decode(wav_b64))
        wav = f.name
    ogg = wav.replace(".wav", ".ogg")
    subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-i", wav, "-c:a", "libopus", ogg], check=True)
    boundary = "twinvoice"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{chat_id}\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"voice\"; filename=\"reply.ogg\"\r\n"
        f"Content-Type: audio/ogg\r\n\r\n"
    ).encode() + open(ogg, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{API}/sendVoice", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    urllib.request.urlopen(req, timeout=60)
    os.unlink(wav)
    os.unlink(ogg)


def ask_twin(text: str, voice: bool) -> dict:
    req = urllib.request.Request(
        f"{ORCH}/chat",
        data=json.dumps({"text": text, "voice": voice}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def main() -> None:
    offset = 0
    print("twin telegram relay up; allowed:", ALLOWED, flush=True)
    while True:
        try:
            updates = tg("getUpdates", offset=offset, timeout=30)["result"]
        except Exception as e:  # noqa: BLE001 — network blips: back off, retry
            print("getUpdates error:", e, flush=True)
            time.sleep(5)
            continue
        for u in updates:
            offset = u["update_id"] + 1
            msg = u.get("message") or {}
            chat_id = (msg.get("chat") or {}).get("id")
            text = msg.get("text")
            if chat_id is None:
                continue
            if not text or (msg.get("from") or {}).get("id") not in ALLOWED:
                continue
            if text.strip() == "/voice":
                voice_mode[chat_id] = not voice_mode.get(chat_id, False)
                tg("sendMessage", chat_id=chat_id,
                   text=f"voice replies: {'on' if voice_mode[chat_id] else 'off'}")
                continue
            if text.strip() == "/start":
                tg("sendMessage", chat_id=chat_id,
                   text="AbhiTwin online (persona-v1 on the Spark). /voice toggles voice notes.")
                continue
            tg("sendChatAction", chat_id=chat_id, action="typing", timeout=10)
            try:
                out = ask_twin(text, voice_mode.get(chat_id, False))
            except Exception as e:  # noqa: BLE001
                tg("sendMessage", chat_id=chat_id, text=f"twin error: {e}")
                continue
            reply = out.get("reply") or "(empty reply)"
            tg("sendMessage", chat_id=chat_id, text=reply[:4000])
            if voice_mode.get(chat_id, False) and out.get("audio_b64"):
                try:
                    send_voice(chat_id, out["audio_b64"])
                except Exception as e:  # noqa: BLE001
                    print("voice send failed:", e, flush=True)


if __name__ == "__main__":
    main()
