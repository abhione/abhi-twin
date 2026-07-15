#!/usr/bin/env python3
"""Phase 4 verify gate: end-to-end voice conversation < 3 s round trip
(mic wav -> STT -> router -> persona -> TTS wav). RUN ON SPARK with the full
voice stack up (make serve-voice)."""

from __future__ import annotations

import io
import json
import math
import struct
import sys
import urllib.request
import uuid

import click


def make_test_wav(seconds: float = 2.0, rate: int = 16000) -> bytes:
    """Tiny dependency-free WAV: a spoken-band tone standing in for mic input."""
    n = int(seconds * rate)
    samples = b"".join(
        struct.pack("<h", int(6000 * math.sin(2 * math.pi * 180 * i / rate))) for i in range(n)
    )
    header = (
        b"RIFF" + struct.pack("<I", 36 + len(samples)) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data" + struct.pack("<I", len(samples))
    )
    return header + samples


def post_multipart(url: str, field: str, filename: str, payload: bytes,
                   timeout: float = 120.0) -> dict:
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        "Content-Type: audio/wav\r\n\r\n".encode()
    )
    body.write(payload)
    body.write(f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        url, data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


@click.command(help=__doc__)
@click.option("--url", default="http://localhost:8080/v1/audio/roundtrip", show_default=True)
@click.option("--gate", is_flag=True)
@click.option("--gate-seconds", default=3.0, show_default=True)
@click.option("--wav", type=click.Path(exists=True), default=None,
              help="real recorded question wav (better than the synthetic tone)")
def main(url: str, gate: bool, gate_seconds: float, wav: str | None) -> None:
    payload = open(wav, "rb").read() if wav else make_test_wav()
    try:
        result = post_multipart(url, "file", "question.wav", payload)
    except OSError as exc:
        click.echo(f"=== RUN ON SPARK === orchestrator unreachable at {url} ({exc}). "
                   "Bring the stack up with `make serve-voice`.")
        sys.exit(2)
    click.echo(json.dumps({k: v for k, v in result.items() if k != "audio_b64"}, indent=2))
    if gate:
        rt = result.get("roundtrip_s", float("inf"))
        ok = rt < gate_seconds
        click.echo(f"=== verify-e2e: {'PASSED' if ok else 'FAILED'} "
                   f"({rt}s vs {gate_seconds}s bar) ===")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
