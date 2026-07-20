#!/usr/bin/env python3
"""Voice eval (spec §8/§12): RTF measurement + the white-noise regression test,
exercised through the LIVE serving path (POST /v1/audio/speech) — a 10 s synth
whose waveform is finite end-to-end. RUN ON SPARK with the voice profile up
(`make serve-voice`); it runs inside the tts container (make verify-voice) or
any host that can reach :8001.

The server asserts no NaN inside the synth pipeline on every request
(serving/tts/server.py assert_finite); a tripped guard turns into an HTTP 500
here, which fails the gate. This client additionally checks the decoded PCM is
finite and non-silent.

Gate: RTF < 1.5x (SDPA should land < 1.0x), measured on a warm model — one
untimed warmup request first, so lazy model load doesn't pollute RTF.
"""

from __future__ import annotations

import io
import json
import sys
import time
import urllib.error
import urllib.request
import wave

import click

TEN_SECOND_TEXT = (
    "Here is a full ten seconds of natural speech for the regression test. "
    "The Spark serves the persona model, the voice clone, and the video twin "
    "from one hundred twenty-eight gigabytes of unified memory, and the epsilon "
    "clamp keeps the vocoder from ever seeing a hard zero on the way in."
)


def measure_rtf(synthesize, text: str, sample_rate: int = 24000) -> tuple[float, float]:
    """Return (rtf, audio_seconds). rtf = wall_time / audio_duration."""
    t0 = time.monotonic()
    wav = synthesize(text)
    wall = time.monotonic() - t0
    audio_s = len(wav) / sample_rate
    return (wall / audio_s if audio_s else float("inf")), audio_s


def wav_stats(wav_bytes: bytes) -> tuple[float, int]:
    """Parse a PCM WAV -> (duration_seconds, peak_abs_amplitude)."""
    import array

    with wave.open(io.BytesIO(wav_bytes)) as w:
        frames, rate, width = w.getnframes(), w.getframerate(), w.getsampwidth()
        raw = w.readframes(frames)
    if width != 2:
        raise ValueError(f"expected 16-bit PCM, got sample width {width}")
    samples = array.array("h", raw)
    peak = max((abs(s) for s in samples), default=0)
    return frames / rate, peak


def _synth(url: str, text: str, timeout: float) -> bytes:
    req = urllib.request.Request(
        url,
        data=json.dumps({"model": "voice-v1", "input": text, "voice": "abhi"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


@click.command(help=__doc__)
@click.option("--url", default="http://localhost:8001/v1/audio/speech", show_default=True)
@click.option("--gate", is_flag=True)
@click.option("--gate-rtf", default=1.5, show_default=True)
@click.option("--timeout", default=600.0, show_default=True,
              help="per-request timeout (first request lazy-loads the model)")
def main(gate: bool, gate_rtf: float, url: str, timeout: float) -> None:
    try:
        t_warm0 = time.monotonic()
        _synth(url, "Warm up.", timeout)  # untimed: absorbs the lazy model load
        warmup_s = time.monotonic() - t_warm0
        t0 = time.monotonic()
        wav_bytes = _synth(url, TEN_SECOND_TEXT, timeout)
        wall = time.monotonic() - t0
    except (urllib.error.URLError, OSError) as exc:
        click.echo(f"=== RUN ON SPARK === TTS unreachable at {url} ({exc}). "
                   "Bring the stack up with `make serve-voice`.")
        sys.exit(2)
    audio_s, peak = wav_stats(wav_bytes)
    rtf = wall / audio_s if audio_s else float("inf")
    results = {
        "rtf": round(rtf, 3),
        "audio_seconds": round(audio_s, 2),
        "wall_s": round(wall, 3),
        "warmup_s": round(warmup_s, 3),
        "peak_amplitude": peak,
        "nan_in_synth_path": False,  # server assert_finite would have 500'd
    }
    click.echo(json.dumps(results, indent=2))
    if gate:
        ok = rtf < gate_rtf and peak > 0
        click.echo("=== verify-voice: PASSED ===" if ok else "=== verify-voice: FAILED ===")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
