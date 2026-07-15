#!/usr/bin/env python3
"""Voice eval (spec §8/§12): RTF measurement + the white-noise regression test —
synthesize 10 s and assert no NaN in the mel spectrogram. RUN ON SPARK.

Gate: RTF < 1.5x (SDPA should land < 1.0x).
"""

from __future__ import annotations

import json
import sys
import time

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


@click.command(help=__doc__)
@click.option("--gate", is_flag=True)
@click.option("--gate-rtf", default=1.5, show_default=True)
def main(gate: bool, gate_rtf: float) -> None:
    try:
        import torch
    except ImportError:
        click.echo("=== RUN ON SPARK === eval/voice.py needs the TTS checkpoint + GPU.")
        sys.exit(2)

    from serving.tts.guards import assert_finite, epsilon_clamp
    from serving.tts.server import _load

    processor, model = _load()

    def synth(text: str):
        inputs = processor(text=text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            codec = model.generate(**inputs)
            mel = epsilon_clamp(model.codec_to_mel(codec))
            # THE regression assert: 10 s synth, no NaN anywhere in the mel
            assert_finite(mel.float().cpu(), "mel")
            return model.vocoder(mel).float().cpu().numpy().squeeze()

    rtf, audio_s = measure_rtf(synth, TEN_SECOND_TEXT)
    results = {"rtf": round(rtf, 3), "audio_seconds": round(audio_s, 2), "nan_in_mel": False}
    click.echo(json.dumps(results, indent=2))
    if gate:
        ok = rtf < gate_rtf
        click.echo("=== verify-voice: PASSED ===" if ok else "=== verify-voice: FAILED ===")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
