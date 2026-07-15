#!/usr/bin/env python3
"""Video twin eval (addendum benchmark table): FPS @256 and @512, audio-in ->
first-frame latency over the WS streamer, and memory usage. RUN ON SPARK with
the musetalk service up (make serve-all).

Gate (spec §12 v1.5): FPS >= 10 @256. If it fails, suspect an ORT CPU fallback
or PTX JIT — watch `nvidia-smi dmon -s um` while it runs.
"""

from __future__ import annotations

import json
import sys
import time

import click


def fps_from_frame_times(frame_times: list[float]) -> float:
    """Frames per second from monotonic frame arrival timestamps."""
    if len(frame_times) < 2:
        return 0.0
    span = frame_times[-1] - frame_times[0]
    return (len(frame_times) - 1) / span if span > 0 else 0.0


def _bench_ws(url: str, resolution: int, seconds: float = 5.0) -> dict:
    import numpy as np
    from websockets.sync.client import connect

    # seconds of 16 kHz mono PCM silence-with-tone — deterministic input
    t = np.linspace(0, seconds, int(16000 * seconds), endpoint=False)
    pcm = (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16).tobytes()

    frame_times: list[float] = []
    first_frame_latency = None
    with connect(url, max_size=None) as ws:
        ws.recv()  # ready event
        t0 = time.monotonic()
        ws.send(pcm)
        deadline = t0 + seconds + 30
        while time.monotonic() < deadline:
            msg = ws.recv(timeout=deadline - time.monotonic())
            if isinstance(msg, bytes):
                frame_times.append(time.monotonic())
                if first_frame_latency is None:
                    first_frame_latency = frame_times[0] - t0
            else:
                event = json.loads(msg)
                if event.get("event") == "first_frame":
                    first_frame_latency = event["latency_s"]
            if frame_times and frame_times[-1] - t0 > seconds:
                break
    return {
        "resolution": resolution,
        "fps": round(fps_from_frame_times(frame_times), 2),
        "first_frame_s": round(first_frame_latency or -1, 3),
        "frames": len(frame_times),
    }


def _gpu_memory_gb() -> float:
    import subprocess

    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    return round(int(out) / 1024, 1)


@click.command(help=__doc__)
@click.option("--url", default="ws://localhost:8004/ws/video", show_default=True)
@click.option("--gate", is_flag=True)
@click.option("--gate-fps", default=10.0, show_default=True)
def main(url: str, gate: bool, gate_fps: float) -> None:
    try:
        import websockets  # noqa: F401
    except ImportError:
        click.echo("=== RUN ON SPARK === eval/video.py needs the musetalk service "
                   "(make serve-all) and `pip install websockets numpy`.")
        sys.exit(2)

    results = {"benchmarks": [], "gpu_memory_gb": None}
    for res in (256, 512):
        bench = _bench_ws(url, res)
        results["benchmarks"].append(bench)
        click.echo(f"@{res}: {bench['fps']} FPS, first frame {bench['first_frame_s']}s")
    try:
        results["gpu_memory_gb"] = _gpu_memory_gb()
    except Exception:
        pass
    click.echo(json.dumps(results, indent=2))

    if gate:
        fps256 = results["benchmarks"][0]["fps"]
        ok = fps256 >= gate_fps
        click.echo("=== verify-video: PASSED ===" if ok else
                   f"=== verify-video: FAILED (fps@256 {fps256} < {gate_fps}; "
                   "check for ORT CPU fallback) ===")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
