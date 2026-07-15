"""Numeric guards for the TTS vocoder path. Host: mac | spark (pure python, no GPU deps).

Blackwell white-noise bug (Martinb): un-clamped zeros in the HiFi-GAN input produce
log(0) -> NaN in the mel spectrogram and the vocoder emits white noise. Every call
into the vocoder MUST go through epsilon_clamp(); ci/preflight.py unit-tests this,
and eval/voice.py synthesizes 10 s on the Spark and asserts no NaN in the mel.
"""

from __future__ import annotations

import math
from typing import Any

EPSILON = 1e-5


def epsilon_clamp(x: Any, eps: float = EPSILON) -> Any:
    """Clamp vocoder input away from zero (min=eps).

    Accepts torch tensors, numpy arrays, or plain (nested) lists of floats so the
    guard is unit-testable on machines without CUDA.
    """
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.clamp(min=eps)
    except ImportError:
        pass
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return np.clip(x, eps, None)
    except ImportError:
        pass
    if isinstance(x, (int, float)):
        return max(float(x), eps)
    if isinstance(x, (list, tuple)):
        return type(x)(epsilon_clamp(v, eps) for v in x)
    raise TypeError(f"epsilon_clamp: unsupported type {type(x)!r}")


def safe_log10(v: float, eps: float = EPSILON) -> float:
    """log10 with the same epsilon floor — the operation the clamp protects."""
    return math.log10(max(v, eps))


def assert_finite(values: Any, name: str = "mel") -> None:
    """Raise ValueError if any element is NaN/inf. Flattens nested lists."""

    def _iter(x):
        if isinstance(x, (list, tuple)):
            for v in x:
                yield from _iter(v)
        else:
            try:  # torch / numpy containers
                yield from (float(v) for v in x.flatten().tolist())
            except AttributeError:
                yield float(x)

    for v in _iter(values):
        if not math.isfinite(v):
            raise ValueError(f"{name} contains non-finite value {v!r} — vocoder guard tripped")
