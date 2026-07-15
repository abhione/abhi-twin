"""Near-duplicate removal: MinHash + LSH, Jaccard >= 0.85 drops the later record
(spec §6 step 1). Pure python — no datasketch dependency. Host: mac | spark."""

from __future__ import annotations

import hashlib
import random
import re
from typing import Iterable

from corpus.pipeline.records import Record

_NUM_PERM = 128
_BANDS = 16  # 16 bands x 8 rows
_PRIME = (1 << 61) - 1
_rng = random.Random(0xAB41)
_A = [_rng.randrange(1, _PRIME) for _ in range(_NUM_PERM)]
_B = [_rng.randrange(0, _PRIME) for _ in range(_NUM_PERM)]

_WORD = re.compile(r"[a-z0-9']+")


def shingles(text: str, k: int = 3) -> frozenset[int]:
    """Word k-gram shingles, hashed to 64-bit ints."""
    words = _WORD.findall(text.lower())
    if not words:
        return frozenset()
    grams = (
        [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]
        if len(words) >= k
        else [" ".join(words)]
    )
    return frozenset(
        int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big") for g in grams
    )


def signature(sh: frozenset[int]) -> tuple[int, ...]:
    if not sh:
        return tuple([0] * _NUM_PERM)
    return tuple(min((a * h + b) % _PRIME for h in sh) for a, b in zip(_A, _B))


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedup(
    records: Iterable[Record], threshold: float = 0.85
) -> tuple[list[Record], list[Record]]:
    """Return (kept, dropped). LSH banding finds candidates; exact Jaccard on the
    shingle sets confirms before dropping. First occurrence wins."""
    rows = _NUM_PERM // _BANDS
    buckets: dict[tuple[int, int], list[int]] = {}
    kept: list[Record] = []
    kept_shingles: list[frozenset[int]] = []
    dropped: list[Record] = []

    for rec in records:
        sh = shingles(f"{rec.prompt or ''}\n{rec.reply}")
        sig = signature(sh)
        band_keys = [
            (band, hash(sig[band * rows : (band + 1) * rows])) for band in range(_BANDS)
        ]
        candidates: set[int] = set()
        for key in band_keys:
            candidates.update(buckets.get(key, ()))
        if any(jaccard(sh, kept_shingles[i]) >= threshold for i in candidates):
            dropped.append(rec)
            continue
        idx = len(kept)
        kept.append(rec)
        kept_shingles.append(sh)
        for key in band_keys:
            buckets.setdefault(key, []).append(idx)
    return kept, dropped
