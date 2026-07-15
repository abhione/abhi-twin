"""Holdout split: 5%, stratified by source, FROZEN once written (spec §6 step 6).

The eval split's identity is captured in holdout.manifest.json (sha256 over sorted
ids). Re-running the pipeline must reproduce the same manifest or fail — the eval
set never silently drifts under the metrics. Host: mac | spark.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from corpus.pipeline.records import Record

MANIFEST_NAME = "holdout.manifest.json"
# sources smaller than this stay in train — a 5% slice of 6 notes isn't an eval set
_MIN_SOURCE_SIZE = 10


def split(
    records: list[Record], frac: float = 0.05, seed: int = 13
) -> tuple[list[Record], list[Record]]:
    """Return (train, eval), stratified by source, deterministic for a given seed."""
    by_source: dict[str, list[Record]] = defaultdict(list)
    for rec in records:
        by_source[rec.source].append(rec)

    rng = random.Random(seed)
    eval_ids: set[str] = set()
    for source in sorted(by_source):
        recs = sorted(by_source[source], key=lambda r: r.id)
        if len(recs) < _MIN_SOURCE_SIZE:
            continue
        k = max(1, int(len(recs) * frac))
        eval_ids.update(r.id for r in rng.sample(recs, k))

    train = [r for r in records if r.id not in eval_ids]
    eval_ = [r for r in records if r.id in eval_ids]
    return train, eval_


def manifest(eval_records: list[Record]) -> dict:
    ids = sorted(r.id for r in eval_records)
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    per_source: dict[str, int] = defaultdict(int)
    for r in eval_records:
        per_source[r.source] += 1
    return {"count": len(ids), "sha256": digest, "per_source": dict(sorted(per_source.items()))}


def freeze(out_dir: Path | str, m: dict) -> Path:
    """Write the manifest, or verify it matches the existing frozen one."""
    path = Path(out_dir) / MANIFEST_NAME
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["sha256"] != m["sha256"]:
            raise RuntimeError(
                f"holdout is FROZEN ({path}): existing sha {existing['sha256'][:12]} != "
                f"new {m['sha256'][:12]}. The eval set must not drift; if you truly "
                "rebuilt the corpus, bump the out dir (corpus-v2) instead."
            )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=2) + "\n")
    return path
