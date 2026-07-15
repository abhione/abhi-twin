#!/usr/bin/env python3
"""Phase 1 verify gate (spec §12): >= 8k pairs, PII-scrubbed with presidio,
holdout frozen and self-consistent. Host: mac | spark."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import click

MIN_PAIRS = 8000


@click.command(help=__doc__)
@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=Path("corpus/data/out"))
@click.option("--min-pairs", default=MIN_PAIRS, show_default=True)
@click.option("--allow-regex-pii", is_flag=True, help="accept a non-presidio smoke build")
def main(out_dir: Path, min_pairs: int, allow_regex_pii: bool) -> None:
    failures: list[str] = []

    stats_path = out_dir / "stats.json"
    manifest_path = out_dir / "holdout.manifest.json"
    train_path = out_dir / "train.jsonl"
    eval_path = out_dir / "eval.jsonl"
    for p in (stats_path, manifest_path, train_path, eval_path):
        if not p.exists():
            click.echo(f"FAIL  missing {p} — run `make corpus` first")
            sys.exit(1)

    stats = json.loads(stats_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    n_train = sum(1 for line in train_path.open() if line.strip())
    n_eval = sum(1 for line in eval_path.open() if line.strip())
    total = n_train + n_eval
    if total >= min_pairs:
        click.echo(f"PASS  {total} pairs (>= {min_pairs})")
    else:
        failures.append(f"only {total} pairs, need >= {min_pairs}")

    if stats.get("pii_engine") == "presidio" or allow_regex_pii:
        click.echo(f"PASS  PII scrubbed (engine: {stats.get('pii_engine')})")
    else:
        failures.append(f"pii_engine is {stats.get('pii_engine')!r}; presidio required")

    eval_ids = sorted(
        json.loads(line)["id"] for line in eval_path.open() if line.strip()
    )
    digest = hashlib.sha256("\n".join(eval_ids).encode()).hexdigest()
    if digest == manifest["sha256"] and len(eval_ids) == manifest["count"]:
        click.echo(f"PASS  holdout frozen + consistent ({manifest['count']} eval pairs)")
    else:
        failures.append("eval.jsonl does not match the frozen holdout manifest")

    if failures:
        for f in failures:
            click.echo(f"FAIL  {f}")
        click.echo("=== verify-corpus: FAILED ===")
        sys.exit(1)
    click.echo("=== verify-corpus: ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
