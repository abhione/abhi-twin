#!/usr/bin/env python3
"""Apply the residual secret/PII scrub to an already-built corpus in place.
Host: mac | spark.

Covers builds produced before scrub_residuals existed: shared credentials
(password:/api_key:/AKIA…/bearer/private-key blocks) and email/phone leftovers
presidio missed. Holdout stays frozen — the manifest hashes eval IDs, and IDs
are never touched. Idempotent; run `corpus/verify.py` afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from corpus.pipeline.pii import scrub_residuals


def rescrub_file(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    out_lines = []
    # NOT .splitlines(): content can hold U+2028/U+2029, which it would split on
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        rec = json.loads(line)
        for msg in rec.get("messages", []):
            msg["content"], c = scrub_residuals(msg["content"])
            for k, v in c.items():
                counts[k] = counts.get(k, 0) + v
        out_lines.append(json.dumps(rec, ensure_ascii=False))
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(out_lines) + "\n")
    tmp.replace(path)
    return counts


@click.command(help=__doc__)
@click.option("--out", "out_dir", type=click.Path(exists=True, path_type=Path),
              default=Path("corpus/data/out"), show_default=True)
def main(out_dir: Path) -> None:
    total: dict[str, int] = {}
    for name in ("train.jsonl", "eval.jsonl"):
        counts = rescrub_file(out_dir / name)
        click.echo(f"{name}: {counts or 'clean'}")
        for k, v in counts.items():
            total[k] = total.get(k, 0) + v

    stats_path = out_dir / "stats.json"
    if stats_path.exists() and total:
        stats = json.loads(stats_path.read_text())
        repl = stats.setdefault("pii_replacements", {})
        for k, v in total.items():
            repl[k] = repl.get(k, 0) + v
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    click.echo(f"total replacements: {sum(total.values())}")


if __name__ == "__main__":
    main()
