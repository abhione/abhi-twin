#!/usr/bin/env python3
"""Corpus cleaning pipeline (spec §6): dedup -> PII scrub -> length -> perplexity
-> chat format -> frozen holdout -> stats (+ optional tarball for cloud burst).

Host: mac (extract + clean, then rsync/upload) | spark. The perplexity stage and
synthetic prompt reconstruction need a model endpoint — on the Mac run with
--skip-ppl and either --skip-reconstruct (drops promptless records) or a reachable
Spark at TWIN_SPARK_HOST.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import click

from corpus.pipeline import dedup as dedup_mod
from corpus.pipeline import filters, formatting, holdout, pii
from corpus.pipeline.records import Record, read_jsonl


def load_extracted(in_dir: Path) -> list[Record]:
    records: list[Record] = []
    for path in sorted(in_dir.glob("*.jsonl")):
        records.extend(read_jsonl(path))
    return records


def write_chat_jsonl(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")


@click.command(help=__doc__)
@click.option("--in-dir", type=click.Path(path_type=Path), default=Path("corpus/data/extracted"))
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("corpus/data/out"))
@click.option(
    "--pii-engine",
    type=click.Choice(["presidio", "regex", "auto"]),
    default="presidio",
    help="presidio is mandatory for a real build; regex only for smoke tests",
)
@click.option("--allow-regex-pii", is_flag=True, help="acknowledge a non-presidio build")
@click.option("--skip-ppl", is_flag=True, help="skip perplexity filter (no GPU on this host)")
@click.option("--skip-reconstruct", is_flag=True, help="drop promptless records instead of "
              "reconstructing prompts via the Spark LLM")
@click.option("--reconstruct-url", default=None, help="OpenAI-compatible /v1 base URL")
@click.option("--min-tokens", default=30, show_default=True)
@click.option("--max-tokens", default=4096, show_default=True)
@click.option("--holdout-frac", default=0.05, show_default=True)
@click.option("--seed", default=13, show_default=True)
@click.option("--tar", "make_tar", is_flag=True, help="write corpus-v1.tar.gz for HF upload")
def main(
    in_dir: Path,
    out_dir: Path,
    pii_engine: str,
    allow_regex_pii: bool,
    skip_ppl: bool,
    skip_reconstruct: bool,
    reconstruct_url: str | None,
    min_tokens: int,
    max_tokens: int,
    holdout_frac: float,
    seed: int,
    make_tar: bool,
) -> None:
    stats: dict = {"stages": {}}

    records = load_extracted(in_dir)
    if not records:
        raise click.ClickException(f"no extracted records in {in_dir} — run the extractors first")
    stats["stages"]["extracted"] = len(records)
    click.echo(f"[1/6] loaded {len(records)} extracted records from {in_dir}")

    records, dropped = dedup_mod.dedup(records)
    stats["stages"]["dedup_dropped"] = len(dropped)
    click.echo(f"[1/6] dedup: dropped {len(dropped)} near-duplicates (Jaccard >= 0.85)")

    scrubber = pii.get_scrubber(pii_engine)
    if scrubber.name != "presidio" and not allow_regex_pii:
        raise click.ClickException(
            "PII engine resolved to regex but presidio is mandatory for a real build. "
            "Install with `pip install -e '.[corpus]'` or pass --allow-regex-pii for a smoke run."
        )
    pii_counts: dict[str, int] = {}
    for rec in records:
        rec.reply, c1 = scrubber.scrub(rec.reply)
        if rec.prompt:
            rec.prompt, c2 = scrubber.scrub(rec.prompt)
            for k, v in c2.items():
                c1[k] = c1.get(k, 0) + v
        for k, v in c1.items():
            pii_counts[k] = pii_counts.get(k, 0) + v
    stats["pii_engine"] = scrubber.name
    stats["pii_replacements"] = pii_counts
    click.echo(f"[2/6] pii ({scrubber.name}): replaced {sum(pii_counts.values())} entities")

    records, dropped = filters.length_filter(records, min_tokens, max_tokens)
    stats["stages"]["length_dropped"] = len(dropped)
    click.echo(f"[3/6] length [{min_tokens}, {max_tokens}]: dropped {len(dropped)}")

    if skip_ppl:
        stats["stages"]["ppl"] = "skipped (=== RUN ON SPARK === for the full build)"
        click.echo("[4/6] perplexity filter SKIPPED — rerun on the Spark for the real corpus")
    else:
        scorer = filters.QwenPerplexityScorer()
        records, dropped = filters.ppl_filter(records, scorer)
        stats["stages"]["ppl_dropped"] = len(dropped)
        click.echo(f"[4/6] perplexity: dropped top {len(dropped)} weirdest")

    reconstructor = None
    if not skip_reconstruct:
        reconstructor = formatting.EndpointReconstructor(base_url=reconstruct_url)
    samples, promptless_dropped, kept_records = [], 0, []
    for rec in records:
        sample = formatting.to_messages(rec, reconstructor)
        if sample is None:
            promptless_dropped += 1
            continue
        samples.append(sample)
        kept_records.append(rec)
    stats["stages"]["promptless_dropped"] = promptless_dropped
    click.echo(f"[5/6] format: {len(samples)} chat pairs ({promptless_dropped} promptless dropped)")

    train_recs, eval_recs = holdout.split(kept_records, holdout_frac, seed)
    eval_ids = {r.id for r in eval_recs}
    train_samples = [s for s in samples if s["id"] not in eval_ids]
    eval_samples = [s for s in samples if s["id"] in eval_ids]
    m = holdout.manifest(eval_recs)
    holdout.freeze(out_dir, m)
    write_chat_jsonl(out_dir / "train.jsonl", train_samples)
    write_chat_jsonl(out_dir / "eval.jsonl", eval_samples)
    stats["train"] = len(train_samples)
    stats["eval"] = len(eval_samples)
    stats["holdout_sha256"] = m["sha256"]
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    click.echo(
        f"[6/6] holdout frozen ({m['count']} eval / {len(train_samples)} train) -> {out_dir}"
    )

    if make_tar:
        tar_path = out_dir / "corpus-v1.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            for name in ("train.jsonl", "eval.jsonl", "stats.json", holdout.MANIFEST_NAME):
                tf.add(out_dir / name, arcname=name)
        click.echo(f"tarball: {tar_path} — upload with training/burst/sync_corpus.sh")


if __name__ == "__main__":
    main()
