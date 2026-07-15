#!/usr/bin/env python3
"""Convert an apple-notes-liberator JSON export to corpus records (spec §6.1:
200-500 notes, long-form voice). Host: mac (vader).

Run https://github.com/HamburgChimps/apple-notes-liberator first; it writes
notes.json. Notes are promptless — the pipeline reconstructs synthetic prompts.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from corpus.extractors.common import make_id, strip_html, word_count
from corpus.pipeline.records import Record, write_jsonl

MIN_WORDS = 20


def extract(notes_path: Path, min_words: int = MIN_WORDS) -> list[Record]:
    notes = json.loads(notes_path.read_text())
    records: list[Record] = []
    for note in notes:
        title = (note.get("title") or "").strip()
        body = note.get("plainText") or note.get("plaintext") or note.get("body") or ""
        if "<" in body and ">" in body:
            body = strip_html(body)
        body = body.strip()
        if word_count(body) < min_words:
            continue
        text = f"{title}\n\n{body}".strip() if title and not body.startswith(title) else body
        records.append(
            Record(
                id=make_id("apple_notes", title, body[:80]),
                source="apple_notes",
                reply=text,
                prompt=None,
                meta={"title": title, "folder": note.get("folder", "")},
            )
        )
    return records


@click.command(help=__doc__)
@click.option("--input", "notes_path", type=click.Path(exists=True, path_type=Path),
              required=True, help="notes.json from apple-notes-liberator")
@click.option("--out", type=click.Path(path_type=Path),
              default=Path("corpus/data/extracted/apple_notes.jsonl"))
@click.option("--min-words", default=MIN_WORDS, show_default=True)
def main(notes_path: Path, out: Path, min_words: int) -> None:
    records = extract(notes_path, min_words)
    n = write_jsonl(out, records)
    click.echo(f"apple_notes: {n} records -> {out}")


if __name__ == "__main__":
    main()
