#!/usr/bin/env python3
"""Extract Apple Notes to corpus records (spec §6.1: 200-500 notes, long-form
voice). Host: mac (vader). Notes are promptless — the pipeline reconstructs
synthetic prompts.

Two input modes:
  --from-db   read NoteStore.sqlite directly (default path below). Note bodies
              are gzipped protobufs; the text lives at proto field path 2.3.2.
              Encrypted (password-protected) and trashed notes are skipped.
  --input     a notes.json from apple-notes-liberator (legacy; the tool NPEs on
              macOS 15+ NoteStore schemas, hence the direct reader).
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

import click

from corpus.extractors.common import make_id, strip_html, word_count
from corpus.pipeline.records import Record, write_jsonl

MIN_WORDS = 20

NOTESTORE_DEFAULT = Path.home() / "Library/Group Containers/group.com.apple.notes/NoteStore.sqlite"

_DB_QUERY = """
SELECT d.Z_PK, d.ZDATA, o.ZTITLE1, f.ZTITLE2
FROM ZICNOTEDATA d
JOIN ZICCLOUDSYNCINGOBJECT o ON o.ZNOTEDATA = d.Z_PK
LEFT JOIN ZICCLOUDSYNCINGOBJECT f ON f.Z_PK = o.ZFOLDER
WHERE d.ZDATA IS NOT NULL
  AND d.ZCRYPTOINITIALIZATIONVECTOR IS NULL
  AND (o.ZMARKEDFORDELETION IS NULL OR o.ZMARKEDFORDELETION = 0)
"""


def _proto_fields(buf: bytes):
    """Yield (field_no, wire_type, value) for one protobuf message; bail on junk."""
    i, n = 0, len(buf)
    while i < n:
        key, shift = 0, 0
        while True:
            if i >= n:
                return
            b = buf[i]
            i += 1
            key |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                break
        field_no, wire = key >> 3, key & 7
        if wire == 0:  # varint
            val, shift = 0, 0
            while True:
                if i >= n:
                    return
                b = buf[i]
                i += 1
                val |= (b & 0x7F) << shift
                shift += 7
                if not b & 0x80:
                    break
        elif wire == 1:
            val, i = buf[i : i + 8], i + 8
        elif wire == 2:
            ln, shift = 0, 0
            while True:
                if i >= n:
                    return
                b = buf[i]
                i += 1
                ln |= (b & 0x7F) << shift
                shift += 7
                if not b & 0x80:
                    break
            val, i = buf[i : i + ln], i + ln
        elif wire == 5:
            val, i = buf[i : i + 4], i + 4
        else:
            return
        yield field_no, wire, val


def _submessage(buf: bytes, field_no: int) -> bytes | None:
    for f, w, v in _proto_fields(buf):
        if f == field_no and w == 2:
            return v
    return None


def decode_note_body(zdata: bytes) -> str | None:
    """gunzip ZICNOTEDATA.ZDATA and pull the note text (proto path 2.3.2)."""
    try:
        raw = gzip.decompress(zdata)
    except OSError:
        return None
    doc = _submessage(raw, 2)
    if doc is None:
        return None
    note = _submessage(doc, 3)
    if note is None:
        return None
    text = _submessage(note, 2)
    if text is None:
        return None
    try:
        return text.decode("utf-8").strip() or None
    except UnicodeDecodeError:
        return None


def extract_db(db_path: Path, min_words: int = MIN_WORDS) -> tuple[list[Record], int]:
    """Return (records, skipped_undecodable) straight from NoteStore.sqlite."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(_DB_QUERY).fetchall()
    conn.close()
    records: list[Record] = []
    skipped = 0
    for pk, zdata, title, folder in rows:
        body = decode_note_body(zdata)
        if body is None:
            skipped += 1
            continue
        title = (title or "").strip()
        if word_count(body) < min_words:
            continue
        text = f"{title}\n\n{body}".strip() if title and not body.startswith(title) else body
        records.append(
            Record(
                id=make_id("apple_notes", str(pk), body[:80]),
                source="apple_notes",
                reply=text,
                prompt=None,
                meta={"title": title, "folder": (folder or "").strip()},
            )
        )
    return records, skipped


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
              default=None, help="notes.json from apple-notes-liberator")
@click.option("--from-db", "db_path", type=click.Path(path_type=Path), is_flag=False,
              flag_value=NOTESTORE_DEFAULT, default=None,
              help=f"read NoteStore.sqlite directly [flag default: {NOTESTORE_DEFAULT}]")
@click.option("--out", type=click.Path(path_type=Path),
              default=Path("corpus/data/extracted/apple_notes.jsonl"))
@click.option("--min-words", default=MIN_WORDS, show_default=True)
def main(notes_path: Path | None, db_path: Path | None, out: Path, min_words: int) -> None:
    if notes_path:
        records = extract(notes_path, min_words)
        skipped = 0
    elif db_path:
        if not db_path.exists():
            raise click.ClickException(f"{db_path} not found")
        records, skipped = extract_db(db_path, min_words)
    else:
        raise click.ClickException("pass --input notes.json or --from-db")
    n = write_jsonl(out, records)
    click.echo(f"apple_notes: {n} records -> {out} ({skipped} undecodable skipped)")


if __name__ == "__main__":
    main()
