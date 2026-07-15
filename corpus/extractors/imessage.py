#!/usr/bin/env python3
"""Extract owner turns from the iMessage chat.db (spec §6.1: 5-10k msgs).
Host: mac (vader — grant the terminal Full Disk Access first).

Coalesces your consecutive messages into one turn; the preceding counterpart
messages become the prompt. Messages that store text only in `attributedBody`
(newer macOS) are decoded best-effort; undecodable rows are skipped and counted.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import click

from corpus.extractors.common import make_id
from corpus.pipeline.records import Record, write_jsonl

_QUERY = """
SELECT c.ROWID AS chat_id, m.ROWID AS msg_id, m.text, m.attributedBody,
       m.is_from_me, m.date
FROM message m
JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
JOIN chat c ON c.ROWID = cmj.chat_id
ORDER BY c.ROWID, m.date
"""


def decode_attributed_body(blob: bytes | None) -> str | None:
    """Best-effort typedstream decode: the string follows an 'NSString' marker."""
    if not blob:
        return None
    idx = blob.find(b"NSString")
    if idx == -1:
        return None
    tail = blob[idx + len(b"NSString") :]
    # skip typedstream framing bytes until the length-prefixed payload
    m = re.search(rb"\+([\x01-\xff])", tail[:16])
    if not m:
        return None
    start = m.end()
    length = m.group(1)[0]
    if length == 0x81:  # two-byte little-endian length
        length = int.from_bytes(tail[start : start + 2], "little")
        start += 2
    text = tail[start : start + length].decode("utf-8", errors="replace").strip()
    return text or None


def _row_text(text: str | None, blob: bytes | None) -> str | None:
    if text and text.strip():
        return text.strip()
    return decode_attributed_body(blob)


def extract(db_path: Path) -> tuple[list[Record], int]:
    """Return (records, skipped_undecodable)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(_QUERY).fetchall()
    conn.close()

    records: list[Record] = []
    skipped = 0
    chat: int | None = None
    prompt_buf: list[str] = []
    reply_buf: list[str] = []
    first_msg_id: int | None = None

    def flush() -> None:
        nonlocal prompt_buf, reply_buf, first_msg_id
        if reply_buf:
            records.append(
                Record(
                    id=make_id("imessage", str(chat), str(first_msg_id)),
                    source="imessage",
                    reply="\n".join(reply_buf),
                    prompt="\n".join(prompt_buf) if prompt_buf else None,
                    meta={"chat_id": chat},
                )
            )
        prompt_buf, reply_buf, first_msg_id = [], [], None

    for chat_id, msg_id, text, blob, is_from_me, _date in rows:
        if chat_id != chat:
            flush()
            prompt_buf = []
            chat = chat_id
        body = _row_text(text, blob)
        if body is None:
            skipped += 1
            continue
        if is_from_me:
            if first_msg_id is None:
                first_msg_id = msg_id
            reply_buf.append(body)
        else:
            if reply_buf:  # my turn ended; emit and start a fresh prompt window
                flush()
            prompt_buf.append(body)
    flush()
    return records, skipped


@click.command(help=__doc__)
@click.option("--db", "db_path", type=click.Path(exists=True, path_type=Path),
              default=Path.home() / "Library/Messages/chat.db", show_default=True)
@click.option("--out", type=click.Path(path_type=Path),
              default=Path("corpus/data/extracted/imessage.jsonl"))
def main(db_path: Path, out: Path) -> None:
    records, skipped = extract(db_path)
    n = write_jsonl(out, records)
    click.echo(f"imessage: {n} turn records -> {out} ({skipped} undecodable rows skipped)")


if __name__ == "__main__":
    main()
