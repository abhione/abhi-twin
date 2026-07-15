#!/usr/bin/env python3
"""Extract Abhi's sent mail from a Gmail Takeout mbox (spec §6.1: 3-5k threads).
Host: mac. Filters length >= 40 words; strips signatures + quoted text.

Takeout: https://takeout.google.com -> Mail -> "Sent" label (or All mail).
"""

from __future__ import annotations

import email.utils
import mailbox
from pathlib import Path

import click

from corpus.extractors.common import make_id, word_count
from corpus.pipeline.records import Record, write_jsonl
from corpus.pipeline.signatures import clean_email_body

MIN_WORDS = 40


def _plain_body(msg) -> str | None:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
        return None
    payload = msg.get_payload(decode=True)
    if payload is None:
        return None
    charset = msg.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def extract(mbox_path: Path, me: str, min_words: int = MIN_WORDS) -> list[Record]:
    records: list[Record] = []
    box = mailbox.mbox(str(mbox_path))
    for msg in box:
        sender = email.utils.parseaddr(msg.get("From", ""))[1].lower()
        if me.lower() not in sender:
            continue
        raw = _plain_body(msg)
        if not raw:
            continue
        body = clean_email_body(raw)
        if word_count(body) < min_words:
            continue
        msg_id = msg.get("Message-ID") or f"{msg.get('Date','')}-{msg.get('Subject','')}"
        records.append(
            Record(
                id=make_id("gmail", msg_id),
                source="gmail",
                reply=body,
                prompt=None,  # Takeout sent-mail has no counterpart; reconstructed later
                meta={
                    "subject": str(msg.get("Subject", "")),
                    "date": str(msg.get("Date", "")),
                },
            )
        )
    return records


@click.command(help=__doc__)
@click.option("--mbox", "mbox_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--me", envvar="TWIN_ME_EMAIL", required=True, help="owner email address")
@click.option("--out", type=click.Path(path_type=Path),
              default=Path("corpus/data/extracted/gmail.jsonl"))
@click.option("--min-words", default=MIN_WORDS, show_default=True)
def main(mbox_path: Path, me: str, out: Path, min_words: int) -> None:
    records = extract(mbox_path, me, min_words)
    n = write_jsonl(out, records)
    click.echo(f"gmail: {n} records -> {out}")


if __name__ == "__main__":
    main()
