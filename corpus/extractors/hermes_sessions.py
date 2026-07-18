#!/usr/bin/env python3
"""Extract Abhi's user turns from Hermes agent sessions (~/.hermes/state.db).
Host: mac (starbase).

His messages to the agent are the training target (`reply`); the preceding
assistant turns become the prompt. The live Hermes agent writes to this db, so
it is opened read-only + immutable — never write to it. FTS mirror tables are
ignored. Harness-injected turns ("[IMPORTANT: ...]"), pasted logs/JSON and
oversized dumps are filtered via common.looks_like_pasted.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import click

from corpus.extractors.common import clip, contains_secret, looks_like_pasted, make_id
from corpus.pipeline.records import Record, write_jsonl

_QUERY = """
SELECT id, session_id, role, content, timestamp
FROM messages
WHERE active = 1 AND role IN ('user', 'assistant') AND content IS NOT NULL
ORDER BY session_id, timestamp, id
"""

_CTX_TURNS = 2  # assistant turns kept as prompt context
_CTX_CLIP = 600  # chars per context turn


def extract(db_path: Path) -> tuple[list[Record], int]:
    """Return (records, filtered_user_msgs). Read-only: mode=ro&immutable=1."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    rows = conn.execute(_QUERY).fetchall()
    conn.close()

    records: list[Record] = []
    filtered = 0
    session: str | None = None
    prompt_buf: list[str] = []
    reply_buf: list[str] = []
    first_id: int | None = None
    first_ts: float | None = None

    def flush() -> None:
        nonlocal prompt_buf, reply_buf, first_id, first_ts
        if reply_buf:
            records.append(
                Record(
                    id=make_id("hermes", str(session), str(first_id)),
                    source="hermes",
                    reply="\n".join(reply_buf),
                    prompt="\n".join(prompt_buf) if prompt_buf else None,
                    meta={"session_id": session, "ts": first_ts},
                )
            )
        prompt_buf, reply_buf, first_id, first_ts = [], [], None, None

    for msg_id, session_id, role, content, ts in rows:
        if session_id != session:
            flush()
            prompt_buf = []
            session = session_id
        body = content.strip() if isinstance(content, str) else None
        if not body:
            continue
        if role == "user":
            if looks_like_pasted(body) or contains_secret(body):
                filtered += 1
                flush()
                prompt_buf = []  # pasted dump breaks the context window
                continue
            if first_id is None:
                first_id, first_ts = msg_id, ts
            reply_buf.append(body)
        else:  # assistant
            if reply_buf:
                flush()
            prompt_buf.append(clip(body, _CTX_CLIP))
            prompt_buf = prompt_buf[-_CTX_TURNS:]
    flush()
    return records, filtered


@click.command(help=__doc__)
@click.option("--db", "db_path", type=click.Path(exists=True, path_type=Path),
              default=Path.home() / ".hermes/state.db", show_default=True)
@click.option("--out", type=click.Path(path_type=Path),
              default=Path("corpus/data/extracted/hermes.jsonl"))
def main(db_path: Path, out: Path) -> None:
    records, filtered = extract(db_path)
    n = write_jsonl(out, records)
    click.echo(f"hermes: {n} turn records -> {out} ({filtered} pasted/log user msgs filtered)")


if __name__ == "__main__":
    main()
