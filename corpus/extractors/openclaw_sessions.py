#!/usr/bin/env python3
"""Extract Abhi's user turns from OpenClaw/Enigma session transcripts.
Host: mac (starbase).

Walks ~/.openclaw/agents/*/sessions/*.jsonl (all agents: main, beta, manus,
meg, claude-code, ...), skipping *.trajectory*.jsonl sidecars. Each transcript
line of type "message" holds {message: {role, content, timestamp}} where
content is either a string or a list of blocks; only "text" blocks are used.
Same turn contract as imessage.py: Abhi's user message is `reply`, prior
assistant turns are the prompt. Malformed lines are skipped and counted.
READ-ONLY on the session files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import click

from corpus.extractors.common import clip, contains_secret, looks_like_pasted, make_id
from corpus.pipeline.records import Record, write_jsonl

_CTX_TURNS = 2
_CTX_CLIP = 600

# gateway wrapper around real Telegram/WhatsApp messages — strip, keep the text
_METADATA_BLOCK = re.compile(
    r"^[A-Za-z ]+\(untrusted metadata\):\s*```json\s*.*?```\s*", re.DOTALL
)
# turns injected by the OpenClaw harness itself, not typed by Abhi
_HARNESS_PREFIXES = (
    "Pre-compaction memory flush",
    "HEARTBEAT",
    "Health check. Reply with exactly",
    "System:",
    "System (untrusted)",
    "Queued messages",
)


def _clean_user_text(text: str) -> str | None:
    """Unwrap gateway metadata; None when the turn is harness-injected/pasted."""
    t = text.strip()
    while True:
        t2 = _METADATA_BLOCK.sub("", t, count=1).strip()
        if t2 == t:
            break
        t = t2
    if not t or t.startswith(_HARNESS_PREFIXES) or looks_like_pasted(t):
        return None
    if contains_secret(t):
        return None
    return t


def _block_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [b.get("text", "") for b in content
                 if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(t for t in texts if t).strip()
    return ""


def _session_files(agents_dir: Path) -> list[Path]:
    return sorted(
        p for p in agents_dir.glob("*/sessions/*.jsonl")
        if ".trajectory" not in p.name
    )


def extract(agents_dir: Path) -> tuple[list[Record], int, int]:
    """Return (records, filtered_user_msgs, malformed_lines)."""
    records: list[Record] = []
    filtered = malformed = 0

    for path in _session_files(agents_dir):
        agent = path.parent.parent.name
        session_id = path.stem
        prompt_buf: list[str] = []
        reply_buf: list[str] = []
        first_ts: str | None = None

        def flush() -> None:
            nonlocal prompt_buf, reply_buf, first_ts
            if reply_buf:
                records.append(
                    Record(
                        id=make_id("openclaw", agent, session_id, str(len(records))),
                        source="openclaw",
                        reply="\n".join(reply_buf),
                        prompt="\n".join(prompt_buf) if prompt_buf else None,
                        meta={"agent": agent, "session_id": session_id, "ts": first_ts},
                    )
                )
            prompt_buf, reply_buf, first_ts = [], [], None

        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            malformed += 1
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(entry, dict) or entry.get("type") != "message":
                continue
            msg = entry.get("message")
            if not isinstance(msg, dict):
                malformed += 1
                continue
            role = msg.get("role")
            body = _block_text(msg.get("content"))
            if role not in ("user", "assistant") or not body:
                continue
            if role == "user":
                cleaned = _clean_user_text(body)
                if cleaned is None:
                    filtered += 1
                    flush()
                    prompt_buf = []
                    continue
                if first_ts is None:
                    first_ts = msg.get("timestamp") or entry.get("timestamp")
                reply_buf.append(cleaned)
            else:
                if reply_buf:
                    flush()
                prompt_buf.append(clip(body, _CTX_CLIP))
                prompt_buf = prompt_buf[-_CTX_TURNS:]
        flush()
    return records, filtered, malformed


@click.command(help=__doc__)
@click.option("--agents-dir", type=click.Path(exists=True, path_type=Path),
              default=Path.home() / ".openclaw/agents", show_default=True)
@click.option("--out", type=click.Path(path_type=Path),
              default=Path("corpus/data/extracted/openclaw.jsonl"))
def main(agents_dir: Path, out: Path) -> None:
    records, filtered, malformed = extract(agents_dir)
    n = write_jsonl(out, records)
    click.echo(f"openclaw: {n} turn records -> {out} "
               f"({filtered} pasted/log user msgs filtered, {malformed} malformed lines)")


if __name__ == "__main__":
    main()
