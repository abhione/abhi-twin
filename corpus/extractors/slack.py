#!/usr/bin/env python3
"""Extract Slack DMs + threads you started from a workspace export (spec §6.1:
1-3k msgs). Host: mac.

Export layout: <export>/dms.json (or mpims.json) lists DM channel ids; each
channel is a directory of day JSON files with message arrays. Per spec, only
DMs and threads whose root message is yours are included.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from corpus.extractors.common import make_id
from corpus.pipeline.records import Record, write_jsonl


def _dm_channel_names(export: Path) -> set[str]:
    names: set[str] = set()
    for listing in ("dms.json", "mpims.json"):
        path = export / listing
        if path.exists():
            for ch in json.loads(path.read_text()):
                names.add(ch.get("id") or ch.get("name", ""))
    return names


def _channel_messages(chan_dir: Path) -> list[dict]:
    msgs: list[dict] = []
    for day in sorted(chan_dir.glob("*.json")):
        msgs.extend(json.loads(day.read_text()))
    msgs.sort(key=lambda m: float(m.get("ts", 0)))
    return msgs


def _my_thread_roots(msgs: list[dict], me: str) -> set[str]:
    return {
        m["ts"]
        for m in msgs
        if m.get("user") == me and m.get("thread_ts") in (None, m.get("ts"))
        and m.get("reply_count")
    }


def extract(export: Path, me: str) -> list[Record]:
    dm_names = _dm_channel_names(export)
    records: list[Record] = []
    for chan_dir in sorted(p for p in export.iterdir() if p.is_dir()):
        msgs = _channel_messages(chan_dir)
        if not msgs:
            continue
        is_dm = chan_dir.name in dm_names
        my_roots = _my_thread_roots(msgs, me)
        prev_other: dict | None = None
        for m in msgs:
            text = (m.get("text") or "").strip()
            if not text or m.get("subtype"):
                prev_other = None if m.get("subtype") else prev_other
                continue
            in_my_thread = m.get("thread_ts") in my_roots
            if m.get("user") == me:
                if is_dm or in_my_thread:
                    records.append(
                        Record(
                            id=make_id("slack", chan_dir.name, m["ts"]),
                            source="slack",
                            reply=text,
                            prompt=(prev_other or {}).get("text"),
                            meta={"channel": chan_dir.name, "dm": is_dm},
                        )
                    )
                prev_other = None
            else:
                prev_other = m
    return records


@click.command(help=__doc__)
@click.option("--export", type=click.Path(exists=True, file_okay=False, path_type=Path),
              required=True, help="unzipped Slack export directory")
@click.option("--me", envvar="TWIN_SLACK_ME", required=True, help="your Slack user id (U…)")
@click.option("--out", type=click.Path(path_type=Path),
              default=Path("corpus/data/extracted/slack.jsonl"))
def main(export: Path, me: str, out: Path) -> None:
    records = extract(export, me)
    n = write_jsonl(out, records)
    click.echo(f"slack: {n} records -> {out}")


if __name__ == "__main__":
    main()
