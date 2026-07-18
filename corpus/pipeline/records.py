"""Corpus record schema + JSONL IO. Host: mac | spark."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable, Iterator


@dataclasses.dataclass
class Record:
    """One turn pair. `reply` is Abhi's side (the training target); `prompt` is the
    counterpart message, or None when it must be reconstructed synthetically."""

    id: str
    source: str  # gmail | imessage | slack | apple_notes | github | hermes | openclaw | brain | manual
    reply: str
    prompt: str | None = None
    meta: dict = dataclasses.field(default_factory=dict)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Record":
        return cls(
            id=d["id"],
            source=d["source"],
            reply=d["reply"],
            prompt=d.get("prompt"),
            meta=d.get("meta", {}),
        )


def write_jsonl(path: Path | str, records: Iterable[Record]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path | str) -> Iterator[Record]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Record.from_json(json.loads(line))
