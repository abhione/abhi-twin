#!/usr/bin/env python3
"""Extract PR descriptions + commit message bodies via the `gh` CLI (spec §6.1:
500-2000 records — prose only, never code). Host: mac (needs `gh auth login`).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

import click

from corpus.extractors.common import make_id, word_count
from corpus.pipeline.records import Record, write_jsonl

MIN_WORDS = 15

Runner = Callable[[list[str]], str]


def _gh(cmd: list[str]) -> str:
    return subprocess.run(
        ["gh", *cmd], check=True, capture_output=True, text=True
    ).stdout


def _commit_body(message: str) -> str:
    """Commit body only — drop the subject line (spec: bodies, not one-liners)."""
    parts = message.split("\n", 1)
    return parts[1].strip() if len(parts) > 1 else ""


def extract(user: str, limit: int = 1000, run: Runner = _gh) -> list[Record]:
    records: list[Record] = []

    prs = json.loads(
        run(["search", "prs", "--author", user, "--limit", str(limit),
             "--json", "title,body,url"])
    )
    for pr in prs:
        body = (pr.get("body") or "").strip()
        title = (pr.get("title") or "").strip()
        text = f"{title}\n\n{body}".strip()
        if word_count(body) < MIN_WORDS:
            continue
        records.append(
            Record(
                id=make_id("github", pr.get("url", title)),
                source="github",
                reply=text,
                prompt=None,
                meta={"kind": "pr", "url": pr.get("url", "")},
            )
        )

    commits = json.loads(
        run(["search", "commits", "--author", user, "--limit", str(limit),
             "--json", "sha,commit"])
    )
    for c in commits:
        body = _commit_body(c.get("commit", {}).get("message", ""))
        if word_count(body) < MIN_WORDS:
            continue
        records.append(
            Record(
                id=make_id("github", c.get("sha", body[:40])),
                source="github",
                reply=body,
                prompt=None,
                meta={"kind": "commit", "sha": c.get("sha", "")},
            )
        )
    return records


def _authed_user() -> str | None:
    try:
        return _gh(["api", "user", "--jq", ".login"]).strip() or None
    except (subprocess.CalledProcessError, OSError):
        return None


@click.command(help=__doc__)
@click.option("--user", envvar="TWIN_GITHUB_USER", default=None,
              help="GitHub login [default: the gh-authed user]")
@click.option("--limit", default=1000, show_default=True)
@click.option("--out", type=click.Path(path_type=Path),
              default=Path("corpus/data/extracted/github.jsonl"))
def main(user: str | None, limit: int, out: Path) -> None:
    user = user or _authed_user()
    if not user:
        raise click.ClickException(
            "no GitHub user — pass --user, set TWIN_GITHUB_USER, or `gh auth login`"
        )
    records = extract(user, limit)
    n = write_jsonl(out, records)
    click.echo(f"github: {n} records -> {out}")


if __name__ == "__main__":
    main()
