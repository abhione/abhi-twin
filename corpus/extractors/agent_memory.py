#!/usr/bin/env python3
"""Extract hermes-mem observations + session summaries as RAG facts.
Host: mac (starbase).

These are FACTS about Abhi and his projects, not dialog — they go to a
separate stream for Qdrant ingestion (serving/rag), NOT persona pairs.
Output deliberately lives OUTSIDE corpus/data/extracted/ because
corpus/build.py globs that whole directory into persona training pairs.

Optional --cloud pulls the supermemory archive: the container tag comes from
~/.hermes/supermemory.json, the API key from $SUPERMEMORY_API_KEY. OFF by
default; never called in tests; the key is never printed.
READ-ONLY on hermes-mem.db (mode=ro&immutable=1).
"""

from __future__ import annotations

import json
import os
import sqlite3
import urllib.request
from pathlib import Path

import click

from corpus.extractors.common import make_id

SUPERMEMORY_URL = "https://api.supermemory.ai/v3/documents/list"


def _fact(kind: str, fact_id: str, text: str, meta: dict) -> dict:
    return {
        "id": make_id("hermes-mem", kind, fact_id),
        "source": "hermes-mem",
        "stream": "rag",  # marker: RAG ingestion only, never persona pairs
        "kind": kind,
        "text": text,
        "meta": meta,
    }


def extract(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    facts: list[dict] = []

    rows = conn.execute(
        "SELECT id, session_id, timestamp, type, title, narrative, facts, concepts "
        "FROM observations ORDER BY timestamp"
    ).fetchall()
    for oid, sid, ts, otype, title, narrative, facts_json, concepts_json in rows:
        try:
            bullets = [str(f) for f in json.loads(facts_json or "[]")]
        except json.JSONDecodeError:
            bullets = []
        parts = [p for p in [title, narrative, *bullets] if p and p.strip()]
        if not parts:
            continue
        try:
            concepts = [str(c) for c in json.loads(concepts_json or "[]")]
        except json.JSONDecodeError:
            concepts = []
        facts.append(_fact("observation", str(oid), "\n".join(parts),
                           {"session_id": sid, "ts": ts, "type": otype,
                            "concepts": concepts}))

    rows = conn.execute(
        "SELECT id, session_id, timestamp, request, investigated, learned, "
        "completed, next_steps, notes FROM session_summaries ORDER BY timestamp"
    ).fetchall()
    for sid_pk, sid, ts, *fields in rows:
        labels = ("request", "investigated", "learned", "completed", "next_steps", "notes")
        parts = [f"{label}: {val.strip()}" for label, val in zip(labels, fields)
                 if val and val.strip()]
        if not parts:
            continue
        facts.append(_fact("session_summary", str(sid_pk), "\n".join(parts),
                           {"session_id": sid, "ts": ts}))
    conn.close()
    return facts


def pull_supermemory(config_path: Path) -> list[dict]:
    """Pull the cloud archive. Requires $SUPERMEMORY_API_KEY; key never logged."""
    api_key = os.environ.get("SUPERMEMORY_API_KEY", "")
    if not api_key:
        raise click.ClickException("--cloud needs SUPERMEMORY_API_KEY in the environment")
    container = json.loads(config_path.read_text()).get("container")
    if not container:
        raise click.ClickException(f"no container tag in {config_path}")

    facts: list[dict] = []
    page = 1
    while page <= 50:  # bounded — no endless polling
        payload = json.dumps({"containerTags": [container], "limit": 200,
                              "page": page}).encode()
        req = urllib.request.Request(
            SUPERMEMORY_URL, data=payload,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
        except OSError as exc:
            raise click.ClickException(f"supermemory pull failed: {exc}") from None
        docs = data.get("memories") or data.get("documents") or []
        for doc in docs:
            text = doc.get("content") or doc.get("summary") or ""
            if not text.strip():
                continue
            facts.append(_fact("supermemory", str(doc.get("id", len(facts))), text,
                               {"ts": doc.get("createdAt"),
                                "title": doc.get("title")}))
        if len(docs) < 200:
            break
        page += 1
    return facts


def write_facts(path: Path, facts: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for fact in facts:
            fh.write(json.dumps(fact, ensure_ascii=False) + "\n")
    return len(facts)


@click.command(help=__doc__)
@click.option("--db", "db_path", type=click.Path(exists=True, path_type=Path),
              default=Path.home() / ".hermes/hermes-mem.db", show_default=True)
@click.option("--out", type=click.Path(path_type=Path),
              default=Path("corpus/data/rag/rag_facts.jsonl"))
@click.option("--cloud", is_flag=True,
              help="also pull the supermemory cloud archive (needs SUPERMEMORY_API_KEY)")
@click.option("--supermemory-config", type=click.Path(path_type=Path),
              default=Path.home() / ".hermes/supermemory.json", show_default=True)
def main(db_path: Path, out: Path, cloud: bool, supermemory_config: Path) -> None:
    facts = extract(db_path)
    n_cloud = 0
    if cloud:
        cloud_facts = pull_supermemory(supermemory_config)
        n_cloud = len(cloud_facts)
        facts.extend(cloud_facts)
    n = write_facts(out, facts)
    suffix = f" (+{n_cloud} supermemory)" if cloud else ""
    click.echo(f"agent_memory: {n} RAG facts -> {out}{suffix} — RAG stream, not persona pairs")


if __name__ == "__main__":
    main()
