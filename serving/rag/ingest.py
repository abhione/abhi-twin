#!/usr/bin/env python3
"""Index the Brain wiki + notes + docs into Qdrant with BGE-M3 embeddings
(spec §7 RAG layer). RUN ON SPARK (needs Qdrant at :6333 and the embedding GPU).

Usage: python -m serving.rag.ingest --source /twin/corpus/brain --collection brain
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import click

from serving.rag.chunking import chunk_text

COLLECTION = "brain"
EMBED_DIM = 1024  # BGE-M3 dense


def load_chunks(path: Path) -> list[tuple[str, dict]]:
    """(text, extra-payload) pairs. `.jsonl` = one pre-chunked fact per line
    (hermes-mem rag_facts stream: {id, source, kind, text}); anything else is
    treated as prose and windowed by chunk_text."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix != ".jsonl":
        return [(chunk, {}) for chunk in chunk_text(raw)]
    pairs: list[tuple[str, dict]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        extra = {k: rec[k] for k in ("id", "source", "kind") if k in rec}
        pairs.append((text, extra))
    return pairs


def _clients():
    from FlagEmbedding import BGEM3FlagModel
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant = QdrantClient(host=host, port=int(os.environ.get("QDRANT_PORT", 6333)))
    model = BGEM3FlagModel("/twin/models/bge-m3", use_fp16=True)
    return qdrant, model, Distance, VectorParams


@click.command(help=__doc__)
@click.option("--source", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--collection", default=COLLECTION, show_default=True)
@click.option("--glob", "pattern", default="**/*.md", show_default=True)
def main(source: Path, collection: str, pattern: str) -> None:
    qdrant, model, Distance, VectorParams = _clients()
    if not qdrant.collection_exists(collection):
        qdrant.create_collection(
            collection,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
    total = 0
    for path in sorted(source.glob(pattern)):
        pairs = load_chunks(path)
        if not pairs:
            continue
        vectors = model.encode([text for text, _ in pairs])["dense_vecs"]
        qdrant.upsert(
            collection,
            points=[
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{path}#{i}")),
                    "vector": vec.tolist(),
                    "payload": {"text": text, "path": str(path), "chunk": i, **extra},
                }
                for i, ((text, extra), vec) in enumerate(zip(pairs, vectors))
            ],
        )
        total += len(pairs)
        click.echo(f"indexed {path} ({len(pairs)} chunks)")
    click.echo(f"OK: {total} chunks in collection {collection!r}")


if __name__ == "__main__":
    main()
