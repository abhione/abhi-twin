"""Top-k retrieval from Qdrant for the orchestrator's RETRIEVER node.
RUN ON SPARK (embedding model + Qdrant); import is cheap, model loads lazily."""

from __future__ import annotations

import os

_model = None
_qdrant = None


def _lazy():
    global _model, _qdrant
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel
        from qdrant_client import QdrantClient

        _qdrant = QdrantClient(
            host=os.environ.get("QDRANT_HOST", "localhost"),
            port=int(os.environ.get("QDRANT_PORT", 6333)),
        )
        _model = BGEM3FlagModel("/twin/models/bge-m3", use_fp16=True)
    return _qdrant, _model


def retrieve(query: str, top_k: int = 8, collection: str = "brain") -> list[str]:
    """Return the top_k chunk texts; [] when the index is missing (degrade, don't die)."""
    try:
        qdrant, model = _lazy()
        if not qdrant.collection_exists(collection):
            return []
        vec = model.encode([query])["dense_vecs"][0].tolist()
        hits = qdrant.search(collection, query_vector=vec, limit=top_k)
        return [h.payload["text"] for h in hits]
    except Exception:
        # RAG is an enhancer, not a dependency: a cold index must not kill chat
        return []
