"""Text chunking for the RAG index. Host: any — pure logic, unit-tested locally."""

from __future__ import annotations

import re

_WORD = re.compile(r"\S+")


def chunk_text(text: str, max_words: int = 350, overlap: int = 50) -> list[str]:
    """Split into ~max_words word windows with overlap, preferring paragraph breaks."""
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            chunks.append("\n\n".join(buf))

    count = 0
    for para in paragraphs:
        words = len(_WORD.findall(para))
        if words > max_words:  # oversized paragraph: hard-split on word windows
            flush()
            buf, count = [], 0
            tokens = _WORD.findall(para)
            step = max_words - overlap
            for start in range(0, len(tokens), step):
                window = tokens[start : start + max_words]
                chunks.append(" ".join(window))
                if start + max_words >= len(tokens):
                    break
            continue
        if count + words > max_words:
            flush()
            buf, count = [], 0
        buf.append(para)
        count += words
    flush()
    return chunks
