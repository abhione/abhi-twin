"""Shared helpers for corpus extractors. Host: mac (starbase/vader)."""

from __future__ import annotations

import hashlib
import re

_WORD = re.compile(r"\S+")


def make_id(source: str, *parts: str) -> str:
    digest = hashlib.sha1("\x1f".join((source, *parts)).encode()).hexdigest()[:16]
    return f"{source}-{digest}"


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def strip_html(text: str) -> str:
    """Cheap tag stripper for Notes HTML bodies — good enough for corpus text."""
    text = re.sub(r"<br\s*/?>|</p>|</div>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
