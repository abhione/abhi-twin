"""Shared helpers for corpus extractors. Host: mac (starbase/vader)."""

from __future__ import annotations

import hashlib
import re

_WORD = re.compile(r"\S+")
_BASE64_RUN = re.compile(r"[A-Za-z0-9+/=]{200,}")
# harness-injected user turns: "[IMPORTANT: ...]", "[Subagent Context]", "[MAGIC KEYWORD ...]"
_TAG_PREFIX = re.compile(r"^\[[A-Z][^\]]{0,80}\]?")


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


def looks_like_pasted(text: str) -> bool:
    """True when a user turn is probably pasted logs/JSON/tool output, not voice.

    Used by the agent-session extractors (hermes, openclaw) where Abhi's real
    instructions sit between harness-injected context blocks and pasted dumps.
    """
    t = text.strip()
    if not t or len(t) > 2000:
        return True
    if _BASE64_RUN.search(t):
        return True
    if t.startswith(("<system-reminder", "```")) or _TAG_PREFIX.match(t):
        return True
    if t[0] in "{[" and len(t) > 120:  # pasted JSON blob
        return True
    if "```" in t:  # code fences dominating the message
        parts = t.split("```")
        inside = sum(len(p) for p in parts[1::2])
        if inside > len(t) / 2:
            return True
    return False


_SECRET = re.compile(
    r"(?i)\b(password|passwd|passphrase)\b\s*(is|:|=)?\s*\S*[0-9!@#$%^&*_+=]\S*"
    r"|AKIA[0-9A-Z]{16}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY"
    r"|\bBearer\s+[A-Za-z0-9._~+/-]{16,}"
)


def contains_secret(text: str) -> bool:
    """Credential heuristic for agent-session turns — presidio won't catch
    passwords, and a persona LoRA must never memorize one. Replace-not-delete
    doesn't apply: the whole turn is dropped."""
    return bool(_SECRET.search(text))


def clip(text: str, limit: int) -> str:
    """Truncate context text to `limit` chars, marking the cut."""
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …"
