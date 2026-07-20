"""Soul file loader: mtime-cached reads of SOUL.md / FACTS.md / MEMORY.md.

host: spark (inside the orchestrator container). Runtime copies live in
TWIN_SOUL_DIR (default /twin/soul, a mounted volume, hot-editable); the repo
copies next to this file are the bundled fallback, so a missing volume never
breaks prompt assembly. Edits apply on the next message, no restart.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

REPO_SOUL_DIR = Path(__file__).resolve().parent
MEMORY_TAIL_CHARS = 4000

_cache: dict[str, tuple[float, str]] = {}


def _runtime_dir() -> Path:
    return Path(os.environ.get("TWIN_SOUL_DIR", "/twin/soul"))


def _resolve(name: str) -> Path | None:
    for base in (_runtime_dir(), REPO_SOUL_DIR):
        p = base / name
        if p.is_file():
            return p
    return None


def load(name: str) -> str:
    """Return the file's text, re-reading only when its mtime changes."""
    path = _resolve(name)
    if path is None:
        return ""
    mtime = path.stat().st_mtime
    key = str(path)
    hit = _cache.get(key)
    if hit is None or hit[0] != mtime:
        _cache[key] = (mtime, path.read_text(encoding="utf-8"))
    return _cache[key][1]


def memory_tail(max_chars: int = MEMORY_TAIL_CHARS) -> str:
    """Last max_chars of MEMORY.md so a fat memory file can't blow the context."""
    text = load("MEMORY.md")
    return text[-max_chars:] if len(text) > max_chars else text


def identity_line() -> str:
    """First body line of SOUL.md (the /whoami one-liner)."""
    for line in load("SOUL.md").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return "AbhiTwin, Abhi Bhattacharya's digital twin."


def system_prompt() -> str:
    """SOUL + FACTS + MEMORY tail, in that order."""
    parts = [load("SOUL.md"), load("FACTS.md")]
    mem = memory_tail()
    if mem:
        parts.append(mem)
    return "\n\n".join(p.strip() for p in parts if p.strip())


def remember(text: str) -> str:
    """Append a timestamped entry to the runtime MEMORY.md (the /remember path).

    Seeds the runtime file from the bundled copy on first write so the header
    survives. Returns the appended line.
    """
    runtime = _runtime_dir() / "MEMORY.md"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    if not runtime.exists():
        runtime.write_text(load("MEMORY.md"), encoding="utf-8")
    stamp = time.strftime("%Y-%m-%d %H:%M")
    line = f"- [{stamp}] {text.strip()}"
    with runtime.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return line
