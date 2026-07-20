"""Per-session conversation history in SQLite. host: spark (orchestrator container).

DB lives at TWIN_SESSIONS_DB (default /twin/soul/sessions.db) on the mounted
volume, WAL mode so the host can read while the container writes. One
connection per call: traffic is single-user chat, correctness beats pooling.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

DEFAULT_DB = "/twin/soul/sessions.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL,
    role    TEXT NOT NULL,
    content TEXT NOT NULL,
    ts      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session, id);
"""


def _db_path() -> str:
    return os.environ.get("TWIN_SESSIONS_DB", DEFAULT_DB)


def _connect(db: str | None = None) -> sqlite3.Connection:
    path = db or _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def append(session: str, role: str, content: str, db: str | None = None) -> None:
    with _connect(db) as conn:
        conn.execute(
            "INSERT INTO messages (session, role, content, ts) VALUES (?, ?, ?, ?)",
            (session, role, content, time.time()),
        )


def history(session: str, max_turns: int = 12, db: str | None = None) -> list[dict[str, str]]:
    """Last max_turns messages for the session, oldest first, chat-message shaped."""
    with _connect(db) as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session = ? ORDER BY id DESC LIMIT ?",
            (session, max_turns),
        ).fetchall()
    return [{"role": role, "content": content} for role, content in reversed(rows)]


def clear(session: str, db: str | None = None) -> int:
    """Delete the session's history; returns rows removed (the /new path)."""
    with _connect(db) as conn:
        cur = conn.execute("DELETE FROM messages WHERE session = ?", (session,))
        return cur.rowcount
