"""serving/orchestrator/memory.py: per-session SQLite history. host: mac (unit)."""

from __future__ import annotations

from pathlib import Path

from serving.orchestrator import memory


def _db(tmp_path: Path) -> str:
    return str(tmp_path / "sessions.db")


def test_append_and_history_roundtrip(tmp_path: Path) -> None:
    db = _db(tmp_path)
    memory.append("s1", "user", "hello", db=db)
    memory.append("s1", "assistant", "hi there", db=db)
    assert memory.history("s1", db=db) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_history_windows_to_last_max_turns(tmp_path: Path) -> None:
    db = _db(tmp_path)
    for i in range(20):
        memory.append("s1", "user", f"m{i}", db=db)
    got = memory.history("s1", max_turns=12, db=db)
    assert len(got) == 12
    assert got[0]["content"] == "m8"  # oldest kept
    assert got[-1]["content"] == "m19"  # newest last


def test_sessions_are_isolated(tmp_path: Path) -> None:
    db = _db(tmp_path)
    memory.append("s1", "user", "mclaren", db=db)
    assert memory.history("s2", db=db) == []


def test_clear_drops_only_that_session(tmp_path: Path) -> None:
    db = _db(tmp_path)
    memory.append("s1", "user", "a", db=db)
    memory.append("s1", "assistant", "b", db=db)
    memory.append("s2", "user", "c", db=db)
    assert memory.clear("s1", db=db) == 2
    assert memory.history("s1", db=db) == []
    assert memory.history("s2", db=db) == [{"role": "user", "content": "c"}]


def test_db_parent_dir_is_created(tmp_path: Path) -> None:
    db = str(tmp_path / "deep" / "nested" / "sessions.db")
    memory.append("s1", "user", "x", db=db)
    assert memory.history("s1", db=db) == [{"role": "user", "content": "x"}]


def test_env_var_selects_default_db(tmp_path: Path, monkeypatch) -> None:
    db = _db(tmp_path)
    monkeypatch.setenv("TWIN_SESSIONS_DB", db)
    memory.append("s1", "user", "via-env")
    assert memory.history("s1")[0]["content"] == "via-env"
