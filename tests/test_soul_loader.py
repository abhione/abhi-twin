"""serving/soul/loader.py: mtime cache, runtime-dir precedence, memory cap. host: mac (unit)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from serving.soul import loader


@pytest.fixture
def soul_dir(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("TWIN_SOUL_DIR", str(tmp_path))
    loader._cache.clear()
    return tmp_path


def test_runtime_dir_wins_over_repo_copy(soul_dir: Path) -> None:
    (soul_dir / "SOUL.md").write_text("# SOUL\n\nruntime identity line\n")
    assert loader.load("SOUL.md") == "# SOUL\n\nruntime identity line\n"


def test_repo_fallback_when_runtime_missing(soul_dir: Path) -> None:
    # nothing in the runtime dir: bundled repo copies must serve
    assert "AbhiTwin" in loader.load("SOUL.md")
    assert "VSP" in loader.load("FACTS.md")


def test_mtime_reload_picks_up_edits(soul_dir: Path) -> None:
    p = soul_dir / "MEMORY.md"
    p.write_text("v1")
    assert loader.load("MEMORY.md") == "v1"
    p.write_text("v2")
    os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 5))
    assert loader.load("MEMORY.md") == "v2"


def test_unchanged_mtime_serves_cache(soul_dir: Path) -> None:
    p = soul_dir / "FACTS.md"
    p.write_text("cached")
    assert loader.load("FACTS.md") == "cached"
    stat = p.stat()
    p.write_text("changed-on-disk")
    os.utime(p, (stat.st_atime, stat.st_mtime))  # same mtime -> cache must win
    assert loader.load("FACTS.md") == "cached"


def test_memory_tail_caps_injection(soul_dir: Path) -> None:
    (soul_dir / "MEMORY.md").write_text("x" * 10_000 + "END")
    tail = loader.memory_tail(max_chars=4000)
    assert len(tail) == 4000
    assert tail.endswith("END")


def test_identity_line_skips_headers(soul_dir: Path) -> None:
    (soul_dir / "SOUL.md").write_text("# SOUL\n\n## Identity\n\nYou are AbhiTwin.\n")
    assert loader.identity_line() == "You are AbhiTwin."


def test_system_prompt_order_soul_facts_memory(soul_dir: Path) -> None:
    (soul_dir / "SOUL.md").write_text("SOUL-PART")
    (soul_dir / "FACTS.md").write_text("FACTS-PART")
    (soul_dir / "MEMORY.md").write_text("MEMORY-PART")
    prompt = loader.system_prompt()
    assert prompt.index("SOUL-PART") < prompt.index("FACTS-PART") < prompt.index("MEMORY-PART")


def test_remember_appends_timestamped_line(soul_dir: Path) -> None:
    line = loader.remember("Telegram user prefers short replies")
    text = (soul_dir / "MEMORY.md").read_text()
    assert text.rstrip().endswith(line)
    assert line.startswith("- [") and "Telegram user prefers short replies" in line
    # first write seeds from the bundled copy so the header survives
    assert text.startswith("# MEMORY")
