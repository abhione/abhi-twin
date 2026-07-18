"""load_chunks: jsonl = one fact per line with payload; prose falls back to chunking."""

import json

from serving.rag.ingest import load_chunks


def test_jsonl_one_point_per_fact(tmp_path):
    path = tmp_path / "facts.jsonl"
    records = [
        {"id": "a1", "source": "hermes-mem", "kind": "observation", "text": "fact one"},
        {"id": "a2", "source": "hermes-mem", "kind": "summary", "text": "fact two"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    pairs = load_chunks(path)
    assert [text for text, _ in pairs] == ["fact one", "fact two"]
    assert pairs[0][1] == {"id": "a1", "source": "hermes-mem", "kind": "observation"}


def test_jsonl_skips_blank_lines_and_empty_text(tmp_path):
    path = tmp_path / "facts.jsonl"
    path.write_text('{"id": "a", "text": ""}\n\n{"id": "b", "text": "keep"}\n')
    pairs = load_chunks(path)
    assert len(pairs) == 1
    assert pairs[0][0] == "keep"


def test_markdown_uses_prose_chunking(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("para one\n\npara two")
    pairs = load_chunks(path)
    assert pairs
    assert all(extra == {} for _, extra in pairs)
