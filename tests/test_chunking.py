import pytest

from serving.rag.chunking import chunk_text


def test_short_text_single_chunk():
    assert chunk_text("one small paragraph") == ["one small paragraph"]


def test_paragraphs_grouped_under_budget():
    paras = [f"paragraph {i} " + "word " * 100 for i in range(6)]  # ~101 words each
    chunks = chunk_text("\n\n".join(paras), max_words=350)
    assert len(chunks) == 2  # 3 paragraphs per chunk
    assert all(len(c.split()) <= 350 for c in chunks)


def test_oversized_paragraph_windows_with_overlap():
    text = " ".join(f"w{i}" for i in range(1000))
    chunks = chunk_text(text, max_words=350, overlap=50)
    assert all(len(c.split()) <= 350 for c in chunks)
    # consecutive windows share the overlap region
    assert chunks[0].split()[-50:] == chunks[1].split()[:50]
    # nothing lost: last word present
    assert "w999" in chunks[-1]


def test_bad_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_text("x", max_words=10, overlap=10)
