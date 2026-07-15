from corpus.pipeline.dedup import dedup, jaccard, shingles
from corpus.pipeline.records import Record

BASE = (
    "We should benchmark the vLLM serving stack on the Spark before committing to the "
    "72B model. The KV cache headroom matters more than raw throughput here, and the "
    "NVFP4 quantization keeps us inside the fifty gigabyte budget with room to spare. "
    "If it gets tight we drop to the 32B and free twenty-eight gigabytes immediately."
)


def rec(i: int, text: str) -> Record:
    return Record(id=f"r{i}", source="test", reply=text)


def test_exact_duplicate_dropped_first_wins():
    kept, dropped = dedup([rec(1, BASE), rec(2, BASE)])
    assert [r.id for r in kept] == ["r1"]
    assert [r.id for r in dropped] == ["r2"]


def test_near_duplicate_dropped():
    tweaked = BASE.replace("benchmark", "profile")
    assert jaccard(shingles(BASE), shingles(tweaked)) >= 0.85
    kept, dropped = dedup([rec(1, BASE), rec(2, tweaked)])
    assert len(kept) == 1 and len(dropped) == 1


def test_distinct_texts_kept():
    other = (
        "Completely different topic: the voice corpus needs thirty to sixty minutes of "
        "clean speech, recorded on the condenser mic at forty-eight kilohertz mono, then "
        "denoised with demucs and segmented into five to fifteen second clips by pyannote."
    )
    kept, dropped = dedup([rec(1, BASE), rec(2, other)])
    assert len(kept) == 2 and not dropped


def test_empty_input():
    kept, dropped = dedup([])
    assert kept == [] and dropped == []
