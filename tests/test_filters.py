from corpus.pipeline.filters import approx_token_count, length_filter, ppl_filter
from corpus.pipeline.records import Record


def rec(i: int, text: str) -> Record:
    return Record(id=f"r{i}", source="test", reply=text)


def test_approx_token_count_uses_tokenizer_when_given():
    assert approx_token_count("a b c", tokenizer=lambda t: [1, 2, 3, 4, 5]) == 5


def test_approx_token_count_estimate():
    assert approx_token_count("one two three") == 3
    # long unbroken text falls back to chars/4
    assert approx_token_count("x" * 400) == 100


def test_length_filter_band():
    short = rec(1, "too short")
    ok = rec(2, " ".join(["word"] * 50))
    long = rec(3, " ".join(["word"] * 5000))
    kept, dropped = length_filter([short, ok, long], min_tokens=30, max_tokens=4096)
    assert [r.id for r in kept] == ["r2"]
    assert {r.id for r in dropped} == {"r1", "r3"}


def test_ppl_filter_drops_top_fraction():
    records = [rec(i, f"text {i}") for i in range(40)]
    scores = {f"text {i}": float(i) for i in range(40)}
    kept, dropped = ppl_filter(records, scorer=lambda t: scores[t], drop_frac=0.05)
    assert len(dropped) == 2  # top 5% of 40
    assert {r.id for r in dropped} == {"r39", "r38"}  # highest ppl = weirdest


def test_ppl_filter_empty():
    assert ppl_filter([], scorer=lambda t: 0.0) == ([], [])
