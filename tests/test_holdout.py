import pytest

from corpus.pipeline.holdout import freeze, manifest, split
from corpus.pipeline.records import Record


def make(n: int, source: str) -> list[Record]:
    return [Record(id=f"{source}-{i}", source=source, reply=f"reply {i}") for i in range(n)]


def test_split_is_stratified_and_five_percent():
    records = make(200, "gmail") + make(80, "imessage") + make(20, "slack")
    train, eval_ = split(records, frac=0.05, seed=13)
    assert len(train) + len(eval_) == 300
    by_source = {s: sum(1 for r in eval_ if r.source == s) for s in ("gmail", "imessage", "slack")}
    assert by_source == {"gmail": 10, "imessage": 4, "slack": 1}


def test_split_deterministic():
    records = make(100, "gmail")
    ids1 = {r.id for r in split(records, seed=13)[1]}
    ids2 = {r.id for r in split(records, seed=13)[1]}
    assert ids1 == ids2


def test_tiny_source_stays_in_train():
    records = make(100, "gmail") + make(3, "brain")
    _, eval_ = split(records)
    assert all(r.source != "brain" for r in eval_)


def test_freeze_writes_then_verifies(tmp_path):
    records = make(100, "gmail")
    _, eval_ = split(records)
    m = manifest(eval_)
    path = freeze(tmp_path, m)
    assert path.exists()
    freeze(tmp_path, m)  # identical manifest re-freezes fine


def test_freeze_refuses_drift(tmp_path):
    records = make(100, "gmail")
    _, eval_ = split(records, seed=13)
    freeze(tmp_path, manifest(eval_))
    _, other = split(records, seed=99)
    with pytest.raises(RuntimeError, match="FROZEN"):
        freeze(tmp_path, manifest(other))
