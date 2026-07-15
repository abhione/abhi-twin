"""End-to-end pipeline run on a synthetic fixture (mac-runnable, no GPU):
extract fixtures -> build.py CLI (regex PII smoke mode, ppl skipped) -> verify.py."""

import json

from click.testing import CliRunner

from corpus.build import main as build_main
from corpus.pipeline.records import Record, write_jsonl
from corpus.verify import main as verify_main


def synthetic_records(n: int) -> list[Record]:
    recs = []
    for i in range(n):
        reply = (
            f"Here's my take number {i}: the memory budget is the whole game on one "
            f"Spark, so we keep the 72B NVFP4 at fifty gigabytes and lazy-load video. "
            f"Topic {i} also touches the corpus build — dedup then scrub then filter — "
            f"and mail me at test{i}@example.com if the gate fails on run {i}."
        )
        recs.append(
            Record(
                id=f"syn-{i:04d}",
                source="gmail" if i % 2 else "imessage",
                reply=reply,
                prompt=f"What's your take on topic {i}?",
            )
        )
    return recs


def run_build(tmp_path, n=60):
    in_dir = tmp_path / "extracted"
    out_dir = tmp_path / "out"
    write_jsonl(in_dir / "synthetic.jsonl", synthetic_records(n))
    runner = CliRunner()
    args = [
        "--in-dir", str(in_dir), "--out-dir", str(out_dir),
        "--pii-engine", "regex", "--allow-regex-pii",
        "--skip-ppl", "--skip-reconstruct",
    ]
    result = runner.invoke(build_main, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return out_dir, result


def test_build_end_to_end(tmp_path):
    out_dir, result = run_build(tmp_path)
    stats = json.loads((out_dir / "stats.json").read_text())
    assert stats["pii_engine"] == "regex"
    assert stats["pii_replacements"]["EMAIL_ADDRESS"] == 60  # scrubbed, not deleted
    assert (out_dir / "train.jsonl").exists() and (out_dir / "eval.jsonl").exists()

    train = [json.loads(x) for x in (out_dir / "train.jsonl").open()]
    eval_ = [json.loads(x) for x in (out_dir / "eval.jsonl").open()]
    assert len(train) + len(eval_) == 60
    assert len(eval_) >= 2  # ~5% stratified over two sources
    sample = train[0]
    assert [m["role"] for m in sample["messages"]] == ["user", "assistant"]
    assert "@example.com" not in sample["messages"][1]["content"]


def test_rebuild_reproduces_frozen_holdout(tmp_path):
    out_dir, _ = run_build(tmp_path)
    manifest1 = json.loads((out_dir / "holdout.manifest.json").read_text())
    out_dir, _ = run_build(tmp_path)  # same inputs, same out dir -> same manifest
    manifest2 = json.loads((out_dir / "holdout.manifest.json").read_text())
    assert manifest1["sha256"] == manifest2["sha256"]


def test_verify_gate_counts_and_pii(tmp_path):
    out_dir, _ = run_build(tmp_path)
    runner = CliRunner()
    # 60 pairs < 8000 -> gate must fail on count even with regex allowance
    result = runner.invoke(verify_main, ["--out", str(out_dir), "--allow-regex-pii"])
    assert result.exit_code == 1
    assert "only 60 pairs" in result.output
    # with a lowered bar it passes; without --allow-regex-pii it fails on engine
    result = runner.invoke(verify_main, ["--out", str(out_dir), "--min-pairs", "50",
                                         "--allow-regex-pii"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(verify_main, ["--out", str(out_dir), "--min-pairs", "50"])
    assert result.exit_code == 1
    assert "presidio required" in result.output
