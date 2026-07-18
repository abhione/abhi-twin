import mailbox
from email.message import EmailMessage
from pathlib import Path

from click.testing import CliRunner

from cli.twin import main

LONG_BODY = " ".join(
    f"sentence {i} about the twin build and the corpus pipeline running locally"
    for i in range(12)
)


def test_help_lists_all_commands():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ("phase0", "corpus", "train", "serve", "video", "verify", "ask",
                "chat", "status"):
        assert cmd in result.output


def test_every_command_has_help():
    for cmd in ("phase0", "corpus", "train", "serve", "video", "verify", "ask", "status"):
        result = CliRunner().invoke(main, [cmd, "--help"])
        assert result.exit_code == 0, cmd


def test_spark_commands_guard_on_mac():
    # this test runs on macOS, where spark-only commands must exit 2 with a marker
    for args in (["phase0"], ["serve"], ["video"], ["verify", "phase0"]):
        result = CliRunner().invoke(main, args)
        assert result.exit_code == 2, args
        assert "RUN ON SPARK" in result.output


def test_corpus_requires_mode():
    result = CliRunner().invoke(main, ["corpus"])
    assert result.exit_code != 0
    assert "--local" in result.output


def test_corpus_local_end_to_end(tmp_path, monkeypatch):
    """twin corpus --local with a real mbox fixture, smoke PII mode, no gh/imessage."""
    monkeypatch.chdir(tmp_path)  # extractor + build defaults are relative paths
    mbox_path = tmp_path / "sent.mbox"
    box = mailbox.mbox(str(mbox_path))
    for i in range(3):
        msg = EmailMessage()
        msg["From"] = "Abhi <abhi@sequoiadigital.io>"
        msg["To"] = "x@example.com"
        msg["Subject"] = f"update {i}"
        msg["Message-ID"] = f"<u{i}@test>"
        msg.set_content(f"Note {i}: {LONG_BODY}")
        box.add(msg)
    box.flush()

    monkeypatch.setenv("TWIN_ME_EMAIL", "abhi@sequoiadigital.io")
    result = CliRunner().invoke(
        main,
        ["corpus", "--local", "--mbox", str(mbox_path), "--no-imessage", "--no-github",
         "--no-hermes", "--no-openclaw", "--no-agent-memory",
         "--pii-engine", "regex", "--allow-regex-pii"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    out = tmp_path / "corpus/data/out"
    assert (out / "train.jsonl").exists()
    assert (out / "holdout.manifest.json").exists()


def test_corpus_local_no_sources_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("cli.twin.shutil.which", lambda _: None)
    # real ~/Library + ~/.hermes + ~/.openclaw must not auto-detect in tests
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = CliRunner().invoke(main, ["corpus", "--local", "--no-imessage", "--no-github",
                                     "--no-hermes", "--no-openclaw", "--no-agent-memory"])
    assert result.exit_code != 0
    assert "no sources available" in result.output
