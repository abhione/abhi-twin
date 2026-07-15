#!/usr/bin/env python3
"""twin — drive the AbhiTwin build phase-by-phase (CLAUDE.md deliverable shape).

Host legend: commands that need the Spark self-guard and print RUN ON SPARK on
this Mac; corpus extraction runs on the Mac and rsyncs to the Spark.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import click

REPO = Path(__file__).resolve().parent.parent
SPARK_HOST = os.environ.get("TWIN_SPARK_HOST", "spark")


def on_spark() -> bool:
    return platform.system() == "Linux"


def require_spark(what: str) -> None:
    if not on_spark():
        click.echo(f"=== RUN ON SPARK === {what} runs on the Spark "
                   f"(ssh {SPARK_HOST}, then rerun this command).")
        sys.exit(2)


def sh(cmd: list[str], cwd: Path | None = None, **kw) -> int:
    """Run a command. make/script targets pin cwd=REPO; extractors and build
    run in the caller's cwd so data lands under the working directory."""
    click.echo(f"$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, cwd=cwd, **kw).returncode


@click.group(help=__doc__)
def main() -> None:
    pass


# ------------------------------------------------------------------- phases


@main.command()
def phase0() -> None:
    """[spark] Bring-up: /twin dirs, nvrtc symlink, then the verify gate."""
    require_spark("phase0")
    sys.exit(sh(["make", "phase0"], cwd=REPO) or sh(["make", "verify-phase0"], cwd=REPO))


@main.command()
@click.option("--local", "local_", is_flag=True, help="extract + clean on this Mac")
@click.option("--sync", "sync_", is_flag=True, help="rsync cleaned corpus to the Spark")
@click.option("--mbox", type=click.Path(exists=True, path_type=Path), default=None,
              help="Gmail Takeout mbox")
@click.option("--slack-export", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--notes-json", type=click.Path(exists=True, path_type=Path), default=None,
              help="apple-notes-liberator output")
@click.option("--imessage/--no-imessage", "imessage_", default=True)
@click.option("--github/--no-github", "github_", default=True)
@click.option("--pii-engine", type=click.Choice(["presidio", "regex", "auto"]),
              default="presidio")
@click.option("--allow-regex-pii", is_flag=True)
def corpus(local_: bool, sync_: bool, mbox: Path | None, slack_export: Path | None,
           notes_json: Path | None, imessage_: bool, github_: bool,
           pii_engine: str, allow_regex_pii: bool) -> None:
    """[mac] Phase 1: run every extractor whose input exists, then build.py."""
    if sync_:
        out = REPO / "corpus/data/out"
        if not out.exists():
            raise click.ClickException("nothing to sync — run `twin corpus --local` first")
        sys.exit(sh(["rsync", "-avz", f"{out}/", f"{SPARK_HOST}:/twin/corpus/out/"]))
    if not local_:
        raise click.ClickException("pass --local (extract on this Mac) or --sync")

    py = sys.executable
    ran = []
    if mbox:
        sh([py, "-m", "corpus.extractors.gmail_takeout", "--mbox", str(mbox)])
        ran.append("gmail")
    if imessage_ and (Path.home() / "Library/Messages/chat.db").exists():
        sh([py, "-m", "corpus.extractors.imessage"])
        ran.append("imessage")
    if slack_export:
        sh([py, "-m", "corpus.extractors.slack", "--export", str(slack_export)])
        ran.append("slack")
    if notes_json:
        sh([py, "-m", "corpus.extractors.apple_notes", "--input", str(notes_json)])
        ran.append("apple_notes")
    if github_ and shutil.which("gh"):
        sh([py, "-m", "corpus.extractors.github_prs"])
        ran.append("github")
    if not ran:
        raise click.ClickException(
            "no sources available — pass --mbox/--slack-export/--notes-json, or enable "
            "iMessage (Full Disk Access) / gh auth"
        )
    click.echo(f"extracted: {', '.join(ran)}")

    build_cmd = [py, "-m", "corpus.build", "--pii-engine", pii_engine,
                 "--skip-ppl", "--skip-reconstruct"]
    if allow_regex_pii:
        build_cmd.append("--allow-regex-pii")
    click.echo("note: perplexity filter + prompt reconstruction are deferred to the "
               "Spark rebuild (RUN ON SPARK markers in build.py)")
    sys.exit(sh(build_cmd))


@main.command()
@click.argument("component", type=click.Choice(["persona", "voice", "musetalk"]))
@click.option("--provider", type=click.Choice(["brev", "vast"]), default="brev",
              show_default=True)
def train(component: str, provider: str) -> None:
    """[mac] Phases 2/3: launch a cloud-burst fine-tune (Brev first, Vast fallback)."""
    config = {
        "persona": "training/configs/persona-lora.yaml",
        "voice": "training/configs/voice-sft.yaml",
        "musetalk": "training/configs/musetalk-identity.yaml",
    }[component]
    name = {"persona": "persona-v1", "voice": "voice-v1",
            "musetalk": "musetalk-identity-v1"}[component]
    sys.exit(sh(["bash", f"training/burst/launch_{provider}.sh", config, name], cwd=REPO))


@main.command()
@click.option("--profile", type=click.Choice(["core", "voice", "all"]), default="core",
              show_default=True)
@click.option("--down", is_flag=True)
def serve(profile: str, down: bool) -> None:
    """[spark] Phase 4: bring the docker compose stack up (or down)."""
    require_spark("serve")
    if down:
        sys.exit(sh(["make", "serve-down"], cwd=REPO))
    target = {"core": "serve", "voice": "serve-voice", "all": "serve-all"}[profile]
    sys.exit(sh(["make", target], cwd=REPO))


@main.command()
def video() -> None:
    """[spark] v1.5: full stack + the <30 min MuseTalk reproducibility demo."""
    require_spark("video")
    sys.exit(sh(["make", "video-demo"], cwd=REPO))


@main.command()
@click.argument("phase", type=click.Choice(
    ["phase0", "corpus", "persona", "voice", "e2e", "video"]))
def verify(phase: str) -> None:
    """Run a phase's verify gate (spec §12 success criteria)."""
    if phase != "corpus":
        require_spark(f"verify {phase}")
    sys.exit(sh(["make", f"verify-{phase}"], cwd=REPO))


# ------------------------------------------------------------------- talk


def _orchestrator_url() -> str:
    host = "localhost" if on_spark() else SPARK_HOST
    return f"http://{host}:8080"


@main.command()
@click.argument("question", nargs=-1, required=True)
@click.option("--voice", is_flag=True, help="also synthesize the reply")
def ask(question: tuple[str, ...], voice: bool) -> None:
    """Ask the twin one question over the orchestrator API."""
    payload = json.dumps({"text": " ".join(question), "voice": voice}).encode()
    req = urllib.request.Request(f"{_orchestrator_url()}/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
    except OSError as exc:
        raise click.ClickException(
            f"orchestrator unreachable at {_orchestrator_url()} ({exc}) — "
            "is the Spark stack up? (twin serve)"
        ) from None
    click.echo(f"[{data['route']}] {data['reply']}")


@main.command()
def chat() -> None:
    """Interactive chat loop against the twin."""
    click.echo(f"twin chat — {_orchestrator_url()} (ctrl-d to exit)")
    while True:
        try:
            line = click.prompt("you", prompt_suffix="> ")
        except (click.Abort, EOFError):
            break
        ctx = click.Context(ask)
        ctx.invoke(ask, question=(line,), voice=False)


@main.command()
def status() -> None:
    """Ping every serving endpoint and report."""
    host = "localhost" if on_spark() else SPARK_HOST
    endpoints = {
        "llm": f"http://{host}:8000/v1/models",
        "tts": f"http://{host}:8001/healthz",
        "stt": f"http://{host}:8003/healthz",
        "video": f"http://{host}:8004/healthz",
        "orchestrator": f"http://{host}:8080/healthz",
        "qdrant": f"http://{host}:6333/readyz",
    }
    for name, url in endpoints.items():
        try:
            with urllib.request.urlopen(url, timeout=3):
                click.echo(f"UP    {name:14s} {url}")
        except OSError:
            click.echo(f"DOWN  {name:14s} {url}")


if __name__ == "__main__":
    main()
