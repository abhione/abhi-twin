#!/usr/bin/env python3
"""Patch a checkpoint's config.json before pushing to HF: point `_name_or_path`
at a local placeholder (Milo gotcha — a hub-shaped value makes offline
`from_pretrained` do an HF lookup and fail on the Spark). Host: cloud | mac.

Every local load on the Spark then uses local_files_only=True and never sees the
hub. Run this on every checkpoint dir before `hf upload`.
"""

from __future__ import annotations

import json
from pathlib import Path

import click


def patch_config(config: dict, placeholder: str) -> tuple[dict, bool]:
    """Return (patched config, changed?)."""
    current = config.get("_name_or_path")
    if current is None or current.startswith(("local://", "/twin")):
        return config, False
    config["_name_or_path"] = placeholder
    return config, True


@click.command(help=__doc__)
@click.option("--checkpoint-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              required=True)
@click.option("--placeholder", default=None,
              help="value for _name_or_path (default: local://<dir name>)")
def main(checkpoint_dir: Path, placeholder: str | None) -> None:
    placeholder = placeholder or f"local://{checkpoint_dir.name}"
    patched_any = False
    configs = list(checkpoint_dir.rglob("config.json"))
    if not configs:
        raise click.ClickException(f"no config.json under {checkpoint_dir}")
    for cfg_path in configs:
        config = json.loads(cfg_path.read_text())
        config, changed = patch_config(config, placeholder)
        if changed:
            cfg_path.write_text(json.dumps(config, indent=2) + "\n")
            click.echo(f"patched {cfg_path}: _name_or_path -> {placeholder}")
            patched_any = True
        else:
            click.echo(f"ok      {cfg_path}: already offline-safe")
    if patched_any:
        click.echo("checkpoint is now safe for local_files_only=True loads")


if __name__ == "__main__":
    main()
