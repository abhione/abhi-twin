import json

from click.testing import CliRunner

from training.burst.patch_checkpoint import main, patch_config


def test_patch_config_strips_hub_pointer():
    cfg, changed = patch_config(
        {"_name_or_path": "Qwen/Qwen3-TTS-12Hz-1.7B", "hidden_size": 2048},
        "local://voice-v1",
    )
    assert changed
    assert cfg["_name_or_path"] == "local://voice-v1"
    assert cfg["hidden_size"] == 2048  # everything else untouched


def test_patch_config_idempotent():
    for safe in ("local://voice-v1", "/twin/checkpoints/voice-v1"):
        cfg, changed = patch_config({"_name_or_path": safe}, "local://x")
        assert not changed and cfg["_name_or_path"] == safe
    cfg, changed = patch_config({"hidden_size": 1}, "local://x")
    assert not changed


def test_cli_patches_nested_configs(tmp_path):
    ckpt = tmp_path / "voice-v1"
    (ckpt / "checkpoint-500").mkdir(parents=True)
    (ckpt / "config.json").write_text(json.dumps({"_name_or_path": "Qwen/Qwen3-TTS-12Hz-1.7B"}))
    (ckpt / "checkpoint-500" / "config.json").write_text(
        json.dumps({"_name_or_path": "Qwen/Qwen3-TTS-12Hz-1.7B"})
    )
    result = CliRunner().invoke(main, ["--checkpoint-dir", str(ckpt)], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    for cfg in ckpt.rglob("config.json"):
        assert json.loads(cfg.read_text())["_name_or_path"] == "local://voice-v1"


def test_cli_fails_without_config(tmp_path):
    (tmp_path / "empty").mkdir()
    result = CliRunner().invoke(main, ["--checkpoint-dir", str(tmp_path / "empty")])
    assert result.exit_code != 0
