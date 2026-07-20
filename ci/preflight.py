#!/usr/bin/env python3
"""AbhiTwin preflight — every known Spark gotcha as an executable check.

Host: mac (`--local-only`) | spark (default = local + spark checks).
Runs at container start and inside `make verify-phase0` / `make test-local`.

Checks (spec §13 + CLAUDE.md hard constraint #1):
  torch            CUDA available + get_device_capability() == (12, 1)
                   (catches silent CPU fallback from a pip-installed torch)
  no-flash-attn    SDPA everywhere; flash_attn must not appear in code or configs
  local-files-only every from_pretrained() in repo code passes local_files_only=True
                   (Milo _name_or_path bug) unless marked `# hub-ok`
  epsilon-clamp    unit test of the HiFi-GAN input guard (white-noise bug)
  configs          training YAMLs parse and encode the recipe gotchas
                   (persona = LoRA rank 32, voice = FULL SFT, sdpa, bf16)
  nvrtc            libnvrtc.so.12.8 symlink exists for CUDA-12.8-pinned packages
  onnxruntime      CUDAExecutionProvider resolves (no silent CPU fallback for
                   insightface face detection in the MuseTalk pipeline);
                   PEND when not installed — the sm_121 build lives in the
                   musetalk image (video profile, v1.5)
  checkpoints      /twin/checkpoints/*/config.json has no HF-hub _name_or_path
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SCAN_EXCLUDE = (".venv", "docs", ".git", "tests")


# --------------------------------------------------------------------- local


def check_epsilon_clamp() -> tuple[bool, str]:
    import math

    from serving.tts.guards import assert_finite, epsilon_clamp, safe_log10

    # ~10 s of silence at ~80 mel frames/s — the worst case for log(0)
    silence = [0.0] * 800
    clamped = epsilon_clamp(silence)
    if min(clamped) < 1e-5:
        return False, "clamp failed to floor zeros"
    mel = [safe_log10(v) for v in clamped]
    if not all(math.isfinite(v) for v in mel):
        return False, "NaN/inf in mel after clamp"
    try:
        assert_finite([1.0, float("nan")])
        return False, "assert_finite missed a NaN"
    except ValueError:
        pass
    return True, "epsilon clamp floors zeros; log-mel finite; NaN detector fires"


def _py_files() -> list[Path]:
    return [
        p
        for p in REPO.rglob("*.py")
        if not any(part in SCAN_EXCLUDE for part in p.relative_to(REPO).parts)
    ]


def check_local_files_only() -> tuple[bool, str]:
    bad: list[str] = []
    for path in _py_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"from_pretrained\s*\(", text):
            window = text[m.start() : m.start() + 400]
            line = text[: m.start()].count("\n") + 1
            if "local_files_only=True" not in window and "# hub-ok" not in window:
                bad.append(f"{path.relative_to(REPO)}:{line}")
    if bad:
        return False, "from_pretrained without local_files_only=True: " + ", ".join(bad)
    return True, "every from_pretrained() uses local_files_only=True (or is marked # hub-ok)"


def check_no_flash_attn() -> tuple[bool, str]:
    bad: list[str] = []
    this = Path(__file__).resolve()
    for path in _py_files() + list(REPO.glob("training/configs/*.yaml")):
        if path.resolve() == this:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            # `flash_attn: sdpa` is LlamaFactory's key NAME for selecting the
            # attention backend — that exact value ENFORCES SDPA, so it's exempt
            if re.search(r"flash_attn:\s*sdpa\b", line):
                continue
            if "flash_attn" in line and "NOT" not in line and "not flash" not in line.lower():
                bad.append(f"{path.relative_to(REPO)}:{i}")
    if bad:
        return False, "flash_attn found (SDPA only on Blackwell): " + ", ".join(bad)
    return True, "no flash_attn anywhere; SDPA it is"


def check_configs() -> tuple[bool, str]:
    import yaml

    cfg_dir = REPO / "training" / "configs"
    errors: list[str] = []

    persona = yaml.safe_load((cfg_dir / "persona-lora.yaml").read_text())
    if persona.get("finetuning_type") != "lora":
        errors.append("persona must be LoRA")
    if persona.get("lora_rank") != 32:
        errors.append("persona lora_rank must be 32")
    voice = yaml.safe_load((cfg_dir / "voice-sft.yaml").read_text())
    if voice.get("finetuning_type") != "full":
        errors.append("voice must be FULL SFT, not LoRA (Milo v8 lesson)")
    for name, cfg in (("persona", persona), ("voice", voice)):
        if cfg.get("attn_implementation", "sdpa") != "sdpa":
            errors.append(f"{name} attn_implementation must be sdpa")
        if cfg.get("bf16") is not True:
            errors.append(f"{name} must train in bf16")
    musetalk = yaml.safe_load((cfg_dir / "musetalk-identity.yaml").read_text())
    if musetalk.get("finetuning_type") != "lora":
        errors.append("musetalk identity must be a LoRA (full SFT only if LoRA plateaus)")
    if errors:
        return False, "; ".join(errors)
    return True, "training configs encode the recipe gotchas"


# --------------------------------------------------------------------- spark


def check_torch() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch not importable — install NGC nightly cu130 aarch64, NEVER pip torch"
    if not torch.cuda.is_available():
        return False, (
            "torch.cuda.is_available() is False — silent CPU fallback; "
            "reinstall from NGC nightly cu130 aarch64"
        )
    cap = torch.cuda.get_device_capability()
    if cap != (12, 1):
        return False, f"get_device_capability() == {cap}, expected (12, 1) (GB10 sm_121)"
    return True, f"CUDA ok, capability (12,1), torch {torch.__version__} cuda {torch.version.cuda}"


def check_nvrtc() -> tuple[bool, str]:
    dst = Path("/usr/local/cuda/lib64/libnvrtc.so.12.8")
    if dst.exists():
        return True, f"{dst} present"
    return False, f"{dst} missing — run phase0/symlink_nvrtc.sh"


def check_onnxruntime() -> tuple[bool | None, str]:
    try:
        import onnxruntime as ort
    except ImportError:
        # The sm_121 source build ships inside docker/musetalk.Dockerfile (v1.5
        # video profile); absence on the host is expected until that phase.
        return None, (
            "onnxruntime not installed — sm_121 build runs in the musetalk image "
            "(video profile, v1.5); required only inside that container"
        )
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        return False, f"CUDAExecutionProvider missing (got {providers}) — CPU fallback would bite"
    return True, f"ORT providers ok: {providers}"


def check_checkpoints() -> tuple[bool, str]:
    # full checkpoints carry config.json/_name_or_path; PEFT adapters carry
    # adapter_config.json/base_model_name_or_path — both must point at /twin
    # or offline loads resolve to the hub (the Milo bug)
    targets = (
        (Path("/twin/checkpoints"), "config.json", "_name_or_path"),
        (Path("/twin/adapters"), "adapter_config.json", "base_model_name_or_path"),
    )
    bad = []
    scanned = 0
    for root, cfg_name, key in targets:
        if not root.exists():
            continue
        for cfg in root.glob(f"*/{cfg_name}"):
            scanned += 1
            data = json.loads(cfg.read_text())
            name = data.get(key, "")
            if name and not (name.startswith("local://") or name.startswith("/twin")):
                bad.append(f"{cfg}: {key}={name!r}")
    if not scanned:
        return True, "no /twin/checkpoints or /twin/adapters yet — nothing to validate"
    if bad:
        return False, "hub-pointing model path (offline-load bug): " + "; ".join(bad)
    return True, "checkpoint configs are offline-safe"


LOCAL_CHECKS = {
    "epsilon-clamp": check_epsilon_clamp,
    "local-files-only": check_local_files_only,
    "no-flash-attn": check_no_flash_attn,
    "configs": check_configs,
}
SPARK_CHECKS = {
    "torch": check_torch,
    "nvrtc": check_nvrtc,
    "onnxruntime": check_onnxruntime,
    "checkpoints": check_checkpoints,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-only", action="store_true", help="skip GPU/Spark checks")
    parser.add_argument("--check", help="run a single named check")
    args = parser.parse_args()

    if args.check:
        fn = {**LOCAL_CHECKS, **SPARK_CHECKS}.get(args.check)
        if fn is None:
            print(f"unknown check {args.check!r}", file=sys.stderr)
            return 2
        checks = {args.check: fn}
    elif args.local_only:
        checks = dict(LOCAL_CHECKS)
    else:
        checks = {**LOCAL_CHECKS, **SPARK_CHECKS}

    failed = 0
    for name, fn in checks.items():
        try:
            ok, msg = fn()
        except Exception as exc:  # a crashed check is a failed check
            ok, msg = False, f"check crashed: {exc}"
        status = "PASS" if ok else ("PEND" if ok is None else "FAIL")
        print(f"{status}  {name:18s} {msg}")
        failed += 1 if ok is False else 0

    if failed and args.local_only is False and not args.check:
        print("\nNote: spark checks are expected to fail on the Mac. "
              "=== RUN ON SPARK === for the full preflight.", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
