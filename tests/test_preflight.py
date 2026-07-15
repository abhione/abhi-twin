import subprocess
import sys
from pathlib import Path

from ci.preflight import (
    check_configs,
    check_epsilon_clamp,
    check_local_files_only,
    check_no_flash_attn,
)
from serving.tts.guards import assert_finite, epsilon_clamp, safe_log10

REPO = Path(__file__).resolve().parent.parent


def test_epsilon_clamp_floors_zeros():
    assert epsilon_clamp(0.0) == 1e-5
    assert epsilon_clamp([0.0, 0.5, -1.0]) == [1e-5, 0.5, 1e-5]
    assert epsilon_clamp(2.0) == 2.0


def test_safe_log10_finite_at_zero():
    import math

    assert math.isfinite(safe_log10(0.0))


def test_assert_finite_raises_on_nan():
    import pytest

    with pytest.raises(ValueError, match="non-finite"):
        assert_finite([0.1, float("nan")])
    assert_finite([0.1, 0.2])  # clean input passes


def test_preflight_local_checks_pass():
    for check in (check_epsilon_clamp, check_local_files_only, check_no_flash_attn,
                  check_configs):
        ok, msg = check()
        assert ok, f"{check.__name__}: {msg}"


def test_preflight_cli_local_only_exit_zero():
    result = subprocess.run(
        [sys.executable, str(REPO / "ci" / "preflight.py"), "--local-only"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
