"""Regression-run every script under apps/.

Each script is executed headless (``MPLBACKEND=Agg`` makes ``plt.show()`` a
no-op).  A script passes when it exits cleanly and, where a stdout baseline
exists in ``tests/app_baselines/``, its stdout matches the baseline.

These take a few minutes, so they are deselected by default (see
``[tool.pytest.ini_options]`` in pyproject.toml).  To run them:

    pytest -m apps

To refresh the stdout baselines after an intentional output change:

    RADLAB_UPDATE_APP_BASELINES=1 pytest -m apps
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APPS_DIR = REPO_ROOT / "apps"
BASELINE_DIR = Path(__file__).parent / "app_baselines"

SCRIPTS = sorted(APPS_DIR.rglob("*.py"))


def _script_id(script: Path) -> str:
    return script.relative_to(REPO_ROOT).as_posix()


@pytest.mark.apps
@pytest.mark.parametrize("script", SCRIPTS, ids=_script_id)
def test_app_script(script: Path) -> None:
    rel = _script_id(script)
    if script.name.endswith("_no_test.py"):
        pytest.skip("script is marked incomplete (_no_test suffix)")

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        env=os.environ | {"MPLBACKEND": "Agg"},
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"{rel} exited {result.returncode}\n{result.stderr[-2000:]}"

    baseline = BASELINE_DIR / (rel.replace("/", "_") + ".out")

    if os.environ.get("RADLAB_UPDATE_APP_BASELINES"):
        BASELINE_DIR.mkdir(exist_ok=True)
        if result.stdout:
            baseline.write_text(result.stdout)
        return

    if baseline.exists():
        assert result.stdout == baseline.read_text(), (
            f"stdout of {rel} differs from its baseline; if the change is "
            "intentional, refresh with: RADLAB_UPDATE_APP_BASELINES=1 pytest -m apps"
        )
