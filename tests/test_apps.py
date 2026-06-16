"""Regression-run every script under apps/.

Each script is executed headless (``MPLBACKEND=Agg`` makes ``plt.show()`` a
no-op) through ``tests/_app_runner.py``, which also saves every figure the
script produces.  A script passes when it:

  * exits cleanly,
  * (where a stdout baseline exists in ``tests/app_baselines/``) reproduces
    that stdout, and
  * (where figure baselines exist in ``tests/app_baselines/figures/``)
    produces the same number of figures, each matching its baseline image
    within an RMS tolerance.

These take a few minutes, so they are deselected by default (see
``[tool.pytest.ini_options]`` in pyproject.toml).  To run them:

    pytest -m apps

To refresh the stdout and image baselines after an intentional change:

    RADLAB_UPDATE_APP_BASELINES=1 pytest -m apps

Set ``RADLAB_SKIP_FIGURE_COMPARE=1`` to keep the script-execution, stdout, and
figure-count checks while skipping the pixel-level image comparison.  CI uses
this because RMS image diffs are sensitive to the rendering environment
(matplotlib bundles a fixed freetype, but ``usetex`` scripts render text via the
host's LaTeX/dvipng, which differs from the machine that generated baselines).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from matplotlib.testing.compare import compare_images

REPO_ROOT = Path(__file__).resolve().parents[1]
APPS_DIR = REPO_ROOT / "apps"
RUNNER = Path(__file__).parent / "_app_runner.py"
BASELINE_DIR = Path(__file__).parent / "app_baselines"
FIGURE_BASELINE_DIR = BASELINE_DIR / "figures"

# RMS pixel tolerance for figure comparison.  Generous enough to absorb
# antialiasing/font jitter, tight enough to catch real plot changes.  A
# matplotlib/freetype upgrade may require refreshing the image baselines.
FIGURE_TOL = 10.0

SCRIPTS = sorted(APPS_DIR.rglob("*.py"))


def _script_id(script: Path) -> str:
    return script.relative_to(REPO_ROOT).as_posix()


def _fig_index(path: Path) -> int:
    """Trailing integer of a ``fig<N>`` / ``..._fig<N>`` filename stem."""
    return int(path.stem.split("fig")[-1])


@pytest.mark.apps
@pytest.mark.parametrize("script", SCRIPTS, ids=_script_id)
def test_app_script(script: Path, tmp_path: Path) -> None:
    rel = _script_id(script)
    if script.name.endswith("_no_test.py"):
        pytest.skip("script is marked incomplete (_no_test suffix)")

    key = rel.replace("/", "_")

    result = subprocess.run(
        [sys.executable, str(RUNNER), str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        env=os.environ | {"MPLBACKEND": "Agg", "RADLAB_FIG_DIR": str(tmp_path)},
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"{rel} exited {result.returncode}\n{result.stderr[-2000:]}"

    produced = sorted(tmp_path.glob("fig*.png"), key=_fig_index)
    stdout_baseline = BASELINE_DIR / (key + ".out")

    if os.environ.get("RADLAB_UPDATE_APP_BASELINES"):
        BASELINE_DIR.mkdir(exist_ok=True)
        if result.stdout:
            stdout_baseline.write_text(result.stdout)
        FIGURE_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        for i, fig in enumerate(produced):
            shutil.copyfile(fig, FIGURE_BASELINE_DIR / f"{key}_fig{i}.png")
        # Drop stale baselines if the script now produces fewer figures.
        for stale in FIGURE_BASELINE_DIR.glob(f"{key}_fig*.png"):
            if _fig_index(stale) >= len(produced):
                stale.unlink()
        return

    if stdout_baseline.exists():
        assert result.stdout == stdout_baseline.read_text(), (
            f"stdout of {rel} differs from its baseline; if the change is "
            "intentional, refresh with: RADLAB_UPDATE_APP_BASELINES=1 pytest -m apps"
        )

    fig_baselines = sorted(FIGURE_BASELINE_DIR.glob(f"{key}_fig*.png"), key=_fig_index)
    if fig_baselines:
        assert len(produced) == len(fig_baselines), (
            f"{rel} produced {len(produced)} figure(s) but has "
            f"{len(fig_baselines)} baseline(s); if intentional, refresh with: "
            "RADLAB_UPDATE_APP_BASELINES=1 pytest -m apps"
        )
        # Pixel comparison is sensitive to the rendering environment (notably
        # LaTeX/dvipng for usetex scripts), so CI sets RADLAB_SKIP_FIGURE_COMPARE
        # to keep the robust count check while skipping the RMS image diff.
        if os.environ.get("RADLAB_SKIP_FIGURE_COMPARE"):
            return
        for expected, actual in zip(fig_baselines, produced):
            msg = compare_images(str(expected), str(actual), tol=FIGURE_TOL)
            assert msg is None, (
                f"figure {actual.name} of {rel} differs from {expected.name}:\n{msg}\n"
                "if intentional, refresh with: RADLAB_UPDATE_APP_BASELINES=1 pytest -m apps"
            )
