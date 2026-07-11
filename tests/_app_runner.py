"""Run an app script headless and save the figures it produces.

Invoked by the apps regression test as a subprocess so each script runs in a
fresh interpreter (isolated RNG and module state).  Usage::

    RADLAB_FIG_DIR=<dir> python tests/_app_runner.py <script.py>

The script runs via ``runpy`` with ``run_name="__main__"`` so it behaves
exactly as ``python <script.py>`` would (its stdout is unchanged and any
``if __name__ == "__main__"`` guard fires).  Under the Agg backend
``plt.show()`` is a no-op and leaves figures open, so every figure is still
available afterward via ``plt.get_fignums()``.
"""

import os
import runpy
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    script = sys.argv[1]
    fig_dir = os.environ["RADLAB_FIG_DIR"]

    runpy.run_path(script, run_name="__main__")

    # Sorted so figure ordering (and thus baseline filenames) is deterministic.
    for i, num in enumerate(sorted(plt.get_fignums())):
        plt.figure(num).savefig(os.path.join(fig_dir, f"fig{i}.png"))


if __name__ == "__main__":
    main()
