#!/usr/bin/env python
"""Generate the docs gallery page from the app baseline figures.

Scans tests/app_baselines/figures/, groups the PNGs by the app script that
produced them, and writes docs/mkdocs/docs/gallery/ (index.md plus copies of
the figures). Run before `mkdocs build`; the output directory is gitignored.

Figure filenames follow the pattern produced by the apps regression:
    apps_<category>_<script_stem>.py_fig<N>.png
e.g. apps_exercises_1_0_range_equation.py_fig0.png comes from
apps/exercises/1_0_range_equation.py.
"""

import ast
import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "tests" / "app_baselines" / "figures"
GALLERY_DIR = Path(__file__).resolve().parent / "docs" / "gallery"
GITHUB_BLOB_URL = "https://github.com/JohnNehls/rad-lab/blob/main"

# Section order and headings; keys are the subdirectories of apps/
CATEGORIES = {
    "exercises": "Exercises",
    "rdms": "RDMs",
    "sar": "SAR",
    "studies": "Studies",
}

FIGURE_RE = re.compile(r"^apps_(?P<rest>.+)\.py_fig(?P<index>\d+)\.png$")
LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")


def _blank_line_before_lists(text):
    """Insert the blank line Python-Markdown needs before a list.

    MkDocs renders these docstrings with Python-Markdown, which will not let a
    list interrupt a paragraph: a docstring like ``"takeaways:\\n- a\\n- b"``
    renders the bullets as literal text. We insert a blank line before any list
    that directly follows a paragraph line so authors can keep writing plain,
    readable docstrings without hand-tuning them for the Markdown parser. A line
    already inside a list (previous line is a list item or an indented
    continuation) is left alone.
    """
    out = []
    for line in text.splitlines():
        if (
            LIST_MARKER_RE.match(line)
            and out
            and out[-1].strip()  # previous line is a non-blank paragraph line...
            and not out[-1].startswith((" ", "\t"))  # ...not a list continuation...
            and not LIST_MARKER_RE.match(out[-1])  # ...and not itself a list item
        ):
            out.append("")
        out.append(line)
    return "\n".join(out)


def script_docstring(script_path):
    """Return the full module docstring with indentation cleaned, or '' if absent.

    ``ast.get_docstring`` runs the text through ``inspect.cleandoc``, so the
    docstring's own Markdown (paragraphs, lists, etc.) survives intact and is
    rendered by MkDocs on the gallery page.
    """
    docstring = ast.get_docstring(ast.parse(script_path.read_text())) or ""
    return _blank_line_before_lists(docstring)


def collect_figures():
    """Map (category, script_stem) -> figure paths sorted by figure index."""
    groups = {}
    for png in sorted(FIGURES_DIR.glob("*.png")):
        match = FIGURE_RE.match(png.name)
        if match is None:
            raise ValueError(f"unexpected baseline figure name: {png.name}")
        rest = match.group("rest")
        category = next((c for c in CATEGORIES if rest.startswith(c + "_")), None)
        if category is None:
            raise ValueError(f"unknown app category in figure name: {png.name}")
        stem = rest[len(category) + 1 :]
        index = int(match.group("index"))
        groups.setdefault((category, stem), []).append((index, png))
    return {key: [png for _, png in sorted(figures)] for key, figures in groups.items()}


def main():
    groups = collect_figures()

    if GALLERY_DIR.exists():
        shutil.rmtree(GALLERY_DIR)
    figures_out = GALLERY_DIR / "figures"
    figures_out.mkdir(parents=True)

    lines = [
        "# Gallery",
        "",
        "Figures produced by the example apps in"
        f" [apps/]({GITHUB_BLOB_URL}/apps), as captured by the apps"
        " regression baselines. Click any figure to enlarge.",
        "",
    ]

    for category, heading in CATEGORIES.items():
        scripts = sorted(stem for cat, stem in groups if cat == category)
        if not scripts:
            continue
        lines += [f"## {heading}", ""]
        for stem in scripts:
            script_rel = f"apps/{category}/{stem}.py"
            docstring = script_docstring(REPO_ROOT / script_rel)
            lines += [f"### [`{stem}.py`]({GITHUB_BLOB_URL}/{script_rel})", ""]
            if docstring:
                lines += [docstring, ""]
            lines.append('<div class="grid-gallery" markdown>')
            for png in groups[(category, stem)]:
                shutil.copy2(png, figures_out / png.name)
                lines.append(f"![{stem} figure](figures/{png.name})")
            lines += ["</div>", ""]

    (GALLERY_DIR / "index.md").write_text("\n".join(lines) + "\n")
    nfigs = sum(len(figs) for figs in groups.values())
    print(f"gallery: wrote {nfigs} figures from {len(groups)} scripts to {GALLERY_DIR}")


if __name__ == "__main__":
    main()
