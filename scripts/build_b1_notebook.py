"""Build the self-contained Kaggle notebook from the README's executable cells."""

import hashlib
import json
import re
from pathlib import Path


def build(root: Path):
    text = (root / "README.md").read_text(encoding="utf-8")
    section = text.split("## Kaggle: B1", 1)[1].split(
        "## Kaggle: B0, seed 42 (first run)", 1
    )[0]
    section = "# B1" + section
    cells = []

    def append(kind, text):
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": text.splitlines(True),
            "id": str(len(cells)) + "-" + hashlib.sha256(text.encode()).hexdigest()[:8],
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    pos = 0
    for match in re.finditer(r"```(?:python|bash)\n(.*?)```", section, re.DOTALL):
        before = section[pos : match.start()].strip()
        if before:
            append("markdown", before)
        append("code", match.group(1))
        pos = match.end()
    if section[pos:].strip():
        append("markdown", section[pos:].strip())
    result = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(result, indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    (root / "notebooks/b1_simple_ocr.ipynb").write_text(build(root), encoding="utf-8")
