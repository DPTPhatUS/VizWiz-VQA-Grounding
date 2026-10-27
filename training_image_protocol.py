"""Explicit image membership and recoverable parent training provenance."""

import json
from pathlib import Path


def image_names(path, available):
    if path is None:
        return list(available)
    names = json.loads(Path(path).read_text())
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(n, str) for n in names)
        or len(set(names)) != len(names)
        or set(names) - set(available)
    ):
        raise ValueError("Invalid training image list")
    return sorted(names)


def parent_provenance(names, previous=None):
    names = sorted(names)
    if (
        previous is not None
        and "training_images" in previous
        and sorted(previous["training_images"]) != names
    ):
        raise ValueError("Resume training image membership differs")
    return {
        "training_images": names,
        "training_lineage_known": previous is None
        or previous.get("training_lineage_known", False),
    }
