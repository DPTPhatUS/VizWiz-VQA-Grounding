"""Atomic JSON records and output directories."""
import json
import os
from pathlib import Path


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Refusing to reuse nonempty output: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path
