"""Run records and recipe guards (lessons from the ACIIDS work, PRIOR_WORK.md items 1-3).

- The resolved recipe is written to run_config.json, every history.jsonl line and every checkpoint.
- --expect KEY=VALUE stops a run before training when the resolved recipe differs (a config flag
  that silently did not apply once cost a whole session).
- Resuming refuses a checkpoint whose recipe differs from the current one.
"""

import json
import math
import subprocess
from pathlib import Path

# Keys that define what is being trained. Paths, worker counts, logging cadence and the epoch
# budget may change on resume; these may not.
RECIPE_KEYS = ("dice_weight", "lr", "batch_size", "seed", "world_size", "grad_clip_norm",
               "freeze_vision_encoder", "freeze_text_encoder",
               "decoder", "attention_residual", "attention_residual_init")


def parse_expectations(items):
    """['dice_weight=0.5', 'decoder=fusion'] -> {'dice_weight': 0.5, 'decoder': 'fusion'} (values parsed as JSON when possible)."""
    expected = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"--expect needs KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        expected[key.strip()] = value
    return expected


def _same(actual, wanted):
    if isinstance(actual, bool) or isinstance(wanted, bool):
        return actual is wanted
    if isinstance(actual, (int, float)) and isinstance(wanted, (int, float)):
        return math.isclose(actual, wanted, rel_tol=1e-12, abs_tol=0.0)
    return actual == wanted


def check_expectations(resolved, expected):
    """Raise ValueError listing every expected key that is missing or different."""
    problems = [f"{k}: expected {v!r}, resolved {resolved.get(k, '<missing>')!r}"
                for k, v in expected.items() if k not in resolved or not _same(resolved[k], v)]
    if problems:
        raise ValueError("Recipe differs from --expect; nothing was trained:\n  " + "\n  ".join(problems))


def check_resume_recipe(checkpoint, recipe):
    """Raise ValueError if the checkpoint lacks a recipe or any RECIPE_KEYS value differs."""
    saved = (checkpoint.get("run_config") or {}).get("recipe") if isinstance(checkpoint, dict) else None
    if saved is None:
        raise ValueError("Checkpoint has no recorded recipe (run_config); start a new run instead of resuming")
    problems = [f"{k}: checkpoint {saved.get(k)!r}, now {recipe.get(k)!r}"
                for k in RECIPE_KEYS if not _same(saved.get(k), recipe.get(k))]
    if problems:
        raise ValueError("Resume recipe differs from the checkpoint:\n  " + "\n  ".join(problems))


def git_state(root="."):
    """Commit hash and whether tracked files differ from it; None values outside a git checkout."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
