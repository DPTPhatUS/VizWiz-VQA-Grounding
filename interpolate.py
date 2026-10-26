"""Export inference-only B5/low_crop10 weight interpolations without training."""

import argparse
import copy
import gc
import hashlib
import json
import math
import shutil
from pathlib import Path

import torch

from utils import (
    checkpoint_model_kwargs,
    checkpoint_storage_bytes,
    read_checkpoint,
    save_checkpoint,
)

VERSION = "pyramid-weight-interpolation-v1"
FROZEN_PREFIXES = ("image_encoder.", "text_encoder.")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alpha_tag(alpha):
    return "alpha_" + format(alpha, ".8g").replace(".", "p")


def alpha_grid(values):
    if not values or any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("Alphas must be finite and in [0, 1]")
    grid = sorted({0.0, 1.0, *values})
    if len({alpha_tag(x) for x in grid}) != len(grid):
        raise ValueError("Alpha values collide after filename formatting")
    return grid


def validate_pair(base, candidate, *, base_sha256):
    """Check provenance plus EVERY unchanged tensor, not just architecture labels."""
    a, b = base.get("run_config", {}), candidate.get("run_config", {})
    if (
        base.get("epoch") != 5
        or a.get("objective_version") != "size-weighted-updates-v1"
        or a.get("focus_alpha") != 0.25
    ):
        raise ValueError("Base must be the full B/mild_focus epoch5 checkpoint")
    recipe = b.get("support_recipe", {})
    if (
        candidate.get("epoch") != 10
        or b.get("objective_version") != "crop-support-updates-v1"
        or recipe.get("variant") != "low_crop"
        or recipe.get("crop_probability") != 0.1
        or recipe.get("retention_weight") != 0
        or recipe.get("large_fp_weight") is not None
    ):
        raise ValueError("Candidate must be the full low_crop epoch10 checkpoint")
    origin = b.get("augmentation_origin", {})
    if origin.get("epoch") != 5 or origin.get("sha256") != base_sha256:
        raise ValueError(
            "Candidate does not record this exact B5 file as its origin; use the original full B5"
        )
    if checkpoint_model_kwargs(base) != checkpoint_model_kwargs(candidate):
        raise ValueError("Model architectures differ")
    for key in (
        "experiment_config",
        "training_config",
        "optimization_config",
        "train_annotation_sha256",
        "batch_size",
        "gradient_accumulation_steps",
        "seed",
        "world_size",
        "lr",
        "focus_alpha",
    ):
        if key not in a or a[key] != b.get(key):
            raise ValueError(f"Source protocols differ: {key}")
    for key in (
        "initial_checkpoint",
        "mask_group_sha256",
        "group_counts",
        "sampling",
        "proposal_weights",
        "updates_per_epoch",
    ):
        if key not in a.get("group_protocol", {}) or a["group_protocol"][key] != b.get(
            "group_protocol", {}
        ).get(key):
            raise ValueError(f"Source data/group protocol differs: {key}")
    controls = []
    for saved in (base, candidate):
        control = saved.get("training_controls", {})
        config = control.get("config", {})
        if (
            control.get("version") != 1
            or not config.get("freeze_vision_encoder")
            or not config.get("freeze_text_encoder")
        ):
            raise ValueError(
                "Both sources must record frozen vision/text training controls"
            )
        controls.append(control)
    names = controls[0].get("trainable_names", [])
    if (
        not names
        or names != controls[1].get("trainable_names")
        or len(set(names)) != len(names)
    ):
        raise ValueError("Trainable parameter layouts differ or are missing")
    if any(
        name.startswith(FROZEN_PREFIXES) or name in {"image_mean", "image_std"}
        for name in names
    ):
        raise ValueError("Only post-encoder parameters may be interpolated")
    left, right = base["model_state_dict"], candidate["model_state_dict"]
    if left.keys() != right.keys() or not set(names) <= left.keys():
        raise ValueError("Model state keys differ")
    changed = set(names)
    squared_distance = 0.0
    for name, tensor in left.items():
        other = right[name]
        if (
            not torch.is_tensor(tensor)
            or not torch.is_tensor(other)
            or tensor.shape != other.shape
            or tensor.dtype != other.dtype
        ):
            raise ValueError(f"Tensor schema differs: {name}")
        if name in changed:
            if (
                tensor.dtype != torch.float32
                or not torch.isfinite(tensor).all()
                or not torch.isfinite(other).all()
            ):
                raise ValueError(
                    f"Interpolation requires finite FP32 body parameters: {name}"
                )
            squared_distance += float((tensor.double() - other.double()).square().sum())
        elif not torch.equal(tensor, other):
            raise ValueError(f"Frozen parameter or buffer differs: {name}")
    return names, math.sqrt(squared_distance)


def blend_state(base_state, candidate_body, names, alpha):
    """Keep frozen tensors untouched; endpoints are exact copies, with no arithmetic."""
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Alpha must be in [0, 1]")
    state = dict(base_state)
    for name in names:
        if alpha == 0:
            state[name] = base_state[name]
        elif alpha == 1:
            state[name] = candidate_body[name]
        else:
            state[name] = torch.lerp(base_state[name], candidate_body[name], alpha)
        if not torch.isfinite(state[name]).all():
            raise FloatingPointError(f"Nonfinite interpolated tensor: {name}")
    return state


def storage_plan(base_state, count, free_bytes, reserve_bytes):
    """Reserve output headroom and serialization margin BEFORE exporting any files."""
    payload = checkpoint_storage_bytes(base_state)
    per_file = payload + max(64 * 1024**2, payload // 20)
    required = count * per_file + reserve_bytes
    if free_bytes < required:
        maximum = max(0, (free_bytes - reserve_bytes) // per_file)
        raise OSError(
            f"Grid needs about {required / 1024**3:.2f} GiB including reserve; {free_bytes / 1024**3:.2f} GiB available for this export. At most {maximum} files fit this conservative budget. Reduce --alphas or increase space; no files are deleted."
        )
    return {
        "tensor_bytes_per_file": payload,
        "estimated_bytes_per_file": per_file,
        "file_count": count,
        "reserve_bytes": reserve_bytes,
        "required_bytes": required,
        "free_bytes": free_bytes,
        "max_files_at_current_free_space": max(
            0, (free_bytes - reserve_bytes) // per_file
        ),
    }


def export_grid(
    base_path,
    candidate_path,
    output_dir,
    alphas,
    reserve_gib=2.0,
    storage_budget_gib=14.0,
):
    base_path, candidate_path, out = (
        Path(base_path),
        Path(candidate_path),
        Path(output_dir),
    )
    grid = alpha_grid(alphas)
    if not math.isfinite(reserve_gib) or reserve_gib < 0:
        raise ValueError("Disk reserve must be finite and nonnegative")
    if not math.isfinite(storage_budget_gib) or storage_budget_gib <= 0:
        raise ValueError("Storage budget must be finite and positive")
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            "Export output must be new or empty; existing files are never overwritten"
        )
    print("Reading B5 and low_crop10 on CPU and checking source hashes...", flush=True)
    base, candidate = read_checkpoint(base_path), read_checkpoint(candidate_path)
    base_hash, candidate_hash = file_hash(base_path), file_hash(candidate_path)
    names, distance = validate_pair(base, candidate, base_sha256=base_hash)
    config = copy.deepcopy(base["run_config"]["experiment_config"])
    provenance = {
        "version": VERSION,
        "base": {"path": str(base_path), "sha256": base_hash, "counter": 5},
        "candidate": {
            "path": str(candidate_path),
            "sha256": candidate_hash,
            "counter": 10,
        },
        "parameter_names": names,
        "parameter_count": sum(base["model_state_dict"][n].numel() for n in names),
        "body_distance_l2": distance,
        "frozen_tensors_exactly_equal": True,
        "source_run_configs": {
            "base": base["run_config"],
            "candidate": candidate["run_config"],
        },
        "exporter_sha256": file_hash(__file__),
    }
    # Keep one complete frozen backbone set plus two small trainable bodies on CPU.
    base_state = base["model_state_dict"]
    candidate_body = {name: candidate["model_state_dict"][name] for name in names}
    del base, candidate
    gc.collect()
    out.mkdir(parents=True, exist_ok=True)
    reserve = math.ceil(reserve_gib * 1024**3)
    budget = math.floor(storage_budget_gib * 1024**3)
    physical_free = shutil.disk_usage(out).free
    plan = storage_plan(base_state, len(grid), min(physical_free, budget), reserve)
    plan["configured_budget_bytes"] = budget
    plan["filesystem_free_bytes"] = physical_free
    print(
        f"Exporting {len(grid)} weights-only models (~{plan['estimated_bytes_per_file'] / 1024**3:.2f} GiB/file). "
        f"Effective budget {plan['free_bytes'] / 1024**3:.2f} GiB, reserve {reserve_gib:.2f} GiB; "
        f"conservative capacity {plan['max_files_at_current_free_space']} files. Alphas: {grid}",
        flush=True,
    )
    manifest = {
        "version": VERSION,
        "complete": False,
        "alphas": grid,
        "disk_plan": plan,
        "provenance": provenance,
        "checkpoints": {},
    }
    for alpha in grid:
        tag = alpha_tag(alpha)
        interpolation = {"alpha": alpha, **provenance}
        artifact = {
            "artifact_type": "inference_weights_only",
            "experiment_config": config,
            "model_state_dict": blend_state(base_state, candidate_body, names, alpha),
            "run_config": {
                "objective_version": VERSION,
                "experiment_config": config,
                "weight_interpolation": interpolation,
                "resume_allowed": False,
            },
        }
        # A final disk check protects the reserved result space even if another task used disk.
        written = sum(entry["bytes"] for entry in manifest["checkpoints"].values())
        remaining = min(shutil.disk_usage(out).free, budget - written)
        storage_plan(base_state, 1, remaining, reserve)
        path = out / f"{tag}.pt"
        save_checkpoint(artifact, path)
        manifest["checkpoints"][tag] = {
            "alpha": alpha,
            "filename": path.name,
            "bytes": path.stat().st_size,
            "weights_only": True,
        }
        manifest["complete"] = len(manifest["checkpoints"]) == len(grid)
        (out / "interpolation.json").write_text(json.dumps(manifest, indent=2))
        print(f"Saved {path} ({path.stat().st_size / 1024**3:.2f} GiB)", flush=True)
        del artifact
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5],
        help="Interior coefficients; exact endpoints0/1 are included automatically",
    )
    parser.add_argument("--disk-reserve-gib", type=float, default=2.0)
    parser.add_argument(
        "--storage-budget-gib",
        type=float,
        default=14.0,
        help="Cap total export/result budget even if filesystem free space exceeds the Kaggle quota",
    )
    args = parser.parse_args(argv)
    export_grid(
        args.base_checkpoint,
        args.candidate_checkpoint,
        args.output_dir,
        args.alphas,
        args.disk_reserve_gib,
        args.storage_budget_gib,
    )


if __name__ == "__main__":
    main()
