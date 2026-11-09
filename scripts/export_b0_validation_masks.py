"""Export and verify B0 epoch-20 validation masks for seeds 41/42/43.

Uses the unchanged eval.py. References may be original result ZIPs or standalone
validation metrics.json files. Only the explicit validation ZIP member is read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (41, 42, 43)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict:
    def unique(pairs: list) -> dict:
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError(f"Duplicate JSON key: {k}")
            result[k] = v
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def verify_source(root: Path = ROOT) -> dict:
    manifest = load(root / "scripts/b0_source_manifest.json")
    for name, expected in manifest["files"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Original B0 source changed: {name}")
    return manifest


def reference(path: Path, seed: int) -> dict:
    if path.suffix.lower() == ".zip":
        member = f"ocrg_b0_s{seed}/eval_val_ep20/metrics.json"
        with ZipFile(path) as z:
            if z.namelist().count(member) != 1:
                raise ValueError(f"Missing/duplicate validation member: {member}")
            result = json.loads(z.read(member))
    else:
        result = load(path)
    validate_metrics(result, seed)
    return result


def validate_metrics(result: dict, seed: int) -> None:
    import math

    s, values = result["summary"], result["per_sample"]
    if (
        s["split"] != "val"
        or s["epoch"] != 20
        or s["num_samples"] != 1131
        or s["text_mode"] != "question"
        or s["answer_in_input"] is not False
        or s["recipe"]["seed"] != seed
        or len(values) != 1131
    ):
        raise ValueError(f"Incompatible validation metrics for seed {seed}")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError("Invalid IoU values")
    if abs(sum(values.values()) / len(values) - s["mean_iou"]) > 1e-7:
        raise ValueError("Mean IoU inconsistent with per-image values")


def validate_checkpoint(c: dict, seed: int, ref: dict, baseline: str) -> None:
    config = c["run_config"]
    if (
        c["epoch"] != 20
        or config["recipe"] != ref["summary"]["recipe"]
        or c["model_config"] != ref["summary"]["model_config"]
        or config["recipe"]["seed"] != seed
        or config["git"]["commit"] != baseline
        or config["git"]["dirty"] is not False
        or not c["model_state_dict"]
    ):
        raise ValueError(f"Checkpoint metadata differs from original B0 seed {seed}")


def gpu_count(requested: str, available: int) -> int:
    count = min(2, available) if requested == "auto" else int(requested)
    if count < 1 or count > available:
        raise ValueError(f"Requested {requested} GPUs; CUDA exposes {available}")
    return count


def verify_masks(data: Path, output: Path, ref: dict, seed: int) -> dict:
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    sys.path.insert(0, str(ROOT))
    from metrics import iou_per_sample

    records = load(data / "val_grounding.json")
    actual = load(output / "metrics.json")
    validate_metrics(actual, seed)
    if set(records) != set(ref["per_sample"]) or set(records) != set(
        actual["per_sample"]
    ):
        raise ValueError("Validation image IDs differ")
    if (
        actual["summary"]["recipe"] != ref["summary"]["recipe"]
        or actual["summary"]["model_config"] != ref["summary"]["model_config"]
    ):
        raise ValueError("Export recipe/architecture differs from reference")
    stems = {Path(n).stem for n in records}
    if len(stems) != 1131 or any(Path(n).name != n for n in records):
        raise ValueError("Invalid/colliding validation filenames")
    if {p.name for p in (output / "masks").iterdir()} != {s + ".png" for s in stems}:
        raise ValueError("Need exactly 1,131 native prediction PNGs")
    samples, max_saved, max_original = {}, 0.0, 0.0
    for name in sorted(records):
        pred_path = output / "masks" / f"{Path(name).stem}.png"
        gt_path = data / "binary_masks_png/val" / pred_path.name
        with Image.open(pred_path) as im:
            pred = np.array(im.convert("L"))
        with Image.open(gt_path) as im:
            gt = np.array(im.convert("L"))
        with Image.open(data / "val" / name) as im:
            native = ImageOps.exif_transpose(im).size
        if (
            pred.shape != gt.shape
            or pred.shape[::-1] != native
            or not np.isin(pred, [0, 255]).all()
        ):
            raise ValueError(f"Mask dimensions/values invalid: {name}")
        score = iou_per_sample(
            torch.from_numpy(pred > 127)[None, None],
            torch.from_numpy(gt > 127)[None, None],
        ).item()
        saved_delta = abs(score - actual["per_sample"][name])
        original_delta = abs(score - ref["per_sample"][name])
        max_saved, max_original = (
            max(max_saved, saved_delta),
            max(max_original, original_delta),
        )
        if saved_delta > 1e-6 or original_delta > 1e-4:
            raise ValueError(
                f"IoU mismatch for {name}: export={saved_delta}, original={original_delta}; preserve outputs and investigate"
            )
        samples[name] = {
            "native_hw": list(pred.shape),
            "mask_sha256": digest(pred_path),
            "gt_sha256": digest(gt_path),
            "question": records[name]["question"],
            "iou": score,
        }
    if abs(actual["summary"]["mean_iou"] - ref["summary"]["mean_iou"]) > 1e-5:
        raise ValueError("Aggregate validation mIoU mismatch")
    return {
        "count": len(samples),
        "max_export_iou_error": max_saved,
        "max_original_iou_error": max_original,
        "samples": samples,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="JSON seeds -> checkpoint/reference paths",
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--gpus", choices=("auto", "1", "2"), default="auto")
    p.add_argument("--batch-size", type=int, default=2, help="Per GPU; FP32 inference")
    p.add_argument("--num-workers", type=int, default=2, help="Per GPU")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="CPU preflight, including trusted checkpoint metadata",
    )
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="CPU verification of existing outputs, no inference",
    )
    args = p.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        p.error("Positive batch and nonnegative worker count required")
    source = verify_source()
    inputs = load(args.inputs)
    if set(inputs) != {str(s) for s in SEEDS}:
        raise ValueError("Input map must specify exactly seeds 41, 42, 43")
    records = load(args.data_root / "val_grounding.json")
    if len(records) != 1131:
        raise ValueError("Expected 1,131 validation records")
    if (
        any(Path(n).name != n for n in records)
        or len({Path(n).stem for n in records}) != 1131
    ):
        raise ValueError("Unsafe/colliding validation image IDs")
    for name in records:
        for path in (
            args.data_root / "val" / name,
            args.data_root / "binary_masks_png/val" / f"{Path(name).stem}.png",
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
    out = args.output_root.resolve()
    if not args.verify_only and (out.exists() or out.with_suffix(".zip").exists()):
        raise FileExistsError(f"Use a new output root and ZIP path: {out}")
    import torch

    refs, checkpoints = {}, {}
    for seed in SEEDS:
        entry = inputs[str(seed)]
        checkpoint, ref_path = (
            Path(entry["checkpoint"]).resolve(),
            Path(entry["reference"]).resolve(),
        )
        if not checkpoint.is_file() or not ref_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint/reference for {seed}")
        if (
            out == checkpoint.parent
            or out in checkpoint.parents
            or out == args.data_root.resolve()
            or out in args.data_root.resolve().parents
        ):
            raise ValueError("Output root overlaps immutable inputs")
        refs[seed] = reference(ref_path, seed)
        if set(refs[seed]["per_sample"]) != set(records):
            raise ValueError(
                f"Reference IDs disagree with validation annotations: {seed}"
            )
        if not args.verify_only:
            # Only load checkpoints produced and retained by the user; torch pickle is not an untrusted input format.
            c = torch.load(checkpoint, map_location="cpu", weights_only=False)
            validate_checkpoint(c, seed, refs[seed], source["baseline_commit"])
            del c
        checkpoints[seed] = checkpoint
    if args.check_only:
        print("CPU preflight passed for all three seeds; no inference performed")
        return
    if args.verify_only:
        for seed in SEEDS:
            verify_masks(args.data_root, out / f"s{seed}_ep20_val", refs[seed], seed)
        print("CPU mask verification passed for all three seeds")
        return
    count = gpu_count(args.gpus, torch.cuda.device_count())
    out.mkdir(parents=True, exist_ok=False)
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(count)],
        "packages": {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
        },
    }
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    metadata = {
        "baseline": source,
        "workflow_commit": git_commit.stdout.strip()
        if git_commit.returncode == 0
        else None,
        "workflow_sha256": digest(Path(__file__)),
        "environment": environment,
        "batch_per_gpu": args.batch_size,
        "num_workers_per_gpu": args.num_workers,
        "gpu_count": count,
        "inputs": inputs,
        "annotations_sha256": digest(args.data_root / "val_grounding.json"),
        "seeds": {},
    }
    (out / "provenance.json").write_text(json.dumps(metadata, indent=2))
    for seed in SEEDS:
        target = out / f"s{seed}_ep20_val"
        cmd = [sys.executable]
        if count == 2:
            cmd += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
        cmd += [
            str(ROOT / "eval.py"),
            "--checkpoint",
            str(checkpoints[seed]),
            "--split",
            "val",
            "--text-mode",
            "question",
            "--data-root",
            str(args.data_root.resolve()),
            "--output-dir",
            str(target),
            "--save-masks",
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
        ]
        print(
            f"Exporting seed {seed} on {count} GPU(s); log: {out / f's{seed}.log'}",
            flush=True,
        )
        checkpoint_hash = digest(checkpoints[seed])
        with (out / f"s{seed}.log").open("w") as log:
            subprocess.run(
                cmd,
                cwd=ROOT,
                env={**os.environ, "OMP_NUM_THREADS": "1"},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        verification = verify_masks(args.data_root, target, refs[seed], seed)
        (target / "original_val_metrics.json").write_text(
            json.dumps(refs[seed], indent=1)
        )
        (target / "verification.json").write_text(json.dumps(verification, indent=1))
        metadata["seeds"][seed] = {
            "checkpoint_sha256": checkpoint_hash,
            "command": cmd,
            "reference_validation_sha256": hashlib.sha256(
                json.dumps(refs[seed], sort_keys=True).encode()
            ).hexdigest(),
            "count": verification["count"],
            "mean_iou": refs[seed]["summary"]["mean_iou"],
        }
        (out / "provenance.json").write_text(json.dumps(metadata, indent=2))
    (out / "COMPLETE.json").write_text(
        json.dumps({"seeds": list(SEEDS), "predictions_per_seed": 1131})
    )
    # Exclusive creation: a failure leaves inspectable output, never an overwritten experiment.
    with ZipFile(out.with_suffix(".zip"), "x", compression=ZIP_DEFLATED) as z:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(out.parent))
    print(f"Verified export complete: {out.with_suffix('.zip')}")


if __name__ == "__main__":
    main()
