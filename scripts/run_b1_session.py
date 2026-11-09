"""Single-session B0 continuation + OCR48/OCR96, then full validation diagnostics.

This command launches the experiment. --preflight-only performs CPU metadata
checks only; there is no mandatory runtime probe and no cross-session split.
"""

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b1_run import EPOCHS, PARENT_SHA, SNAPSHOTS, VIEWS, load_parent
from models.simple_ocr import ARMS
from scripts.export_b0_validation_masks import digest, reference, verify_source


def plan_commands(root, data, parent, features, gpus=2, workers=2):
    """All train jobs precede every evaluation job, by construction."""
    prefix = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpus}",
        str(ROOT / "b1_run.py"),
    ]
    common = [
        "--data",
        str(data),
        "--parent",
        str(parent),
        "--features",
        str(features),
        "--workers",
        str(workers),
    ]
    train = [
        prefix + ["train", *common, "--arm", a, "--output", str(root / "training" / a)]
        for a in ARMS
    ]
    evaluations = [
        prefix
        + [
            "evaluate",
            *common,
            "--parent-eval",
            "--output",
            str(root / "analysis/evaluations/original_b0"),
        ]
    ]
    for arm in ARMS:
        for epoch in SNAPSHOTS:
            for view in ("clean",) if arm == "b0_continued" else VIEWS:
                evaluations.append(
                    prefix
                    + [
                        "evaluate",
                        *common,
                        "--checkpoint",
                        str(root / "training" / arm / f"checkpoint_epoch{epoch}.pt"),
                        "--view",
                        view,
                        "--output",
                        str(root / "analysis/evaluations" / f"{arm}_ep{epoch}_{view}"),
                    ]
                )
    return train, evaluations


def training_barrier(root):
    for arm in ARMS:
        folder = root / "training" / arm
        if json.loads((folder / "TRAINING_COMPLETE.json").read_text()) != {
            "epochs": EPOCHS,
            "snapshots": list(SNAPSHOTS),
        }:
            raise ValueError(
                "All three arms must complete ten epochs before diagnostics"
            )
        h = [json.loads(l) for l in (folder / "history.jsonl").read_text().splitlines()]
        if [r["epoch"] for r in h] != list(range(1, 11)):
            raise ValueError("Incomplete/duplicate training epochs")
        for epoch in SNAPSHOTS:
            path = folder / f"checkpoint_epoch{epoch}.pt"
            if digest(path) != path.with_suffix(".sha256").read_text().strip():
                raise ValueError("Saved snapshot hash mismatch")


def materialize_cache(source: Path, target: Path) -> Path:
    """Kaggle may unpack uploaded archives. Repack bytes, never rerun OCR."""
    if source.is_file():
        return source
    folder = (
        source / "ocr_cache"
        if (source / "ocr_cache/manifest.json").is_file()
        else source
    )
    if not (folder / "manifest.json").is_file():
        raise FileNotFoundError("OCR manifest missing")
    with ZipFile(target, "x", compression=ZIP_DEFLATED) as z:
        files = (
            [folder / "manifest.json"]
            + sorted((folder / "train").glob("*.json"))
            + sorted((folder / "val").glob("*.json"))
        )
        from zipfile import ZipInfo

        for path in files:
            info = ZipInfo("ocr_cache/" + str(path.relative_to(folder)))
            info.compress_type = ZIP_DEFLATED
            z.writestr(info, path.read_bytes())
    return target


def package(root: Path) -> Path:
    """Whitelist analysis plus lightweight training metadata; never serialize weights."""
    target = root.parent / f"{root.name}_analysis.zip"
    paths = [p for p in (root / "analysis").rglob("*") if p.is_file()]
    paths += [
        p
        for p in (root / "training").rglob("*")
        if p.is_file() and p.suffix in {".json", ".jsonl", ".sha256"}
    ]
    paths += [
        root / "provenance.json",
        root / "original_val_metrics.json",
        root / "COMPLETE.json",
    ]
    paths += list((root / "logs").glob("*.log"))
    if any(p.suffix in {".pt", ".bin", ".safetensors"} for p in paths):
        raise ValueError("Weight file in analysis whitelist")
    manifest = {str(p.relative_to(root)): digest(p) for p in paths}
    with ZipFile(target, "x", compression=ZIP_DEFLATED) as z:
        for p in sorted(paths):
            z.write(p, p.relative_to(root))
        z.writestr("archive_manifest.json", json.dumps(manifest, indent=2))
    with ZipFile(target) as z:
        if z.testzip() is not None or len(z.namelist()) != len(set(z.namelist())):
            raise ValueError("ZIP integrity failure")
        import hashlib

        if any(
            hashlib.sha256(z.read(name)).hexdigest() != expected
            for name, expected in manifest.items()
        ):
            raise ValueError("ZIP digest verification failed")
    target.with_suffix(".zip.sha256").write_text(digest(target) + "\n")
    return target


def run_command(cmd, log):
    print("Running:", " ".join(cmd), flush=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with (
        log.open("w") as stream,
        subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process,
    ):
        for line in process.stdout:
            stream.write(line)
            if not any(token in line for token in ("it/s", "s/it", "%|")):
                print(line, end="", flush=True)
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--ocr-cache", type=Path, required=True)
    p.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="original seed42 result ZIP or validation metrics.json",
    )
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--gpus", type=int, choices=(1, 2), default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    if args.num_workers < 0:
        p.error("Invalid worker count")
    root = args.output_root.resolve()
    if (
        root.exists()
        or (root.parent / f"{root.name}_analysis.zip").exists()
        or (root.parent / f"{root.name}_analysis.zip.sha256").exists()
    ):
        raise FileExistsError(
            "Choose a new output root; previous experiments are never overwritten"
        )
    baseline = verify_source()
    import torch

    parent = load_parent(args.parent)
    ref = reference(args.reference, 42)
    if (
        parent["model_config"] != ref["summary"]["model_config"]
        or parent["run_config"]["recipe"] != ref["summary"]["recipe"]
    ):
        raise ValueError("Parent/reference metadata differ")
    for split, n in (("train", 6494), ("val", 1131)):
        records = json.loads((args.data_root / f"{split}_grounding.json").read_text())
        if len(records) != n:
            raise ValueError(f"Wrong {split} counts")
        for name in records:
            if (
                Path(name).name != name
                or not (args.data_root / split / name).is_file()
                or not (
                    args.data_root
                    / "binary_masks_png"
                    / split
                    / f"{Path(name).stem}.png"
                ).is_file()
            ):
                raise ValueError(f"Missing/unsafe image or mask: {name}")
    if set(records) != set(ref["per_sample"]):
        raise ValueError("Validation IDs differ from original reference")
    if not args.ocr_cache.exists():
        raise FileNotFoundError(args.ocr_cache)
    if (
        root.is_relative_to(args.data_root.resolve())
        or args.data_root.resolve().is_relative_to(root)
        or root.is_relative_to(args.parent.resolve().parent)
    ):
        raise ValueError(
            "Output root must not overlap immutable dataset/checkpoint inputs"
        )
    if args.preflight_only:
        print("CPU metadata preflight passed; no model inference or training performed")
        return
    if torch.cuda.device_count() < args.gpus:
        raise ValueError("Select the requested Kaggle GPU accelerator")
    # Storage acceptance, not a runtime-feasibility probe. Six full weight-only
    # snapshots plus one atomic temporary and artifact margin must fit.
    tensor_bytes = sum(
        t.numel() * t.element_size() for t in parent["model_state_dict"].values()
    )
    if shutil.disk_usage(root.parent).free < tensor_bytes * 7 + 2 * 1024**3:
        raise ValueError(
            "Insufficient free working disk for six weight-only snapshots plus atomic-save/artifact margin"
        )
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    (root / "analysis/evaluations").mkdir(parents=True)
    input_paths = {
        "data_root": str(args.data_root.resolve()),
        "parent": str(args.parent.resolve()),
        "original_ocr_input": str(args.ocr_cache.resolve()),
        "original_validation_reference": str(args.reference.resolve()),
    }
    args.ocr_cache = materialize_cache(args.ocr_cache, root / "ocr_cache_input.zip")
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    required = {
        "b1_run.py",
        "b1_data.py",
        "b1_analysis.py",
        "grounding_diagnostics.py",
        "models/simple_ocr.py",
        "scripts/run_b1_session.py",
    }
    if not required <= set(tracked):
        raise ValueError(
            "Required workflow modules must be Git-tracked before execution"
        )
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise ValueError(
            "Commit tracked workflow changes before running the experiment"
        )
    metadata = {
        "input_paths": input_paths,
        "workflow_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "workflow_files_sha256": {
            name: digest(ROOT / name)
            for name in tracked
            if name.endswith(".py")
            or name
            in (
                "uv.lock",
                "pyproject.toml",
                "README.md",
                "notebooks/b1_simple_ocr.ipynb",
            )
        },
        "baseline_source": baseline,
        "parent_sha256": PARENT_SHA,
        "ocr_cache_sha256": digest(args.ocr_cache),
        "annotation_sha256": {
            s: digest(args.data_root / f"{s}_grounding.json") for s in ("train", "val")
        },
        "gpus": [torch.cuda.get_device_name(i) for i in range(args.gpus)],
        "python": sys.version,
        "packages": {
            x.metadata["Name"]: x.version for x in importlib.metadata.distributions()
        },
        "policy": {
            "arms": list(ARMS),
            "epochs": 10,
            "snapshots": [5, 10],
            "diagnostics_after_all_training": True,
            "weight_only_no_optimizer_resume": True,
            "text_mode": "question",
            "test_access": False,
        },
    }
    (root / "provenance.json").write_text(json.dumps(metadata, indent=2))
    (root / "original_val_metrics.json").write_text(json.dumps(ref, indent=1))
    # Build the same frozen evidence bank once for all arms. No grounding mask
    # or answer enters feature generation, and the verified original parent is used.
    from b1_data import build_features
    from models.text_encoder import TextEncoder

    encoder = TextEncoder(parent["model_config"]["clip_name"])
    state = {
        k.removeprefix("text_encoder."): v
        for k, v in parent["model_state_dict"].items()
        if k.startswith("text_encoder.")
    }
    encoder.load_state_dict(state, strict=True)
    del parent, state
    encoder.requires_grad_(False).to("cuda:0").eval()
    bank = build_features(
        args.data_root, args.ocr_cache, encoder, torch.device("cuda:0")
    )
    train_hashes = {r["pixel_sha256"] for r in bank["train"].values()}
    overlap = {
        "val_images_with_train_pixel_hash": [
            name for name, r in bank["val"].items() if r["pixel_sha256"] in train_hashes
        ],
        "val_pixel_hashes": {
            name: r["pixel_sha256"] for name, r in bank["val"].items()
        },
        "note": "Official results unchanged; use grouped resampling for future confirmatory uncertainty.",
    }
    # Hold the preflight overlap record outside analysis until all training is done.
    (root / "dataset_overlap.json").write_text(json.dumps(overlap, indent=2))
    features = root / "ocr_features.pt"
    torch.save(bank, features)
    metadata["features_sha256"] = digest(features)
    metadata["ocr_feature_policy"] = {
        "cap": 64,
        "normalization": "NFC+strip",
        "coordinate_frame": "upright native normalized xyxy",
        "score": "recognition confidence, not calibrated",
        "frozen_text": "parent final EOS hidden",
    }
    with ZipFile(args.ocr_cache) as cache_archive:
        metadata["ocr_manifest"] = json.loads(
            cache_archive.read("ocr_cache/manifest.json")
        )
    (root / "provenance.json").write_text(json.dumps(metadata, indent=2))
    del bank, encoder
    torch.cuda.empty_cache()
    training, evaluation = plan_commands(
        root,
        args.data_root.resolve(),
        args.parent.resolve(),
        features,
        args.gpus,
        args.num_workers,
    )
    for arm, cmd in zip(ARMS, training, strict=True):
        run_command(cmd, root / "logs" / f"train_{arm}.log")
    training_barrier(root)
    metadata["training_barrier_passed"] = True
    shutil.copyfile(
        root / "dataset_overlap.json", root / "analysis/dataset_overlap.json"
    )
    metadata["commands"] = {"training": training, "evaluation": evaluation}
    (root / "provenance.json").write_text(json.dumps(metadata, indent=2))
    for i, cmd in enumerate(evaluation):
        run_command(cmd, root / "logs" / f"evaluate_{i:02d}.log")
    from b1_analysis import analyze_all

    summary = analyze_all(root, args.data_root, args.ocr_cache, ref)
    (root / "COMPLETE.json").write_text(json.dumps(summary, indent=2))
    print("Verified analysis ZIP:", package(root), flush=True)
    print(
        "Keep the six training/*/checkpoint_epoch{5,10}.pt files separately in saved Kaggle outputs."
    )


if __name__ == "__main__":
    main()
