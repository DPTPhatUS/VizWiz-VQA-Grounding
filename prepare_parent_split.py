"""Prepare disjoint parent/refiner image lists; no labels/images/checkpoints are copied."""

import argparse
import hashlib
import json
from pathlib import Path

from ocr_cache import atomic_json, file_hash


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-fraction", type=float, default=0.15)
    p.add_argument(
        "--cache-root",
        help="Optional OCR cache: keep identical upright pixels in the same parent/head group",
    )
    args = p.parse_args(argv)
    if not 0 < args.holdout_fraction < 1:
        p.error("holdout-fraction must be between zero and one")
    source = Path(args.data_root) / "train_grounding.json"
    names = list(json.loads(source.read_text()))
    if len(names) < 2:
        raise ValueError("Need at least two train images")
    ordered = sorted(
        names,
        key=lambda n: hashlib.sha256(
            f"native-parent-split-v1:{args.seed}:{n}".encode()
        ).hexdigest(),
    )
    count = min(len(names) - 1, max(1, round(args.holdout_fraction * len(names))))
    if args.cache_root:
        groups = {}
        for name in names:
            record = json.loads(
                (
                    Path(args.cache_root) / "train" / Path(name).with_suffix(".json")
                ).read_text()
            )
            groups.setdefault(record["pixel_sha256"], []).append(name)
        if len(groups) < 2:
            raise ValueError("Need at least two distinct pixel groups")
        keys = sorted(
            groups,
            key=lambda k: hashlib.sha256(
                f"parent-pixel-split-v1:{args.seed}:{k}".encode()
            ).hexdigest(),
        )
        held = []
        for key in keys[:-1]:
            held.extend(groups[key])
            if len(held) >= count:
                break
        ordered = sorted(held) + sorted(set(names) - set(held))
        count = len(held)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Split output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(sorted(ordered[count:]), out / "parent_fit.json")
    atomic_json(sorted(ordered[:count]), out / "refiner_fit.json")
    atomic_json(
        {
            "version": "native-parent-split-v1",
            "seed": args.seed,
            "annotation_sha256": file_hash(source),
            "parent_count": len(names) - count,
            "refiner_count": count,
            "holdout_fraction": args.holdout_fraction,
            "pixel_grouped": bool(args.cache_root),
        },
        out / "split.json",
    )
    print(f"Parent: {len(names) - count} images; refiner: {count} disjoint images")


if __name__ == "__main__":
    main()
