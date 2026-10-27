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
    args = p.parse_args(argv)
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
    count = max(1, round(0.15 * len(names)))
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
        },
        out / "split.json",
    )
    print(f"Parent: {len(names) - count} images; refiner: {count} disjoint images")


if __name__ == "__main__":
    main()
