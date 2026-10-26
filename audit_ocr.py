"""CPU candidate coverage/oracles on native masks. Oracle scores are diagnostics, not model results."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from ocr_cache import atomic_json, candidates, pixel_hash, polygon_mask, read_json


def iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--predictions", help="Optional native B+5 prediction directory")
    p.add_argument("--max-candidates", type=int, default=24)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args(argv)
    if args.max_candidates < 1:
        p.error("max-candidates must be positive")
    root, cache, out = (
        Path(args.data_root),
        Path(args.cache_root),
        Path(args.output_dir),
    )
    if out.exists() and any(out.iterdir()):
        raise ValueError("Audit output must be new or empty")
    rows = {}
    for name in tqdm(
        read_json(root / f"{args.split}_grounding.json"), desc="OCR coverage"
    ):
        record = read_json(cache / args.split / Path(name).with_suffix(".json"))
        with Image.open(root / args.split / name) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if record["size"] != list(image.size) or record["pixel_sha256"] != pixel_hash(
            image
        ):
            raise ValueError(f"OCR/image mismatch: {name}")
        with Image.open(
            root / "binary_masks_png" / args.split / Path(name).with_suffix(".png")
        ) as source:
            truth = np.asarray(source.convert("L")) > 127
        if truth.shape != (image.height, image.width):
            raise ValueError(f"Mask/image coordinates disagree: {name}")
        selected = candidates(record, args.max_candidates)
        support, union = np.zeros_like(truth), np.zeros_like(truth)
        best, all_best = 0.0, 0.0
        for region in selected:
            mask = polygon_mask(record, region)
            best = max(best, iou(mask, truth))
            union |= mask
            x1, y1, x2, y2 = region["box"]
            # Same pixel-center support rule as render_regions.
            xx = (np.arange(image.width) + 0.5) / image.width
            yy = (np.arange(image.height) + 0.5) / image.height
            support |= ((xx >= x1) & (xx < x2))[None] & ((yy >= y1) & (yy < y2))[
                :, None
            ]
        for region in record["regions"]:
            all_best = max(all_best, iou(polygon_mask(record, region), truth))
        area = float(truth.mean())
        group = (
            "empty"
            if area == 0
            else "small_lt_1pct"
            if area < 0.01
            else "medium_1_to_10pct"
            if area < 0.1
            else "large_ge_10pct"
        )
        row = {
            "group": group,
            "candidates": len(selected),
            "raw_candidates": len(record["regions"]),
            "ocr_seconds": record.get("seconds"),
            "best_polygon_iou": best,
            "all_polygon_best_iou": all_best,
            "union_polygon_iou": iou(union, truth),
            "support_gt_coverage": float((support & truth).sum() / max(1, truth.sum())),
            "support_area_fraction": float(support.mean()),
        }
        if args.predictions:
            with Image.open(
                Path(args.predictions) / Path(name).with_suffix(".png")
            ) as source:
                base = np.asarray(source.convert("L")) > 127
            if base.shape != truth.shape:
                raise ValueError(f"Prediction dimensions disagree: {name}")
            row.update(
                base_iou=iou(base, truth),
                candidate_set_oracle=max(iou(base, truth), best, iou(union, truth)),
                perfect_local_correction_oracle=iou(
                    (base & ~support) | (truth & support), truth
                ),
            )
        rows[name] = row

    def summary(names):
        numeric = [
            key
            for key, value in rows[names[0]].items()
            if isinstance(value, (int, float))
        ]
        return dict(
            n=len(names),
            **{key: float(np.mean([rows[k][key] for k in names])) for key in numeric},
        )

    groups = sorted({r["group"] for r in rows.values()})
    result = {
        "oracle_warning": "GT-dependent candidate choice and perfect local correction are diagnostic upper bounds, never inference.",
        "max_candidates": args.max_candidates,
        "overall": summary(list(rows)),
        "groups": {
            g: summary([k for k, v in rows.items() if v["group"] == g]) for g in groups
        },
        "per_sample": rows,
    }
    atomic_json(result, out / "coverage.json")
    print(result["overall"])


if __name__ == "__main__":
    main()
