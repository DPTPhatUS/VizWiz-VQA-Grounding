"""CPU-only error analysis of exported native-resolution validation masks."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        required=True,
        help="eval_val directory containing metrics.json and masks",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    predictions, out = Path(args.predictions), Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Diagnostic output must be new or empty")
    metrics = json.loads((predictions / "metrics.json").read_text())
    if (
        metrics["summary"]["dataset"] != "val"
        or metrics["summary"]["metric_resolution"] != "original"
    ):
        raise ValueError("Diagnostic requires native-resolution validation predictions")
    annotation = json.loads((Path(args.data_root) / "val_grounding.json").read_text())
    if set(annotation) != set(metrics["per_sample"]):
        raise ValueError("Prediction IDs must match the entire validation split")
    rows = []
    for name in annotation:
        path = Path(name).with_suffix(".png")
        with Image.open(predictions / path) as image:
            raw = np.asarray(image.convert("L"))
        if not np.isin(raw, (0, 255)).all():
            raise ValueError(f"Expected a binary exported mask for {name}")
        with Image.open(Path(args.data_root) / "binary_masks_png/val" / path) as image:
            truth = np.asarray(image.convert("L")) > 127
        pred = raw > 127
        if pred.shape != truth.shape:
            raise ValueError(f"Native prediction/GT shape mismatch for {name}")
        tp, p, g = int((pred & truth).sum()), int(pred.sum()), int(truth.sum())
        union = p + g - tp
        iou = tp / union if union else 1.0
        if abs(iou - metrics["per_sample"][name]) > 1e-6:
            raise ValueError(f"Exported mask and stored IoU disagree for {name}")
        area = g / truth.size
        group = (
            "empty"
            if not g
            else "small_lt_1pct"
            if area < 0.01
            else "medium_1_to_10pct"
            if area < 0.1
            else "large_ge_10pct"
        )
        rows.append(
            dict(
                filename=name,
                question=annotation[name].get("question", ""),
                iou=iou,
                precision=tp / p if p else 0.0,
                recall=tp / g if g else 0.0,
                gt_area_fraction=area,
                predicted_area_fraction=p / truth.size,
                area_ratio=p / g if g else None,
                area_group=group,
                true_positive_pixels=tp,
                false_positive_pixels=p - tp,
                false_negative_pixels=g - tp,
            )
        )

    def summarize(selected):
        intersection = sum(r["true_positive_pixels"] for r in selected)
        union = sum(
            r["true_positive_pixels"]
            + r["false_positive_pixels"]
            + r["false_negative_pixels"]
            for r in selected
        )
        result = {
            "num_samples": len(selected),
            "global_iou": (intersection / union if union else 1.0)
            if selected
            else None,
            "total_intersection": intersection,
            "total_union": union,
        }
        for key in (
            "iou",
            "precision",
            "recall",
            "gt_area_fraction",
            "predicted_area_fraction",
            "area_ratio",
        ):
            values = [r[key] for r in selected if r[key] is not None]
            result[f"mean_{key}"] = float(np.mean(values)) if values else None
            if key == "area_ratio":
                result["median_area_ratio"] = (
                    float(np.median(values)) if values else None
                )
        result["iou_lt_0.1_count"] = sum(r["iou"] < 0.1 for r in selected)
        result["area_ratio_gt_10_count"] = sum(
            r["area_ratio"] is not None and r["area_ratio"] > 10 for r in selected
        )
        return result

    report = dict(
        protocol=metrics["summary"],
        overall=summarize(rows),
        by_gt_area={
            g: summarize([r for r in rows if r["area_group"] == g])
            for g in ("empty", "small_lt_1pct", "medium_1_to_10pct", "large_ge_10pct")
        },
        interpretation="Precision/recall use zero for an empty denominator; empty-union IoU is one. No checkpoint or GT answer is used.",
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    with (out / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
