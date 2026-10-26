"""CPU diagnosis separating evidence-mask quality from routing; GT is never used for inference."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ocr_cache import atomic_json, read_json
from ocr_routing_data import OCRRoutingDataset, routing_targets


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    metrics = read_json(Path(args.predictions) / "metrics.json")
    summary = metrics["summary"]
    if (
        summary["dataset"] != "val"
        or summary["view"] != "full"
        or not metrics.get("evidence_per_sample")
    ):
        raise ValueError(
            "Use normal routing validation predictions, with evidence branch scores"
        )
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Diagnostic output must be new or empty")
    config = summary["experiment_config"]
    data = OCRRoutingDataset(
        args.data_root,
        "val",
        args.cache_root,
        config["parent_experiment"]["image_size"],
        detail_size=config["parent_experiment"]["detail_size"],
        max_candidates=len(next(iter(metrics["evidence"].values()))["boxes"]),
    )
    if set(data.names) != set(metrics["per_sample"]):
        raise ValueError("Diagnostic IDs do not cover the full validation split")
    if data.cache_digest != summary["cache_digest"]:
        raise ValueError("OCR cache differs from evaluation")
    rows = []
    for i, name in enumerate(data.names):
        sample = data[i]
        with Image.open(
            Path(args.data_root)
            / "binary_masks_png/val"
            / Path(name).with_suffix(".png")
        ) as source:
            mask = source.convert("L")
        targets = routing_targets(mask, sample)
        area = float((np.asarray(mask) > 127).mean())
        group = (
            "empty"
            if not area
            else "small_lt_1pct"
            if area < 0.01
            else "medium_1_to_10pct"
            if area < 0.1
            else "large_ge_10pct"
        )
        e = metrics["evidence"][name]
        base, evidence, final = (
            metrics["base_per_sample"][name],
            metrics["evidence_per_sample"][name],
            metrics["per_sample"][name],
        )
        rows.append(
            {
                "filename": name,
                "group": group,
                "base_iou": base,
                "evidence_iou": evidence,
                "final_iou": final,
                "route_probability": e["route_probability"],
                "coverage_target": int(targets["ocr_route_target"]),
                "coverage": float(targets["ocr_route_coverage"]),
                "evidence_gain_pp": 100 * (evidence - base),
                "final_gain_pp": 100 * (final - base),
            }
        )

    def aggregate(selected):
        return {
            "n": len(selected),
            "mean_base_iou": float(np.mean([r["base_iou"] for r in selected])),
            "mean_evidence_iou": float(np.mean([r["evidence_iou"] for r in selected])),
            "mean_final_iou": float(np.mean([r["final_iou"] for r in selected])),
            "branch_choice_oracle_iou": float(
                np.mean([max(r["base_iou"], r["evidence_iou"]) for r in selected])
            ),
            "mean_route_probability": float(
                np.mean([r["route_probability"] for r in selected])
            ),
            "coverage_positive": sum(r["coverage_target"] for r in selected),
            "route_above_half": sum(r["route_probability"] >= 0.5 for r in selected),
            "evidence_better_by_5pp": sum(r["evidence_gain_pp"] > 5 for r in selected),
            "missed_better_evidence_5pp": sum(
                r["evidence_gain_pp"] > 5 and r["route_probability"] < 0.5
                for r in selected
            ),
            "routed_to_worse_evidence_5pp": sum(
                r["evidence_gain_pp"] < -5 and r["route_probability"] >= 0.5
                for r in selected
            ),
        }

    result = {
        "overall": aggregate(rows),
        "groups": {
            g: aggregate([r for r in rows if r["group"] == g])
            for g in sorted({r["group"] for r in rows})
        },
        "interpretation": "Coverage target is the training heuristic, not a manually annotated semantic route. Branch-choice oracle uses GT and is not a deployable score or upper bound on arbitrary soft mixtures.",
    }
    atomic_json(result, out / "summary.json")
    with (out / "per_image.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(result["overall"])


if __name__ == "__main__":
    main()
