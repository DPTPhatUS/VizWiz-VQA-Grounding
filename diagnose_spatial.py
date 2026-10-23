"""Paired native-resolution pre/post audit; GT is used only after prediction."""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from dataset import VizWizGroundingDataset, make_loader
from metrics import accumulate_pixel_counts, global_iou_summary
from models.model import GroundingModel
from utils import to_device


def summarize(rows):
    result = {"num_images": len(rows)}
    if not rows:
        return result
    for view in ("coarse", "final"):
        totals = {
            key: sum(r[view + "_" + key] for r in rows)
            for key in ("intersection", "union")
        }
        result[view] = {
            "mean_iou": sum(r[view + "_iou"] for r in rows) / len(rows),
            **global_iou_summary(totals),
        }
    result["delta_mean_iou"] = (
        result["final"]["mean_iou"] - result["coarse"]["mean_iou"]
    )
    result["improved_gt_5pp"] = sum(r["delta_iou"] > 0.05 for r in rows)
    result["hurt_gt_5pp"] = sum(r["delta_iou"] < -0.05 for r in rows)
    result["mean_destroyed_tp_fraction"] = sum(
        r["destroyed_tp_fraction"] for r in rows
    ) / len(rows)
    return result


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid loader configuration")
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new/empty diagnostic directory")
    model = GroundingModel.from_checkpoint(args.checkpoint, args.device)
    dataset = VizWizGroundingDataset(
        args.data_root, "val", model.image_size, detail_size=model.detail_size
    )
    rows = []
    for batch in make_loader(dataset, args, args.batch_size):
        batch = to_device(batch, args.device)
        output = model(batch)
        for i, name in enumerate(batch["filename"]):
            truth = batch["original_mask"][i][0].to(args.device) > 0.5
            area = truth.float().mean().item()
            group = (
                "empty"
                if area == 0
                else "small_lt_1pct"
                if area < 0.01
                else "medium_1_to_10pct"
                if area < 0.1
                else "large_ge_10pct"
            )
            row = {"filename": name, "area_group": group, "gt_area_fraction": area}
            masks = {}
            for view, key in (("coarse", "coarse_logits"), ("final", "logits")):
                logits = F.interpolate(
                    output[key][i : i + 1].float(),
                    size=truth.shape,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                pred = logits > 0
                totals = {}
                accumulate_pixel_counts(pred, truth, totals)
                row.update({view + "_" + k: v for k, v in totals.items()})
                row[view + "_iou"] = (
                    totals["intersection"] / totals["union"] if totals["union"] else 1.0
                )
                row[view + "_area_fraction"] = pred.float().mean().item()
                masks[view] = pred
            row["delta_iou"] = row["final_iou"] - row["coarse_iou"]
            parent_tp = masks["coarse"] & truth
            row["destroyed_tp_fraction"] = (
                (parent_tp & ~masks["final"]).sum() / parent_tp.sum().clamp_min(1)
            ).item()
            row["added_tp_pixels"] = (
                ((~masks["coarse"]) & masks["final"] & truth).sum().item()
            )
            if "predicted_area" in output:
                row["predicted_soft_area"] = output["predicted_area"][i].item()
                row["actual_soft_area"] = output["logits"][i].sigmoid().mean().item()
            rows.append(row)
    report = {
        "checkpoint": args.checkpoint,
        "experiment": model.experiment_config,
        "metric_resolution": "native",
        "overall": summarize(rows),
        "by_gt_area": {
            g: summarize([r for r in rows if r["area_group"] == g])
            for g in ("empty", "small_lt_1pct", "medium_1_to_10pct", "large_ge_10pct")
        },
        "interpretation": "Coarse and final share this trained model; this is not comparison against an independently trained parent. GT is diagnostic-only. No oracle selection or test threshold tuning.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    with (out / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
