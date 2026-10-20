"""Read-only grounding-first diagnostics at native GT resolution."""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from dataset import VizWizGroundingDataset, make_loader
from metrics import metric_samples
from models.model import GroundingModel
from utils import (
    checkpoint_model_kwargs,
    load_model_weights,
    read_checkpoint,
    to_device,
)


def mean(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return sum(values) / len(values) if values else None


@torch.no_grad()
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
        raise ValueError("Invalid batch size or worker count")
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Diagnostic output must be new or empty")
    saved = read_checkpoint(args.checkpoint)
    model = GroundingModel(**checkpoint_model_kwargs(saved))
    load_model_weights(model, saved)
    device = torch.device(args.device)
    model.to(device).eval()
    dataset = VizWizGroundingDataset(
        args.data_root, "val", saved["run_config"].get("image_size", 336)
    )
    loader = make_loader(dataset, args, args.batch_size)
    rows = []
    for batch in loader:
        batch = to_device(batch, device)
        result = model(dict(batch, read_answer=True))
        local = model.reader.encode(batch["image"])
        full = model.reader.read(
            local, torch.ones_like(batch["mask"]), result["question_vec"], frozen=True
        )
        scores = {
            key: F.cosine_similarity(
                value.float(), result["target_answer"].float(), dim=-1, eps=1e-6
            ).tolist()
            for key, value in [
                ("gt_region_cosine", result["gt_region_answer"]),
                ("predicted_region_cosine", result["predicted_answer"]),
                ("full_image_cosine", full),
            ]
        }
        for index, (name, iou, _) in enumerate(metric_samples(result["logits"], batch)):
            area = batch["original_mask"][index].float().mean().item()
            available = bool(batch["answer_available"][index]) and bool(
                result["region_nonempty"][index]
            )
            row = dict(
                filename=name,
                iou=iou,
                gt_area_fraction=area,
                answer_available=available,
            )
            row["area_group"] = (
                "empty"
                if area == 0
                else "small_lt_1pct"
                if area < 0.01
                else "medium_1_to_10pct"
                if area < 0.1
                else "large_ge_10pct"
            )
            row.update(
                {
                    key: values[index] if available else None
                    for key, values in scores.items()
                }
            )
            rows.append(row)

    def summary(selected):
        return dict(
            num_samples=len(selected),
            mean_iou=mean(selected, "iou"),
            gt_region_cosine=mean(selected, "gt_region_cosine"),
            predicted_region_cosine=mean(selected, "predicted_region_cosine"),
            full_image_cosine=mean(selected, "full_image_cosine"),
        )

    report = dict(
        checkpoint=str(args.checkpoint),
        epoch=saved.get("epoch"),
        metric_resolution="original",
        experiment_config=model.experiment_config,
        training_config=saved["run_config"].get("training_config"),
        source_sha256=saved["run_config"].get("source_sha256"),
        overall=summary(rows),
        by_gt_area={
            group: summary([row for row in rows if row["area_group"] == group])
            for group in [
                "empty",
                "small_lt_1pct",
                "medium_1_to_10pct",
                "large_ge_10pct",
            ]
        },
        interpretation=[
            "Mask inference uses image/question only. GT masks/answers are read only after mask prediction.",
            "Reader cosine is embedding alignment, not textual answer accuracy or proof of causal grounding.",
            "Local reader features have a 5x5 receptive field, including a two-pixel context halo around selected pixels.",
            "Full-image versus selected-region readout is diagnostic, not an additional training loss.",
        ],
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
