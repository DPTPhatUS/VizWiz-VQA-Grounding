"""Validate train-vocabulary coverage and visual dependence of the dense answer head."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from dataset import VizWizGroundingDataset, make_loader
from models.model import GroundingModel
from utils import (
    checkpoint_model_kwargs,
    load_model_weights,
    read_checkpoint,
    to_device,
)


def point_statistics(output, selected):
    targets, logits = output["answer_targets"], output["answer_logits"].float()
    valid = (targets > 0) if selected == "foreground" else (targets == 0)
    ce = F.cross_entropy(
        logits.transpose(1, 2), targets, reduction="none", ignore_index=-100
    )
    count = valid.sum(1)
    return (
        (ce * valid).sum(1) / count.clamp_min(1),
        (((logits.argmax(-1) == targets) & valid).sum(1).float() / count.clamp_min(1)),
        count > 0,
    )


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
        raise ValueError("Diagnostic output must be new or empty")
    saved = read_checkpoint(args.checkpoint)
    model = GroundingModel(**checkpoint_model_kwargs(saved))
    load_model_weights(model, saved)
    model.to(args.device).eval()
    dataset = VizWizGroundingDataset(
        args.data_root,
        "val",
        model.image_size,
        detail_size=model.detail_size,
        include_answers=True,
    )
    rows = []
    with torch.inference_mode():
        for batch in make_loader(dataset, args, args.batch_size):
            batch = to_device(batch, args.device)
            output = model({**batch, "inspect_answer": True})
            feature = output["answer_features"]
            variants = {
                "original": feature,
                "zero_visual": torch.zeros_like(feature),
                "flipped_visual": feature.flip((-2, -1)),
            }
            stats = {}
            for name, local in variants.items():
                # Deterministic point coordinates and hard negatives are identical across probes.
                answer = model.answer_head(
                    local,
                    batch["mask"],
                    batch["answer_text"],
                    prediction=output["logits"],
                )
                stats[name] = {
                    group: point_statistics(answer, group)
                    for group in ("foreground", "background")
                }
            for i, filename in enumerate(batch["filename"]):
                row = {"filename": filename}
                for name, groups in stats.items():
                    row[name] = {
                        group: {
                            "ce": float(loss[i]),
                            "class_accuracy": float(accuracy[i]),
                            "valid": bool(valid[i]),
                        }
                        for group, (loss, accuracy, valid) in groups.items()
                    }
                rows.append(row)
    valid = [r for r in rows if r["original"]["foreground"]["valid"]]
    summaries = {}
    for name in ("original", "zero_visual", "flipped_visual"):
        summaries[name] = {}
        for group in ("foreground", "background"):
            available = [r for r in valid if r[name][group]["valid"]]
            summaries[name][group] = {
                "num_images": len(available),
                **{
                    key: sum(r[name][group][key] for r in available) / len(available)
                    if available
                    else None
                    for key in ("ce", "class_accuracy")
                },
            }
    report = {
        "checkpoint": args.checkpoint,
        "num_images": len(rows),
        "num_valid_answers": len(valid),
        "vocabulary_size": len(model.answer_head.lookup),
        "views": summaries,
        "interpretation": "Auxiliary point classification, not VQA generation. Validation OOV answers are skipped. Zero/rotated features are diagnostic interventions after I+Q mask prediction, not deployable image inputs.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    (out / "per_image.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
