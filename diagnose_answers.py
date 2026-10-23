"""Validation-only auxiliary answer check; answer labels never alter predicted masks."""

import argparse
import json
from pathlib import Path

import torch

from dataset import VizWizGroundingDataset, make_loader
from losses import TRAINING_DEFAULTS, answer_statistics
from models.model import GroundingModel
from utils import (
    checkpoint_model_kwargs,
    load_model_weights,
    read_checkpoint,
    to_device,
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
    settings = {
        **TRAINING_DEFAULTS,
        **saved.get("run_config", {}).get("training_config", {}),
    }
    rows = []
    with torch.inference_mode():
        for batch in make_loader(dataset, args, args.batch_size):
            batch = to_device(batch, args.device)
            output = model({**batch, "inspect_answer": True})
            feature = output["answer_features"]
            # All views reuse exactly the same image encoding, mask logits and answer prefix.
            views = {
                "gt_region": batch["mask"],
                "full_image": torch.ones_like(batch["mask"]),
                "predicted_region": output["logits"].float().sigmoid(),
                "zero_visual": batch["mask"],
            }
            stats = {}
            for name, mask in views.items():
                local = torch.zeros_like(feature) if name == "zero_visual" else feature
                stats[name] = answer_statistics(
                    model.answer_head(local, mask, batch["answer_text"])
                )
            for i, filename in enumerate(batch["filename"]):
                row = {"filename": filename}
                for name, (loss, accuracy, valid) in stats.items():
                    row[name] = {
                        "ce": float(loss[i]),
                        "teacher_forced_accuracy": float(accuracy[i]),
                        "valid": bool(valid[i]),
                    }
                gain = row["full_image"]["ce"] - row["gt_region"]["ce"]
                row["reader_visual_gain"] = gain
                row["reader_reliable"] = (
                    row["gt_region"]["valid"] and gain > settings["reader_margin"]
                )
                row["mask_answer_gap"] = max(
                    0.0,
                    row["predicted_region"]["ce"]
                    - row["gt_region"]["ce"]
                    - settings["mask_answer_tolerance"],
                )
                row["guidance_active"] = (
                    row["reader_reliable"] and row["mask_answer_gap"] > 0
                )
                rows.append(row)
    valid = [r for r in rows if r["gt_region"]["valid"]]
    report = {
        "checkpoint": args.checkpoint,
        "num_images": len(rows),
        "num_valid_answers": len(valid),
        "views": {
            name: {
                key: sum(r[name][key] for r in valid) / len(valid) if valid else None
                for key in ("ce", "teacher_forced_accuracy")
            }
            for name in ("gt_region", "full_image", "predicted_region", "zero_visual")
        },
        "reader_margin": settings["reader_margin"],
        "mask_answer_tolerance": settings["mask_answer_tolerance"],
        "reader_reliable_fraction": sum(r["reader_reliable"] for r in rows) / len(rows)
        if rows
        else None,
        "guidance_active_fraction": sum(r["guidance_active"] for r in rows) / len(rows)
        if rows
        else None,
        "interpretation": "Auxiliary teacher-forced byte accuracy, not free-generation VQA accuracy. GT/answer used only for this diagnostic; grounding logits were computed first from I+Q.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    (out / "per_image.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
