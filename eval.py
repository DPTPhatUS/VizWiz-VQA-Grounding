"""Evaluate this branch's checkpoint with question-only text and export masks."""

import argparse
import json
from pathlib import Path

import torch

from dataset import VizWizGroundingDataset, make_loader
from metrics import evaluate, global_iou_summary
from models.model import GroundingModel
from utils import checkpoint_model_kwargs, load_model_weights, read_checkpoint


def eval_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Question-only evaluation using saved model architecture"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Raw model weights or a training checkpoint"
    )
    parser.add_argument("--data-root", default="data/vizwiz")
    parser.add_argument("--dataset", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count")
    saved = read_checkpoint(args.checkpoint)
    config = saved["run_config"]
    model = GroundingModel(**checkpoint_model_kwargs(saved))
    load_model_weights(model, saved)
    device = torch.device(args.device)
    model.to(device)
    dataset = VizWizGroundingDataset(
        args.data_root, args.dataset, model.image_size, detail_size=model.detail_size
    )
    loader = make_loader(dataset, args, args.batch_size)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Evaluation output must be a new or empty directory")
    out.mkdir(parents=True, exist_ok=True)
    pixel_totals, views, scope = {}, {}, {}
    scores = evaluate(
        model,
        loader,
        device,
        out,
        pixel_totals=pixel_totals,
        view_results=views,
        scope_diagnostics=scope,
    )
    summary = {
        **global_iou_summary(pixel_totals),
        "num_samples": len(scores),
        "mean_iou": sum(scores.values()) / len(scores),
        "dataset": args.dataset,
        "metric_resolution": "original",
        "text_mode": "question",
        "checkpoint": str(args.checkpoint),
        "grounding_mode": "question",
        "training_config": config.get("training_config"),
        "source_sha256": config.get("source_sha256"),
        "experiment_config": model.experiment_config,
        "inference_precision": "float32",
        "threshold_logit": 0,
        "empty_union_iou": 1,
    }
    for result in views.values():
        values = result["per_sample"]
        result["summary"] = {
            **global_iou_summary(result.pop("pixel_totals")),
            "mean_iou": sum(values.values()) / len(values),
            "num_samples": len(values),
        }
    summary["selection_fraction"] = sum(
        row["selected"] for row in scope.values()
    ) / len(scope)
    summary["parent_mean_iou"] = views["parent"]["summary"]["mean_iou"]
    summary["candidate_mean_iou"] = views["candidate"]["summary"]["mean_iou"]
    for name, row in scope.items():
        row["candidate_gain"] = (
            views["candidate"]["per_sample"][name] - views["parent"]["per_sample"][name]
        )
        row["selected_gain"] = scores[name] - views["parent"]["per_sample"][name]
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "per_sample": scores,
                "views": views,
                "scope_diagnostics": scope,
            },
            indent=2,
        )
    )
    print(json.dumps(summary), flush=True)


main = eval_main

if __name__ == "__main__":
    main()
