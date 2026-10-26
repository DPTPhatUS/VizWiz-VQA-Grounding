"""Evaluate adapter and robustness views at native resolution; GT is used only by metrics."""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from dataset import make_loader
from metrics import global_iou_summary, metric_samples
from ocr_cache import atomic_json
from ocr_data import OCRGroundingDataset
from training.ocr_checkpoint import load_ocr_model
from utils import to_device


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-checkpoint", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--dataset", choices=("val", "test"), default="val")
    p.add_argument(
        "--view",
        choices=("full", "base", "missing", "geometry", "shuffled_text"),
        default="full",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        p.error("Invalid batch/worker count")
    model, saved, digest = load_ocr_model(args.parent_checkpoint, args.checkpoint)
    config = saved["run_config"]
    model.inference_view = args.view
    model.to(args.device).eval()
    data = OCRGroundingDataset(
        args.data_root,
        args.dataset,
        args.cache_root,
        model.image_size,
        detail_size=model.detail_size,
        max_candidates=config["max_candidates"],
    )
    if data.cache_protocol != config["cache_protocol"]:
        raise ValueError(
            "OCR provider/protocol differs from training; export a separate transfer experiment"
        )
    shuffled_regions = (
        data.shuffle_recognized_text(config["seed"] + 17)
        if args.view == "shuffled_text"
        else 0
    )
    loader = make_loader(data, args, args.batch_size)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Evaluation output must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    scores, totals, base_scores, base_totals, groups, evidence = {}, {}, {}, {}, {}, {}
    elapsed, measured_images = 0.0, 0
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            batch = to_device(batch, args.device)
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(batch)
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            if (
                step
            ):  # Exclude first batch warmup, report separately from OCR cache timing.
                elapsed += time.perf_counter() - start
                measured_images += len(batch["filename"])
            for name, score, prediction in metric_samples(
                output["logits"], batch, pixel_totals=totals
            ):
                scores[name] = score
                Image.fromarray(prediction.cpu().numpy().astype("uint8") * 255).save(
                    out / Path(name).with_suffix(".png")
                )
            baseline = output.get("base_logits", output["logits"])
            for i, (name, score, _) in enumerate(
                metric_samples(baseline, batch, pixel_totals=base_totals)
            ):
                base_scores[name] = score
                area = batch["original_mask"][i].float().mean().item()
                group = (
                    "empty"
                    if area == 0
                    else "small_lt_1pct"
                    if area < 0.01
                    else "medium_1_to_10pct"
                    if area < 0.1
                    else "large_ge_10pct"
                )
                groups.setdefault(group, []).append(name)
                evidence[name] = {
                    "ocr_candidates": int(batch["ocr_valid"][i].sum()),
                    "null_probability": output["null_probability"][i].item()
                    if "null_probability" in output
                    else None,
                }
    summary = dict(
        num_samples=len(scores),
        mean_iou=sum(scores.values()) / len(scores),
        **global_iou_summary(totals),
        dataset=args.dataset,
        metric_resolution="original",
        text_mode="question",
        view=args.view,
        shuffled_text_changed_regions=shuffled_regions,
        parent_sha256=digest,
        base_mean_iou=sum(base_scores.values()) / len(base_scores),
        base_metrics=global_iou_summary(base_totals),
        checkpoint=str(args.checkpoint),
        experiment_config=model.experiment_config,
        cache_protocol=data.cache_protocol,
        cache_digest=data.cache_digest,
        adapter_parameters=sum(p.numel() for p in model.adapter.parameters()),
        frozen_parent_parameters=sum(p.numel() for p in model.parent.parameters()),
        cached_grounding_seconds_per_image=elapsed / measured_images
        if measured_images
        else None,
        timing_note="Excludes OCR, data decode/transfer, metric calculation and first batch; not end-to-end latency",
        inference_precision="float32",
        threshold_logit=0,
        empty_union_iou=1,
    )
    summary["groups"] = {
        group: {
            "n": len(names),
            "mean_iou": sum(scores[k] for k in names) / len(names),
            "base_mean_iou": sum(base_scores[k] for k in names) / len(names),
        }
        for group, names in groups.items()
    }
    atomic_json(
        {
            "summary": summary,
            "per_sample": scores,
            "base_per_sample": base_scores,
            "evidence": evidence,
        },
        out / "metrics.json",
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
