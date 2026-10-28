"""Native-resolution evaluation and OCR interventions for the matched joint experiment."""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from dataset import make_loader
from metrics import global_iou_summary
from native_refinement_data import NativeRefinementDataset
from native_refinement_ops import native_metric_samples
from ocr_cache import atomic_json, file_hash
from training.joint_checkpoint import read_joint
from utils import to_device

VIEWS = ("full", "pyramid", "global", "missing", "geometry", "shuffled_text")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--dataset", choices=("val", "test"), default="val")
    p.add_argument("--view", choices=VIEWS, default="full")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        p.error("Invalid batch/worker count")
    model, saved = read_joint(args.checkpoint)
    config = saved["run_config"]
    model.inference_view = args.view
    model.to(args.device).eval()
    data = NativeRefinementDataset(
        args.data_root,
        args.dataset,
        args.cache_root,
        model.image_size,
        detail_size=model.detail_size,
        max_candidates=config["max_candidates"],
        mode="native_crop"
        if model.variant == "joint-crop" and args.view != "pyramid"
        else "native_point",
    )
    if data.cache_protocol != config["protocol"]["cache_protocol"]:
        raise ValueError("OCR provider/protocol differs from training")
    changed = (
        data.shuffle_recognized_text(config["seed"] + 17)
        if args.view == "shuffled_text"
        else 0
    )
    loader = make_loader(data, args, args.batch_size)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Evaluation output must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    scores, base_scores, totals, base_totals, groups, evidence = {}, {}, {}, {}, {}, {}
    elapsed, measured = 0.0, 0
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            batch = to_device(batch, args.device)
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(batch)
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            if step:
                elapsed += time.perf_counter() - start
                measured += len(batch["filename"])
            for name, score, prediction in native_metric_samples(
                output, batch, pixel_totals=totals
            ):
                scores[name] = score
                Image.fromarray(prediction.cpu().numpy().astype("uint8") * 255).save(
                    out / Path(name).with_suffix(".png")
                )
            baseline = {"native_logits": output["pyramid_native"]}
            for i, (name, score, _) in enumerate(
                native_metric_samples(baseline, batch, pixel_totals=base_totals)
            ):
                base_scores[name] = score
                area = batch["original_mask"][i].double().mean().item()
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
                    "area_fraction": area,
                }
    summary = dict(
        num_samples=len(scores),
        mean_iou=sum(scores.values()) / len(scores),
        **global_iou_summary(totals),
        dataset=args.dataset,
        view=args.view,
        base_mean_iou=sum(base_scores.values()) / len(base_scores),
        base_metrics=global_iou_summary(base_totals),
        groups={
            g: {
                "n": len(names),
                "mean_iou": sum(scores[n] for n in names) / len(names),
                "base_mean_iou": sum(base_scores[n] for n in names) / len(names),
            }
            for g, names in groups.items()
        },
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=file_hash(args.checkpoint),
        epoch=saved["epoch"],
        experiment_config=model.experiment_config,
        total_parameters=sum(p.numel() for p in model.parameters()),
        added_crop_parameters=sum(p.numel() for p in model.crop.parameters())
        if hasattr(model, "crop")
        else 0,
        cache_protocol=data.cache_protocol,
        cache_digest=data.cache_digest,
        shuffled_text_changed_regions=changed,
        metric_resolution="original",
        text_mode="question",
        threshold_logit=0,
        empty_union_iou=1,
        inference_precision="float32",
        cached_grounding_seconds_per_image=elapsed / measured if measured else None,
        timing_note="Excludes OCR, data loading, external batch transfer, metrics and first batch; not end-to-end latency",
    )
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
