"""Paired Q-only / inferred-answer / real-answer diagnostics for latent checkpoints."""
import argparse
import csv
import html
import json
import math
import random
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from dataset import VizWizGroundingDataset, make_loader
from metrics import metric_samples
from models.model import CompactGroundingModel, GroundingModel
from models.text_encoder import masked_mean
from utils import is_tiny_state, load_model_weights, read_checkpoint, to_device

MODES = ("question", "predicted_answer", "oracle_answer")


def load_diagnostic_model(path, device):
    saved = read_checkpoint(path)
    state, config = saved["model_state_dict"], saved["run_config"]
    if config.get("teacher_training"):
        raise ValueError("Use a latent student checkpoint, not an external teacher")
    model_type = GroundingModel if any(k.startswith("coarse.") for k in state) else CompactGroundingModel
    model = model_type(tiny=config.get("tiny", is_tiny_state(state)))
    load_model_weights(model, saved)
    model.to(device=device, dtype=torch.float32).eval()
    metadata = {"checkpoint": str(path), "epoch": saved.get("epoch"),
                "objective_version": config.get("objective_version", "unknown (not stored)"),
                "image_size": config.get("image_size", 336)}
    return model, metadata


@torch.no_grad()
def diagnostic_views(model, batch):
    """Reuse encodings; all decoder calls use eval mode and the checkpoint's weights."""
    if model.training:
        raise ValueError("Diagnostic requires model.eval()")
    core = model.coarse if hasattr(model, "coarse") else model
    features = core.encode_image(batch["image"])
    text = core.text_encoder(batch["question_text"])
    q = masked_mean(text.tokens, text.valid_mask)
    predicted = core.decode(features, text, question_vec=q)
    question = core.decode(features, text, question_vec=q, question_only=True)
    target_features = core.text_encoder(batch["answer_only_text"])
    target = masked_mean(target_features.tokens, target_features.valid_mask)
    oracle = core.decode(features, text, target, question_vec=q,
                         answer_available=batch["answer_available"])
    latent = predicted["latent_answer"].float()
    target = target.float()
    return {
        "logits": {"question": question["logits"], "predicted_answer": predicted["logits"],
                   "oracle_answer": oracle["logits"]},
        "cosine": F.cosine_similarity(latent, target, dim=-1, eps=1e-6),
        "smooth_l1": F.smooth_l1_loss(latent, target, reduction="none").mean(-1),
        "latent_norm": latent.norm(dim=-1), "target_norm": target.norm(dim=-1),
    }


def save_mask(mask, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.cpu().numpy().astype(np.uint8) * 255).save(path)


def area_group(fraction):
    if fraction == 0:
        return "empty"
    if fraction < 0.01:
        return "small_lt_1pct"
    if fraction < 0.1:
        return "medium_1_to_10pct"
    return "large_ge_10pct"


def paired_rows(views, batch, records, mask_output=None):
    scores = {}
    for mode, logits in views["logits"].items():
        scores[mode] = {}
        for i, (name, score, mask) in enumerate(metric_samples(logits, batch)):
            available = bool(batch["answer_available"][i])
            scores[mode][name] = score if mode != "oracle_answer" or available else None
            if mask_output is not None and (mode != "oracle_answer" or available):
                save_mask(mask, mask_output / mode / Path(name).with_suffix(".png"))
    rows = []
    for i, name in enumerate(batch["filename"]):
        available = bool(batch["answer_available"][i])
        truth = batch["original_mask"][i]
        h, w = truth.shape[-2:]
        fraction = truth.float().mean().item()
        q, p, a = (scores[mode][name] for mode in MODES)
        rows.append({"filename": name, "question": str(records[name].get("question", "")),
                     "answer": str(records[name].get("most_common_answer") or ""),
                     "answer_available": available, "height": h, "width": w,
                     "gt_area_fraction": fraction, "area_group": area_group(fraction),
                     "aspect_ratio": max(h / w, w / h),
                     "iou_question": q, "iou_predicted_answer": p, "iou_oracle_answer": a,
                     "predicted_minus_question": p - q,
                     "oracle_minus_predicted": a - p if available else None,
                     "oracle_minus_question": a - q if available else None,
                     "answer_cosine": views["cosine"][i].item() if available else None,
                     "answer_smooth_l1": views["smooth_l1"][i].item() if available else None,
                     "latent_norm": views["latent_norm"][i].item(),
                     "target_answer_norm": views["target_norm"][i].item() if available else None})
    return rows


def average(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return mean(values) if values else None


def pearson(rows, x, y):
    pairs = [(r[x], r[y]) for r in rows if r[x] is not None and r[y] is not None]
    if len(pairs) < 2:
        return None
    mx, my = mean(p[0] for p in pairs), mean(p[1] for p in pairs)
    covariance = sum((a - mx) * (b - my) for a, b in pairs)
    scale = math.sqrt(sum((a - mx) ** 2 for a, _ in pairs) * sum((b - my) ** 2 for _, b in pairs))
    return covariance / scale if scale else None


def score_summary(rows):
    paired = [r for r in rows if r["answer_available"]]
    return {"num_samples": len(rows), "num_with_answer": len(paired),
            "all_samples": {"mean_iou_question": average(rows, "iou_question"),
                            "mean_iou_predicted_answer": average(rows, "iou_predicted_answer"),
                            "mean_predicted_minus_question": average(rows, "predicted_minus_question")},
            "answer_available_subset": {key: average(paired, key) for key in
                ("iou_question", "iou_predicted_answer", "iou_oracle_answer", "predicted_minus_question",
                 "oracle_minus_predicted", "oracle_minus_question", "answer_cosine", "answer_smooth_l1")},
            "predicted_helps_count": sum(r["predicted_minus_question"] > 1e-6 for r in rows),
            "predicted_hurts_count": sum(r["predicted_minus_question"] < -1e-6 for r in rows),
            "predicted_ties_count": sum(abs(r["predicted_minus_question"]) <= 1e-6 for r in rows),
            "cosine_vs_predicted_iou_pearson": pearson(paired, "answer_cosine", "iou_predicted_answer"),
            "cosine_vs_predicted_gain_pearson": pearson(paired, "answer_cosine", "predicted_minus_question")}


def select_cases(rows, top_k, seed):
    def ranked(key, eligible, reverse):
        return [r["filename"] for r in sorted((r for r in rows if eligible(r)),
                key=lambda r: (r[key], r["filename"]), reverse=reverse)[:top_k]]
    return {
        "predicted_helps": ranked("predicted_minus_question", lambda r: r["predicted_minus_question"] > 1e-6, True),
        "predicted_hurts": ranked("predicted_minus_question", lambda r: r["predicted_minus_question"] < -1e-6, False),
        "oracle_gap": ranked("oracle_minus_predicted", lambda r: r["answer_available"] and r["oracle_minus_predicted"] > 1e-6, True),
        "small_gt_failures": ranked("iou_predicted_answer", lambda r: 0 < r["gt_area_fraction"] < .01 and r["iou_predicted_answer"] < .5, False),
        "high_alignment_low_iou": ranked("answer_cosine", lambda r: r["answer_available"] and r["answer_cosine"] >= .8 and r["iou_predicted_answer"] < .5, True),
        "random": random.Random(seed).sample([r["filename"] for r in rows], min(top_k, len(rows))),
    }


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_gallery(out, selections, rows, summary):
    by_name = {r["filename"]: r for r in rows}
    parts = ["<!doctype html><meta charset='utf-8'><title>Latent answer diagnostic</title>",
             "<style>body{font:16px system-ui;margin:24px;background:#f6f6f6}article{background:white;padding:16px;margin:16px 0}"
             ".images{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0}img{max-width:280px;max-height:280px}"
             "a{color:#1556a3}p{overflow-wrap:anywhere}</style>",
             "<h1>Latent answer diagnostic</h1><p>Metrics use the whole split. Ranked examples are selected cases, "
             "not estimates of failure frequency. Oracle uses a real answer and is unavailable at Q/I inference. "
             "Q-only here is a same-checkpoint counterfactual, not a separately trained baseline.</p>",
             "<p>Green overlay: ground truth. Red overlay: each predicted mask.</p>",
             "<p>Review rules: small GT &lt; 1% and predicted IoU &lt; 0.5; high-alignment cases use cosine ≥ 0.8 "
             "and predicted IoU &lt; 0.5. These are descriptive thresholds, not answer-correctness criteria.</p>",
             f"<p>Checkpoint: {html.escape(summary['checkpoint'])}<br>Objective: {html.escape(summary['objective_version'])}</p>"]
    parts.append("<table border='1' cellpadding='8'><tr><th>Subset</th><th>N</th><th>Q-only mIoU</th>"
                 "<th>Predicted-answer mIoU</th><th>Oracle mIoU</th></tr>")
    subsets = [("Whole split", len(rows), rows, False),
               ("Same answer-available subset", summary["num_with_answer"],
                [r for r in rows if r["answer_available"]], True)]
    for label, count, subset, has_oracle in subsets:
        values = [average(subset, "iou_" + mode) if mode != "oracle_answer" or has_oracle else None for mode in MODES]
        cells = "".join(f"<td>{value:.2%}</td>" if value is not None else "<td>—</td>" for value in values)
        parts.append(f"<tr><td>{label}</td><td>{count}</td>{cells}</tr>")
    parts.append("</table><nav>" + " | ".join(f"<a href='#{html.escape(g)}'>{html.escape(g)}</a>" for g in selections) + "</nav>")
    for group, names in selections.items():
        parts.append(f"<h2 id='{html.escape(group)}'>{html.escape(group)}</h2>")
        for name in names:
            row = by_name[name]
            identifier = row["review_id"]
            metrics = ", ".join(f"{mode}: {row['iou_' + mode]:.4f}" if row["iou_" + mode] is not None
                                else f"{mode}: unavailable" for mode in MODES)
            cosine = f"{row['answer_cosine']:.4f}" if row["answer_cosine"] is not None else "unavailable"
            parts.append(f"<article><h3>{html.escape(name)}</h3><p>Q: {html.escape(row['question'])}<br>"
                         f"A (annotation): {html.escape(row['answer'])}</p><p>{metrics}<br>"
                         f"A′ gain: {row['predicted_minus_question']:+.4f}; cosine: {cosine}; "
                         f"GT fraction: {row['gt_area_fraction']:.4%}</p><div class='images'>")
            panels = ["image", "gt", "question", "predicted_answer"]
            if row["answer_available"]:
                panels.append("oracle_answer")
            for panel in panels:
                parts.append(f"<figure><img loading='lazy' src='cases/{identifier}_{panel}.png'>"
                             f"<figcaption>{panel}</figcaption></figure>")
            parts.append("</div></article>")
    (out / "gallery.html").write_text("\n".join(parts), encoding="utf-8")


@torch.no_grad()
def export_review_cases(model, dataset, rows, selections, out, device, args):
    selected = set(name for names in selections.values() for name in names)
    cases = out / "cases"
    cases.mkdir(exist_ok=True)
    by_name = {r["filename"]: r for r in rows}
    for index, row in enumerate(rows):
        row["review_id"] = f"{index:05d}" if row["filename"] in selected else ""
    if not selected:
        return
    # Preserve batch composition/padding from scoring, rather than rerunning selected
    # images alone and potentially changing floating-point threshold decisions.
    for batch in tqdm(make_loader(dataset, args, args.batch_size), desc="Review images", dynamic_ncols=True):
        if not selected.intersection(batch["filename"]):
            continue
        batch = to_device(batch, device)
        views = diagnostic_views(model, batch)
        predictions = {}
        for mode, logits in views["logits"].items():
            predictions[mode] = {name: mask.cpu() for name, _, mask in metric_samples(logits, batch)}
        for i, name in enumerate(batch["filename"]):
            if name not in selected:
                continue
            row = by_name[name]
            # Resize only for display; native masks and metrics remain unchanged.
            with Image.open(dataset.root / dataset.split / name) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((384, 384))
            prefix = cases / row["review_id"]
            image.save(f"{prefix}_image.png")
            masks = {"gt": batch["original_mask"][i][0].bool()}
            masks.update({mode: predictions[mode][name] for mode in MODES
                          if mode != "oracle_answer" or row["answer_available"]})
            for panel, mask in masks.items():
                mask_image = Image.fromarray(mask.numpy().astype(np.uint8) * 255)
                mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)
                overlay = Image.new("RGB", image.size, (0, 200, 0) if panel == "gt" else (255, 0, 0))
                tinted = Image.blend(image, overlay, .4)
                Image.composite(tinted, image, mask_image).save(f"{prefix}_{panel}.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/vizwiz")
    parser.add_argument("--dataset", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=8, help="Examples per ranked group and random group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-masks", action="store_true", help="Export all native masks, not only review overlays")
    parser.add_argument("--review-list", help="Extra filenames to visualize, one per line; whole-split scoring is unchanged")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0 or args.top_k < 0:
        parser.error("Batch must be positive; workers/top-k must be nonnegative")
    device = torch.device(args.device)
    model, metadata = load_diagnostic_model(args.checkpoint, device)
    dataset = VizWizGroundingDataset(args.data_root, args.dataset, metadata["image_size"])
    extra = []
    if args.review_list:
        extra = list(dict.fromkeys(line.strip() for line in Path(args.review_list).read_text(encoding="utf-8").splitlines()
                                  if line.strip() and not line.strip().startswith("#")))
        unknown = set(extra) - set(dataset.names)
        if unknown:
            parser.error(f"Review names not in {args.dataset}: {sorted(unknown)}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Prevent stale masks/gallery panels being mixed with another checkpoint's output.
    if any(out.iterdir()):
        parser.error("Diagnostic output directory must be empty; use a new directory")
    rows = []
    model.eval()
    for batch in tqdm(make_loader(dataset, args, args.batch_size), desc="Paired diagnostic", dynamic_ncols=True):
        batch = to_device(batch, device)
        views = diagnostic_views(model, batch)
        rows.extend(paired_rows(views, batch, dataset.records, out / "masks" if args.save_masks else None))
    selections = select_cases(rows, args.top_k, args.seed)
    if extra:
        selections["manual_review"] = extra
    summary = {**metadata, "dataset": args.dataset, "metric_resolution": "original",
               "inference_precision": "float32", "threshold_logit": 0, "empty_union_iou": 1, "batch_size": args.batch_size,
               "selection_seed": args.seed, "top_k": args.top_k, **score_summary(rows),
               "by_gt_area": {g: score_summary([r for r in rows if r["area_group"] == g])
                              for g in ("empty", "small_lt_1pct", "medium_1_to_10pct", "large_ge_10pct")},
               "interpretation": ["Oracle is a real-answer counterfactual, not deployable Q/I inference.",
                   "Question mode is the same checkpoint without an answer token, not a trained Q-only baseline.",
                   "Alignment uses this checkpoint's answer encoder in eval mode; cosine is not answer accuracy.",
                   "Ranked cases do not estimate error prevalence; aggregate metrics use the whole split.",
                   "Unknown objective_version must not be assumed to be latent-gain-v2."]}
    write_json(out / "summary.json", summary)
    write_json(out / "selected_cases.json", selections)
    export_review_cases(model, dataset, rows, selections, out, device, args)
    write_gallery(out, selections, rows, summary)
    with (out / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "per_image.json", rows)
    print(json.dumps({"output_dir": str(out), **score_summary(rows)}, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
