"""Test whether grounding depends on the content of the predicted answer vector.

Image and question stay fixed. Every answer intervention keeps the answer-token
slot and uses the replacement for both cross-attention and skip conditioning.
"""
import argparse
import csv
import html
import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from dataset import VizWizGroundingDataset, make_loader
from diagnose import load_diagnostic_model
from metrics import metric_samples
from probe_latent_conditioning import (
    capture_conditioning, donor_mapping, file_digest, git_revision, relative_norm,
    save_review, state_digest, stats, write_json,
)
from utils import to_device


MODES = ("predicted_answer", "cached_replay", "question", "shuffled_latent",
         "different_answer_latent", "mean_latent", "zero_latent")
ROOT = Path(__file__).resolve().parent


def latent_donors(records, seed):
    """One global derangement, plus a distinct-label control sampled with replacement."""
    names = sorted(records)
    random.Random(seed).shuffle(names)
    different = donor_mapping(records, seed)
    return {name: {
        "shuffled_latent": names[(i + 1) % len(names)] if len(names) > 1 else None,
        "different_answer_latent": different[name]["answer_donor"],
    } for i, name in enumerate(names)}


@torch.inference_mode()
def cache_latents(model, loader, device):
    """Cache predicted vectors only, not image features or ground-truth answer vectors."""
    if model.training:
        raise ValueError("Latent cache requires model.eval()")
    core = model.coarse if hasattr(model, "coarse") else model
    bank = {}
    for batch in tqdm(loader, desc="Caching predicted answers", unit="batch"):
        batch = to_device(batch, device)
        features = core.encode_image(batch["image"])
        _, q = core.encode_question(batch["question_text"])
        latent, _ = core.infer_latent_answer(features[-1], q)
        for name, vector in zip(batch["filename"], latent):
            if name in bank:
                raise ValueError(f"Duplicate cache filename: {name}")
            bank[name] = vector.float().cpu().clone()
    if not bank:
        raise ValueError("Cannot build a latent cache from an empty dataset")
    mean = torch.stack([bank[name] for name in sorted(bank)]).mean(0)
    return bank, mean


@torch.inference_mode()
def intervention_views(model, batch, bank, mean, mapping):
    """Reuse I/Q encodings; replace only A-prime, including its skip-conditioning input."""
    if model.training:
        raise ValueError("Interventions require model.eval()")
    core = model.coarse if hasattr(model, "coarse") else model
    features = core.encode_image(batch["image"])
    text, q = core.encode_question(batch["question_text"])
    names = batch["filename"]
    own = torch.stack([bank[name] for name in names]).to(q.device)
    replacements = {"cached_replay": own,
                    "mean_latent": mean.to(q.device).unsqueeze(0).expand_as(own),
                    "zero_latent": torch.zeros_like(own)}
    for mode in ("shuffled_latent", "different_answer_latent"):
        replacements[mode] = torch.stack([bank[mapping[name][mode] or name] for name in names]).to(q.device)
    outputs, traces = {}, {}
    for mode in MODES:
        with capture_conditioning(core) as trace:
            if mode == "predicted_answer":
                outputs[mode] = core.decode(features, text, question_vec=q)
            elif mode == "question":
                outputs[mode] = core.decode(features, text, question_vec=q, question_only=True)
            else:
                # No answer_available mask: the slot stays present even for a zero vector.
                outputs[mode] = core.decode(features, text, replacements[mode], question_vec=q)
        traces[mode] = trace
    return outputs, traces, replacements


def summarize(rows):
    """Paired statistics; exclude unavailable donors instead of counting identity fallbacks."""
    pairs = {}
    for mode in MODES[1:]:
        selected = [r for r in rows if r[f"{mode}_available"]]
        keys = ("iou_delta", "iou_abs_delta", "logit_mean_abs_delta", "logit_max_abs_delta", "probability_mean_abs_delta",
                "native_changed_fraction", "input_cosine", "input_norm_ratio", "gate_mean_abs_delta",
                "conditioned_skip_relative_delta")
        pairs[mode] = {"n": len(selected), "metrics": {
            key: stats([r[f"{mode}_{key}"] for r in selected if r[f"{mode}_{key}"] is not None]) for key in keys},
            "wins": sum(r[f"{mode}_iou_delta"] > 0 for r in selected),
            "losses": sum(r[f"{mode}_iou_delta"] < 0 for r in selected),
            "abs_iou_delta_gt_0.1pp": sum(abs(r[f"{mode}_iou_delta"]) > .001 for r in selected),
            "abs_iou_delta_gt_1pp": sum(abs(r[f"{mode}_iou_delta"]) > .01 for r in selected),
            "native_changed_pixels": sum(r[f"{mode}_native_changed_pixels"] for r in selected),
            "native_total_pixels": sum(r["native_total_pixels"] for r in selected)}
    return {"num_samples": len(rows),
            "means": {mode: stats([r[f"iou_{mode}"] for r in rows])["mean"] for mode in MODES}, "pairs": pairs}


def write_gallery(out, rows, selected):
    parts = ["<!doctype html><meta charset='utf-8'><title>Predicted-answer interventions</title>",
             "<style>body{font:16px system-ui;margin:24px}.images{display:flex;flex-wrap:wrap;gap:12px}img{max-width:240px}figure{margin:0}</style>",
             "<h1>Predicted-answer content probes</h1><p>Image and question stay fixed. Green: original GT; red: predictions.</p>"]
    for row in rows:
        if row["filename"] not in selected:
            continue
        parts.append(f"<h2>{html.escape(row['filename'])}</h2><p>Q: {html.escape(row['question'])}"
                     f"<br>A annotation (analysis only): {html.escape(row['answer'])}"
                     f"<br>Latent donor: {html.escape(row['shuffled_latent_donor'] or 'unavailable')}"
                     f"<br>Different-answer donor: {html.escape(row['different_answer_latent_donor'] or 'unavailable')}</p><div class='images'>")
        stem = Path(row["filename"]).stem
        for mode in ("image", "gt", *MODES):
            parts.append(f"<figure><img src='review/{stem}_{mode}.png'><figcaption>{mode}</figcaption></figure>")
        parts.append("</div>")
    (out / "gallery.html").write_text("\n".join(parts), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use one process, not torchrun")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Batch size must be positive and workers nonnegative")
    if not Path(args.checkpoint).is_file():
        parser.error("Checkpoint file does not exist")
    out = Path(args.output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error("Output directory must be new or empty")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, metadata = load_diagnostic_model(args.checkpoint, device)
    dataset = VizWizGroundingDataset(args.data_root, "val", metadata["image_size"])
    annotation = dataset.root / "val_grounding.json"
    annotation_hash = file_digest(annotation)
    checkpoint_stat = (Path(args.checkpoint).stat().st_size, Path(args.checkpoint).stat().st_mtime_ns)
    before = state_digest(model)
    loader = make_loader(dataset, args, args.batch_size)
    bank, mean = cache_latents(model, loader, device)
    if set(bank) != set(dataset.names):
        raise ValueError("Latent cache does not cover the validation dataset")
    mapping = latent_donors(dataset.records, args.seed)
    selected = set(random.Random(args.seed).sample(sorted(bank), min(8, len(bank))))
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "donor_mapping.json", {"seed": args.seed, "mapping": mapping,
        "shuffled_latent_policy": "Seeded global cyclic derangement, same-answer donors allowed",
        "different_answer_latent_policy": "Different normalized annotation, sampled with replacement",
        "mean_policy": "One fixed label-free mean over predicted validation latents; diagnostic only"})
    rows = []
    for batch in tqdm(loader, desc="Predicted-answer interventions", unit="batch"):
        batch = to_device(batch, device)
        outputs, traces, replacements = intervention_views(model, batch, bank, mean, mapping)
        logits = {mode: outputs[mode]["logits"].float() for mode in MODES}
        scores = {mode: list(metric_samples(value, batch)) for mode, value in logits.items()}
        for i, name in enumerate(batch["filename"]):
            record = dataset.records[name]
            truth = batch["original_mask"][i][0].bool()
            masks = {mode: scores[mode][i][2] for mode in MODES}
            original = outputs["predicted_answer"]["latent_answer"][i].float()
            row = {"filename": name, "question": str(record.get("question") or ""),
                   "answer": str(record.get("most_common_answer") or ""),
                   "gt_area_fraction": truth.float().mean().item(), "native_total_pixels": truth.numel()}
            row.update({f"iou_{mode}": scores[mode][i][1] for mode in MODES})
            for mode in MODES[1:]:
                donor = mapping[name].get(mode)
                available = donor is not None if mode in ("shuffled_latent", "different_answer_latent") else True
                delta = (logits[mode][i] - logits["predicted_answer"][i]).abs()
                changed = (masks[mode] != masks["predicted_answer"]).sum().item()
                replacement = replacements[mode][i].float() if mode in replacements else None
                trace, base_trace = traces[mode], traces["predicted_answer"]
                row.update({f"{mode}_available": available, f"{mode}_donor": donor,
                    f"{mode}_iou_delta": row[f"iou_{mode}"] - row["iou_predicted_answer"],
                    f"{mode}_iou_abs_delta": abs(row[f"iou_{mode}"] - row["iou_predicted_answer"]),
                    f"{mode}_logit_mean_abs_delta": delta.mean().item(), f"{mode}_logit_max_abs_delta": delta.max().item(),
                    f"{mode}_probability_mean_abs_delta": (logits[mode][i].sigmoid() - logits["predicted_answer"][i].sigmoid()).abs().mean().item(),
                    f"{mode}_native_changed_pixels": int(changed), f"{mode}_native_changed_fraction": changed / truth.numel(),
                    f"{mode}_input_cosine": F.cosine_similarity(original, replacement, dim=0).item()
                        if replacement is not None and replacement.norm().item() > 1e-12 and original.norm().item() > 1e-12 else None,
                    f"{mode}_input_norm_ratio": (replacement.norm() / original.norm().clamp_min(1e-12)).item() if replacement is not None else None,
                    f"{mode}_gate_mean_abs_delta": (trace["gate"][i] - base_trace["gate"][i]).abs().mean().item(),
                    f"{mode}_conditioned_skip_relative_delta": relative_norm(
                        trace["conditioned_skips"][i:i+1] - base_trace["conditioned_skips"][i:i+1],
                        base_trace["conditioned_skips"][i:i+1]).item()})
            rows.append(row)
            if name in selected:
                save_review(out, row, dict(masks, gt=truth), dataset)
    after = state_digest(model)
    if before != after or annotation_hash != file_digest(annotation) or checkpoint_stat != (
            Path(args.checkpoint).stat().st_size, Path(args.checkpoint).stat().st_mtime_ns):
        raise RuntimeError("Model, annotations or checkpoint metadata changed during diagnostic")
    summary = dict(metadata, status="complete", dataset="val", metric_resolution="original",
        threshold_logit=0, inference_precision="float32", seed=args.seed, batch_size=args.batch_size,
        device=str(device), torch_version=str(torch.__version__),
        cuda_device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        **summarize(rows))
    summary["by_gt_area"] = {label: summarize([r for r in rows if low <= r["gt_area_fraction"] < high])
        for label, low, high in (("small_lt_1pct", 0, .01), ("medium_1_to_10pct", .01, .1), ("large_ge_10pct", .1, 1.01))}
    summary["cache_replay_max_abs_logit_delta"] = max(r["cached_replay_logit_max_abs_delta"] for r in rows)
    summary["cache_replay_changed_pixels"] = sum(r["cached_replay_native_changed_pixels"] for r in rows)
    summary["provenance"] = {"git_commit": git_revision(), "script_sha256": file_digest(__file__),
        "source_sha256": {name: file_digest(ROOT / name) for name in
            ("models/model.py", "models/text_encoder.py", "models/mask_decoder.py", "dataset.py", "metrics.py", "diagnose.py", "probe_latent_conditioning.py")},
        "annotation_sha256": annotation_hash, "model_state_sha256_before": before,
        "model_state_sha256_after": after, "weights_unchanged": before == after}
    summary["interpretation"] = [
        "Check cached_replay before interpreting intervention effects; numerical mismatch is reported, never hidden.",
        "All answer replacements preserve the slot and replace both attention and skip inputs; question mode removes the slot.",
        "Donor latent comes from the donor image/question, not the donor GT answer encoding.",
        "The unconstrained shuffle preserves the cached latent multiset; distinct-label sampling need not.",
        "Mean is one fixed label-free validation-cache vector, not a deployable trained constant.",
        "Zero/mean and swapped I/Q/latent combinations can be out of distribution; sensitivity alone is not semantic correctness.",
        "Different normalized answer strings are not verified semantic negatives. All scores use original GT."]
    with (out / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "summary.json", summary)
    write_gallery(out, rows, selected)
    print(json.dumps({"output_dir": str(out), "means": summary["means"],
        "cache_replay_max_abs_logit_delta": summary["cache_replay_max_abs_logit_delta"],
        "cache_replay_changed_pixels": summary["cache_replay_changed_pixels"],
        "weights_unchanged": before == after}, indent=2))


if __name__ == "__main__":
    main()
