"""Read-only Q/A interventions and representation probes for latent checkpoints.

Run from the repository root. Donor annotations are substituted only in memory;
scores for mismatched inputs remain comparisons against the ORIGINAL GT.
"""
import argparse
import csv
import hashlib
import html
import json
import os
import random
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from dataset import VizWizGroundingDataset, make_loader
from diagnose import load_diagnostic_model
from metrics import metric_samples
from models.text_encoder import TextEncoder, masked_mean
from utils import to_device

ROOT = Path(__file__).resolve().parent

REFERENCE_MODEL = "openai/clip-vit-large-patch14-336"
MODES = ("predicted_answer", "repeat", "question", "oracle_answer", "shuffled_answer", "shuffled_question")
PAIRS = {
    "repeat_vs_predicted": ("predicted_answer", "repeat", None),
    "answer_intervention": ("oracle_answer", "shuffled_answer", "answer_probe_available"),
    "question_intervention": ("predicted_answer", "shuffled_question", "question_probe_available"),
    "predicted_vs_question": ("question", "predicted_answer", None),
    "oracle_vs_predicted": ("predicted_answer", "oracle_answer", "answer_available"),
}


def normalize(text):
    return " ".join(str(text or "").casefold().split())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "min": float(values.min()), "max": float(values.max())}


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def donor_mapping(records, seed):
    """Fixed global sampling WITH replacement; different labels, independent of batches.

    A true permutation may be impossible when a label occupies over half the split.
    Different answer strings are not guaranteed to be semantically false answers.
    """
    names = sorted(records)
    result = {name: {} for name in names}
    for offset, field in enumerate(("question", "most_common_answer")):
        rng = random.Random(seed + offset)
        labels = {name: normalize(records[name].get(field)) for name in names}
        for name in names:
            candidates = [other for other in names if labels[other] and labels[other] != labels[name]] if labels[name] else []
            key = "question_donor" if field == "question" else "answer_donor"
            result[name][key] = rng.choice(candidates) if candidates else None
    return result


def relative_norm(delta, original):
    return delta.flatten(1).norm(dim=1) / original.flatten(1).norm(dim=1).clamp_min(1e-12)


@contextmanager
def capture_conditioning(core):
    """Hooks observe existing activations only and are always removed."""
    captured = {}

    def attention(module, inputs, output):
        visual, attended = inputs[0].float(), output[0].float()
        captured["attention_relative_norm"] = relative_norm(core.residual_scale.float() * attended, visual).cpu()

    def gate(module, inputs, output):
        captured["gate"] = output.sigmoid().float().cpu()

    def affine(index):
        def hook(module, inputs, output):
            captured[f"affine_{index}"] = output.float().cpu()
        return hook

    def skips(module, inputs, output):
        before = torch.cat([x.float().flatten(1) for x in inputs[0]], dim=1)
        after = torch.cat([x.float().flatten(1) for x in output], dim=1)
        captured["skip_effect_relative_norm"] = relative_norm(after - before, before).cpu()
        captured["conditioned_skips"] = after.cpu()

    conditioner = core.decoder.conditioner
    handles = [core.cross_attn.register_forward_hook(attention),
               conditioner.qa_gate.register_forward_hook(gate), conditioner.register_forward_hook(skips)]
    handles.extend(head.register_forward_hook(affine(i)) for i, head in enumerate(conditioner.affine))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def probe_batch(model, batch, records, mapping):
    if model.training:
        raise ValueError("Probe requires model.eval()")
    core = model.coarse if hasattr(model, "coarse") else model
    features = core.encode_image(batch["image"])
    question = core.text_encoder(batch["question_text"])
    q = masked_mean(question.tokens, question.valid_mask)
    answer_features = core.text_encoder(batch["answer_only_text"])
    answer = masked_mean(answer_features.tokens, answer_features.valid_mask)
    donor_answers, donor_questions = [], []
    for i, name in enumerate(batch["filename"]):
        donors = mapping[name]
        a = records[donors["answer_donor"]]["most_common_answer"] if donors["answer_donor"] else None
        question_text = records[donors["question_donor"]]["question"] if donors["question_donor"] else None
        donor_answers.append(f"A: {str(a).strip()}" if a is not None else batch["answer_only_text"][i])
        donor_questions.append(f"Q: {str(question_text).strip()}" if question_text is not None else batch["question_text"][i])
    answer_features = core.text_encoder(donor_answers)
    shuffled_answer = masked_mean(answer_features.tokens, answer_features.valid_mask)
    shuffled_question = core.text_encoder(donor_questions)
    outputs, traces = {}, {}
    calls = {
        "predicted_answer": lambda: core.decode(features, question, question_vec=q),
        "repeat": lambda: model(batch),  # ordinary full inference, including repeated image/text encoding
        "question": lambda: core.decode(features, question, question_vec=q, question_only=True),
        "oracle_answer": lambda: core.decode(features, question, answer, question_vec=q,
                                              answer_available=batch["answer_available"]),
        "shuffled_answer": lambda: core.decode(features, question, shuffled_answer, question_vec=q,
                                                answer_available=batch["answer_available"]),
        "shuffled_question": lambda: core.decode(features, shuffled_question),  # recompute A' with changed Q
    }
    for mode in MODES:
        with capture_conditioning(core) as trace:
            outputs[mode] = calls[mode]()
        traces[mode] = trace
    return outputs, traces, answer


def trace_rows(traces, index):
    row = {}
    for mode, trace in traces.items():
        row[f"{mode}_attention_relative_norm"] = trace["attention_relative_norm"][index].item()
        row[f"{mode}_skip_effect_relative_norm"] = trace["skip_effect_relative_norm"][index].item()
        row[f"{mode}_mean_gate_toward_question"] = trace["gate"][index].mean().item()
    for pair, (base, changed, _) in PAIRS.items():
        a, b = traces[base], traces[changed]
        row[f"{pair}_gate_mean_abs_difference"] = (a["gate"][index] - b["gate"][index]).abs().mean().item()
        for component, half in (("gamma", 0), ("beta", 1)):
            delta = torch.cat([(a[f"affine_{i}"][index].chunk(2)[half] -
                                b[f"affine_{i}"][index].chunk(2)[half]) for i in range(3)])
            row[f"{pair}_{component}_mean_abs_difference"] = delta.abs().mean().item()
        original, intervention = a["conditioned_skips"][index], b["conditioned_skips"][index]
        row[f"{pair}_conditioned_skip_relative_difference"] = (
            (intervention - original).norm() / original.norm().clamp_min(1e-12)).item()
    return row


def structure(vectors, labels):
    x = vectors.detach().cpu().double()
    if not len(x):
        return {"n": 0, "status": "no_valid_answers"}
    centered = x - x.mean(0, keepdim=True)
    energy = torch.linalg.svdvals(centered).square() if len(x) > 1 else torch.zeros(1, dtype=x.dtype)
    probabilities = energy / energy.sum().clamp_min(1e-30)
    rank = float(torch.exp(-torch.xlogy(probabilities, probabilities).sum())) if energy.sum() > 1e-30 else 0.
    normalized = F.normalize(x, dim=1)
    similarities = normalized @ normalized.T
    different = torch.tensor([[a != b for b in labels] for a in labels], dtype=torch.bool)
    return {"n": len(x), "dimension": x.shape[1], "different_label_pair_cosine": stats(similarities[different].tolist()),
            "mean_vector_norm": x.mean(0).norm().item(), "individual_norm": stats(x.norm(dim=1).tolist()),
            "mean_centered_variance_per_dimension": centered.square().mean().item(),
            "centered_rms_l2": centered.square().sum(1).mean().sqrt().item(),
            "effective_rank_covariance_energy": rank}


def unique_prototypes(vectors, labels):
    unique = sorted(set(labels))
    lookup = {label: i for i, label in enumerate(unique)}
    ids = torch.tensor([lookup[label] for label in labels])
    prototypes = torch.stack([vectors[ids == i].mean(0) for i in range(len(unique))])
    return unique, ids, prototypes


def retrieval(latent, target, labels, mismatched_labels):
    if not len(labels):
        return {"n": 0, "status": "no_valid_answers"}
    unique, ids, prototypes = unique_prototypes(target, labels)
    lookup = {label: i for i, label in enumerate(unique)}
    result = {"n": len(labels), "num_unique_answers": len(unique),
              "uniform_class_chance": 1 / len(unique),
              "majority_class_baseline": max(labels.count(x) for x in unique) / len(labels), "spaces": {}}
    for space in ("raw", "target_centered"):
        center = target.mean(0, keepdim=True) if space == "target_centered" else 0
        scores = F.normalize(latent - center, dim=1) @ F.normalize(prototypes - center, dim=1).T
        top = scores.topk(min(5, len(unique)), dim=1).indices
        correct = scores[torch.arange(len(ids)), ids]
        ties = (scores - scores.max(1, keepdim=True).values).abs() <= 1e-6
        tie_credit = ties[torch.arange(len(ids)), ids].float() / ties.sum(1)
        wrong_scores = scores.clone()
        wrong_scores[torch.arange(len(ids)), ids] = -torch.inf
        mismatched = [(i, lookup[label]) for i, label in enumerate(mismatched_labels) if label in lookup]
        result["spaces"][space] = {
            "top1": (top[:, 0] == ids).float().mean().item(),
            "topk": (top == ids[:, None]).any(1).float().mean().item(), "k": top.shape[1],
            "tie_aware_top1": tie_credit.mean().item(), "tie_tolerance": 1e-6,
            "macro_top1": float(np.mean([(top[ids == i, 0] == i).float().mean().item() for i in range(len(unique))])),
            "positive_cosine": stats(correct.tolist()),
            "positive_minus_best_wrong": stats((correct - wrong_scores.max(1).values).tolist()) if len(unique) > 1 else stats([]),
            "positive_minus_mismatched": stats([float(correct[i] - scores[i, j]) for i, j in mismatched]),
            "argmax_ties_note": "Exact top1/topk may break ties arbitrarily; inspect tie-aware top1 and margins."}
    return result


def representation_report(latent, target, labels, mismatched_labels):
    if not len(labels):
        return {"status": "no_valid_answers"}
    unique, _, prototypes = unique_prototypes(target, labels)
    _, _, latent_prototypes = unique_prototypes(latent, labels)
    return {"labels_normalization": "casefold and whitespace; no synonym merging",
            "rank_definition": "exp entropy of normalized squared singular values of centered RAW vectors; zero variance -> 0",
            "sample_weighted": {"latent": structure(latent, labels), "target": structure(target, labels)},
            "unique_answer_weighted": {"latent_centroids": structure(latent_prototypes, unique),
                                       "target_centroids": structure(prototypes, unique)},
            "retrieval": retrieval(latent, target, labels, mismatched_labels)}


def pair_summaries(rows):
    summaries = {}
    for pair, (_, _, eligibility) in PAIRS.items():
        selected = [r for r in rows if eligibility is None or r[eligibility]]
        keys = [key for key in rows[0] if key.startswith(pair + "_") and key != pair + "_native_total_pixels"]
        summaries[pair] = {"n": len(selected), "metrics": {key[len(pair) + 1:]: stats([r[key] for r in selected]) for key in keys}}
        total = sum(r[pair + "_native_total_pixels"] for r in selected)
        changed = sum(r[pair + "_native_changed_pixels"] for r in selected)
        summaries[pair].update(native_changed_pixels=changed, native_total_pixels=total,
                               native_changed_pixel_fraction=changed / total if total else None)
    return summaries


def save_review(out, row, masks, dataset):
    directory = out / "review"
    directory.mkdir(exist_ok=True)
    stem = Path(row["filename"]).stem
    with Image.open(dataset.root / dataset.split / row["filename"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((320, 320))
    image.save(directory / f"{stem}_image.png")
    for mode, tensor in masks.items():
        foreground = Image.fromarray(tensor.cpu().numpy().astype(np.uint8) * 255).resize(image.size, Image.Resampling.NEAREST)
        color = (0, 200, 0) if mode == "gt" else (255, 0, 0)
        tint = Image.blend(image, Image.new("RGB", image.size, color), .4)
        Image.composite(tint, image, foreground).save(directory / f"{stem}_{mode}.png")


def write_gallery(out, rows, review_names):
    parts = ["<!doctype html><meta charset='utf-8'><title>Q/A interventions</title>",
             "<style>body{font:16px system-ui;margin:24px}.images{display:flex;flex-wrap:wrap;gap:12px}img{max-width:240px}figure{margin:0}</style>",
             "<h1>Seeded review cases</h1><p>Green: original GT. Red: predictions. Mismatched Q/A are interventions, not newly labeled examples. Display thumbnails do not affect metrics.</p>"]
    for row in rows:
        if row["filename"] not in review_names:
            continue
        stem = Path(row["filename"]).stem
        parts.append(f"<h2>{html.escape(row['filename'])}</h2><p>Q: {html.escape(row['question'])}<br>A: {html.escape(row['answer'])}"
                     f"<br>Donor Q: {html.escape(row['intervention_question'])}<br>Donor A: {html.escape(row['intervention_answer'])}</p><div class='images'>")
        for mode in ("image", "gt", "predicted_answer", "question", "oracle_answer", "shuffled_answer", "shuffled_question"):
            parts.append(f"<figure><img src='review/{stem}_{mode}.png'><figcaption>{mode}</figcaption></figure>")
        parts.append("</div>")
    (out / "gallery.html").write_text("\n".join(parts), encoding="utf-8")


@torch.inference_mode()
def pretrained_reference(labels, texts, batch_size, skip, tiny):
    if skip or tiny or not labels:
        return {"status": "skipped_by_flag" if skip else "not_applicable_tiny_checkpoint" if tiny else "no_valid_answers"}
    try:
        encoder = TextEncoder(model_name=REFERENCE_MODEL).cpu().float().eval()
        vectors = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Pretrained text reference (CPU)"):
            features = encoder(texts[start:start + batch_size])
            vectors.append(masked_mean(features.tokens, features.valid_mask).float().cpu())
        target = torch.cat(vectors)
        unique, _, prototypes = unique_prototypes(target, labels)
        return {"status": "complete", "model": REFERENCE_MODEL, "device": "cpu",
                "sample_weighted": structure(target, labels), "unique_answer_weighted": structure(prototypes, unique),
                "note": "Compare structure within each encoder space; no cross-encoder raw-coordinate retrieval."}
    except Exception as error:
        return {"status": "unavailable", "model": REFERENCE_MODEL, "error_type": type(error).__name__,
                "note": "Reference load/encoding failed; learned-encoder probes are still saved. Check model cache/network before repeating."}


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=["val"], default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-pretrained-reference", action="store_true", help="Skip the CPU pretrained-text reference; status remains explicit")
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
    device = torch.device(args.device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    model, metadata = load_diagnostic_model(args.checkpoint, device)
    core = model.coarse if hasattr(model, "coarse") else model
    dataset = VizWizGroundingDataset(args.data_root, args.dataset, metadata["image_size"])
    mapping = donor_mapping(dataset.records, args.seed)
    annotation = dataset.root / f"{dataset.split}_grounding.json"
    annotation_hash = file_digest(annotation)
    checkpoint_stat = (Path(args.checkpoint).stat().st_size, Path(args.checkpoint).stat().st_mtime_ns)
    before = state_digest(model)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "shuffle_mapping.json", {"seed": args.seed, "sampling": "with replacement, independent of batch size",
                "mismatched_answer_note": "Different normalized annotation string, not a verified false answer", "mapping": mapping})
    rows, conditioning, names, answers, target_texts, latents, targets = [], [], [], [], [], [], []
    review_names = set(random.Random(args.seed).sample(sorted(dataset.names), min(8, len(dataset))))
    for batch in tqdm(make_loader(dataset, args, args.batch_size), desc="Q/A conditioning probe", dynamic_ncols=True):
        batch = to_device(batch, device)
        outputs, traces, target = probe_batch(model, batch, dataset.records, mapping)
        logits = {mode: outputs[mode]["logits"].float() for mode in MODES}
        scores = {mode: list(metric_samples(value, batch)) for mode, value in logits.items()}
        for i, name in enumerate(batch["filename"]):
            record, donors = dataset.records[name], mapping[name]
            gt = batch["original_mask"][i][0].bool().to(device)
            native_masks = {mode: scores[mode][i][2] for mode in MODES}
            row = {"filename": name, "question": str(record.get("question") or ""), "answer": str(record.get("most_common_answer") or ""),
                   "answer_available": bool(batch["answer_available"][i]),
                   "answer_probe_available": donors["answer_donor"] is not None,
                   "question_probe_available": donors["question_donor"] is not None,
                   "answer_donor": donors["answer_donor"], "question_donor": donors["question_donor"],
                   "intervention_answer": str(dataset.records[donors["answer_donor"]]["most_common_answer"]) if donors["answer_donor"] else "",
                   "intervention_question": str(dataset.records[donors["question_donor"]]["question"]) if donors["question_donor"] else "",
                   "height": gt.shape[0], "width": gt.shape[1], "gt_area_fraction": gt.float().mean().item()}
            predicted_latent = outputs["predicted_answer"]["latent_answer"][i].float()
            changed_latent = outputs["shuffled_question"]["latent_answer"][i].float()
            row.update(answer_cosine=F.cosine_similarity(predicted_latent, target[i], dim=0).item() if row["answer_available"] else None,
                       latent_norm=predicted_latent.norm().item(),
                       target_answer_norm=target[i].norm().item() if row["answer_available"] else None,
                       shuffled_question_latent_cosine=F.cosine_similarity(predicted_latent, changed_latent, dim=0).item(),
                       shuffled_question_latent_mean_abs_difference=(predicted_latent - changed_latent).abs().mean().item())
            row.update({"iou_" + mode: scores[mode][i][1] for mode in MODES})
            for pair, (base, changed, _) in PAIRS.items():
                delta = (logits[changed][i] - logits[base][i]).abs()
                probability_delta = (logits[changed][i].sigmoid() - logits[base][i].sigmoid()).abs()
                pixel_count = (native_masks[base] != native_masks[changed]).sum().item()
                row.update({pair + "_iou_difference": row["iou_" + changed] - row["iou_" + base],
                            pair + "_logit_mean_abs_difference": delta.mean().item(),
                            pair + "_logit_max_abs_difference": delta.max().item(),
                            pair + "_probability_mean_abs_difference": probability_delta.mean().item(),
                            pair + "_probability_max_abs_difference": probability_delta.max().item(),
                            pair + "_native_changed_pixels": int(pixel_count), pair + "_native_total_pixels": gt.numel(),
                            pair + "_native_changed_fraction": pixel_count / gt.numel()})
            rows.append(row)
            conditioning.append(dict(filename=name, **trace_rows(traces, i)))
            if name in review_names:
                save_review(out, row, dict(native_masks, gt=gt), dataset)
            if row["answer_available"]:
                names.append(name)
                answers.append(normalize(row["answer"]))
                target_texts.append(batch["answer_only_text"][i])
                latents.append(outputs["predicted_answer"]["latent_answer"][i].float().cpu())
                targets.append(target[i].float().cpu())
    after = state_digest(model)
    if before != after or annotation_hash != file_digest(annotation) or checkpoint_stat != (
            Path(args.checkpoint).stat().st_size, Path(args.checkpoint).stat().st_mtime_ns):
        raise RuntimeError("Model, annotation or checkpoint metadata changed during diagnostic")
    latent = torch.stack(latents) if latents else torch.empty(0, core.text_dim)
    target = torch.stack(targets) if targets else torch.empty(0, core.text_dim)
    mismatched_labels = [normalize(dataset.records[mapping[name]["answer_donor"]]["most_common_answer"])
                         if mapping[name]["answer_donor"] else None for name in names]
    representation = representation_report(latent, target, answers, mismatched_labels)
    representation["pretrained_reference"] = pretrained_reference(answers, target_texts, args.batch_size,
                                                                 args.skip_pretrained_reference, core.text_encoder.tiny)
    torch.save({"filenames": names, "normalized_answers": answers, "latent": latent, "target": target,
                "note": "Analysis embeddings only; this is NOT a model checkpoint"}, out / "embeddings.pt")
    conditioning_summary = {"residual_scale": core.residual_scale.item(), "per_mode": {}, "pairs": {}}
    for mode in MODES:
        conditioning_summary["per_mode"][mode] = {
            suffix: stats([r[f"{mode}_{suffix}"] for r in conditioning]) for suffix in
            ("attention_relative_norm", "skip_effect_relative_norm", "mean_gate_toward_question")}
    for pair, (_, _, eligibility) in PAIRS.items():
        selected = [c for c, r in zip(conditioning, rows) if eligibility is None or r[eligibility]]
        conditioning_summary["pairs"][pair] = {key[len(pair) + 1:]: stats([r[key] for r in selected])
                                                for key in conditioning[0] if key.startswith(pair + "_")}
    valid_answers = [r for r in rows if r["answer_available"]]
    summary = dict(metadata, status="complete", dataset=args.dataset, num_samples=len(rows), num_with_answer=len(valid_answers),
        metric_resolution="original", threshold_logit=0, inference_precision="float32", batch_size=args.batch_size, seed=args.seed,
        device=str(device), torch_version=str(torch.__version__), cpu_threads=torch.get_num_threads(),
        cuda_device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        logit_difference_grid="decoder output grid BEFORE native resize, no threshold",
        means={mode: stats([r["iou_" + mode] for r in rows])["mean"] for mode in MODES},
        answer_available_subset={mode: stats([r["iou_" + mode] for r in valid_answers])["mean"] for mode in MODES},
        pairs=pair_summaries(rows), provenance={"git_commit": git_revision(), "script_sha256": file_digest(__file__),
            "model_py_sha256": file_digest(ROOT / "models/model.py"), "annotation_sha256": annotation_hash,
            "source_sha256": {name: file_digest(ROOT / name) for name in
                ("dataset.py", "metrics.py", "diagnose.py", "models/model.py", "models/text_encoder.py", "models/mask_decoder.py")},
            "model_state_sha256_before": before, "model_state_sha256_after": after, "weights_unchanged": before == after},
        interpretation=["Mismatched inputs are in-memory interventions, never new labeled triplets.",
            "Intervention IoU uses ORIGINAL GT and is not accuracy for a newly posed question.",
            "Inspect repeat_vs_predicted to separate repeat numerical differences from interventions.",
            "A different answer string can still be semantically valid; inspect donor mappings.",
            "Q-only is a same-checkpoint view, not an independently trained baseline.",
            "No automatic collapse verdict: use retrieval, margins, centered variance and pretrained reference together."])
    for filename, values in (("per_image.csv", rows), ("conditioning_per_image.csv", conditioning)):
        with (out / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    write_json(out / "representation_summary.json", representation)
    write_json(out / "conditioning_summary.json", conditioning_summary)
    write_json(out / "summary.json", summary)
    write_gallery(out, rows, review_names)
    print(json.dumps({"output_dir": str(out), "num_samples": len(rows), "means": summary["means"],
                      "weights_unchanged": before == after, "pretrained_reference": representation["pretrained_reference"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
