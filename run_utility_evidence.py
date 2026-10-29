"""Matched utility heads: train on parent-unseen fit, select on tune, audit on safety."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from tqdm.auto import tqdm

from models.utility_evidence import UtilityEvidence
from ocr_cache import atomic_json, file_hash, read_json
from train import seed_everything
from training.utility_evidence_objective import utility_objective
from utility_evidence import (
    THRESHOLDS,
    VERSION,
    UtilityDataset,
    certify,
    choose_actions,
    collate_utility,
    render_action,
    unpack_mask,
    validate_manifest,
)
from utils import save_checkpoint, to_device

RECIPES = ("gain", "risk")


def make_loader(root, names, args, training=False, labels=False):
    data = UtilityDataset(root, names, training or labels)
    sampler = RandomSampler(data, generator=torch.Generator()) if training else None
    return DataLoader(
        data,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_utility,
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed + 7919),
    )


@torch.inference_mode()
def decision_table(model, cache, names, args):
    model.eval()
    rows = []
    for batch, target in make_loader(cache, names, args, labels=True):
        scores = model(to_device(batch, args.device)).cpu()
        for i, name in enumerate(batch["filename"]):
            n = int(batch["valid"][i].sum())
            rows.append(
                {
                    "image": name,
                    "scores": scores[i, :n].tolist(),
                    "quality": target["quality"][i, :n].tolist(),
                    "quality_counts": target["quality_counts"][i, :n].tolist(),
                    "area_fraction": float(target["area_fraction"][i]),
                }
            )
    return rows


def apply_table(rows, threshold):
    result = []
    for row in rows:
        best = int(np.argmax(row["scores"])) if threshold is not None else 0
        selected = (
            best if threshold is not None and row["scores"][best] > threshold else 0
        )
        result.append(
            {
                "image": row["image"],
                "area_fraction": row["area_fraction"],
                "action": selected,
                "iou": row["quality"][selected],
                "base_iou": row["quality"][0],
                "counts": row["quality_counts"][selected],
                "base_counts": row["quality_counts"][0],
            }
        )
    return result


def policy_stats(rows, threshold):
    applied = apply_table(rows, threshold)
    delta = np.array([r["iou"] - r["base_iou"] for r in applied], dtype=np.float64)
    area = np.array([r["area_fraction"] for r in applied])
    small, medium, large = area < 0.01, (area >= 0.01) & (area < 0.1), area >= 0.1
    group = lambda mask: float(delta[mask].mean()) if mask.any() else None
    a, b, c = group(small), group(medium), group(large)
    feasible = threshold is None or (
        a is not None
        and b is not None
        and c is not None
        and c >= 0
        and delta.mean() >= 0
    )
    return {
        "threshold": threshold,
        "objective_small_medium": (a + b) / 2
        if a is not None and b is not None
        else 0.0,
        "mean_delta": float(delta.mean()),
        "large_delta": c,
        "small_delta": a,
        "medium_delta": b,
        "switches": sum(r["action"] != 0 for r in applied),
        "feasible": bool(feasible),
    }


def policy_key(policy):
    return policy["objective_small_medium"], policy["mean_delta"], -policy["switches"]


def train(recipe, args, manifest):
    seed_everything(args.seed)
    root = Path(args.output_dir) / recipe
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Use a new/empty run folder: {root}")
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "version": VERSION,
        "recipe": recipe,
        "dimensions": manifest["dimensions"],
        "width": 64,
        "manifest_sha256": file_hash(Path(args.feature_cache) / "manifest.json"),
        "parent_sha256": manifest["parent_sha256"],
        "max_candidates": manifest["max_candidates"],
        "protocol": manifest["protocol"],
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "amp": args.amp,
        "source_sha256": {
            p: file_hash(p)
            for p in (
                "run_utility_evidence.py",
                "models/utility_evidence.py",
                "training/utility_evidence_objective.py",
                "utility_evidence.py",
            )
        },
    }
    model = UtilityEvidence(**manifest["dimensions"]).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=str(args.device).startswith("cuda"),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and str(args.device).startswith("cuda")
    )
    start, history, candidates = 0, [], []
    if args.resume_checkpoint:
        saved = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=True
        )
        if saved.get("kind") != "resume" or saved["config"] != config:
            raise ValueError(
                "Resume lacks optimizer or recipe/cache/source/control settings differ"
            )
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start, history, candidates = (
            saved["epoch"],
            saved["history"],
            saved["candidates"],
        )
    if start > args.num_epochs:
        raise ValueError("num-epochs is a total target and must exceed resumed epoch")
    atomic_json(config, root / "run_config.json")
    train_loader = make_loader(
        Path(args.feature_cache) / "train", manifest["cohorts"]["fit"], args, True
    )
    for epoch in range(start, args.num_epochs):
        model.train()
        train_loader.sampler.generator.manual_seed(args.seed + epoch)
        totals = torch.zeros(3, device=args.device, dtype=torch.float64)
        n = 0
        skipped = 0
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        begin = time.perf_counter()
        for batch, target in tqdm(train_loader, desc=f"{recipe} {epoch + 1}"):
            batch, target = (
                to_device(batch, args.device),
                to_device(target, args.device),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                torch.device(args.device).type,
                enabled=scaler.is_enabled(),
                dtype=torch.float16,
            ):
                scores = model(batch)
                loss, pieces = utility_objective(scores, target, batch["valid"], recipe)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not scaler.is_enabled() and not torch.isfinite(norm):
                raise FloatingPointError(
                    "Nonfinite gradient; refusing optimizer update"
                )
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += scaler.get_scale() < before
            count = len(batch["filename"])
            totals += (
                torch.stack(
                    (loss.detach(), pieces["regression"], pieces["regret"])
                ).double()
                * count
            )
            n += count
        values = (totals / n).tolist()
        if not all(math.isfinite(v) for v in values):
            raise FloatingPointError("Nonfinite epoch loss; checkpoint not written")
        history.append(
            {
                "epoch": epoch + 1,
                "loss": values[0],
                "regression": values[1],
                "regret": values[2],
                "skipped_updates": skipped,
                "epoch_seconds": time.perf_counter() - begin,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        common = {"config": config, "epoch": epoch + 1, "model": model.state_dict()}
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.num_epochs:
            save_checkpoint(
                dict(common, kind="weights"), root / f"checkpoint_epoch{epoch + 1}.pt"
            )
            rows = decision_table(
                model,
                Path(args.feature_cache) / "train",
                manifest["cohorts"]["tune"],
                args,
            )
            policies = [policy_stats(rows, t) for t in THRESHOLDS]
            best = max((p for p in policies if p["feasible"]), key=policy_key)
            candidates.append({"epoch": epoch + 1, "policy": best})
            atomic_json(
                {"rows": rows, "policies": policies}, root / f"tune_ep{epoch + 1}.json"
            )
        save_checkpoint(
            dict(
                common,
                kind="resume",
                optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),
                history=history,
                candidates=candidates,
            ),
            root / "latest_resume.pt",
        )
        atomic_json(history, root / "history.json")
    selected = max(candidates, key=lambda c: (*policy_key(c["policy"]), -c["epoch"]))
    checkpoint = root / f"checkpoint_epoch{selected['epoch']}.pt"
    if not checkpoint.is_file() and args.previous_checkpoint_dir:
        checkpoint = Path(args.previous_checkpoint_dir) / checkpoint.name
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Selected checkpoint missing; attach earlier snapshots: {checkpoint.name}"
        )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["config"] != config:
        raise ValueError("Selected old checkpoint belongs to a different run")
    model.load_state_dict(saved["model"], strict=True)
    proposed = selected["policy"]["threshold"]
    # Safety labels are first inspected AFTER checkpoint and threshold are frozen.
    safety_rows = apply_table(
        decision_table(
            model,
            Path(args.feature_cache) / "train",
            manifest["cohorts"]["safety"],
            args,
        ),
        proposed,
    )
    audit = certify(safety_rows)
    if proposed is None:
        audit.update(accepted=True, harm_rate_upper=0.0, identity_policy=True)
    deployed = proposed if audit["accepted"] else None
    policy = {
        "version": VERSION,
        "epoch": selected["epoch"],
        "checkpoint_sha256": file_hash(checkpoint),
        "proposed_threshold": proposed,
        "deployed_threshold": deployed,
        "audit": audit,
        "selected_on": "parent-unseen tune cohort only",
    }
    atomic_json(policy, root / "policy.json")
    atomic_json(safety_rows, root / "safety_rows.json")
    for split in ("val", "test"):
        evaluate(model, args, manifest, root, split, proposed, "proposed")
        evaluate(model, args, manifest, root, split, deployed, "certified")


@torch.inference_mode()
def evaluate(model, args, manifest, root, split, threshold, view):
    model.eval()
    out = root / f"eval_{split}_{view}"
    out.mkdir()
    scores, base_scores, evidence, groups = {}, {}, {}, {}
    intersection = union = base_intersection = base_union = 0
    for batch, _ in make_loader(
        Path(args.feature_cache) / split, manifest["splits"][split]["names"], args
    ):
        inputs = to_device(batch, args.device)
        pred = model(inputs)
        actions = choose_actions(pred, inputs["valid"], threshold).cpu().tolist()
        for i, name in enumerate(batch["filename"]):
            record = torch.load(
                Path(args.feature_cache) / split / f"{Path(name).stem}.pt",
                weights_only=True,
            )
            parent = unpack_mask(record["parent"])
            mask = render_action(record, actions[i])
            with Image.open(
                Path(args.data_root)
                / "binary_masks_png"
                / split
                / Path(name).with_suffix(".png")
            ) as source:
                truth = np.asarray(source) > 127
            if parent.shape != truth.shape:
                raise ValueError("Native image/GT size mismatch")
            a, b = int((mask & truth).sum()), int((mask | truth).sum())
            c, d = int((parent & truth).sum()), int((parent | truth).sum())
            intersection += a
            union += b
            base_intersection += c
            base_union += d
            scores[name] = a / b if b else 1.0
            base_scores[name] = c / d if d else 1.0
            area = float(truth.mean())
            group = (
                "small_lt_1pct"
                if area < 0.01
                else "medium_1_to_10pct"
                if area < 0.1
                else "large_ge_10pct"
            )
            groups.setdefault(group, []).append(name)
            evidence[name] = {
                "action": actions[i],
                "operation": int(record["features"]["operation"][actions[i]]),
                "predicted_utility": float(pred[i, actions[i]]),
                "area_fraction": area,
            }
            Image.fromarray(mask.astype("uint8") * 255).save(
                out / Path(name).with_suffix(".png")
            )
    summary = {
        "num_samples": len(scores),
        "mean_iou": sum(scores.values()) / len(scores),
        "global_iou": intersection / union if union else 1.0,
        "total_intersection": intersection,
        "total_union": union,
        "global_iou_definition": "foreground_sum_intersection_over_sum_union",
        "base_mean_iou": sum(base_scores.values()) / len(scores),
        "base_global_iou": base_intersection / base_union if base_union else 1.0,
        "view": view,
        "threshold": threshold,
        "metric_resolution": "original",
        "groups": {
            g: {
                "n": len(names),
                "mean_iou": sum(scores[n] for n in names) / len(names),
                "base_mean_iou": sum(base_scores[n] for n in names) / len(names),
            }
            for g, names in groups.items()
        },
        "switches": sum(v["action"] > 0 for v in evidence.values()),
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


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--feature-cache", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--recipe", choices=(*RECIPES, "both"), default="both")
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume-checkpoint")
    p.add_argument("--previous-checkpoint-dir")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if (
        args.batch_size < 1
        or args.num_workers < 0
        or args.num_epochs < 1
        or not math.isfinite(args.lr)
        or args.lr <= 0
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0
    ):
        p.error("Invalid batch/epoch/optimizer setting")
    if args.resume_checkpoint and args.recipe == "both":
        p.error("Resume one recipe at a time")
    manifest = read_json(Path(args.feature_cache) / "manifest.json")
    validate_manifest(manifest)
    for split, data in manifest["splits"].items():
        if (
            file_hash(Path(args.data_root) / f"{split}_grounding.json")
            != data["annotation_sha256"]
        ):
            p.error("Dataset differs from cache")
    for recipe in RECIPES if args.recipe == "both" else (args.recipe,):
        train(recipe, args, manifest)


if __name__ == "__main__":
    main()
