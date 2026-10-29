"""Train two independent cached-feature experiments, select on val, evaluate native masks."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm.auto import tqdm

from dataset import seed_worker
from evidence_transport_data import VERSION, FeatureDataset, collate_features
from metrics import accumulate_pixel_counts, global_iou_summary
from models.evidence_transport import EvidenceTransport, compose_native
from ocr_cache import atomic_json, file_hash, read_json
from train import seed_everything
from training.evidence_transport_objective import transport_objective
from utils import save_checkpoint, to_device

RECIPES = ("transport", "intervention")
SNAPSHOTS = (5, 10, 15, 20)


def shuffled_words(cache, names, seed):
    """Global permutation of real slots, independent of evaluation batch size/padding."""
    rows = [
        torch.load(Path(cache) / f"{Path(n).stem}.pt", weights_only=True)["features"][
            "words"
        ]
        for n in names
    ]
    sizes = [len(row) for row in rows]
    words = torch.cat(rows)
    order = torch.randperm(len(words), generator=torch.Generator().manual_seed(seed))
    permuted = words[order]
    changed = int((words != permuted).any(1).sum())
    return dict(zip(names, permuted.split(sizes))), changed


def loader(cache, manifest, split, args, epoch=0, training=False):
    data = FeatureDataset(
        Path(cache) / split, manifest["splits"][split]["names"], training
    )
    generator = torch.Generator().manual_seed(args.seed + epoch)
    return DataLoader(
        data,
        batch_size=args.batch_size,
        sampler=RandomSampler(data, generator=generator) if training else None,
        num_workers=args.num_workers,
        collate_fn=collate_features,
        pin_memory=str(args.device).startswith("cuda"),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed + 7919),
        persistent_workers=args.num_workers > 0,
    )


def train(recipe, args, manifest):
    seed_everything(args.seed)
    root = Path(args.output_dir) / recipe
    saved = (
        torch.load(args.resume_checkpoint, map_location="cpu", weights_only=True)
        if args.resume_checkpoint
        else None
    )
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"{root} must be new/empty; no automatic overwrite")
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "version": VERSION,
        "recipe": recipe,
        "dimensions": manifest["dimensions"],
        "width": 64,
        "feature_manifest_sha256": file_hash(
            Path(args.feature_cache) / "manifest.json"
        ),
        "parent_sha256": manifest["parent_sha256"],
        "max_candidates": manifest["max_candidates"],
        "cache_protocol": manifest["cache_protocol"],
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "amp": args.amp,
        "optimizer": "adamw",
        "schedule": "constant",
        "mask_loss_resolution": manifest["canvas"],
        "local_mask_resolution": manifest["crop"],
        "source_sha256": {
            p: file_hash(p)
            for p in (
                "run_evidence_transport.py",
                "models/evidence_transport.py",
                "training/evidence_transport_objective.py",
                "evidence_transport_data.py",
            )
        },
    }
    if saved and (saved["config"] != config or saved["kind"] != "resume"):
        raise ValueError(
            "Resume recipe/cache/source/training controls differ, or snapshot lacks optimizer"
        )
    model = EvidenceTransport(**manifest["dimensions"]).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=str(args.device).startswith("cuda"),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and str(args.device).startswith("cuda")
    )
    history, start = [], 0
    if saved:
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        history, start = saved["history"], saved["epoch"]
    if start >= args.num_epochs:
        raise ValueError("Resume has already reached the requested total epochs")
    atomic_json(config, root / "run_config.json")
    training_loader = loader(args.feature_cache, manifest, "train", args, training=True)
    for epoch in range(start, args.num_epochs):
        training_loader.sampler.generator.manual_seed(args.seed + epoch)
        model.train()
        sums = torch.zeros(6, device=args.device, dtype=torch.float64)
        count, skipped = 0, 0
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        begin = time.perf_counter()
        for batch, target in tqdm(
            training_loader,
            desc=f"{recipe} {epoch + 1}",
        ):
            batch, target = (
                to_device(batch, args.device),
                to_device(target, args.device),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=torch.device(args.device).type,
                enabled=scaler.is_enabled(),
                dtype=torch.float16,
            ):
                output = model(batch)
                loss, pieces = transport_objective(output, target, batch, recipe)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not scaler.is_enabled() and not torch.isfinite(norm):
                raise FloatingPointError(
                    "Nonfinite gradient without AMP; stop instead of corrupting weights"
                )
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += scaler.get_scale() < before
            size = len(batch["filename"])
            sums += torch.stack([loss.detach(), *pieces.values()]).double() * size
            count += size
        values = (sums / count).tolist()  # One end-of-epoch synchronization.
        if not all(math.isfinite(v) for v in values):
            raise FloatingPointError("Nonfinite epoch loss; checkpoint not written")
        record = dict(
            epoch=epoch + 1,
            loss=values[0],
            **dict(zip(pieces, values[1:])),
            skipped_updates=skipped,
            epoch_seconds=time.perf_counter() - begin,
        )
        history.append(record)
        print(json.dumps(record), flush=True)
        common = {"config": config, "epoch": epoch + 1, "model": model.state_dict()}
        if epoch + 1 in SNAPSHOTS or epoch + 1 == args.num_epochs:
            save_checkpoint(
                dict(common, kind="weights"), root / f"checkpoint_epoch{epoch + 1}.pt"
            )
        # Small adapters: full resume saved EVERY epoch, atomically replacing one file.
        save_checkpoint(
            dict(
                common,
                kind="resume",
                optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),
                history=history,
            ),
            root / "latest_resume.pt",
        )
        atomic_json(history, root / "history.json")
    return root


def evaluate(checkpoint, args, manifest, split="val", view="full"):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["config"]["feature_manifest_sha256"] != file_hash(
        Path(args.feature_cache) / "manifest.json"
    ):
        raise ValueError("Evaluation feature cache differs from training")
    model = EvidenceTransport(
        **saved["config"]["dimensions"], width=saved["config"]["width"]
    ).to(args.device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    replacements, changed = (
        shuffled_words(
            Path(args.feature_cache) / split,
            manifest["splits"][split]["names"],
            args.seed + 17,
        )
        if view == "shuffled_text"
        else ({}, 0)
    )
    root = (
        Path(args.output_dir)
        / saved["config"]["recipe"]
        / f"eval_{split}_{view}_ep{saved['epoch']}"
    )
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Evaluation output is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    scores, bases, totals, base_totals, groups, detail = {}, {}, {}, {}, {}, {}
    with torch.inference_mode():
        for batch, _ in tqdm(
            loader(args.feature_cache, manifest, split, args),
            desc=f"{split} {view} ep{saved['epoch']}",
        ):
            if view == "shuffled_text":
                for i, name in enumerate(batch["filename"]):
                    batch["words"][i, : len(replacements[name])] = replacements[name]
            batch = to_device(batch, args.device)
            if view == "missing":
                batch["valid"].zero_()
            elif view == "geometry":
                batch["words"].zero_()
            output = model(batch)
            for i, name in enumerate(batch["filename"]):
                with Image.open(
                    Path(args.data_root)
                    / "binary_masks_png"
                    / split
                    / Path(name).with_suffix(".png")
                ) as mask:
                    truth = torch.from_numpy(np.asarray(mask).copy() > 127).to(
                        args.device
                    )
                if tuple(truth.shape) != tuple(batch["native_size"][i]):
                    raise ValueError(f"Native image/GT mask dimensions differ: {name}")
                base = torch.load(
                    Path(args.feature_cache) / split / "base" / f"{Path(name).stem}.pt",
                    weights_only=True,
                )["logits"].to(args.device)
                base = F.interpolate(
                    base[None], truth.shape, mode="bilinear", align_corners=False
                )[0]
                evidence = compose_native(output, batch, i, truth.shape)
                gate = output["gate"][i]
                probability = (1 - gate) * base.sigmoid() + gate * evidence
                original = base[0] > 0
                prediction = torch.where(gate == 0, original, probability[0] > 0.5)
                accumulate_pixel_counts(prediction, truth, totals)
                accumulate_pixel_counts(original, truth, base_totals)
                intersection, union = (
                    (prediction & truth).sum().item(),
                    (prediction | truth).sum().item(),
                )
                bi, bu = (
                    (original & truth).sum().item(),
                    (original | truth).sum().item(),
                )
                scores[name], bases[name] = (
                    intersection / union if union else 1.0,
                    bi / bu if bu else 1.0,
                )
                area = truth.double().mean().item()
                group = (
                    "small_lt_1pct"
                    if area < 0.01
                    else "medium_1_to_10pct"
                    if area < 0.1
                    else "large_ge_10pct"
                )
                groups.setdefault(group, []).append(name)
                detail[name] = {
                    "gate": float(gate),
                    "candidates": int(batch["valid"][i].sum()),
                    "area_fraction": area,
                    "scores": output["scores"][i, batch["valid"][i]]
                    .float()
                    .cpu()
                    .tolist(),
                    "evidence_iou": float(
                        ((evidence[0] > 0.5) & truth).sum()
                        / ((evidence[0] > 0.5) | truth).sum().clamp_min(1)
                    ),
                }
                Image.fromarray(prediction.cpu().numpy().astype("uint8") * 255).save(
                    root / Path(name).with_suffix(".png")
                )
    summary = dict(
        num_samples=len(scores),
        mean_iou=sum(scores.values()) / len(scores),
        **global_iou_summary(totals),
        base_mean_iou=sum(bases.values()) / len(bases),
        base_metrics=global_iou_summary(base_totals),
        epoch=saved["epoch"],
        recipe=saved["config"]["recipe"],
        split=split,
        view=view,
        parent_sha256=manifest["parent_sha256"],
        checkpoint_sha256=file_hash(checkpoint),
        groups={
            g: {
                "n": len(names),
                "mean_iou": sum(scores[n] for n in names) / len(names),
                "base_mean_iou": sum(bases[n] for n in names) / len(names),
            }
            for g, names in groups.items()
        },
        metric_resolution="original",
        parameter_count=sum(p.numel() for p in model.parameters()),
        shuffled_text_changed_regions=changed,
        shuffled_text_protocol="global permutation of real slots, seed+17"
        if view == "shuffled_text"
        else None,
    )
    summary["evidence_diagnostic"] = {
        "oracle_parent_or_evidence_mean_iou": sum(
            max(bases[n], detail[n]["evidence_iou"]) for n in scores
        )
        / len(scores),
        "helpful_evidence_gt_5pp": sum(
            detail[n]["evidence_iou"] > bases[n] + 0.05 for n in scores
        ),
        "helpful_but_gate_lt_half": sum(
            detail[n]["evidence_iou"] > bases[n] + 0.05 and detail[n]["gate"] < 0.5
            for n in scores
        ),
        "harmful_but_gate_ge_half": sum(
            detail[n]["evidence_iou"] < bases[n] - 0.05 and detail[n]["gate"] >= 0.5
            for n in scores
        ),
        "rescued_base_lt01_to_ge05": sum(
            bases[n] < 0.1 and scores[n] >= 0.5 for n in scores
        ),
        "broken_base_ge05_to_lt01": sum(
            bases[n] >= 0.5 and scores[n] < 0.1 for n in scores
        ),
        "oracle_note": "GT chooses between two predicted masks for diagnosis only; never deployed or used for selection",
    }
    atomic_json(
        {
            "summary": summary,
            "per_sample": scores,
            "base_per_sample": bases,
            "evidence": detail,
        },
        root / "metrics.json",
    )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--feature-cache", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--recipe", choices=(*RECIPES, "both"), default="both")
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume-checkpoint")
    p.add_argument(
        "--evaluate-checkpoints",
        nargs="+",
        help="Evaluate/select existing adapter snapshots without training",
    )
    p.add_argument(
        "--previous-checkpoint-dir",
        help="Earlier snapshots from the SAME recipe for selection after resume",
    )
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
        p.error("Resume exactly one recipe at a time")
    if args.evaluate_checkpoints and (args.recipe == "both" or args.resume_checkpoint):
        p.error("Evaluation-only requires one recipe and no resume checkpoint")
    manifest = read_json(Path(args.feature_cache) / "manifest.json")
    if manifest.get("version") != VERSION or not manifest.get("complete"):
        p.error("Incomplete/unsupported feature cache")
    for split, meta in manifest["splits"].items():
        if (
            file_hash(Path(args.data_root) / f"{split}_grounding.json")
            != meta["annotation_sha256"]
        ):
            p.error(f"{split} annotation differs from the feature-cache dataset")
    for recipe in RECIPES if args.recipe == "both" else (args.recipe,):
        root = Path(args.output_dir) / recipe
        checkpoints = {}
        if args.evaluate_checkpoints:
            for path in args.evaluate_checkpoints:
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if saved["config"]["recipe"] != recipe or saved["epoch"] in checkpoints:
                    raise ValueError(
                        "Wrong recipe or duplicate epoch in evaluation checkpoints"
                    )
                checkpoints[saved["epoch"]] = Path(path)
        else:
            root = train(recipe, args, manifest)
        for epoch in sorted(
            []
            if args.evaluate_checkpoints
            else {e for e in SNAPSHOTS if e <= args.num_epochs} | {args.num_epochs}
        ):
            paths = [root / f"checkpoint_epoch{epoch}.pt"]
            if args.previous_checkpoint_dir:
                paths.append(Path(args.previous_checkpoint_dir) / paths[0].name)
            checkpoint = next((p for p in paths if p.is_file()), None)
            if checkpoint is None:
                raise FileNotFoundError(
                    f"Need previous checkpoint epoch {epoch} for full validation selection"
                )
            checkpoints[epoch] = checkpoint
        validation = {
            e: evaluate(path, args, manifest) for e, path in checkpoints.items()
        }
        best = max(validation, key=lambda e: (validation[e]["mean_iou"], -e))
        atomic_json(
            {
                "selected_epoch": best,
                "validation": validation,
                "checkpoint_paths": {str(e): str(p) for e, p in checkpoints.items()},
            },
            root / "selection.json",
        )
        for epoch in sorted({best, max(checkpoints)}):
            evaluate(checkpoints[epoch], args, manifest, "test")
        for view in ("missing", "geometry", "shuffled_text"):
            evaluate(checkpoints[best], args, manifest, "val", view)


if __name__ == "__main__":
    main()
