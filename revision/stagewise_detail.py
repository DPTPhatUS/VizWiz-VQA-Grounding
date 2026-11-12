"""Session R: stage-by-stage use of the detail maps in a trained QGDP-Net (inference only, no training).

Session P zeroed all detail maps at once. This script asks two finer questions on one split:

1. Which stage, if any, uses its map? Each condition changes the maps P_0..P_4 that the pathway
   returns (P_0..P_3 enter decoder stages 0..3, P_4 enters the refiner) and leaves everything else
   as trained:
     original    unchanged (must reproduce eval.py's per-image IoU)
     zero_p<k>   only P_k set to zero, k = 0..4
     scale_x2    all maps multiplied by 2
     scale_x4    all maps multiplied by 4
   The scale conditions ask whether the maps carry information the trained model under-weights:
   if amplifying them improved the masks, the information would be there but too weak. Like
   Session P's interventions, all conditions shift the input distribution of the trained model, so
   they are sensitivity diagnostics, not estimates of what a retrained model would do.
   Per image and condition: IoU against the ground truth, and the mask IoU and pixel disagreement
   with the original mask (same rule as revision/mask_agreement.py). Each image also records the
   share of the image its ground-truth target covers, for size groups in the analysis.

2. How much of the training signal reaches the maps? Under "original", with the model's training
   loss (pixel BCE + 0.5 Dice at 768 px, summed over images so that every activation gradient is
   that image's own), per image and stage k:
     grad_share_k  sum|P_k * dL/dP_k| / (sum|P_k * dL/dP_k| + sum|x_k * dL/dx_k|)
                   (gradient x input attribution of the map against the decoder features x_k that
                   enter the same fusion block; for k = 4, x_4 is the 384 px logits entering the refiner)
     grad_norm_map_k, grad_norm_x_k   Frobenius norms of the two gradients
   and per batch, for each parameter group (detail_pyramid, detail_fusion, decoder_rest, refiner),
   ||dL/dtheta|| / ||theta||. Adam rescales gradients per parameter, so these describe where the
   loss signal flows, not the size of the updates. The encoders get no gradient in this pass.

Output: one JSON per checkpoint and split; read-only on the checkpoint; refuses to overwrite.

Usage:
  python revision/stagewise_detail.py --checkpoint CKPT --data-root DATA --dataset val --output OUT.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root

import torch  # noqa: E402

from dataset import VizWizGroundingDataset, make_loader  # noqa: E402
from losses import segmentation_loss  # noqa: E402
from models.model import GroundingModel  # noqa: E402
from revision.mask_agreement import agreement, mask_iou, native_mask  # noqa: E402
from utils import checkpoint_model_kwargs, load_model_weights, read_checkpoint, to_device  # noqa: E402

NUM_MAPS = 5
CONDITIONS = ("original", *(f"zero_p{k}" for k in range(NUM_MAPS)), "scale_x2", "scale_x4")
PARAM_GROUPS = ("detail_pyramid", "detail_fusion", "decoder_rest", "refiner")


def transform_maps(maps: tuple, condition: str) -> tuple:
    """Apply one condition to the coarse-to-fine maps (P_0 at 48 px ... P_4 at 768 px)."""
    if len(maps) != NUM_MAPS:
        raise ValueError(f"Expected {NUM_MAPS} detail maps, got {len(maps)}")
    if condition == "original":
        return maps
    if condition.startswith("zero_p"):
        k = int(condition[len("zero_p"):])
        if not 0 <= k < NUM_MAPS:
            raise ValueError(f"No detail map {k}")
        return tuple(torch.zeros_like(m) if i == k else m for i, m in enumerate(maps))
    if condition.startswith("scale_x"):
        factor = float(condition[len("scale_x"):])
        return tuple(m * factor for m in maps)
    raise ValueError(f"Unknown condition {condition}")


def predict(model: GroundingModel, batch: dict, condition: str) -> torch.Tensor:
    """Logits (B, 1, 768, 768) under one condition; the model is restored afterwards."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition}")
    if model.detail_input != "image":
        raise ValueError("Interventions apply to a model trained with real detail maps")
    if condition == "original":
        return model(batch)["logits"]
    method = model._detail_maps
    model._detail_maps = lambda *a: transform_maps(method(*a), condition)  # instance attribute shadows the method
    try:
        return model(batch)["logits"]
    finally:
        del model._detail_maps


def parameter_groups(model: GroundingModel) -> dict:
    fusion = {id(p) for p in model.decoder.detail_fusion.parameters()}
    return {
        "detail_pyramid": list(model.detail_pyramid.parameters()),
        "detail_fusion": list(model.decoder.detail_fusion.parameters()),
        "decoder_rest": [p for p in model.decoder.parameters() if id(p) not in fusion],
        "refiner": list(model.refiner.parameters()),
    }


def gradient_stats(model: GroundingModel, batch: dict) -> tuple[list, dict, torch.Tensor]:
    """Per-image attribution rows, per-batch parameter-gradient ratios, and the logits (unchanged)."""
    groups = parameter_groups(model)
    flags = {p: p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for params in groups.values():
        for p in params:
            p.requires_grad_(True)
    method = model._detail_maps
    maps, inputs, handles = [], {}, []

    def keep_maps(*a):
        out = method(*a)
        for m in out:
            m.retain_grad()
        maps.extend(out)
        return out

    def keep_input(k):
        def hook(_module, args, _output):
            args[0].retain_grad()
            inputs[k] = args[0]
        return hook

    model._detail_maps = keep_maps
    for k, fusion in enumerate(model.decoder.detail_fusion):
        handles.append(fusion.register_forward_hook(keep_input(k)))
    handles.append(model.refiner.register_forward_hook(keep_input(NUM_MAPS - 1)))
    try:
        with torch.enable_grad():
            logits = model(batch)["logits"]
            target = batch["mask"]
            if logits.shape[-2:] != target.shape[-2:]:
                raise ValueError("Logits and training mask grids differ")
            loss = sum(segmentation_loss(logits[i:i + 1], target[i:i + 1]) for i in range(logits.shape[0]))
            loss.backward()
        rows = []
        for i in range(logits.shape[0]):
            row = {}
            for k in range(NUM_MAPS):
                p, gp = maps[k][i].detach().float(), maps[k].grad[i].float()
                x, gx = inputs[k][i].detach().float(), inputs[k].grad[i].float()
                ap, ax = float((p * gp).abs().sum()), float((x * gx).abs().sum())
                row[f"grad_share_{k}"] = ap / (ap + ax) if ap + ax > 0 else 0.0
                row[f"grad_norm_map_{k}"] = float(gp.norm())
                row[f"grad_norm_x_{k}"] = float(gx.norm())
            rows.append(row)
        ratios = {}
        for name, params in groups.items():
            grads = [p.grad.float().flatten() for p in params if p.grad is not None]
            g = torch.cat(grads).norm() if grads else torch.tensor(0.0)
            w = torch.cat([p.detach().float().flatten() for p in params]).norm()
            ratios[name] = float(g / w.clamp_min(1e-12))
        return rows, ratios, logits.detach()
    finally:
        del model._detail_maps
        for h in handles:
            h.remove()
        for p, flag in flags.items():
            p.grad = None
            p.requires_grad_(flag)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=["val", "test"], required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="Smoke tests only: stop after N images")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise ValueError(f"Refusing to overwrite {out}")

    saved = read_checkpoint(args.checkpoint)
    model = GroundingModel(**checkpoint_model_kwargs(saved))
    load_model_weights(model, saved)
    device = torch.device(args.device)
    model.to(device).eval()
    if model.detail_input != "image":
        raise ValueError(f"Checkpoint was trained with detail_input={model.detail_input!r}")

    dataset = VizWizGroundingDataset(args.data_root, args.dataset, model.image_size, detail_size=model.detail_size)
    loader = make_loader(dataset, args, args.batch_size)
    rows, batch_ratios = {}, []
    for batch in loader:
        batch = to_device(batch, device)
        grads, ratios, grad_logits = gradient_stats(model, batch)
        batch_ratios.append({"n": len(batch["filename"]), **ratios})
        with torch.inference_mode():
            logits = {c: predict(model, batch, c) for c in CONDITIONS}
        if not torch.allclose(grad_logits, logits["original"], atol=1e-3):
            raise RuntimeError("The gradient pass changed the model's output")
        for i, name in enumerate(batch["filename"]):
            if name in rows:
                raise ValueError(f"Duplicate sample {name}")
            truth = batch["original_mask"][i].to(device)[0] > 0.5
            base = native_mask(logits["original"][i:i + 1], truth.shape)
            row = {"iou_original": mask_iou(base, truth), "target_area": float(truth.float().mean()), **grads[i]}
            for c in CONDITIONS[1:]:
                mask = native_mask(logits[c][i:i + 1], truth.shape)
                agree = agreement(base, mask)
                row[c] = {"iou": mask_iou(mask, truth), "mask_iou": agree["mask_iou"],
                          "disagreement": agree["disagreement"]}
            rows[name] = row
        if args.limit and len(rows) >= args.limit:
            break

    names = list(rows)
    mean = lambda f: sum(f(rows[n]) for n in names) / len(names)
    total = sum(r["n"] for r in batch_ratios)
    summary = {
        "n": len(names),
        "mean_iou_original": mean(lambda r: r["iou_original"]),
        **{c: {k: mean(lambda r, c=c, k=k: r[c][k]) for k in ("iou", "mask_iou", "disagreement")}
           for c in CONDITIONS[1:]},
        **{k: mean(lambda r, k=k: r[k]) for k in rows[names[0]] if k.startswith("grad_")},
        "param_grad_ratio": {g: sum(r[g] * r["n"] for r in batch_ratios) / total for g in PARAM_GROUPS},
        "residual_scale": [float(f.residual_scale.detach()) for f in model.decoder.detail_fusion],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": str(args.checkpoint), "dataset": args.dataset,
                               "conditions": list(CONDITIONS), "experiment_config": model.experiment_config,
                               "summary": summary, "per_sample": rows, "param_grad_ratio_per_batch": batch_ratios}))
    print(json.dumps({"dataset": args.dataset, **summary}), flush=True)


if __name__ == "__main__":
    main()
