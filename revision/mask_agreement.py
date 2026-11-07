"""Session Q: mask-level agreement between the trained QGDP-Net and its detail-pathway interventions,
and how much of each detail residual depends on the detail maps (inference only, no training).

Session P compared mean IoU against the ground truth. Similar means can hide different masks, so this
script compares the predicted masks themselves, per image, at native resolution and with eval.py's
rule (bilinear resize of the 768 px logits, threshold 0). For each intervention c in zeros,
misaligned, low_res (defined in revision/detail_intervention.py):
  mask_iou        IoU between the original and the intervened mask (1 if both are empty)
  disagreement    pixels where the two masks differ, as a share of the image
  disagreement_rel  the same count divided by the original mask's area (1 pixel minimum)
  iou             IoU of the intervened mask against the ground truth (and of the original)

Map dependence, under the unchanged model only. For decoder stage i the fusion block adds
  r_i = lambda_i * delta_i(x_i, P_i)  to the decoder features x_i.
With the same x_i, the counterfactual r0_i = lambda_i * delta_i(x_i, 0) is what the block adds without
its detail map. Recorded per image:
  map_share_i = ||r_i - r0_i|| / ||r_i||   (share of the residual that depends on the map)
  map_rel_i   = ||r_i - r0_i|| / ||x_i||   (that part relative to the decoder features)
and for the refiner, ||c - c0|| / ||c|| for its logit correction c with and without P_4.
Frobenius norms over each image. The hooks do not change any output.

Output: one JSON per checkpoint and split; refuses to overwrite.

Usage:
  python revision/mask_agreement.py --checkpoint CKPT --data-root DATA --dataset val --output OUT.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from dataset import VizWizGroundingDataset, make_loader  # noqa: E402
from models.model import GroundingModel  # noqa: E402
from revision.detail_intervention import predict  # noqa: E402
from utils import checkpoint_model_kwargs, load_model_weights, read_checkpoint, to_device  # noqa: E402

INTERVENTIONS = ("zeros", "misaligned", "low_res")


def native_mask(logits: torch.Tensor, shape) -> torch.Tensor:
    """eval.py's rule for one image: (1, 1, H', W') logits -> (H, W) boolean mask."""
    return F.interpolate(logits.float(), size=shape, mode="bilinear", align_corners=False)[0, 0] > 0


def mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    union = int((a | b).sum())
    return 1.0 if union == 0 else int((a & b).sum()) / union


def agreement(original: torch.Tensor, other: torch.Tensor) -> dict:
    differ = int((original ^ other).sum())
    return {"mask_iou": mask_iou(original, other), "disagreement": differ / original.numel(),
            "disagreement_rel": differ / max(int(original.sum()), 1)}


class MapDependence:
    """Forward hooks computing each block's residual with and without its detail map (same inputs)."""

    def __init__(self, model: GroundingModel):
        self.current, self.handles, self._inside = {}, [], False
        for i, fusion in enumerate(model.decoder.detail_fusion):
            self.handles.append(fusion.register_forward_hook(self._fusion(i)))
        self.handles.append(model.refiner.register_forward_hook(self._refiner))

    def _fusion(self, i):
        def hook(module, inputs, output):
            visual, detail, _question = inputs
            residual = (output - visual).float()
            without = (module.residual_scale * module.refine(torch.cat((visual, torch.zeros_like(detail)), 1))).float()
            diff = (residual - without).flatten(1).norm(dim=1)
            self.current[f"map_share_{i}"] = diff / residual.flatten(1).norm(dim=1).clamp_min(1e-12)
            self.current[f"map_rel_{i}"] = diff / visual.float().flatten(1).norm(dim=1).clamp_min(1e-12)
        return hook

    def _refiner(self, module, inputs, output):
        if self._inside:
            return
        logits, semantic, detail, question = inputs
        self._inside = True
        try:
            without = module(logits, semantic, torch.zeros_like(detail), question)
        finally:
            self._inside = False
        up = F.interpolate(logits.float(), size=detail.shape[-2:], mode="bilinear", align_corners=False)
        correction = (output - up).float()
        diff = (output - without).float().flatten(1).norm(dim=1)
        self.current["refiner_map_share"] = diff / correction.flatten(1).norm(dim=1).clamp_min(1e-12)

    def take(self) -> dict:
        out, self.current = self.current, {}
        return out

    def close(self):
        for h in self.handles:
            h.remove()


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
    rows = {}
    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            hooks = MapDependence(model)
            try:
                original = predict(model, batch, "original")
            finally:
                hooks.close()
            dependence = hooks.take()
            others = {c: predict(model, batch, c) for c in INTERVENTIONS}
            for i, name in enumerate(batch["filename"]):
                if name in rows:
                    raise ValueError(f"Duplicate sample {name}")
                truth = batch["original_mask"][i].to(device)[0] > 0.5
                base = native_mask(original[i:i + 1], truth.shape)
                row = {"iou_original": mask_iou(base, truth),
                       **{k: float(v[i]) for k, v in dependence.items()}}
                for c, logits in others.items():
                    mask = native_mask(logits[i:i + 1], truth.shape)
                    row[c] = {"iou": mask_iou(mask, truth), **agreement(base, mask)}
                rows[name] = row
            if args.limit and len(rows) >= args.limit:
                break

    names = list(rows)
    mean = lambda f: sum(f(rows[n]) for n in names) / len(names)
    summary = {"n": len(names), "mean_iou_original": mean(lambda r: r["iou_original"]),
               **{c: {k: mean(lambda r, c=c, k=k: r[c][k]) for k in ("iou", "mask_iou", "disagreement", "disagreement_rel")}
                  for c in INTERVENTIONS},
               **{k: mean(lambda r, k=k: r[k]) for k in rows[names[0]] if k.startswith(("map_", "refiner_"))}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": str(args.checkpoint), "dataset": args.dataset,
                               "experiment_config": model.experiment_config, "summary": summary, "per_sample": rows}))
    print(json.dumps({"dataset": args.dataset, **summary}), flush=True)


if __name__ == "__main__":
    main()
