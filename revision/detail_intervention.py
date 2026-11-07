"""Inference-time interventions on the detail pathway of a trained QGDP-Net (no training).

Question: does an already-trained full model (detail_input="image") use its detail maps?
The checkpoint is evaluated on one split under four conditions, with the same prediction rule
as eval.py (float32 logits, bilinear resize to the native mask grid, threshold 0):

  original    the trained model as published (must reproduce eval.py's mean IoU)
  zeros       all detail maps P0..P4 set to zero (the model's own "zeros" switch)
  misaligned  the detail view rotated by 180 degrees: same image statistics and content, but the
              maps no longer line up with the ViT features
  low_res     the detail CNN sees the 336 px view upsampled to 768 px (the model's "low_res"
              switch): same pathway, no information above 336 px

The last three shift the input distribution of a model trained with real maps, so they are
sensitivity diagnostics, not estimates of what a retrained model would do.

Under "original" the script also records, per decoder stage i, the learned residual scale
lambda_i, the mean spatial gate g_i and the ratio ||lambda_i * delta_i|| / ||x_i|| of the detail
residual to the decoder features it is added to (per image, Frobenius norms), and for the refiner
the ratio mean|correction| / mean|upsampled logits|.

Output: one JSON per checkpoint and split with per-image IoU for every condition and the
statistics above. Read-only on the checkpoint; refuses to overwrite its output.

Usage:
  python revision/detail_intervention.py --checkpoint CKPT --data-root DATA --dataset test \
      --output OUT.json [--conditions original,zeros,misaligned,low_res]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root, as for the other revision/ scripts

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from dataset import VizWizGroundingDataset, make_loader  # noqa: E402
from models.model import GroundingModel  # noqa: E402
from utils import checkpoint_model_kwargs, load_model_weights, read_checkpoint, to_device  # noqa: E402

CONDITIONS = ("original", "zeros", "misaligned", "low_res")


def predict(model: GroundingModel, batch: dict, condition: str) -> torch.Tensor:
    """Logits (B, 1, 768, 768) under one condition; the model's switches are restored afterwards."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition}")
    if model.detail_input != "image":
        raise ValueError("Interventions apply to a model trained with real detail maps")
    switch = {"zeros": "zeros", "low_res": "low_res"}.get(condition, "image")
    if condition == "misaligned":
        batch = {**batch, "detail_image": torch.flip(batch["detail_image"], dims=(-2, -1))}
    model.detail_input = switch
    try:
        return model(batch)["logits"]
    finally:
        model.detail_input = "image"


class PathwayStats:
    """Forward hooks on the four fusion stages and the refiner (used under "original" only)."""

    def __init__(self, model: GroundingModel):
        self.rows, self.handles, self.current = [], [], {}
        for i, fusion in enumerate(model.decoder.detail_fusion):
            self.handles.append(fusion.guide.register_forward_hook(self._gate(i)))
            self.handles.append(fusion.register_forward_hook(self._residual(i)))
        self.handles.append(model.refiner.correction.register_forward_hook(self._correction))
        self.handles.append(model.refiner.register_forward_hook(self._refiner))

    def _gate(self, i):
        def hook(_m, _inp, out):  # out: (B, 1, H, W) in (0, 1)
            self.current[f"gate_{i}"] = out.float().mean(dim=(1, 2, 3))
        return hook

    def _residual(self, i):
        def hook(_m, inp, out):  # out = x + lambda * delta
            x = inp[0].float()
            term = out.float() - x
            self.current[f"residual_ratio_{i}"] = term.flatten(1).norm(dim=1) / x.flatten(1).norm(dim=1).clamp_min(1e-12)
        return hook

    def _correction(self, _m, _inp, out):
        self.current["_delta"] = out.float()

    def _refiner(self, _m, _inp, out):
        delta = self.current.pop("_delta")
        up = out.float() - delta
        self.current["refiner_ratio"] = delta.abs().mean(dim=(1, 2, 3)) / up.abs().mean(dim=(1, 2, 3)).clamp_min(1e-12)

    def collect(self, n: int):
        keys = sorted(self.current)
        for b in range(n):
            self.rows.append({k: float(self.current[k][b]) for k in keys})
        self.current = {}

    def close(self):
        for h in self.handles:
            h.remove()


def native_iou(logits: torch.Tensor, truth: torch.Tensor) -> float:
    """eval.py's rule: bilinear resize to the native grid, threshold 0; empty union counts as 1."""
    resized = F.interpolate(logits.float(), size=truth.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    pred, gt = resized > 0, truth[0, 0] > 0.5
    union = int((pred | gt).sum())
    return 1.0 if union == 0 else int((pred & gt).sum()) / union


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=["val", "test"], required=True)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="Smoke tests only: stop after N images")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise ValueError(f"Refusing to overwrite {out}")
    conditions = args.conditions.split(",")
    if conditions[0] != "original" or any(c not in CONDITIONS for c in conditions):
        raise ValueError(f"Conditions must start with 'original' and come from {CONDITIONS}")

    saved = read_checkpoint(args.checkpoint)
    model = GroundingModel(**checkpoint_model_kwargs(saved))
    load_model_weights(model, saved)
    device = torch.device(args.device)
    model.to(device).eval()
    if model.detail_input != "image":
        raise ValueError(f"Checkpoint was trained with detail_input={model.detail_input!r}")

    dataset = VizWizGroundingDataset(args.data_root, args.dataset, model.image_size, detail_size=model.detail_size)
    loader = make_loader(dataset, args, args.batch_size)
    stats = PathwayStats(model)
    per_sample = {c: {} for c in conditions}
    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            for condition in conditions:
                logits = predict(model, batch, condition)
                if condition == "original":
                    stats.collect(len(batch["filename"]))
                for i, name in enumerate(batch["filename"]):
                    if name in per_sample[condition]:
                        raise ValueError(f"Duplicate sample {name}")
                    per_sample[condition][name] = native_iou(logits[i:i + 1], batch["original_mask"][i].unsqueeze(0).to(device))
            if args.limit and len(per_sample["original"]) >= args.limit:
                break
    stats.close()

    names = list(per_sample["original"])
    stat_keys = sorted(stats.rows[0]) if stats.rows else []
    summary = {
        "n": len(names),
        "mean_iou": {c: sum(per_sample[c][n] for n in names) / len(names) for c in conditions},
        "residual_scale": [float(f.residual_scale.detach()) for f in model.decoder.detail_fusion],
        "pathway_stats_mean": {k: sum(r[k] for r in stats.rows) / len(stats.rows) for k in stat_keys},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "dataset": args.dataset, "conditions": conditions,
        "experiment_config": model.experiment_config, "seed": saved.get("run_config", {}).get("seed"),
        "summary": summary, "per_sample": per_sample,
        "pathway_stats": dict(zip(names, stats.rows))}))
    print(json.dumps({"dataset": args.dataset, **summary}), flush=True)


if __name__ == "__main__":
    main()
