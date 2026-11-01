"""Mask-threshold sweep: per-image overlap counts at several logit thresholds (inference only).

Works inside this branch and inside the single-scale baseline commit (2079c8a): run it with the
target repository as the working directory, so that its own dataset, model and checkpoint code
are imported. Prediction follows each repository's eval.py exactly (float32 logits, bilinear
resize to the native mask grid), except that the logits are thresholded at every value in
THRESHOLDS instead of only at 0.

For every image the output stores the ground-truth area and, per threshold, the intersection and
predicted area in native pixels. IoU, precision, recall and error types at any threshold can be
recomputed from these counts. Threshold 0 reproduces eval.py.

Usage (from the target repository):
  python /path/to/revision/threshold_sweep.py --checkpoint CKPT --data-root DATA --dataset val \
      --output OUT.json
"""

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from dataset import VizWizGroundingDataset, make_loader  # noqa: E402
from models.model import GroundingModel  # noqa: E402
from utils import checkpoint_model_kwargs, load_model_weights, read_checkpoint, to_device  # noqa: E402

THRESHOLDS = [-4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0,
              0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


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
    state, config = saved["model_state_dict"], saved["run_config"]
    model_type = GroundingModel
    try:  # same rule as eval.py at commit 2079c8a; this branch has no TeacherModel
        from models.model import TeacherModel
        if config.get("teacher_training", False) or (state and not any(k.startswith("coarse.") for k in state)):
            model_type = TeacherModel
    except ImportError:
        pass
    model = model_type(**checkpoint_model_kwargs(saved))
    load_model_weights(model, saved)
    device = torch.device(args.device)
    model.to(device).eval()

    image_size = getattr(model, "image_size", saved["run_config"].get("image_size", 336))
    kwargs = {}
    if "detail_size" in inspect.signature(VizWizGroundingDataset).parameters and hasattr(model, "detail_size"):
        kwargs["detail_size"] = model.detail_size
    dataset = VizWizGroundingDataset(args.data_root, args.dataset, image_size, **kwargs)
    loader = make_loader(dataset, args, args.batch_size)

    records = {}
    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            logits = model(batch)["logits"]
            for i, name in enumerate(batch["filename"]):
                truth = batch["original_mask"][i].unsqueeze(0).to(device)  # (1, 1, H, W)
                resized = F.interpolate(logits[i:i + 1].float(), size=truth.shape[-2:],
                                        mode="bilinear", align_corners=False)[0, 0]  # (H, W)
                gt = truth[0, 0] > 0.5
                inter, pred = [], []
                for t in THRESHOLDS:
                    p = resized > t
                    inter.append(int((p & gt).sum()))
                    pred.append(int(p.sum()))
                if name in records:
                    raise ValueError(f"Duplicate sample {name}")
                records[name] = {"gt": int(gt.sum()), "inter": inter, "pred": pred}
                if args.limit and len(records) >= args.limit:
                    break
            if args.limit and len(records) >= args.limit:
                break

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "dataset": args.dataset, "thresholds": THRESHOLDS,
        "repository": os.getcwd(), "experiment_config": getattr(model, "experiment_config", None),
        "records": records}))
    zero = THRESHOLDS.index(0.0)
    ious = [r["inter"][zero] / u if (u := r["pred"][zero] + r["gt"] - r["inter"][zero]) else 1.0
            for r in records.values()]
    print(json.dumps({"dataset": args.dataset, "n": len(records),
                      "mean_iou_at_0": sum(ious) / len(ious)}), flush=True)


if __name__ == "__main__":
    main()
