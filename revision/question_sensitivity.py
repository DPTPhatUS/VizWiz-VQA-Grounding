"""A5: how much QGDP-Net's predicted mask depends on the question (inference only).

The image is fixed and only the question text changes. The primary measures compare
each intervention's prediction with the reference prediction (original question) at
native resolution, so they quantify sensitivity, not correctness:
  - agreement IoU between the two predicted masks (1 = unchanged),
  - fraction of native pixels whose label flips,
  - log2 ratio of predicted foreground areas.
IoU against the ground truth is also stored, as a descriptive secondary measure only:
for a donor question the ground truth no longer matches the question.

Interventions (fixed seed, recorded per image):
  repeat         original question again (determinism check)
  donor_random   question of another image of the split
  donor_matched  question of another image whose target is in the same size group
                 (<1%, 1-10%, >=10%), so the donor does not carry a different size prior
  generic        "What is this?"
  empty          no question words (just the "Q:" prefix used in training)

Usage:
  python revision/question_sensitivity.py --checkpoint CKPT --data-root ROOT \
      --dataset val --output-dir OUT [--batch-size 4] [--device cuda] [--seed 2027]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import VizWizGroundingDataset, collate_samples  # noqa: E402
from models.model import GroundingModel  # noqa: E402
from utils import checkpoint_model_kwargs, load_model_weights, read_checkpoint, to_device  # noqa: E402

CONDITIONS = ("repeat", "donor_random", "donor_matched", "generic", "empty")


def size_group(area):
    return 0 if area < 0.01 else 1 if area < 0.1 else 2


def donors(names, questions, groups, seed):
    """Donor indices with a different question; matched donors share the size group."""
    rng = random.Random(seed)
    index = list(range(len(names)))
    random_donor, matched_donor = {}, {}
    by_group = {g: [i for i in index if groups[i] == g] for g in set(groups)}
    for i in index:
        pool = [j for j in index if questions[j].strip().lower() != questions[i].strip().lower()]
        random_donor[i] = rng.choice(pool)
        same = [j for j in by_group[groups[i]] if questions[j].strip().lower() != questions[i].strip().lower()]
        matched_donor[i] = rng.choice(same) if same else random_donor[i]
    return random_donor, matched_donor


@torch.inference_mode()
def native_masks(model, batch, questions, device):
    out = model(dict(batch, question_text=questions))["logits"]
    masks = []
    for i, truth in enumerate(batch["original_mask"]):
        resized = F.interpolate(out[i : i + 1].float(), size=truth.shape[-2:], mode="bilinear", align_corners=False)
        masks.append(resized[0, 0] > 0)
    return masks


def iou(a, b):
    union = (a | b).sum().item()
    return 1.0 if union == 0 else (a & b).sum().item() / union


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--limit", type=int, default=None, help="Optional subset for smoke tests")
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be new or empty")
    saved = read_checkpoint(args.checkpoint)
    model = GroundingModel(**checkpoint_model_kwargs(saved))
    load_model_weights(model, saved)
    device = torch.device(args.device)
    model.to(device).eval()

    dataset = VizWizGroundingDataset(args.data_root, args.dataset, model.image_size, detail_size=model.detail_size)
    count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    names = dataset.names[:count]
    questions = [str(dataset.records[n].get("question") or "").strip() for n in names]

    # Size groups come from the native ground-truth masks (needed only to match donors).
    groups = [size_group(float(dataset[i]["original_mask"].mean())) for i in range(count)]
    random_donor, matched_donor = donors(names, questions, groups, args.seed)
    text = {
        "repeat": lambda i: f"Q: {questions[i]}",
        "donor_random": lambda i: f"Q: {questions[random_donor[i]]}",
        "donor_matched": lambda i: f"Q: {questions[matched_donor[i]]}",
        "generic": lambda i: "Q: What is this?",
        "empty": lambda i: "Q:",
    }

    rows = {}
    for start in range(0, count, args.batch_size):
        idx = list(range(start, min(start + args.batch_size, count)))
        batch = to_device(collate_samples([dataset[i] for i in idx]), device)
        batch["original_mask"] = [m.to(device) for m in batch["original_mask"]]
        reference = native_masks(model, batch, [f"Q: {questions[i]}" for i in idx], device)
        per_condition = {c: native_masks(model, batch, [text[c](i) for i in idx], device) for c in CONDITIONS}
        for k, i in enumerate(idx):
            truth = batch["original_mask"][k][0] > 0.5
            ref = reference[k]
            row = {"group": groups[i], "question": questions[i],
                   "donor_random_image": names[random_donor[i]], "donor_matched_image": names[matched_donor[i]],
                   "reference_gt_iou": iou(ref, truth), "reference_area": ref.float().mean().item()}
            for c in CONDITIONS:
                pred = per_condition[c][k]
                ref_area, pred_area = ref.sum().item(), pred.sum().item()
                row[c] = {
                    "agreement_iou": iou(pred, ref),
                    "changed_fraction": (pred ^ ref).float().mean().item(),
                    "log2_area_ratio": float(np.log2((pred_area + 1) / (ref_area + 1))),
                    "gt_iou_descriptive": iou(pred, truth),
                }
            rows[names[i]] = row

    def summarize(sel):
        res = {"n": len(sel), "reference_gt_miou": 100 * float(np.mean([r["reference_gt_iou"] for r in sel]))}
        for c in CONDITIONS:
            res[c] = {
                "mean_agreement_iou": 100 * float(np.mean([r[c]["agreement_iou"] for r in sel])),
                "median_agreement_iou": 100 * float(np.median([r[c]["agreement_iou"] for r in sel])),
                "mean_changed_pct": 100 * float(np.mean([r[c]["changed_fraction"] for r in sel])),
                "share_agreement_below_50": float(np.mean([r[c]["agreement_iou"] < 0.5 for r in sel])),
                "mean_abs_log2_area_ratio": float(np.mean([abs(r[c]["log2_area_ratio"]) for r in sel])),
                "gt_miou_descriptive": 100 * float(np.mean([r[c]["gt_iou_descriptive"] for r in sel])),
            }
        return res

    summary = {"checkpoint": args.checkpoint, "dataset": args.dataset, "seed": args.seed,
               "experiment_config": model.experiment_config,
               "all": summarize(list(rows.values())),
               "by_group": {name: summarize([r for r in rows.values() if r["group"] == g])
                            for g, name in enumerate(("<1%", "1-10%", ">=10%"))
                            if any(r["group"] == g for r in rows.values())}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "per_image.json").write_text(json.dumps(rows))
    print(json.dumps(summary["all"], indent=2))


if __name__ == "__main__":
    main()
