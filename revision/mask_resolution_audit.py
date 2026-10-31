"""A6: how much target information survives resizing (CPU only, read-only).

For every ground-truth mask of a split, at native resolution:
  1. Training-target fidelity: the mask is nearest-resized to 768x768 (exactly as in
     dataset.py), then nearest-resized back. We report whether it vanishes, its
     area ratio, and its IoU with the native mask.
  2. Grid ceiling: the mask is area-averaged to an s x s grid, bilinearly upsampled to
     native size and thresholded at 0.5, mirroring how predicted logits are resized
     before thresholding. The IoU with the native mask is an approximate upper bound
     for a prediction made on that grid (s = 24: ViT patch grid; 96, 384, 768).
Results are summarised by native area bracket.

Usage: python revision/mask_resolution_audit.py MASK_DIR OUT_JSON [--limit N]
MASK_DIR holds the binary PNG masks of one split (e.g. .../binary_masks_png/train).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

GRIDS = (24, 96, 384, 768)
BRACKETS = [("<1%", 0, .01), ("1-5%", .01, .05), ("5-25%", .05, .25), ("25-50%", .25, .5), (">=50%", .5, 2)]


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def audit(mask):
    h, w = mask.shape
    row = {"area": float(mask.mean())}
    small = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((768, 768), Image.Resampling.NEAREST)) > 127
    back = np.array(Image.fromarray(small.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 127
    row["train768_area"] = float(small.mean())
    row["train768_vanished"] = bool(mask.any() and not small.any())
    row["train768_iou"] = iou(mask, back)
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    for s in GRIDS:
        coarse = F.interpolate(t, size=(s, s), mode="area")
        up = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy() > 0.5
        row[f"ceiling_{s}"] = iou(mask, up)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mask_dir")
    parser.add_argument("out")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    paths = sorted(Path(args.mask_dir).glob("*.png"))[: args.limit]
    rows = {}
    for path in paths:
        with Image.open(path) as m:
            rows[path.name] = audit(np.array(m.convert("L")) > 127)
    summary = {"n": len(rows), "brackets": {}}
    for name, lo, hi in BRACKETS:
        sel = [r for r in rows.values() if lo <= r["area"] < hi]
        if not sel:
            continue
        entry = {"n": len(sel),
                 "vanished_at_768": int(sum(r["train768_vanished"] for r in sel)),
                 "median_area_ratio_768": float(np.median([r["train768_area"] / r["area"] for r in sel if r["area"] > 0])),
                 "mean_iou_768_roundtrip": 100 * float(np.mean([r["train768_iou"] for r in sel]))}
        for s in GRIDS:
            entry[f"mean_ceiling_iou_{s}"] = 100 * float(np.mean([r[f"ceiling_{s}"] for r in sel]))
        summary["brackets"][name] = entry
    summary["all"] = {f"mean_ceiling_iou_{s}": 100 * float(np.mean([r[f"ceiling_{s}"] for r in rows.values()])) for s in GRIDS}
    summary["all"]["vanished_at_768"] = int(sum(r["train768_vanished"] for r in rows.values()))
    summary["all"]["empty_masks"] = int(sum(r["area"] == 0 for r in rows.values()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "per_image": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
