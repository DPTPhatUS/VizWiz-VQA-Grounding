"""Native-resolution per-image/global foreground IoU and mask export."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from losses import per_image_iou
from models.zoom import prediction_at_size
from utils import to_device


def accumulate_pixel_counts(prediction, truth, totals):
    """Accumulate integer foreground counts on the native mask grid."""
    if totals is None:
        return
    intersection, union = torch.stack(
        [(prediction & truth).sum(), (prediction | truth).sum()]
    ).tolist()
    totals["intersection"] = totals.get("intersection", 0) + intersection
    totals["union"] = totals.get("union", 0) + union


def global_iou_summary(totals):
    """Dataset foreground IoU; an empty union scores one, as with per-image IoU."""
    intersection, union = totals.get("intersection", 0), totals.get("union", 0)
    return {
        "global_iou": intersection / union if union else 1.0,
        "total_intersection": intersection,
        "total_union": union,
        "global_iou_definition": "foreground_sum_intersection_over_sum_union",
    }


def metric_samples(output, batch, diagnostics=None, *, pixel_totals=None):
    device = output["logits"].device
    for i, name in enumerate(batch["filename"]):
        truth = (
            batch["original_mask"][i].unsqueeze(0).to(device, non_blocking=True)
        )  # (1, 1, H_orig, W_orig)
        resized = prediction_at_size(output, i, truth.shape[-2:])
        score = per_image_iou(resized, truth).item()
        if diagnostics is not None:
            coarse = prediction_at_size(output, i, truth.shape[-2:], global_only=True)
            boxes = output["zoom"]["boxes"][i]
            height, width = truth.shape[-2:]
            y = (torch.arange(height, device=device).float() + 0.5) / height
            x = (torch.arange(width, device=device).float() + 0.5) / width
            covered = torch.zeros((height, width), dtype=torch.bool, device=device)
            for box in boxes:
                covered |= ((y >= box[1]) & (y <= box[3]))[:, None] & (
                    (x >= box[0]) & (x <= box[2])
                )[None, :]
            foreground = truth[0, 0] > 0.5
            coverage = (
                (covered & foreground).sum().float() / foreground.sum().clamp_min(1)
            ).item()
            diagnostics[name] = dict(
                global_iou=per_image_iou(coarse, truth).item(),
                zoom_iou=score,
                crop_gt_coverage=coverage,
                gt_empty=not bool(foreground.any()),
                boxes_xyxy=boxes.cpu().tolist(),
                crop_quality=output["zoom"]["quality"][i]
                .float()
                .sigmoid()
                .cpu()
                .tolist(),
            )
        prediction = resized[0, 0] > 0  # (H_orig, W_orig)
        accumulate_pixel_counts(prediction, truth[0, 0] > 0.5, pixel_totals)
        yield name, score, prediction


@torch.inference_mode()
def evaluate(
    model, loader, device, output=None, diagnostics=None, *, pixel_totals=None
):
    model.eval()
    scores = {}
    for batch in tqdm(loader, desc="Evaluating", unit="batch", dynamic_ncols=True):
        batch = to_device(batch, device)
        # Dataset validation/test always supplies question-only text.
        prediction_output = model(batch)
        for name, score, prediction in metric_samples(
            prediction_output, batch, diagnostics, pixel_totals=pixel_totals
        ):
            if name in scores:
                raise ValueError(f"Duplicate evaluation sample: {name}")
            scores[name] = score
            if output is not None:
                path = Path(output) / Path(name).with_suffix(".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((prediction.cpu().numpy().astype(np.uint8) * 255)).save(
                    path
                )
    return scores
