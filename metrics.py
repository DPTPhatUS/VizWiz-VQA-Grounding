"""Native-resolution per-image/global foreground IoU and mask export."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm.auto import tqdm

from losses import per_image_iou
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


def metric_samples(logits, batch, *, pixel_totals=None):
    for i, name in enumerate(batch["filename"]):
        truth = (
            batch["original_mask"][i].unsqueeze(0).to(logits.device, non_blocking=True)
        )  # (1, 1, H_orig, W_orig)
        resized = F.interpolate(
            logits[i : i + 1].float(),
            size=truth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, H_orig, W_orig)
        score = per_image_iou(resized, truth).item()
        prediction = resized[0, 0] > 0  # (H_orig, W_orig)
        accumulate_pixel_counts(prediction, truth[0, 0] > 0.5, pixel_totals)
        yield name, score, prediction


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
    output=None,
    *,
    pixel_totals=None,
    view_results=None,
    scope_diagnostics=None,
):
    model.eval()
    scores = {}
    for batch in tqdm(loader, desc="Evaluating", unit="batch", dynamic_ncols=True):
        batch = to_device(batch, device)
        # Dataset validation/test always supplies question-only text.
        result = model(batch)
        logits = result["logits"]
        for name, score, prediction in metric_samples(
            logits, batch, pixel_totals=pixel_totals
        ):
            if name in scores:
                raise ValueError(f"Duplicate evaluation sample: {name}")
            scores[name] = score
            if output is not None:
                path = Path(output) / Path(name).with_suffix(".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(prediction.cpu().numpy().astype(np.uint8) * 255).save(
                    path
                )
        if view_results is not None:
            for view in ("parent", "candidate"):
                state = view_results.setdefault(
                    view, {"per_sample": {}, "pixel_totals": {}}
                )
                for name, score, _ in metric_samples(
                    result[view + "_logits"], batch, pixel_totals=state["pixel_totals"]
                ):
                    state["per_sample"][name] = score
        if scope_diagnostics is not None:
            selection = result["scope"]
            for i, name in enumerate(batch["filename"]):
                truth = batch["original_mask"][i].unsqueeze(0).to(logits.device) > 0.5
                shape = truth.shape[-2:]
                original = (
                    F.interpolate(
                        result["parent_logits"][i : i + 1].float(),
                        size=shape,
                        mode="bilinear",
                        align_corners=False,
                    )
                    > 0
                )
                candidate = (
                    F.interpolate(
                        result["candidate_logits"][i : i + 1].float(),
                        size=shape,
                        mode="bilinear",
                        align_corners=False,
                    )
                    > 0
                )
                count = int((original & truth).sum())
                destroyed = int((original & truth & ~candidate).sum())
                scope_diagnostics[name] = {
                    "selected": bool(selection.selected[i]),
                    "predicted_gain": float(selection.gain[i]),
                    "predicted_damage": float(selection.damage_logits[i].sigmoid()),
                    "actual_damage": destroyed / count if count else 0.0,
                    "removed_pixels": int((original & ~candidate).sum()),
                    "gt_area_fraction": float(truth.float().mean()),
                }
    return scores
