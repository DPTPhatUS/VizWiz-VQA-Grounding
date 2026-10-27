"""Shared native output contract for evaluation and annotation-free prediction."""

import torch
from torch.nn import functional as F

from metrics import accumulate_pixel_counts
from ocr_selection_data import native_hypothesis


def selected_native_mask(output, batch, index):
    size = tuple(int(v) for v in batch["original_size"][index])
    selected = int(output["selected_index"][index])
    if not selected:
        return (
            F.interpolate(
                output["base_logits"][index : index + 1].float(),
                size,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            > 0
        )
    members = batch["hypothesis_members"][index, selected].nonzero().flatten().tolist()
    mask = native_hypothesis(size[::-1], batch["evidence_polygons"][index], members)
    return torch.from_numpy(mask.copy()).to(output["base_logits"].device)


def selection_metric_samples(output, batch, *, pixel_totals=None):
    for i, name in enumerate(batch["filename"]):
        prediction = selected_native_mask(output, batch, i)
        target = batch["original_mask"][i][0].to(prediction.device).bool()
        intersection, union = (prediction & target).sum(), (prediction | target).sum()
        score = (intersection / union).item() if union else 1.0
        accumulate_pixel_counts(prediction, target, pixel_totals)
        yield name, score, prediction
