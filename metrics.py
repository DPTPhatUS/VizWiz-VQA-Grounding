"""Original-resolution per-image IoU and mask export."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from losses import per_image_iou
from models.zoom import prediction_at_size
from utils import to_device


def metric_samples(output, batch, diagnostics=None):
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
        yield name, score, resized[0, 0] > 0  # (H_orig, W_orig)


@torch.inference_mode()
def evaluate(model, loader, device, output=None, diagnostics=None):
    model.eval()
    scores = {}
    for batch in tqdm(loader, desc="Evaluating", unit="batch", dynamic_ncols=True):
        batch = to_device(batch, device)
        # Dataset validation/test always supplies question-only text.
        prediction_output = model(batch)
        for name, score, prediction in metric_samples(
            prediction_output, batch, diagnostics
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
