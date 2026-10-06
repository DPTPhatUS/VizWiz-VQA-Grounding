# compute IoU metric

import torch
import torch.nn.functional as F

def compute_iou(pred_mask, true_mask, threshold=0.5):
    # 1. Resize pred_mask to match true_mask
    if pred_mask.shape != true_mask.shape:
        pred_mask = F.interpolate(pred_mask, size=true_mask.shape[2:], mode="bilinear", align_corners=False)

    pred_mask = (pred_mask > threshold).float()
    true_mask = (true_mask > threshold).float()

    intersection = (pred_mask * true_mask).sum(dim=(1,2,3))
    union = (pred_mask + true_mask - pred_mask * true_mask).sum(dim=(1,2,3))
    iou = intersection / (union + 1e-6)
    return iou.mean().item()


def compute_iou_per_sample(pred_mask, true_mask, threshold=0.5):
    """Compute IoU for each sample in a batch separately. Returns a 1D tensor of
    per-sample IoU values (shape [B])."""
    # 1. Resize pred_mask to match true_mask
    if pred_mask.shape != true_mask.shape:
        pred_mask = F.interpolate(pred_mask, size=true_mask.shape[2:], mode="bilinear", align_corners=False)

    pred_mask = (pred_mask > threshold).float()
    true_mask = (true_mask > threshold).float()

    intersection = (pred_mask * true_mask).sum(dim=(1, 2, 3))
    union = (pred_mask + true_mask - pred_mask * true_mask).sum(dim=(1, 2, 3))
    iou = intersection / (union + 1e-6)
    return iou  # [B]

from pathlib import Path
import numpy as np
from PIL import Image
from utils import to_device
from losses import per_image_iou


def metric_samples(logits, batch, resolution):
    for i, name in enumerate(batch["filename"]):
        truth = batch["original_mask"][i].unsqueeze(0).to(logits.device) if resolution == "original" else batch["mask"][i:i+1]
        resized = F.interpolate(logits[i:i+1].float(), size=truth.shape[-2:],
                                mode="bilinear", align_corners=False)
        score = per_image_iou(resized, truth).item()
        yield name, score, resized[0,0] > 0


@torch.no_grad()
def evaluate(model, loader, device, resolution, output=None):
    model.eval()
    scores = {}
    for batch in loader:
        batch = to_device(batch, device)
        # Dataset validation/test always supplies question-only text.
        logits = model(batch)["logits"]
        for name, score, prediction in metric_samples(logits, batch, resolution):
            scores[name] = score
            if output is not None:
                path = Path(output) / Path(name).with_suffix(".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((prediction.cpu().numpy().astype(np.uint8)*255)).save(path)
    return scores
