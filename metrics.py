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