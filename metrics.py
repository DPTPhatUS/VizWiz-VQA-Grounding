# compute IoU metric

import torch
import torch.nn.functional as F

def compute_iou(pred_mask, true_mask, threshold=0.5):
    return compute_iou_per_sample(pred_mask, true_mask, threshold).mean().item()


def compute_iou_per_sample(pred_mask, true_mask, threshold=0.5, *, legacy=False):
    """Compute IoU for each sample in a batch separately. Returns a 1D tensor of
    per-sample IoU values (shape [B])."""
    # 1. Resize pred_mask to match true_mask
    if pred_mask.shape[-2:] != true_mask.shape[-2:]:
        pred_mask = F.interpolate(pred_mask.float(), size=true_mask.shape[2:], mode="bilinear", align_corners=False)  # (B, 1, H, W)

    pred_mask = (pred_mask > threshold).float()  # (B, 1, H, W)
    true_mask = (true_mask > threshold).float()  # (B, 1, H, W)

    intersection = (pred_mask * true_mask).sum(dim=(1, 2, 3))  # (B,)
    union = (pred_mask + true_mask - pred_mask * true_mask).sum(dim=(1, 2, 3))  # (B,)
    if legacy:
        return intersection / (union + 1e-6)
    # Match the distillation branches: two empty masks have IoU 1.
    return torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union))
