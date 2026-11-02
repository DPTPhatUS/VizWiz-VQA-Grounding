"""Per-image IoU, the official VizWiz-VQA-Grounding metric (mean over images)."""

import torch


def iou_per_sample(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
    """Binary masks (B, 1, H, W) of equal size -> IoU per image (B,). Two empty masks score 1."""
    pred, true = pred_mask.bool(), true_mask.bool()
    intersection = (pred & true).sum(dim=(1, 2, 3)).float()  # (B,)
    union = (pred | true).sum(dim=(1, 2, 3)).float()  # (B,)
    return torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union))
