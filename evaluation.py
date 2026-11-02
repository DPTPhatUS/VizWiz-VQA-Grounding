"""The one per-image scoring rule, shared by eval.py and validation during training."""

import torch
import torch.nn.functional as F

from metrics import compute_iou_per_sample


def score_sample(logits: torch.Tensor, truth: torch.Tensor, *, legacy: bool = False):
    """Threshold one image's logits at the ground-truth resolution and score it.

    logits: (1, 1, h, w) model output for one image. truth: (1, 1, H, W) binary mask, native size for
    the official protocol (legacy=False: empty union scores 1) or 336 px for the legacy protocol.
    Returns (prediction (1, 1, H, W) float {0, 1}, IoU as a Python float).
    """
    logits = F.interpolate(logits.float(), size=truth.shape[-2:], mode="bilinear", align_corners=False)
    prediction = (logits > 0).float()
    score = compute_iou_per_sample(prediction, truth, legacy=legacy).item()
    return prediction, score
