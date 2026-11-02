"""Training objective: pixel BCE plus an optional per-image soft Dice term."""

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, masks: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Mean over images of 1 - (2*sum(p*m) + s) / (sum(p) + sum(m) + s), with p = sigmoid(logits).

    logits, masks: (B, 1, H, W). Computed in float32 so AMP does not change the sums. The formula
    and smoothing match the ACIIDS QGDP-Net objective (per-image Dice, smoothing 1), so an image
    with an empty mask contributes a small loss that pushes its prediction towards empty.
    """
    probs = torch.sigmoid(logits.float()).flatten(1)  # (B, H*W)
    target = masks.float().flatten(1)  # (B, H*W)
    intersection = (probs * target).sum(1)  # (B,)
    dice = (2 * intersection + smooth) / (probs.sum(1) + target.sum(1) + smooth)  # (B,)
    return (1 - dice).mean()


def segmentation_loss(logits: torch.Tensor, masks: torch.Tensor, dice_weight: float = 0.0) -> torch.Tensor:
    """BCE-with-logits averaged over pixels, plus dice_weight * per-image soft Dice.

    dice_weight = 0 returns exactly the upstream baseline loss (nn.BCEWithLogitsLoss()).
    """
    loss = F.binary_cross_entropy_with_logits(logits, masks)
    if dice_weight:
        loss = loss + dice_weight * soft_dice_loss(logits, masks)
    return loss
