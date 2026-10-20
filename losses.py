"""Segmentation and region-answer supervision for grounding first."""

import math

import torch
import torch.distributed as dist
from torch.nn import functional as F


def resize_logits(logits, target):
    if logits.shape[-2:] == target.shape[-2:]:
        return logits
    return F.interpolate(
        logits, size=target.shape[-2:], mode="bilinear", align_corners=False
    )


def segmentation_loss(logits, target, dice_weight=0.0, *, reduction="mean"):
    """Mean batch loss, or one scalar per image for scheduled supervision."""
    logits = resize_logits(logits, target).float()  # (B, 1, H, W)
    target = target.float()  # (B, 1, H, W)
    axes = tuple(range(1, target.ndim))
    if reduction == "mean":
        bce = F.binary_cross_entropy_with_logits(logits, target)
    elif reduction == "none":
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(
            axes
        )
    else:
        raise ValueError("Segmentation reduction must be mean or none")
    if not dice_weight:
        return bce
    probability = logits.sigmoid()  # (B, 1, H, W)
    dice = 1 - (2 * (probability * target).sum(axes) + 1) / (
        probability.sum(axes) + target.sum(axes) + 1
    )  # (B,)
    return bce + dice_weight * (dice.mean() if reduction == "mean" else dice)


def masked_loss_mean(values, available=None, *, valid_mean=False):
    """Legacy batch mean or mean over examples with a valid privileged target."""
    if available is None:
        return values.mean()
    mask = torch.as_tensor(available, dtype=values.dtype, device=values.device).view(-1)
    total = (values * mask).sum()
    if not valid_mean:
        return (values * mask).mean()
    count = mask.sum().detach().clone()
    world = 1
    if dist.is_initialized():
        dist.all_reduce(count)
        world = dist.get_world_size()
    # DDP averages gradients; each rank contributes to the global valid-example mean.
    return total * world / count.clamp_min(1)


def per_image_iou(logits, target):
    prediction = resize_logits(logits, target) > 0  # (B, 1, H, W)
    truth = target > 0.5  # (B, 1, H, W)
    axes = tuple(range(1, target.ndim))
    intersection = (prediction & truth).sum(axes).float()  # (B,)
    union = (prediction | truth).sum(axes).float()  # (B,)
    return torch.where(
        union > 0, intersection / union.clamp_min(1), torch.ones_like(union)
    )


def answer_embedding_loss(latent, target, available=None, *, loss_version="stable"):
    """Mean cosine-plus-SmoothL1 error over globally valid region/answer pairs."""
    if loss_version != "stable":
        raise ValueError("This experiment uses valid-answer reduction only")
    latent, target = latent.float(), target.detach().float()  # (B, D_t), (B, D_t)
    cosine = 1 - F.cosine_similarity(latent, target, dim=-1, eps=1e-6)  # (B,)
    l1 = F.smooth_l1_loss(latent, target, reduction="none").mean(dim=-1)  # (B,)
    loss = cosine + l1  # (B,)
    return masked_loss_mean(loss, available, valid_mean=loss_version == "stable")


class GroundingAnswerObjective:
    """Mask supervision plus a GT-trained reader and mask-only answer gradients."""

    def __init__(
        self,
        dice_weight=0.5,
        reader_weight=0.5,
        answer_weight=0.5,
        answer_warmup_epochs=2,
    ):
        for name, value in (
            ("dice_weight", dice_weight),
            ("reader_weight", reader_weight),
            ("answer_weight", answer_weight),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(answer_warmup_epochs) is not int or answer_warmup_epochs < 0:
            raise ValueError("answer_warmup_epochs must be a nonnegative integer")
        if answer_weight and not reader_weight:
            raise ValueError("Mask-answer supervision requires GT reader supervision")
        self.dice_weight = dice_weight
        self.reader_weight = reader_weight
        self.answer_weight = answer_weight
        self.answer_warmup_epochs = answer_warmup_epochs

    def __call__(self, model, batch, epoch=0):
        output = model(batch)
        segmentation = segmentation_loss(
            output["logits"], batch["mask"], self.dice_weight
        )
        available = (
            batch["answer_available"].to(output["region_nonempty"].device)
            & output["region_nonempty"]
        )
        reader = answer_embedding_loss(
            output["gt_region_answer"],
            output["target_answer"],
            available,
            loss_version="stable",
        )
        answer = answer_embedding_loss(
            output["predicted_answer"],
            output["target_answer"],
            available,
            loss_version="stable",
        )
        effective = self.answer_weight if epoch >= self.answer_warmup_epochs else 0.0
        total = segmentation + self.reader_weight * reader + effective * answer
        return total, {
            "segmentation": segmentation.detach(),
            "reader_gt_loss": reader.detach(),
            "mask_answer_loss": answer.detach(),
            "weighted_reader_loss": (self.reader_weight * reader).detach(),
            "weighted_mask_answer_loss": (effective * answer).detach(),
            "effective_answer_weight": segmentation.new_tensor(effective),
            "valid_region_answer_fraction": available.float().mean().detach(),
        }
