"""Float32 segmentation losses with area-preserving auxiliary supervision."""

import math

import torch
from torch.nn import functional as F

OBJECTIVE_VERSION = "per-image-lovasz-v1"

TRAINING_DEFAULTS = {
    "overlap_loss": "lovasz",
    "overlap_weight": 0.5,
    "fp_weight": 0.7,
    "aux96_weight": 0.1,
    "aux192_weight": 0.2,
    "aux384_weight": 0.3,
}


def resize_logits(logits, target):
    return F.interpolate(
        logits.float(), size=target.shape[-2:], mode="bilinear", align_corners=False
    )


def per_image_iou(logits, target):
    prediction, truth = resize_logits(logits, target) > 0, target > 0.5
    axes = tuple(range(1, truth.ndim))
    intersection = (prediction & truth).sum(axes).float()
    union = (prediction | truth).sum(axes).float()
    return torch.where(
        union > 0, intersection / union.clamp_min(1), torch.ones_like(union)
    )


def segmentation_loss(
    logits, target, *, overlap_loss="dice", overlap_weight=0.5, fp_weight=0.7
):
    """Per-image overlap averaged over batch, alongside mean pixel BCE."""
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    axes = tuple(range(1, target.ndim))
    tp = (probability * target).sum(axes)
    fp = (probability * (1 - target)).sum(axes)
    fn = ((1 - probability) * target).sum(axes)
    if overlap_loss == "dice":
        score = (2 * tp + 1) / (2 * tp + fp + fn + 1)
    elif overlap_loss == "tversky":
        score = (tp + 1) / (tp + fp_weight * fp + (1 - fp_weight) * fn + 1)
    else:
        raise ValueError("overlap_loss must be dice or tversky")
    return bce + overlap_weight * (1 - score).mean()


def lovasz_hinge_per_image(logits, target):
    """Binary foreground Jaccard surrogate, independently sorted per image.

    Implements the Lovasz-extension construction of Berman et al. (CVPR 2018).
    Targets must be binary. Empty foreground uses the maximum negative-class
    margin error, so an empty image is not silently discarded.
    """
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("Expected matching Bx1xHxW binary masks")
    losses = []
    for score, truth in zip(
        logits.float().flatten(1), target.detach().float().flatten(1)
    ):
        labels = truth > 0.5
        errors = 1 - score * (2 * labels.float() - 1)
        order = torch.argsort(errors.detach(), descending=True, stable=True)
        sorted_truth = labels[order].float()
        positives = sorted_truth.sum()
        intersection = positives - sorted_truth.cumsum(0)
        union = positives + (1 - sorted_truth).cumsum(0)
        jaccard = 1 - intersection / union.clamp_min(1)
        increments = torch.cat((jaccard[:1], jaccard[1:] - jaccard[:-1]))
        losses.append((F.relu(errors[order]) * increments).sum())
    return torch.stack(losses)


def regional_loss_per_image(
    logits, target, *, overlap_loss="dice", overlap_weight=0.5, fp_weight=0.7
):
    """Original BCE/overlap objective, retaining one scalar per original image."""
    logits, target = logits.float(), target.float()
    axes = tuple(range(1, target.ndim))
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(
        axes
    )
    if overlap_loss == "lovasz":
        overlap = lovasz_hinge_per_image(logits, target)
    else:
        p = logits.sigmoid()
        tp, fp, fn = (
            (p * target).sum(axes),
            (p * (1 - target)).sum(axes),
            ((1 - p) * target).sum(axes),
        )
        if overlap_loss == "dice":
            overlap = 1 - (2 * tp + 1) / (2 * tp + fp + fn + 1)
        elif overlap_loss == "tversky":
            overlap = 1 - (tp + 1) / (tp + fp_weight * fp + (1 - fp_weight) * fn + 1)
        else:
            raise ValueError("overlap_loss must be dice, tversky or lovasz")
    return bce + overlap_weight * overlap


class SegmentationObjective:
    """Change only the main overlap term; fractional auxiliary targets keep Dice."""

    def __init__(
        self,
        overlap_loss="lovasz",
        overlap_weight=0.5,
        fp_weight=0.7,
        aux96_weight=0.1,
        aux192_weight=0.2,
        aux384_weight=0.3,
    ):
        if overlap_loss not in {"dice", "tversky", "lovasz"}:
            raise ValueError("overlap_loss must be dice, tversky or lovasz")
        for name, value in {
            "overlap_weight": overlap_weight,
            "fp_weight": fp_weight,
            "aux96_weight": aux96_weight,
            "aux192_weight": aux192_weight,
            "aux384_weight": aux384_weight,
        }.items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < fp_weight < 1:
            raise ValueError("fp_weight must be strictly between zero and one")
        self.loss_kwargs = {
            "overlap_loss": overlap_loss,
            "overlap_weight": overlap_weight,
            "fp_weight": fp_weight,
        }
        self.auxiliary_weights = dict(
            zip(("96", "192", "384"), (aux96_weight, aux192_weight, aux384_weight))
        )

    def per_image(self, output, batch):
        if output["logits"].shape != batch["mask"].shape:
            raise ValueError("Main logits and high-resolution target must match")
        main = regional_loss_per_image(
            output["logits"], batch["mask"], **self.loss_kwargs
        )
        total, components = main, {"segmentation": main.detach().mean()}
        for name, weight in self.auxiliary_weights.items():
            logits = output["auxiliary_logits"][name]
            target = F.interpolate(
                batch["mask"].float(), size=logits.shape[-2:], mode="area"
            )
            # Preserve the parent's area-aware auxiliary BCE + Dice recipe.
            loss = regional_loss_per_image(
                logits,
                target,
                overlap_loss="dice",
                overlap_weight=self.loss_kwargs["overlap_weight"],
                fp_weight=self.loss_kwargs["fp_weight"],
            )
            total = total + weight * loss
            components[f"aux{name}_loss"] = loss.detach().mean()
        return total, components

    def __call__(self, model, batch, epoch=0):
        total, components = self.per_image(model(batch), batch)
        return total.mean(), components
