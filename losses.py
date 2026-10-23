"""Float32 segmentation losses with area-preserving auxiliary supervision."""

import math

import torch
from torch.nn import functional as F

TRAINING_DEFAULTS = {
    "overlap_loss": "dice",
    "overlap_weight": 0.5,
    "fp_weight": 0.7,
    "aux96_weight": 0.1,
    "aux192_weight": 0.2,
    "aux384_weight": 0.3,
    "coarse_weight": 0.2,
    "area_weight": 0.1,
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


class SegmentationObjective:
    """Full-resolution mask plus three auxiliary heads; no GT answer is consumed."""

    def __init__(
        self,
        overlap_loss="dice",
        overlap_weight=0.5,
        fp_weight=0.7,
        aux96_weight=0.1,
        aux192_weight=0.2,
        aux384_weight=0.3,
        coarse_weight=0.2,
        area_weight=0.1,
    ):
        if overlap_loss not in {"dice", "tversky"}:
            raise ValueError("overlap_loss must be dice or tversky")
        for name, value in {
            "overlap_weight": overlap_weight,
            "fp_weight": fp_weight,
            "aux96_weight": aux96_weight,
            "aux192_weight": aux192_weight,
            "aux384_weight": aux384_weight,
            "coarse_weight": coarse_weight,
            "area_weight": area_weight,
        }.items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < fp_weight < 1:
            raise ValueError("fp_weight must be strictly between zero and one")
        self.coarse_weight, self.area_weight = coarse_weight, area_weight
        self.loss_kwargs = {
            "overlap_loss": overlap_loss,
            "overlap_weight": overlap_weight,
            "fp_weight": fp_weight,
        }
        self.auxiliary_weights = dict(
            zip(("96", "192", "384"), (aux96_weight, aux192_weight, aux384_weight))
        )

    def __call__(self, model, batch, epoch=0):
        output = model(batch)
        if output["logits"].shape != batch["mask"].shape:
            raise ValueError(
                "Main logits and high-resolution target must have matching shapes"
            )
        main = segmentation_loss(output["logits"], batch["mask"], **self.loss_kwargs)
        total, components = main, {"segmentation": main.detach()}
        for name, weight in self.auxiliary_weights.items():
            logits = output["auxiliary_logits"][name]
            # Area targets preserve fractional coverage of tiny regions at coarse grids.
            target = F.interpolate(
                batch["mask"].float(), size=logits.shape[-2:], mode="area"
            )
            loss = segmentation_loss(logits, target, **self.loss_kwargs)
            total = total + weight * loss
            components[f"aux{name}_loss"] = loss.detach()
        coarse = segmentation_loss(
            output["coarse_logits"], batch["mask"], **self.loss_kwargs
        )
        target_area = batch["mask"].float().mean((1, 2, 3)).clamp(1e-6, 1 - 1e-6)
        area = F.smooth_l1_loss(output["area_logit"], torch.logit(target_area))
        total = total + self.coarse_weight * coarse + self.area_weight * area
        components.update(
            coarse_segmentation=coarse.detach(),
            area_loss=area.detach(),
            area_absolute_error=(output["predicted_area"].detach() - target_area)
            .abs()
            .mean(),
        )
        return total, components
