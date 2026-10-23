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
    "answer_weight": 0.01,
    "answer_warmup_epochs": 0,
    "answer_ramp_epochs": 1,
    "mask_answer_weight": 0.01,
    "mask_answer_warmup_epochs": 5,
    "mask_answer_ramp_epochs": 5,
    "reader_margin": 0.05,
    "mask_answer_tolerance": 0.05,
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


def answer_statistics(output):
    """Per-image CE; missing answers contribute zero, with explicit coverage.

    The outer mean counts original images, matching DDP and partial-window
    sample weighting. This is intentionally not a mean over valid tokens globally.
    """
    logits, targets = output["answer_logits"].float(), output["answer_targets"]
    valid = targets != -100
    count = valid.sum(1)
    ce = F.cross_entropy(
        logits.transpose(1, 2), targets, reduction="none", ignore_index=-100
    )
    loss = ce.sum(1) / count.clamp_min(1)
    correct = ((logits.argmax(-1) == targets) & valid).sum(1)
    accuracy = correct.float() / count.clamp_min(1)
    return loss, accuracy, count > 0


class SegmentationObjective:
    """Segmentation plus a GT-trained reader and reliability-gated mask guidance."""

    def __init__(
        self,
        overlap_loss="dice",
        overlap_weight=0.5,
        fp_weight=0.7,
        aux96_weight=0.1,
        aux192_weight=0.2,
        aux384_weight=0.3,
        answer_weight=0.01,
        answer_warmup_epochs=0,
        answer_ramp_epochs=1,
        mask_answer_weight=0.01,
        mask_answer_warmup_epochs=5,
        mask_answer_ramp_epochs=5,
        reader_margin=0.05,
        mask_answer_tolerance=0.05,
    ):
        if overlap_loss not in {"dice", "tversky"}:
            raise ValueError("overlap_loss must be dice or tversky")
        for name, value in {
            "overlap_weight": overlap_weight,
            "fp_weight": fp_weight,
            "aux96_weight": aux96_weight,
            "aux192_weight": aux192_weight,
            "aux384_weight": aux384_weight,
            "answer_weight": answer_weight,
            "mask_answer_weight": mask_answer_weight,
            "reader_margin": reader_margin,
            "mask_answer_tolerance": mask_answer_tolerance,
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
        if type(answer_warmup_epochs) is not int or answer_warmup_epochs < 0:
            raise ValueError("answer_warmup_epochs must be a nonnegative integer")
        if type(answer_ramp_epochs) is not int or answer_ramp_epochs <= 0:
            raise ValueError("answer_ramp_epochs must be a positive integer")
        if answer_weight == 0 and mask_answer_weight > 0:
            raise ValueError(
                "Mask guidance requires GT-reader training; set both answer weights to zero to disable it"
            )
        self.answer_weight = answer_weight
        self.answer_warmup_epochs = answer_warmup_epochs
        self.answer_ramp_epochs = answer_ramp_epochs
        if type(mask_answer_warmup_epochs) is not int or mask_answer_warmup_epochs < 0:
            raise ValueError("mask_answer_warmup_epochs must be a nonnegative integer")
        if type(mask_answer_ramp_epochs) is not int or mask_answer_ramp_epochs <= 0:
            raise ValueError("mask_answer_ramp_epochs must be a positive integer")
        self.mask_answer_weight = mask_answer_weight
        self.mask_answer_warmup_epochs = mask_answer_warmup_epochs
        self.mask_answer_ramp_epochs = mask_answer_ramp_epochs
        self.reader_margin = reader_margin
        self.mask_answer_tolerance = mask_answer_tolerance

    def __call__(self, model, batch, epoch=0):
        mask_progress = min(
            1.0,
            max(
                0.0,
                (epoch - self.mask_answer_warmup_epochs + 1)
                / self.mask_answer_ramp_epochs,
            ),
        )
        mask_weight = self.mask_answer_weight * mask_progress
        output = model(
            {
                **batch,
                "answer_supervision": True,
                "mask_answer_supervision": mask_weight > 0,
            }
        )
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
        answer, accuracy, available = answer_statistics(output)
        progress = min(
            1.0,
            max(0.0, (epoch - self.answer_warmup_epochs + 1) / self.answer_ramp_epochs),
        )
        effective = self.answer_weight * progress
        total = total + effective * answer.mean()
        components.update(
            answer_loss=answer.mean().detach(),
            answer_accuracy=accuracy.mean().detach(),
            answer_valid_fraction=available.float().mean(),
            effective_answer_weight=answer.new_tensor(effective),
        )
        if "predicted_answer" in output:
            guidance, metrics = mask_answer_statistics(
                output,
                reader_margin=self.reader_margin,
                tolerance=self.mask_answer_tolerance,
            )
            total = total + mask_weight * guidance.mean()
            components.update(metrics)
        else:
            components.update(
                {
                    key: main.detach().new_zeros(())
                    for key in (
                        "mask_answer_loss",
                        "reader_reliable_fraction",
                        "reader_visual_gain",
                        "predicted_answer_loss",
                        "mask_answer_active_fraction",
                    )
                }
            )
        components["effective_mask_answer_weight"] = main.detach().new_tensor(
            mask_weight
        )
        return total, components


def mask_answer_statistics(output, *, reader_margin, tolerance):
    """One detached reliability decision per image; preserve original-batch averaging.

    A reader must prefer GT to the full image before guiding a predicted mask.
    The hinge stops pushing once predicted-region CE reaches the GT reference
    within tolerance. Invalid GT answers/regions never contribute to guidance.
    """
    reference, _, valid = answer_statistics(output)
    full, _, _ = answer_statistics(output["full_answer"])
    predicted, _, _ = answer_statistics(output["predicted_answer"])
    gain = full.detach() - reference.detach()
    reliable = valid & (gain > reader_margin)
    gap = F.relu(predicted - reference.detach() - tolerance)
    loss = gap * reliable.to(gap.dtype)
    return loss, {
        "mask_answer_loss": loss.mean().detach(),
        "reader_reliable_fraction": reliable.float().mean(),
        "reader_visual_gain": (gain * valid).mean(),
        "predicted_answer_loss": (predicted.detach() * valid).mean(),
        "mask_answer_active_fraction": (reliable & (gap.detach() > 0)).float().mean(),
    }
