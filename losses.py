"""Per-image deletion/retention supervision and detached whole-mask edit targets."""

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from models.scope_correction import ScopePrediction

TRAINING_DEFAULTS = {
    "removal_weight": 1.0,
    "retention_weight": 2.0,
    "segmentation_weight": 0.5,
    "gain_weight": 1.0,
    "damage_weight": 0.2,
    "corruption_weight": 0.5,
    "corruption_probability": 0.5,
}


def resize_logits(logits: Tensor, target: Tensor) -> Tensor:
    return F.interpolate(
        logits.float(), size=target.shape[-2:], mode="bilinear", align_corners=False
    )


def per_image_iou(logits: Tensor, target: Tensor) -> Tensor:
    prediction, truth = resize_logits(logits, target) > 0, target > 0.5
    axes = tuple(range(1, truth.ndim))
    intersection, union = (
        (prediction & truth).sum(axes).float(),
        (prediction | truth).sum(axes).float(),
    )
    return torch.where(
        union > 0, intersection / union.clamp_min(1), torch.ones_like(union)
    )


def masked_per_image_mean(values: Tensor, mask: Tensor) -> Tensor:
    axes = tuple(range(1, values.ndim))
    # Empty pixel sets contribute zero for that image, preserving DDP sample weighting.
    return (values * mask).sum(axes) / mask.sum(axes).clamp_min(1)


def segmentation_loss(logits: Tensor, target: Tensor) -> Tensor:
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    axes = tuple(range(1, target.ndim))
    score = (2 * (probability * target).sum(axes) + 1) / (
        probability.sum(axes) + target.sum(axes) + 1
    )
    return bce + 0.5 * (1 - score).mean()


@torch.no_grad()
def edit_targets(
    view: ScopePrediction, target: Tensor, native_targets: list[Tensor] | None = None
) -> tuple[Tensor, Tensor]:
    """Whole-image gain and fraction of the parent's true positives destroyed."""
    if native_targets is not None:
        gains, damages = [], []
        for index, native in enumerate(native_targets):
            truth = native.unsqueeze(0).to(view.source_logits.device) > 0
            source = resize_logits(view.source_logits[index : index + 1], truth)
            candidate = resize_logits(view.candidate_logits[index : index + 1], truth)
            count = ((source > 0) & truth).sum().float()
            destroyed = ((source > 0) & truth & (candidate <= 0)).sum().float()
            gains.append(
                (per_image_iou(candidate, truth) - per_image_iou(source, truth))[0]
            )
            damages.append(
                torch.where(
                    count > 0, destroyed / count.clamp_min(1), torch.zeros_like(count)
                )
            )
        return torch.stack(gains), torch.stack(damages)
    original, candidate, truth = (
        view.source_logits > 0,
        view.candidate_logits > 0,
        target > 0.5,
    )
    axes = tuple(range(1, target.ndim))
    correct = original & truth
    count = correct.sum(axes).float()
    destroyed = (correct & ~candidate).sum(axes).float()
    damage = torch.where(
        count > 0, destroyed / count.clamp_min(1), torch.zeros_like(count)
    )
    gain = per_image_iou(view.candidate_logits, target) - per_image_iou(
        view.source_logits, target
    )
    return gain, damage


class SegmentationObjective:
    def __init__(self, **settings):
        unknown = settings.keys() - TRAINING_DEFAULTS.keys()
        if unknown:
            raise ValueError(f"Unknown scope loss settings: {sorted(unknown)}")
        self.settings = {**TRAINING_DEFAULTS, **settings}
        for name, value in self.settings.items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.settings["corruption_probability"] > 1:
            raise ValueError("corruption_probability must be in [0, 1]")

    def view_loss(
        self,
        view: ScopePrediction,
        target: Tensor,
        native_targets: list[Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if view.source_logits.shape != target.shape:
            raise ValueError(
                "Scope logits and training masks must have matching shapes"
            )
        parent_foreground, truth = view.source_logits > 0, target > 0.5
        remove, retain = parent_foreground & ~truth, parent_foreground & truth
        removal = masked_per_image_mean(
            F.softplus(-view.removal_logits.float()), remove
        ).mean()
        retention = masked_per_image_mean(
            F.softplus(view.removal_logits.float()), retain
        ).mean()
        gain_target, damage_target = edit_targets(view, target, native_targets)
        components = {
            "removal": removal,
            "retention": retention,
            "segmentation": segmentation_loss(view.candidate_logits, target),
            "gain": F.smooth_l1_loss(view.gain, gain_target, beta=0.05),
            "damage": F.binary_cross_entropy_with_logits(
                view.damage_logits, damage_target
            ),
        }
        total = sum(
            self.settings[name + "_weight"] * loss for name, loss in components.items()
        )
        metrics = {name + "_loss": loss.detach() for name, loss in components.items()}
        metrics.update(
            actual_gain=gain_target.mean(),
            predicted_gain=view.gain.detach().mean(),
            actual_damage=damage_target.mean(),
            predicted_damage=view.damage_logits.detach().sigmoid().mean(),
            selection_fraction=view.selected.float().mean(),
            removal_valid_fraction=remove.flatten(1).any(1).float().mean(),
            retention_valid_fraction=retain.flatten(1).any(1).float().mean(),
        )
        return total, metrics

    def __call__(self, model, batch, epoch=0):
        # The model computes its genuine inference view before this separate GT-derived view.
        training_batch = batch
        if (
            self.settings["corruption_weight"] > 0
            and self.settings["corruption_probability"] > 0
        ):
            training_batch = {
                **batch,
                "scope_training": {
                    "corruption_probability": self.settings["corruption_probability"]
                },
            }
        output = model(training_batch)
        native = batch.get("native_mask")
        loss, metrics = self.view_loss(output["scope"], batch["mask"], native)
        if output["training_scope"] is not None:
            extra, extra_metrics = self.view_loss(
                output["training_scope"], batch["mask"], native
            )
            weight = self.settings["corruption_weight"]
            # Fixed view weights: one statistical observation per original image.
            loss = (loss + weight * extra) / (1 + weight)
            metrics.update(
                {"corrupt_" + key: value for key, value in extra_metrics.items()}
            )
        return loss, metrics
