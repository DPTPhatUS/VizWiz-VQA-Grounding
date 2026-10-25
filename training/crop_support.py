"""Preset crop experiments with training-only large/full-image recall support."""

from copy import deepcopy
from dataclasses import asdict, dataclass

import torch
from torch import nn

from models.model import GroundingModel


@dataclass(frozen=True)
class SupportRecipe:
    variant: str
    crop_probability: float
    retention_weight: float = 0.0
    large_fp_weight: float | None = None

    @property
    def config(self):
        return {
            "version": "crop-support-v1",
            **asdict(self),
            "scope": "source_ge_10pct_and_full_image",
            "recall_head": "final_only",
            "retention": "parent_correct_foreground_probability_shortfall_squared",
        }


RECIPES = {
    "control": SupportRecipe("control", 0.25),
    "low_crop": SupportRecipe("low_crop", 0.10),
    "retain": SupportRecipe("retain", 0.25, retention_weight=1.0),
    "recall": SupportRecipe("recall", 0.25, large_fp_weight=0.3),
    "retain_recall": SupportRecipe("retain_recall", 0.25, 1.0, 0.3),
}


def support_scope(batch):
    """Never apply large-image support to a magnified small crop."""
    return (batch["source_area_fraction"] >= 0.1) & (batch["crop_applied"] == 0)


def recall_overlap_delta(logits, target, *, fp_weight, overlap_weight):
    """Replace only final Dice overlap by Tversky; BCE/auxiliaries stay unchanged.

    Reuse the existing loss's smoothing exactly: Dice eps1, Tversky eps1.
    fp_weight0.3 means FN weight0.7. Return one signed adjustment per image.
    """
    probability, truth = logits.float().sigmoid(), target.float()
    axes = tuple(range(1, truth.ndim))
    tp = (probability * truth).sum(axes)
    fp = (probability * (1 - truth)).sum(axes)
    fn = ((1 - probability) * truth).sum(axes)
    dice = (2 * tp + 1) / (2 * tp + fp + fn + 1)
    tversky = (tp + 1) / (tp + fp_weight * fp + (1 - fp_weight) * fn + 1)
    return overlap_weight * (dice - tversky)


class FrozenGroundingBody(nn.Module):
    """Copy only post-encoder modules; never instantiate or store another backbone."""

    def __init__(self, model):
        super().__init__()
        for name, module in model.named_children():
            if name not in {"image_encoder", "text_encoder"}:
                self.add_module(name, deepcopy(module))
        for name, parameter in model.named_parameters(recurse=False):
            self.register_parameter(
                name, nn.Parameter(parameter.detach().clone(), requires_grad=False)
            )
        for name, buffer in model.named_buffers(recurse=False):
            self.register_buffer(name, buffer.detach().clone())
        self.requires_grad_(False)
        self.eval()

    @staticmethod
    def _run(block, *inputs):
        return block(*inputs)

    @torch.no_grad()
    def forward(self, batch, features, text):
        return GroundingModel.decode_features(self, batch, features, text)["logits"]


class CropSupportObjective:
    """All optional training behavior lives here, outside the inference model."""

    def __init__(self, base, model, recipe, saved=None):
        self.base, self.recipe = base, recipe
        if base.loss_kwargs["overlap_loss"] != "dice":
            raise ValueError(
                "Controlled crop-support experiments retain the B Dice objective"
            )
        if saved is not None and saved.get("recipe") != recipe.config:
            raise ValueError("Resume support-reference recipe differs")
        self.reference = None
        if recipe.retention_weight:
            if any(
                p.requires_grad
                for encoder in (model.image_encoder, model.text_encoder)
                for p in encoder.parameters()
            ):
                raise ValueError(
                    "Shared-feature retention requires both encoders frozen"
                )
            self.reference = FrozenGroundingBody(model)
            if saved is not None:
                if not isinstance(saved.get("reference"), dict):
                    raise ValueError("Resume is missing the original B5 reference body")
                self.reference.load_state_dict(saved["reference"], strict=True)
        elif saved is not None and saved.get("reference") is not None:
            raise ValueError("Unexpected reference for a teacher-free recipe")

    def state_dict(self):
        return {
            "recipe": self.recipe.config,
            "reference": self.reference.state_dict()
            if self.reference is not None
            else None,
        }

    @property
    def reference_parameters(self):
        return (
            sum(p.numel() for p in self.reference.parameters())
            if self.reference is not None
            else 0
        )

    def __call__(self, model, batch):
        eligible = support_scope(batch)
        if (
            not self.recipe.retention_weight and self.recipe.large_fp_weight is None
        ) or not bool(eligible.any()):
            return self.base(model, batch)
        output = (
            model({**batch, "_return_backbone": True})
            if self.reference is not None
            else model(batch)
        )
        base, components = self.base.per_image(output, batch)
        delta, keep = torch.zeros_like(base), torch.zeros_like(base)
        coverage = torch.zeros_like(base)
        if self.recipe.large_fp_weight is not None:
            delta = (
                recall_overlap_delta(
                    output["logits"],
                    batch["mask"],
                    fp_weight=self.recipe.large_fp_weight,
                    overlap_weight=self.base.loss_kwargs["overlap_weight"],
                )
                * eligible
            )
        if self.reference is not None:
            parent = self.reference(
                batch, output.pop("backbone_features"), output.pop("backbone_text")
            )
            truth = batch["mask"] > 0.5
            protected = truth & (parent > 0) & eligible[:, None, None, None]
            count = protected.flatten(1).sum(1)
            shortfall = (
                parent.float().sigmoid() - output["logits"].float().sigmoid()
            ).relu()
            keep = (shortfall.square() * protected).flatten(1).sum(1) / count.clamp_min(
                1
            )
            coverage = count / truth.flatten(1).sum(1).clamp_min(1)
        term = self.recipe.retention_weight * keep
        components.update(
            base_objective=base.detach().mean(),
            retention_term=term.detach().mean(),
            recall_delta=delta.detach().mean(),
            support_fraction=eligible.float().mean(),
            protected_coverage=coverage.mean(),
            retention_shortfall=keep.detach().mean(),
        )
        return (base + delta + term).mean(), components
