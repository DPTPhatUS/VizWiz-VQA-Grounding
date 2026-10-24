"""Training-only parent references: selected foreground or L2-SP parameter anchors."""

import math
from copy import deepcopy

import torch
from torch import nn

from models.model import GroundingModel


class ParentPredictionHead(nn.Module):
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


def protected_foreground_loss(student_logits, parent_logits, target):
    """Per-image one-sided probability penalty, restricted to correct parent positives."""
    if (
        student_logits.ndim != 4
        or student_logits.shape[1] != 1
        or student_logits.shape != parent_logits.shape
        or student_logits.shape != target.shape
    ):
        raise ValueError("Student, parent and GT masks must have matching shapes")
    truth = target.detach() > 0.5
    large = truth.float().mean((1, 2, 3)) >= 0.1
    protected = truth & (parent_logits.detach() > 0) & large[:, None, None, None]
    count = protected.flatten(1).sum(1)
    shortfall = (
        parent_logits.detach().float().sigmoid() - student_logits.float().sigmoid()
    ).relu()
    loss = (shortfall.square() * protected).flatten(1).sum(1) / count.clamp_min(1)
    coverage = count / truth.flatten(1).sum(1).clamp_min(1)
    return loss, coverage, (count > 0).float()


class ParentPreservation:
    """Keep reference tensors outside model/DDP so inference state keys stay unchanged."""

    def __init__(self, model, parameters, mode, strength, saved=None):
        if mode not in {"none", "foreground", "parameters"}:
            raise ValueError("Unknown parent-preservation mode")
        if (
            not math.isfinite(strength)
            or strength < 0
            or (mode == "none" and strength != 0)
        ):
            raise ValueError(
                "Preservation strength must be finite/nonnegative; none uses zero"
            )
        self.mode, self.strength = mode, strength
        self.active = mode != "none" and strength > 0
        self.named = [
            (name, p) for name, p in model.named_parameters() if p.requires_grad
        ]
        if [id(p) for _, p in self.named] != [id(p) for p in parameters]:
            raise ValueError("Anchor parameter order differs from the optimizer layout")
        self.reference, self.anchors = None, {}
        if saved is not None and (
            saved.get("version") != 1
            or saved.get("mode") != mode
            or saved.get("strength") != strength
            or saved.get("trainable_names") != [name for name, _ in self.named]
        ):
            raise ValueError("Resume parent-preservation state differs from this run")
        if self.active:
            if any(
                p.requires_grad
                for encoder in (model.image_encoder, model.text_encoder)
                for p in encoder.parameters()
            ):
                raise ValueError("Parent preservation requires both encoders frozen")
            if mode == "foreground":
                self.reference = ParentPredictionHead(model)
                if saved is not None:
                    if not isinstance(saved.get("reference"), dict):
                        raise ValueError(
                            "Resume is missing the original parent prediction head"
                        )
                    self.reference.load_state_dict(saved["reference"], strict=True)
            else:
                state = saved.get("anchors") if saved is not None else None
                if saved is not None and (
                    not isinstance(state, dict)
                    or set(state) != {n for n, _ in self.named}
                ):
                    raise ValueError("Resume is missing original parameter anchors")
                for name, p in self.named:
                    anchor = state[name] if state is not None else p.detach()
                    if anchor.shape != p.shape:
                        raise ValueError(f"Anchor shape differs: {name}")
                    self.anchors[name] = (
                        anchor.detach().to(device=p.device, dtype=p.dtype).clone()
                    )

    @property
    def metadata(self):
        count = (
            sum(p.numel() for p in self.reference.parameters())
            if self.reference is not None
            else sum(t.numel() for t in self.anchors.values())
        )
        return {
            "mode": self.mode,
            "strength": self.strength,
            "training_only_reference_parameters": count,
            "foreground_scope": "gt_ge_10pct_and_parent_positive"
            if self.mode == "foreground"
            else None,
            "normalization": "per_image_protected_mean"
            if self.mode == "foreground"
            else "half_sum_squared_parameter_distance",
        }

    def state_dict(self):
        return {
            "version": 1,
            "mode": self.mode,
            "strength": self.strength,
            "trainable_names": [name for name, _ in self.named],
            "reference": self.reference.state_dict()
            if self.reference is not None
            else None,
            "anchors": self.anchors,
        }

    def objective(self, base):
        if self.reference is None:
            return base
        reference, strength = self.reference, self.strength

        def with_foreground(model, batch):
            large = batch["mask"].float().mean((1, 2, 3)) >= 0.1
            if not bool(large.any()):
                return base(model, batch)
            output = model({**batch, "_return_backbone": True})
            parent = reference(
                batch, output.pop("backbone_features"), output.pop("backbone_text")
            )
            segmentation, components = base.per_image(output, batch)
            keep, coverage, active = protected_foreground_loss(
                output["logits"], parent, batch["mask"]
            )
            components.update(
                base_objective=segmentation.detach().mean(),
                preservation_loss=keep.detach().mean(),
                protected_foreground_fraction=coverage.detach().mean(),
                preservation_active_fraction=active.detach().mean(),
            )
            return (segmentation + strength * keep).mean(), components

        return with_foreground

    @torch.no_grad()
    def parameter_regularizer(self):
        """Exact L2-SP derivative, added once to the unscaled proposal before clipping."""
        if not self.anchors:
            return None, 0.0
        delta = torch.cat(
            [
                (p.detach() - self.anchors[name]).float().flatten()
                for name, p in self.named
            ]
        )
        penalty = float(delta.double().square().sum()) * 0.5
        return delta * self.strength, penalty
