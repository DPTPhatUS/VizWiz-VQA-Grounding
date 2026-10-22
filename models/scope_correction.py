"""Small question-conditioned deletion expert and detached edit evaluator."""

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.mask_decoder import conv_block
from models.text_encoder import TextFeatures, masked_mean

SELECTION_DEFAULTS = {"gain_margin": 0.01, "max_damage": 0.02}
CORRUPTION_PROTOCOL = {
    "kind": "train-only-expanded-gt-support",
    "box_scale": 1.5,
    "dilation_radius_at_768": 16,
    "foreground_logit": 4.0,
}


class ScopePrediction(NamedTuple):
    source_logits: Tensor
    candidate_logits: Tensor
    removal_logits: Tensor
    gain: Tensor
    damage_logits: Tensor
    selected: Tensor


def suppress_logits(source: Tensor, removal: Tensor) -> Tensor:
    """Stable logit(sigmoid(source) * (1-sigmoid(removal))), in float32.

    No clipping/probability round-trip: the candidate cannot exceed the source,
    including at saturated logits. Bilinear resizing preserves this ordering.
    """
    source, removal = source.float(), removal.float()
    return source - F.softplus(removal + F.softplus(source))


class ScopeVerifier(nn.Module):
    def __init__(
        self,
        semantic_dim: int,
        detail_dim: int,
        text_dim: int,
        *,
        tiny=False,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        width = 8 if tiny else 32
        channels = (8, 4, 4) if tiny else (32, 16, 8)
        self.input = nn.Sequential(
            nn.Conv2d(semantic_dim + detail_dim + 1, width, 1, bias=False),
            nn.GroupNorm(4, width),
            nn.GELU(),
        )
        self.text = nn.Linear(text_dim, width)
        self.attention = nn.MultiheadAttention(
            width, 2 if tiny else 4, batch_first=True
        )
        self.norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList(
            [
                conv_block(a + detail_dim, b, separable=True)
                for a, b in zip((width, *channels[:-1]), channels)
            ]
        )
        self.removal = nn.Conv2d(channels[-1], 1, 1)
        nn.init.normal_(self.removal.weight, std=0.01)
        nn.init.constant_(self.removal.bias, -4.0)
        self.evidence_dim = channels[0]

    def forward(
        self,
        region: Tensor,
        detail: tuple[Tensor, ...],
        text: TextFeatures,
        source_logits: Tensor,
    ) -> tuple[Tensor, Tensor]:
        probability = F.interpolate(
            source_logits.float().sigmoid(),
            size=region.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).to(region.dtype)
        x = self.input(torch.cat((region, detail[1], probability), dim=1))
        tokens = x.flatten(2).transpose(1, 2)
        question = self.text(text.tokens).to(tokens.dtype)
        attended, _ = self.attention(
            tokens,
            question,
            question,
            key_padding_mask=~text.attention_mask,
            need_weights=False,
        )
        x = self.norm(tokens + attended).transpose(1, 2).reshape_as(x)
        evidence = None
        for index, (block, local) in enumerate(zip(self.blocks, detail[2:])):
            x = F.interpolate(
                x, size=local.shape[-2:], mode="bilinear", align_corners=False
            )
            joined = torch.cat((x, local), dim=1)
            x = (
                checkpoint(block, joined, use_reentrant=False)
                if self.training and self.gradient_checkpointing
                else block(joined)
            )
            if index == 0:
                evidence = x
        if evidence is None:
            raise ValueError(
                "Scope verifier requires three high-resolution detail levels"
            )
        return self.removal(x), evidence


class EditEvaluator(nn.Module):
    """Score the proposed edit; these losses never update the verifier."""

    grid = 8

    def __init__(self, evidence_dim: int, text_dim: int, *, tiny=False):
        super().__init__()
        width = 8 if tiny else 32
        self.question = nn.Linear(text_dim, width)
        self.features = nn.Sequential(
            nn.Linear(2 * evidence_dim + width + 3 * self.grid**2 + 3, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.gain = nn.Linear(32, 1)
        self.damage = nn.Linear(32, 1)
        nn.init.normal_(self.gain.weight, std=0.01)
        nn.init.constant_(self.gain.bias, -0.01)
        nn.init.normal_(self.damage.weight, std=0.01)
        nn.init.constant_(self.damage.bias, -4.0)

    @staticmethod
    def pool(feature: Tensor, mask: Tensor) -> Tensor:
        weights = F.interpolate(mask.float(), size=feature.shape[-2:], mode="area")
        mass = weights.sum((-2, -1))
        total = (feature.float() * weights).sum((-2, -1))
        denominator = torch.where(mass > 0, mass, torch.ones_like(mass))
        return total / denominator

    def forward(
        self, evidence: Tensor, text: TextFeatures, source: Tensor, candidate: Tensor
    ) -> tuple[Tensor, Tensor]:
        evidence = evidence.detach()
        original, retained = (
            source.detach().float().sigmoid(),
            candidate.detach().float().sigmoid(),
        )
        removed = original - retained
        maps = torch.cat((original, retained, removed), dim=1)
        layout = F.adaptive_avg_pool2d(maps, self.grid).flatten(1)
        question = self.question(masked_mean(text.tokens.detach(), text.valid_mask))
        descriptors = torch.cat(
            (
                self.pool(evidence, retained),
                self.pool(evidence, removed),
                question,
                layout,
                maps.mean((-2, -1)),
            ),
            dim=1,
        )
        hidden = self.features(descriptors.to(question.dtype))
        return self.gain(hidden).squeeze(1).float(), self.damage(hidden).squeeze(
            1
        ).float()


def expanded_training_logits(
    source: Tensor, target: Tensor, probability: float
) -> Tensor:
    """Auxiliary train-only over-coverage; does not modify deployment predictions."""
    batch, _, height, width = target.shape
    truth = target > 0.5
    radius = max(
        1,
        round(min(height, width) * CORRUPTION_PROTOCOL["dilation_radius_at_768"] / 768),
    )
    dilated = F.max_pool2d(
        truth.float(), (1, 2 * radius + 1), stride=1, padding=(0, radius)
    )
    dilated = F.max_pool2d(
        dilated, (2 * radius + 1, 1), stride=1, padding=(radius, 0)
    ).bool()
    y = torch.arange(height, device=target.device).view(1, 1, height, 1)
    x = torch.arange(width, device=target.device).view(1, 1, 1, width)
    y0 = torch.where(truth, y, height).amin((-2, -1), keepdim=True)
    y1 = torch.where(truth, y + 1, 0).amax((-2, -1), keepdim=True)
    x0 = torch.where(truth, x, width).amin((-2, -1), keepdim=True)
    x1 = torch.where(truth, x + 1, 0).amax((-2, -1), keepdim=True)
    margin_y = (y1 - y0).clamp_min(0) * (CORRUPTION_PROTOCOL["box_scale"] - 1) / 2
    margin_x = (x1 - x0).clamp_min(0) * (CORRUPTION_PROTOCOL["box_scale"] - 1) / 2
    envelope = (
        (y >= y0 - margin_y)
        & (y < y1 + margin_y)
        & (x >= x0 - margin_x)
        & (x < x1 + margin_x)
    )
    kind = torch.rand(batch, 1, 1, 1, device=target.device) < 0.5
    support = torch.where(kind, dilated, envelope)
    active = torch.rand(batch, 1, 1, 1, device=target.device) < probability
    active &= truth.flatten(1).any(1).view(batch, 1, 1, 1)
    added = torch.maximum(
        source.float(),
        source.new_tensor(CORRUPTION_PROTOCOL["foreground_logit"]).float(),
    )
    return torch.where(active & support, added, source.float()).detach()


def validate_selection(gain_margin: float, max_damage: float) -> None:
    for name, value in (("gain_margin", gain_margin), ("max_damage", max_damage)):
        if (
            type(value) not in (float, int)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be finite and in [0, 1]")
