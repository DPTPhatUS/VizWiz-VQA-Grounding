"""Backbone-independent, bounded local transport of segmentation logits."""

import torch
from torch import nn
from torch.nn import functional as F

OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def neighbor(tensor, dy, dx):
    """Return y+dy,x+dx neighbors and a validity mask; never wrap image borders."""
    shifted = torch.roll(tensor, shifts=(-dy, -dx), dims=(-2, -1))
    valid = torch.ones(
        (1, 1, *tensor.shape[-2:]), dtype=torch.bool, device=tensor.device
    )
    if dy > 0:
        valid[:, :, -dy:, :] = False
    if dy < 0:
        valid[:, :, :-dy, :] = False
    if dx > 0:
        valid[:, :, :, -dx:] = False
    if dx < 0:
        valid[:, :, :, :-dx] = False
    return shifted, valid


class AffinityTransport(nn.Module):
    """Refine from arbitrary semantic/detail features and a question vector.

    No unfold of Cx9xHxW features, attention kernel, external encoder, backbone
    name, patch count or fixed image shape. A single local update is bounded by
    two logits and remains between the center and the weighted neighborhood.
    """

    def __init__(self, semantic_channels, detail_channels, question_dim, *, width=16):
        super().__init__()
        self.semantic = nn.Conv2d(semantic_channels, width // 2, 1)
        self.detail = nn.Conv2d(detail_channels, width // 2, 1)
        self.question = nn.Linear(question_dim, width // 2)
        self.fusion = nn.Sequential(
            nn.Conv2d(width + width // 2, width, 1, bias=False),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
            nn.GroupNorm(4, width),
            nn.GELU(),
        )
        self.affinity = nn.Conv2d(width, 8, 1)
        self.gate = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.affinity.weight)
        nn.init.constant_(self.affinity.bias, -2)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2)
        self.register_buffer("format_version", torch.tensor(1))

    def forward(self, logits, semantic, detail, question):
        size = logits.shape[-2:]
        meaning = F.interpolate(
            self.semantic(semantic), size=size, mode="bilinear", align_corners=False
        )
        local = self.detail(detail)
        if local.shape[-2:] != size:
            local = F.interpolate(
                local, size=size, mode="bilinear", align_corners=False
            )
        qmap = self.question(question)[:, :, None, None].expand(-1, -1, *size)
        feature = self.fusion(torch.cat((meaning, local, qmap), 1))
        scores = self.affinity(feature)
        # Accumulate FP32. The center has fixed unnormalized weight one.
        logits = logits.float()
        numerator, denominator = torch.zeros_like(logits), torch.ones_like(logits)
        for index, (dy, dx) in enumerate(OFFSETS):
            other, valid = neighbor(logits, dy, dx)
            weight = scores[:, index : index + 1].float().sigmoid() * valid
            numerator = numerator + weight * (other - logits)
            denominator = denominator + weight
        movement = numerator / denominator
        correction = self.gate(feature).float().sigmoid() * 2 * (movement / 2).tanh()
        return {
            "logits": logits + correction,
            "coarse_logits": logits,
            "affinity_logits": scores,
        }


def affinity_loss(scores, target):
    """Balance same/different valid neighbor pairs inside each original image.

    Labels are derived only from the current training mask. No synthetic mask
    corruption or direction-to-boundary oracle is needed at inference.
    """
    truth = target.float() > 0.5
    positive_sum = scores.new_zeros(len(scores), dtype=torch.float32)
    negative_sum = torch.zeros_like(positive_sum)
    positive_count = torch.zeros_like(positive_sum)
    negative_count = torch.zeros_like(positive_sum)
    for index, (dy, dx) in enumerate(OFFSETS):
        other, valid = neighbor(truth, dy, dx)
        same = truth == other
        loss = F.binary_cross_entropy_with_logits(
            scores[:, index : index + 1].float(), same.float(), reduction="none"
        )
        positives = same & valid
        negatives = (~same) & valid
        positive_sum = positive_sum + (loss * positives).flatten(1).sum(1)
        negative_sum = negative_sum + (loss * negatives).flatten(1).sum(1)
        positive_count = positive_count + positives.flatten(1).sum(1)
        negative_count = negative_count + negatives.flatten(1).sum(1)
    loss = positive_sum / positive_count.clamp_min(
        1
    ) + negative_sum / negative_count.clamp_min(1)
    classes = (positive_count > 0).float() + (negative_count > 0).float()
    return (loss / classes.clamp_min(1)).mean()
