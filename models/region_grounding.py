"""Token-aware pixels and training-only, area-matched hard-negative region ranking."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TokenPixelInteraction(nn.Module):
    """Each spatial query attends to the full question sequence, not its mean."""

    def __init__(self, channels: int, text_dim: int, width: int = 64):
        super().__init__()
        self.query = nn.Conv2d(channels, width, 1)
        self.text = nn.Linear(text_dim, width)
        self.query_norm = nn.LayerNorm(width)
        self.text_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.output = nn.Conv2d(width, channels, 1)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self, feature: Tensor, tokens: Tensor, attention_mask: Tensor
    ) -> Tensor:
        if not attention_mask.any(dim=1).all():
            raise ValueError("Question attention requires at least one unmasked token")
        query = self.query_norm(self.query(feature).flatten(2).transpose(1, 2))
        text = self.text_norm(self.text(tokens))
        context, _ = self.attention(
            query, text, text, key_padding_mask=~attention_mask, need_weights=False
        )
        context = context.transpose(1, 2).reshape(
            feature.shape[0], -1, *feature.shape[-2:]
        )
        return feature + self.residual_scale * self.output(context)


@torch.no_grad()
def mine_translated_regions(
    target: Tensor, logits: Tensor, size: tuple[int, int], count: int = 3
) -> tuple[Tensor, Tensor, Tensor]:
    """Translate GT's soft support without wrapping, truncation or overlap with its halo.

    A deterministic 5x5 placement grid plus eight high-probability distractor
    centers supply candidates. The top `count` by detached prediction confidence
    become negatives. Large/empty regions may have no valid translation.
    """
    positive = F.interpolate(target.float(), size=size, mode="area")
    probability = F.interpolate(
        logits.detach().float(), size=size, mode="bilinear", align_corners=False
    ).sigmoid()
    batch, _, height, width = positive.shape
    negatives = positive.new_zeros(batch, count, height, width)
    valid = torch.zeros(batch, count, dtype=torch.bool, device=target.device)
    for index in range(batch):
        weights = positive[index, 0]
        support = weights > 0
        coordinates = support.nonzero()
        if not coordinates.numel():
            continue
        y0, x0 = coordinates.amin(0).tolist()
        y1, x1 = (coordinates.amax(0) + 1).tolist()
        template = weights[y0:y1, x0:x1]
        th, tw = template.shape
        ymax, xmax = height - th, width - tw
        halo = F.max_pool2d(support.float()[None, None], 3, stride=1, padding=1)[
            0, 0
        ].bool()
        ys = torch.linspace(0, ymax, 5).round().long().tolist()
        xs = torch.linspace(0, xmax, 5).round().long().tolist()
        positions = {(y, x) for y in ys for x in xs}
        scores = probability[index, 0].masked_fill(halo, -1)
        centers = scores.flatten().topk(min(8, scores.numel())).indices.tolist()
        positions.update(
            (
                max(0, min(ymax, c // width - th // 2)),
                max(0, min(xmax, c % width - tw // 2)),
            )
            for c in centers
        )
        positions = sorted(positions)
        candidates = weights.new_zeros(len(positions), height, width)
        for candidate, (y, x) in zip(candidates, positions):
            candidate[y : y + th, x : x + tw] = template
        overlaps = ((candidates > 0) & halo).flatten(1).any(1)
        candidates = candidates[~overlaps]
        if not len(candidates):
            continue
        confidence = (candidates * probability[index, 0]).sum((-2, -1)) / weights.sum()
        chosen = confidence.topk(min(count, len(candidates))).indices
        negatives[index, : len(chosen)] = candidates[chosen]
        valid[index, : len(chosen)] = True
    return positive, negatives, valid


class RegionRanker(nn.Module):
    """Shared nonlinear region/Q scorer; absent from inference computation."""

    def __init__(self, channels: int, text_dim: int, hidden: int = 64):
        super().__init__()
        self.visual = nn.Linear(channels, hidden)
        self.question = nn.Linear(text_dim, hidden)
        # A shared additive score bias cancels in every pairwise difference.
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(
        self, feature: Tensor, question: Tensor, logits: Tensor, target: Tensor
    ) -> dict[str, Tensor]:
        positive, negatives, valid = mine_translated_regions(
            target, logits, feature.shape[-2:]
        )
        regions = torch.cat((positive, negatives), dim=1).flatten(2)
        # Float32 pooling/reduction; masks and mining are detached, features remain differentiable.
        pooled = torch.bmm(regions.float(), feature.float().flatten(2).transpose(1, 2))
        pooled = pooled / regions.sum(-1, keepdim=True).clamp_min(1e-6)
        hidden = F.gelu(self.visual(pooled) + self.question(question)[:, None, :])
        scores = self.score(hidden).squeeze(-1).float()
        return {"positive": scores[:, :1], "negative": scores[:, 1:], "valid": valid}
