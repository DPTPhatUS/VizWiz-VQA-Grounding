"""Backbone-independent OCR evidence adapter: token selection and local signed refinement."""

import torch
from torch import nn
from torch.nn import functional as F

from models.region_ops import PATCH_SIZE, render_regions, sample_regions


def block(a, b, stride=1):
    return nn.Sequential(
        nn.Conv2d(a, b, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(4, b),
        nn.GELU(),
    )


class OCREvidenceAdapter(nn.Module):
    def __init__(self, text_dim, detail_dim, width=128):
        super().__init__()
        self.text_projection = nn.Linear(text_dim, width)
        self.native_reader = nn.Sequential(block(3, 16), block(16, 32, 2))
        self.visual_projection = nn.Linear(32 + detail_dim, width)
        self.geometry = nn.Sequential(
            nn.Linear(7, width), nn.GELU(), nn.Linear(width, width)
        )
        self.null = nn.Parameter(torch.zeros(width))
        self.question_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.candidate_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.score = nn.Sequential(
            nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1)
        )
        self.condition = nn.Linear(width, 16)
        self.local = nn.Sequential(block(32 + detail_dim + 1 + 16, 32), block(32, 16))
        self.correction = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(
        self,
        *,
        detail,
        base_logits,
        question_tokens,
        question_mask,
        recognized_tokens,
        patches,
        boxes,
        confidence,
        valid,
    ):
        b, n = boxes.shape[:2]
        native = self.native_reader((patches.flatten(0, 1) - 0.5) / 0.5)
        native = F.interpolate(native, PATCH_SIZE, mode="bilinear", align_corners=False)
        region_detail = sample_regions(detail, boxes)
        region_base = sample_regions(base_logits, boxes)
        visual = torch.cat(
            (
                native.reshape(b, n, 32, *PATCH_SIZE).mean((-1, -2)),
                region_detail.mean((-1, -2)),
            ),
            -1,
        )
        geometry = torch.cat(
            (boxes, boxes[..., 2:] - boxes[..., :2], confidence[..., None]), -1
        )
        candidates = (
            self.visual_projection(visual)
            + self.geometry(geometry)
            + self.text_projection(recognized_tokens)
        )
        candidates = torch.cat(
            (candidates, self.null.view(1, 1, -1).expand(b, -1, -1)), 1
        )
        available = torch.cat(
            (valid, torch.ones(b, 1, device=valid.device, dtype=torch.bool)), 1
        )
        question = self.text_projection(question_tokens)
        attended, _ = self.question_attention(
            candidates,
            question,
            question,
            key_padding_mask=~question_mask,
            need_weights=False,
        )
        candidates = self.norm1(candidates + attended)
        attended, _ = self.candidate_attention(
            candidates,
            candidates,
            candidates,
            key_padding_mask=~available,
            need_weights=False,
        )
        candidates = self.norm2(candidates + attended)
        scores = (
            self.score(candidates).squeeze(-1).float().masked_fill(~available, -1e4)
        )
        weights = scores.softmax(-1)[:, :n] * valid.float()
        condition = self.condition(candidates[:, :n]).flatten(0, 1)[..., None, None]
        local = torch.cat(
            (
                native,
                region_detail.flatten(0, 1),
                region_base.flatten(0, 1),
                condition.expand(-1, -1, *PATCH_SIZE),
            ),
            1,
        )
        delta = self.correction(self.local(local)).reshape(b, n, 1, *PATCH_SIZE).float()
        correction = render_regions(delta, boxes, weights, base_logits.shape[-2:])
        return {
            "logits": base_logits.float() + correction,
            "base_logits": base_logits,
            "candidate_logits": scores,
            "candidate_valid": valid,
            "region_logits": region_base + delta,
            "boxes": boxes,
            "null_probability": scores.softmax(-1)[:, -1],
        }
