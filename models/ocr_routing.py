"""Independent evidence masks and optional multi-scale OCR fusion, without a second backbone."""

import torch
from torch import nn
from torch.nn import functional as F

from models.evidence_ops import compose_evidence, rasterize_tokens
from models.ocr_evidence import block
from models.region_ops import PATCH_SIZE, sample_regions


class CandidateBank(nn.Module):
    """Question-aware multi-label candidate features and a separate global route token."""

    def __init__(self, text_dim, detail_dim, width):
        super().__init__()
        self.text = nn.Linear(text_dim, width)
        self.reader = nn.Sequential(block(4, 16), block(16, 32, 2))
        self.visual = nn.Linear(32 + detail_dim, width)
        self.geometry = nn.Sequential(
            nn.Linear(7, width), nn.GELU(), nn.Linear(width, width)
        )
        self.scene = nn.Linear(detail_dim, width)
        self.route_token = nn.Parameter(torch.zeros(width))
        self.question_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.set_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.inclusion = nn.Linear(width, 1)
        self.router = nn.Linear(width, 1)
        nn.init.constant_(self.inclusion.bias, 1.0)
        nn.init.zeros_(self.router.weight)
        nn.init.constant_(self.router.bias, -3.0)

    def forward(
        self,
        detail,
        question_tokens,
        question_mask,
        recognized_tokens,
        patches,
        polygons,
        boxes,
        confidence,
        valid,
    ):
        b, n = boxes.shape[:2]
        native = self.reader(
            torch.cat(((patches.flatten(0, 1) - 0.5) / 0.5, polygons.flatten(0, 1)), 1)
        )
        native = F.interpolate(native, PATCH_SIZE, mode="bilinear", align_corners=False)
        local_detail = sample_regions(detail, boxes)
        visual = torch.cat(
            (
                native.reshape(b, n, 32, *PATCH_SIZE).mean((-1, -2)),
                local_detail.mean((-1, -2)),
            ),
            -1,
        )
        geometry = torch.cat(
            (boxes, boxes[..., 2:] - boxes[..., :2], confidence[..., None]), -1
        )
        candidates = (
            self.visual(visual) + self.geometry(geometry) + self.text(recognized_tokens)
        )
        global_token = self.route_token + self.scene(detail.mean((-1, -2)))
        tokens = torch.cat((candidates, global_token[:, None]), 1)
        available = torch.cat(
            (valid, torch.ones(b, 1, device=valid.device, dtype=torch.bool)), 1
        )
        question = self.text(question_tokens)
        attended, _ = self.question_attention(
            tokens,
            question,
            question,
            key_padding_mask=~question_mask,
            need_weights=False,
        )
        tokens = self.norm1(tokens + attended)
        attended, _ = self.set_attention(
            tokens, tokens, tokens, key_padding_mask=~available, need_weights=False
        )
        tokens = self.norm2(tokens + attended)
        inclusion = self.inclusion(tokens[:, :n]).squeeze(-1).float()
        route = self.router(tokens[:, -1]).squeeze(-1).float()
        return tokens[:, :n], native, local_detail, inclusion, route


class EvidencePyramid(nn.Module):
    """OCR is fused during 96/192/384 decoding, followed by a small full-resolution head."""

    def __init__(self, detail_dim, token_dim, width=24):
        super().__init__()
        self.token_projection = nn.Linear(token_dim, 8)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    block(detail_dim + 8 + 1 + width, width), block(width, width)
                )
                for _ in range(3)
            ]
        )
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.final = nn.Sequential(
            block(detail_dim + width + 1, 16), nn.Conv2d(16, 1, 1)
        )
        nn.init.zeros_(self.final[-1].weight)
        nn.init.zeros_(self.final[-1].bias)
        self.width = width

    def forward(self, levels, detail, tokens, local_probability, boxes, inclusion):
        state = None
        for level, block_ in zip(levels, self.blocks):
            shape = level.shape[-2:]
            field = rasterize_tokens(
                self.token_projection(tokens),
                local_probability,
                boxes,
                inclusion,
                shape,
            )
            prior = compose_evidence(local_probability, boxes, inclusion, shape)
            if state is None:
                state = level.new_zeros(level.shape[0], self.width, *shape)
            else:
                state = F.interpolate(
                    state, shape, mode="bilinear", align_corners=False
                )
            state = block_(torch.cat((level, field, prior, state), 1))
        coarse_prior = compose_evidence(
            local_probability, boxes, inclusion, state.shape[-2:]
        )
        logits = torch.logit(coarse_prior.clamp(0.02, 0.98)) + self.head(state).float()
        shape = detail.shape[-2:]
        logits = F.interpolate(logits, shape, mode="bilinear", align_corners=False)
        state = F.interpolate(state, shape, mode="bilinear", align_corners=False)
        return logits + self.final(torch.cat((detail, state, logits), 1)).float()


class EvidenceRoutingAdapter(nn.Module):
    def __init__(
        self,
        text_dim,
        detail_dim,
        width=128,
        architecture="proposal",
        semantic_dim=None,
    ):
        super().__init__()
        if architecture not in {"proposal", "pyramid"}:
            raise ValueError("Unknown evidence architecture")
        self.architecture = architecture
        self.semantic_projection = nn.Sequential(
            nn.Conv2d(semantic_dim or detail_dim, detail_dim, 1, bias=False),
            nn.GroupNorm(4, detail_dim),
            nn.GELU(),
        )
        self.bank = CandidateBank(text_dim, detail_dim, width)
        self.condition = nn.Linear(width, 16)
        self.local = nn.Sequential(
            block(32 + detail_dim + 16 + 1, 32), block(32, 16), nn.Conv2d(16, 1, 1)
        )
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)
        if architecture == "pyramid":
            self.pyramid = EvidencePyramid(detail_dim, width)

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
        polygons,
        detail_levels=None,
        semantic=None,
    ):
        b, n = boxes.shape[:2]
        meaning = self.semantic_projection(detail if semantic is None else semantic)
        detail = detail + F.interpolate(
            meaning, detail.shape[-2:], mode="bilinear", align_corners=False
        )
        if detail_levels is not None:
            detail_levels = tuple(
                level
                + F.interpolate(
                    meaning, level.shape[-2:], mode="bilinear", align_corners=False
                )
                for level in detail_levels
            )
        tokens, native, local_detail, inclusion_logits, route_logits = self.bank(
            detail,
            question_tokens,
            question_mask,
            recognized_tokens,
            patches,
            polygons,
            boxes,
            confidence,
            valid,
        )
        condition = self.condition(tokens).flatten(0, 1)[..., None, None]
        local_input = torch.cat(
            (
                native,
                local_detail.flatten(0, 1),
                condition.expand(-1, -1, *PATCH_SIZE),
                polygons.flatten(0, 1),
            ),
            1,
        )
        # Independent mask: never inherits a parent's mistaken foreground.
        prior = torch.logit(polygons.float().clamp(0.01, 0.99))
        local_logits = (
            prior + self.local(local_input).reshape(b, n, 1, *PATCH_SIZE).float()
        )
        inclusion = inclusion_logits.sigmoid() * valid.float()
        if self.architecture == "pyramid":
            if detail_levels is None or len(detail_levels) != 3:
                raise ValueError("Pyramid routing requires three aligned detail levels")
            evidence_logits = self.pyramid(
                detail_levels, detail, tokens, local_logits.sigmoid(), boxes, inclusion
            )
            evidence_probability = evidence_logits.sigmoid()
        else:
            evidence_probability = compose_evidence(
                local_logits.sigmoid(), boxes, inclusion, base_logits.shape[-2:]
            )
            evidence_logits = torch.logit(evidence_probability.clamp(1e-6, 1 - 1e-6))
        has_evidence = valid.any(1)
        route = route_logits.sigmoid() * has_evidence
        probability = (
            1 - route[:, None, None, None]
        ) * base_logits.float().sigmoid() + route[
            :, None, None, None
        ] * evidence_probability
        logits = torch.logit(probability.clamp(1e-6, 1 - 1e-6))
        logits = torch.where(
            has_evidence[:, None, None, None], logits, base_logits.float()
        )
        return {
            "logits": logits,
            "base_logits": base_logits,
            "evidence_logits": evidence_logits,
            "candidate_valid": valid,
            "candidate_logits": inclusion_logits,
            "inclusion_probability": inclusion,
            "route_logits": route_logits,
            "route_probability": route,
            "null_probability": 1 - route,
            "region_logits": local_logits,
            "boxes": boxes,
        }
