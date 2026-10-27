"""Question-conditioned native OCR pointer with an explicit defer alternative."""

import torch
from torch import nn
from torch.nn import functional as F

from models.ocr_evidence import block
from models.ocr_scope import hypothesis_validity
from models.region_ops import sample_regions


def pointer_selection(logits, valid):
    """Defer competes with total OCR mass, not each member of a near-tied group."""
    logits = logits.float().masked_fill(~valid, -1e4)
    probability = logits.softmax(1)
    ocr_valid = valid.clone()
    ocr_valid[:, 0] = False
    best = logits.masked_fill(~ocr_valid, -1e4).argmax(1)
    selected = torch.where((probability[:, 0] < 0.5) & ocr_valid.any(1), best, 0)
    return logits, probability, selected, best


class EvidencePointer(nn.Module):
    """Slot zero means OCR is insufficient, never an estimate of parent quality."""

    def __init__(
        self, text_dim, detail_dim, semantic_dim, width=128, recipe="categorical"
    ):
        super().__init__()
        if recipe not in {"categorical", "structured"}:
            raise ValueError("Invalid pointer recipe")
        self.recipe = recipe
        channels = 16 if width < 64 else 32
        self.image_features = block(detail_dim + semantic_dim + 3, channels)
        self.reader = nn.Sequential(block(4, 16, 2), block(16, 32, 2))
        self.text = nn.Linear(text_dim, width)
        self.word_visual = nn.Linear(32 + channels, width)
        self.word_geometry = nn.Linear(7, width)
        self.scene = nn.Linear(channels, width)
        self.region = nn.Sequential(nn.Linear(3 * channels + 3, width), nn.GELU())
        self.null_token = nn.Parameter(torch.zeros(1, 1, width))
        self.question_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.context_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.member_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.word_norm1, self.word_norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.mask_norm1, self.mask_norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.pointer = nn.Sequential(
            nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1)
        )
        self.token_head = nn.Linear(width, 1)
        # Both recipes start with identical predictions; B learns this correction.
        nn.init.zeros_(self.token_head.weight)
        nn.init.zeros_(self.token_head.bias)

    def _words(
        self,
        feature,
        question,
        question_mask,
        patches,
        polygons,
        recognized_tokens,
        boxes,
        confidence,
        valid,
    ):
        b, n = valid.shape
        pixels = F.interpolate(
            torch.cat(((patches.flatten(0, 1) - 0.5) / 0.5, polygons.flatten(0, 1)), 1),
            (24, 64),
            mode="bilinear",
            align_corners=False,
        )
        crop = self.reader(pixels).mean((-1, -2)).reshape(b, n, -1)
        local = sample_regions(feature, boxes, (4, 8)).mean((-1, -2))
        geometry = torch.cat(
            (boxes, boxes[..., 2:] - boxes[..., :2], confidence[..., None]), -1
        )
        words = (
            self.word_visual(torch.cat((crop, local), -1))
            + self.text(recognized_tokens)
            + self.word_geometry(geometry)
        )
        context, _ = self.question_attention(
            words,
            question,
            question,
            key_padding_mask=~question_mask,
            need_weights=False,
        )
        words = self.word_norm1(words + context)
        scene = self.scene(feature.mean((-1, -2)))[:, None]
        keys = torch.cat((words, scene), 1)
        available = torch.cat(
            (valid, torch.ones(b, 1, device=valid.device, dtype=torch.bool)), 1
        )
        context, _ = self.context_attention(
            words, keys, keys, key_padding_mask=~available, need_weights=False
        )
        return self.word_norm2(words + context), scene

    def _regions(self, feature, masks, scene):
        b, k, h, w = masks.shape
        # Slot zero has no mask descriptor, even if supplied by a caller.
        masks = torch.cat((torch.zeros_like(masks[:, :1]), masks[:, 1:]), 1)
        flat = masks.flatten(2)
        mass = flat.sum(-1).clamp_min(1e-6)
        pixels = feature.flatten(2).transpose(1, 2)
        inside = torch.bmm(flat, pixels) / mass[..., None]
        ring = (
            (
                F.max_pool2d(masks.reshape(b * k, 1, h, w), 7, 1, 3).reshape(b, k, h, w)
                - masks
            )
            .clamp_min(0)
            .flatten(2)
        )
        around = torch.bmm(ring, pixels) / ring.sum(-1).clamp_min(1e-6)[..., None]
        xs = (
            ((torch.arange(w, device=masks.device, dtype=torch.float32) + 0.5) / w)[
                None
            ]
            .expand(h, w)
            .flatten()
        )
        ys = (
            ((torch.arange(h, device=masks.device, dtype=torch.float32) + 0.5) / h)[
                :, None
            ]
            .expand(h, w)
            .flatten()
        )
        geometry = torch.stack(
            (mass / (h * w), (flat * xs).sum(-1) / mass, (flat * ys).sum(-1) / mass), -1
        )
        regions = self.region(
            torch.cat(
                (
                    inside,
                    around,
                    feature.mean((-1, -2))[:, None].expand(-1, k, -1),
                    geometry,
                ),
                -1,
            )
        )
        return torch.cat((scene + self.null_token, regions[:, 1:]), 1)

    def forward(
        self,
        *,
        detail,
        semantic,
        image,
        base_logits,
        question_tokens,
        question_mask,
        recognized_tokens,
        patches,
        boxes,
        confidence,
        valid,
        polygons,
        members,
        masks,
    ):
        shape = masks.shape[-2:]
        feature = self.image_features(
            torch.cat(
                [
                    F.adaptive_avg_pool2d(detail, shape),
                    F.adaptive_avg_pool2d(semantic, shape),
                    F.adaptive_avg_pool2d((image - 0.5) / 0.5, shape),
                ],
                1,
            )
        )
        question = self.text(question_tokens)
        words, scene = self._words(
            feature,
            question,
            question_mask,
            patches,
            polygons,
            recognized_tokens,
            boxes,
            confidence,
            valid,
        )
        regions = self._regions(feature, masks, scene)
        context, _ = self.question_attention(
            regions,
            question,
            question,
            key_padding_mask=~question_mask,
            need_weights=False,
        )
        regions = self.mask_norm1(regions + context)
        b, k, d = regions.shape
        n = valid.shape[1]
        keys = torch.cat((words, scene), 1)
        # Each phrase attends to its members; null can inspect the whole OCR set.
        allowed = members & valid[:, None]
        allowed = torch.cat((valid[:, None], allowed[:, 1:]), 1)
        allowed = torch.cat(
            (allowed, torch.ones(b, k, 1, device=valid.device, dtype=torch.bool)), -1
        )
        keys = keys[:, None].expand(-1, k, -1, -1).reshape(b * k, n + 1, d)
        context, _ = self.member_attention(
            regions.reshape(b * k, 1, d),
            keys,
            keys,
            key_padding_mask=~allowed.reshape(b * k, n + 1),
            need_weights=False,
        )
        regions = self.mask_norm2(regions + context.reshape(b, k, d))
        token_logits = self.token_head(words).squeeze(-1)
        logits = self.pointer(regions).squeeze(-1).float()
        if self.recipe == "structured":
            # Bernoulli set log likelihood, up to a constant shared by OCR sets.
            # Extra irrelevant words reduce the score; multiple relevant words help.
            set_score = torch.bmm(
                members.float(), (token_logits.float() * valid).unsqueeze(-1)
            ).squeeze(-1)
            set_score = set_score / valid.sum(1).clamp_min(1).sqrt()[:, None]
            logits = logits + torch.cat(
                (torch.zeros_like(set_score[:, :1]), set_score[:, 1:]), 1
            )
        available = hypothesis_validity(members, valid)
        logits, probability, selected, evidence = pointer_selection(logits, available)
        return {
            "base_logits": base_logits,
            "selection_logits": logits,
            "selection_probability": probability,
            "null_probability": probability[:, 0],
            "selected_index": selected,
            "evidence_index": evidence,
            "hypothesis_valid": available,
            "candidate_valid": valid,
            "token_logits": token_logits,
        }
