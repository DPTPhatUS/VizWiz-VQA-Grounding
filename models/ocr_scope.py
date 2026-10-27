"""Score each mask from its own visual/language evidence, without a reference-mask proxy."""

import torch
from torch import nn
from torch.nn import functional as F

from models.ocr_evidence import block
from models.region_ops import sample_regions


def binary_mask_grid(logits, shape):
    """Geometry-only supersampled descriptor, without a parent's confidence cue."""
    fine = (
        F.interpolate(
            logits.float(),
            (2 * shape[0], 2 * shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        > 0
    )
    return F.avg_pool2d(fine.float(), 2)


def hypothesis_validity(members, valid_words):
    valid = members.any(-1) & ~(members & ~valid_words[:, None]).any(-1)
    valid[:, 0] = True
    return valid


class IndependentEvidenceScorer(nn.Module):
    """A shared, permutation-equivariant mask scorer; source/index is never a feature."""

    def __init__(self, text_dim, detail_dim, semantic_dim, width=128, min_gain=0.02):
        super().__init__()
        self.min_gain = min_gain
        channels = 16 if width < 64 else 32
        self.image_features = block(detail_dim + semantic_dim + 3, channels)
        self.reader = nn.Sequential(block(4, 16, 2), block(16, 32, 2))
        self.text = nn.Linear(text_dim, width)
        self.word_visual = nn.Linear(32 + channels, width)
        self.word_geometry = nn.Linear(7, width)
        self.scene = nn.Linear(channels, width)
        self.question_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.word_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.word_norm1, self.word_norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.region = nn.Sequential(nn.Linear(3 * channels + 6, width), nn.GELU())
        self.mask_norm = nn.LayerNorm(width)
        self.quality = nn.Sequential(
            nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1)
        )

    def _encode_words(
        self,
        feature,
        question,
        question_mask,
        patches,
        polygons,
        recognized,
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
        native = self.reader(pixels).mean((-1, -2)).reshape(b, n, -1)
        local = sample_regions(feature, boxes, (4, 8)).mean((-1, -2))
        geometry = torch.cat(
            (boxes, boxes[..., 2:] - boxes[..., :2], confidence[..., None]), -1
        )
        words = (
            self.word_visual(torch.cat((native, local), -1))
            + self.text(recognized)
            + self.word_geometry(geometry)
        )
        attended, _ = self.question_attention(
            words,
            question,
            question,
            key_padding_mask=~question_mask,
            need_weights=False,
        )
        words = self.word_norm1(words + attended)
        # The scene key prevents all-masked attention when a rank has no OCR.
        context = torch.cat((words, self.scene(feature.mean((-1, -2)))[:, None]), 1)
        available = torch.cat(
            (valid, torch.ones(b, 1, device=valid.device, dtype=torch.bool)), 1
        )
        attended, _ = self.word_attention(
            words, context, context, key_padding_mask=~available, need_weights=False
        )
        return self.word_norm2(words + attended)

    def _describe_masks(self, feature, masks, word_masks, words, valid_words):
        b, k, h, w = masks.shape
        flat = masks.flatten(2)
        feature_flat = feature.flatten(2).transpose(1, 2)
        mass = flat.sum(-1).clamp_min(1e-6)
        inside = torch.bmm(flat, feature_flat) / mass[..., None]
        dilated = F.max_pool2d(masks.reshape(b * k, 1, h, w), 7, stride=1, padding=3)
        ring = (dilated.reshape(b, k, h, w) - masks).clamp_min(0).flatten(2)
        context = (
            torch.bmm(ring, feature_flat) / ring.sum(-1).clamp_min(1e-6)[..., None]
        )
        global_context = feature.mean((-1, -2))[:, None].expand(-1, k, -1)
        xs = (torch.arange(w, device=masks.device, dtype=torch.float32) + 0.5) / w
        ys = (torch.arange(h, device=masks.device, dtype=torch.float32) + 0.5) / h
        xx = xs[None].expand(h, w).flatten()
        yy = ys[:, None].expand(h, w).flatten()
        cx, cy = (flat * xx).sum(-1) / mass, (flat * yy).sum(-1) / mass
        vx = ((flat * xx.square()).sum(-1) / mass - cx.square()).clamp_min(0)
        vy = ((flat * yy.square()).sum(-1) / mass - cy.square()).clamp_min(0)
        geometry = torch.stack(
            (mass / (h * w), cx, cy, vx, vy, (flat * (1 - flat)).mean(-1)), -1
        )
        # Every hypothesis, including the base, gets OCR content by spatial support.
        word_flat = word_masks.float().flatten(2)
        covered = (
            torch.bmm(flat, word_flat.transpose(1, 2))
            / word_flat.sum(-1).clamp_min(1e-6)[:, None]
        )
        covered = covered * valid_words[:, None]
        language = (
            torch.bmm(covered, words) / covered.sum(-1).clamp_min(1e-6)[..., None]
        )
        own = self.region(torch.cat((inside, context, global_context, geometry), -1))
        return own + language

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
        word_masks,
    ):
        shape = masks.shape[-2:]
        inputs = [
            F.adaptive_avg_pool2d(x.float(), shape)
            for x in (detail, semantic, image * 2 - 1)
        ]
        feature = self.image_features(torch.cat(inputs, 1))
        question = self.text(question_tokens)
        words = self._encode_words(
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
        base = binary_mask_grid(base_logits, shape)[:, 0]
        masks = torch.cat((base[:, None], masks[:, 1:]), 1).float()
        query = self._describe_masks(feature, masks, word_masks, words, valid)
        attended, _ = self.question_attention(
            query,
            question,
            question,
            key_padding_mask=~question_mask,
            need_weights=False,
        )
        logits = self.quality(self.mask_norm(query + attended)).squeeze(-1).float()
        quality = logits.sigmoid()
        available = hypothesis_validity(members, valid)
        masked = quality.masked_fill(~available, -1)
        evidence = masked[:, 1:].argmax(-1) + 1
        evidence_quality = masked.gather(1, evidence[:, None]).squeeze(1)
        selected = torch.where(
            evidence_quality > quality[:, 0] + self.min_gain, evidence, 0
        )
        return {
            "base_logits": base_logits,
            "quality_logits": logits,
            "predicted_quality": quality,
            "hypothesis_valid": available,
            "candidate_valid": valid,
            "selected_index": selected,
            "evidence_index": evidence,
        }


@torch.no_grad()
def quality_training_masks(logits, masks, word_valid, recipe):
    """Mask-only training counterexamples. All draws are matched across recipes.

    ranking: the previous distribution of geometric shifts/morphology.
    scope: tight OCR, expanded context, or OCR+distractor unions.
    Targets are measured AFTER augmentation; no GT chooses a counterexample.
    """
    draws = torch.rand(len(logits), 8, device=logits.device)
    result = logits.clone()
    for i, random in enumerate(draws):
        if random[0] >= 0.5:
            continue
        mode = min(2, int(random[1] * 3))
        if recipe == "ranking":
            source = logits[i : i + 1].float()
            if mode == 0:
                theta = torch.eye(2, 3, device=logits.device)[None]
                theta[0, :, 2] = (random[2:4] - 0.5) * 0.4
                grid = F.affine_grid(theta, source.shape, align_corners=False)
                changed = F.grid_sample(source + 20, grid, align_corners=False) - 20
            else:
                kernel = max(
                    3, int(min(source.shape[-2:]) * (0.02 + 0.06 * random[4])) | 1
                )
                sign = 1 if mode == 1 else -1
                changed = F.max_pool2d(sign * source, (1, kernel), 1, (0, kernel // 2))
                changed = sign * F.max_pool2d(changed, (kernel, 1), 1, (kernel // 2, 0))
            result[i : i + 1] = changed
            continue
        ids = word_valid[i].nonzero().flatten()
        if not len(ids):
            continue
        anchor = ids[min(len(ids) - 1, int(random[2] * len(ids)))]
        probability = masks[i, int(anchor) + 1].float()
        if mode == 1:
            points = (probability > 0).nonzero()
            if not len(points):
                continue
            lo, hi = points.amin(0).float(), points.amax(0).float() + 1
            center, half = (lo + hi) / 2, (hi - lo) / 2 * (2 + 2 * random[4])
            h, w = probability.shape
            top, left = (center - half).floor().clamp_min(0).long().tolist()
            bottom, right = (
                torch.minimum((center + half).ceil(), probability.new_tensor([h, w]))
                .long()
                .tolist()
            )
            probability = torch.zeros_like(probability)
            probability[top:bottom, left:right] = 1
        elif mode == 2 and len(ids) > 1:
            others = ids[ids != anchor]
            other = others[min(len(others) - 1, int(random[3] * len(others)))]
            probability = torch.maximum(probability, masks[i, int(other) + 1])
        probability = F.interpolate(
            probability[None, None],
            logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        result[i : i + 1] = torch.logit(probability.clamp(0.002, 0.998))
    return result
