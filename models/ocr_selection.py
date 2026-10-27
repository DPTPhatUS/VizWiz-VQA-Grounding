"""Compact question-conditioned quality ranking of fixed segmentation hypotheses."""

import torch
from torch import nn
from torch.nn import functional as F

from models.ocr_evidence import block
from models.region_ops import sample_regions


class HypothesisSelector(nn.Module):
    def __init__(
        self, text_dim, detail_dim, width=128, semantic_dim=None, min_gain=0.02
    ):
        super().__init__()
        self.min_gain = min_gain
        self.text = nn.Linear(text_dim, width)
        self.reader = nn.Sequential(block(4, 16, 2), block(16, 32, 2))
        self.visual = nn.Linear(32 + detail_dim, width)
        self.geometry = nn.Linear(7, width)
        self.question_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.set_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.scene = nn.Linear(semantic_dim or detail_dim, width)
        self.base_token = nn.Parameter(torch.zeros(width))
        self.mask_features = nn.Sequential(
            nn.Linear(2 * (semantic_dim or detail_dim) + 7, width), nn.GELU()
        )
        self.quality = nn.Sequential(
            nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1)
        )
        self.answer = nn.Linear(width, 1)

    def forward(
        self,
        *,
        detail,
        semantic,
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
        b, n = valid.shape
        small = F.interpolate(
            torch.cat(((patches.flatten(0, 1) - 0.5) / 0.5, polygons.flatten(0, 1)), 1),
            (24, 64),
            mode="bilinear",
            align_corners=False,
        )
        appearance = self.reader(small).mean((-1, -2)).reshape(b, n, -1)
        local = sample_regions(detail, boxes, (4, 8)).mean((-1, -2))
        geometry = torch.cat(
            (boxes, boxes[..., 2:] - boxes[..., :2], confidence[..., None]), -1
        )
        candidate = (
            self.visual(torch.cat((appearance, local), -1))
            + self.text(recognized_tokens)
            + self.geometry(geometry)
        )
        scene = self.scene(semantic.mean((-1, -2))) + self.base_token
        tokens = torch.cat((scene[:, None], candidate), 1)
        available = torch.cat(
            (torch.ones(b, 1, device=valid.device, dtype=torch.bool), valid), 1
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
        candidate = tokens[:, 1:]
        member_count = members.sum(-1)
        hypothesis_valid = (member_count > 0) & ~(members & ~valid[:, None]).any(-1)
        hypothesis_valid[:, 0] = True
        weights = members.float() / member_count.clamp_min(1)[..., None]
        grouped = torch.bmm(weights, candidate)
        grouped = torch.cat((tokens[:, :1], grouped[:, 1:]), 1)
        base = F.adaptive_avg_pool2d(base_logits.float().sigmoid(), masks.shape[-2:])[
            :, 0
        ]
        masks = torch.cat((base[:, None], masks[:, 1:]), 1).float()
        feature = F.adaptive_avg_pool2d(semantic.float(), masks.shape[-2:]).flatten(2)
        flat = masks.flatten(2)
        mass = flat.sum(-1).clamp_min(1e-6)
        pooled = torch.bmm(flat, feature.transpose(1, 2)) / mass[..., None]
        background = (
            torch.bmm(1 - flat, feature.transpose(1, 2))
            / (flat.shape[-1] - mass).clamp_min(1e-6)[..., None]
        )
        base_flat = base.flatten(1)[:, None]
        intersection = (flat * base_flat).sum(-1)
        union = (flat + base_flat - flat * base_flat).sum(-1).clamp_min(1e-6)
        summary = torch.stack(
            (
                mass / flat.shape[-1],
                intersection / union,
                intersection / mass,
                intersection / base_flat.sum(-1).clamp_min(1e-6),
                (flat - base_flat).abs().mean(-1),
                member_count.float() / n,
                (flat * (1 - flat)).mean(-1),
            ),
            -1,
        )
        fused = grouped + self.mask_features(
            torch.cat((pooled, background, summary), -1)
        )
        logits = self.quality(fused).squeeze(-1).float()
        quality = logits.sigmoid()
        masked = quality.masked_fill(~hypothesis_valid, -1)
        best_evidence = masked[:, 1:].argmax(-1) + 1
        best_quality = masked.gather(1, best_evidence[:, None]).squeeze(1)
        choice = torch.where(
            best_quality > quality[:, 0] + self.min_gain, best_evidence, 0
        )
        return {
            "base_logits": base_logits,
            "quality_logits": logits,
            "predicted_quality": quality,
            "hypothesis_valid": hypothesis_valid,
            "selected_index": choice,
            "evidence_index": best_evidence,
            "candidate_logits": self.answer(candidate).squeeze(-1).float(),
            "candidate_valid": valid,
        }


@torch.no_grad()
def augment_quality_masks(logits):
    """Train-only mask-quality coverage; never changes the image or uses GT.

    Frozen parents can fit train well. Shifts and separable morphology expose the
    scorer to imperfect parent-shaped masks rather than an always-good base type.
    Half of the masks remain untouched. Targets are recomputed from the result.
    """
    result = logits.clone()
    for i in range(len(result)):
        if torch.rand((), device=logits.device).item() >= 0.5:
            continue
        mode = int(torch.randint(3, (), device=logits.device))
        source = logits[i : i + 1].float()
        if mode == 0:
            theta = torch.eye(2, 3, device=logits.device)[None]
            theta[0, :, 2] = (torch.rand(2, device=logits.device) - 0.5) * 0.4
            grid = F.affine_grid(theta, source.shape, align_corners=False)
            changed = F.grid_sample(source + 20, grid, align_corners=False) - 20
        else:
            fraction = 0.02 + 0.06 * torch.rand((), device=logits.device).item()
            kernel = max(3, int(min(source.shape[-2:]) * fraction) | 1)
            sign = 1 if mode == 1 else -1
            changed = F.max_pool2d(
                sign * source, (1, kernel), stride=1, padding=(0, kernel // 2)
            )
            changed = sign * F.max_pool2d(
                changed, (kernel, 1), stride=1, padding=(kernel // 2, 0)
            )
        result[i : i + 1] = changed
    return result
