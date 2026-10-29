"""Question-conditioned evidence masks with an explicit background competitor."""

import torch
from torch import nn
from torch.nn import functional as F

from native_refinement_ops import local_window


def render_candidates(patches, boxes, valid, size):
    """Batched inverse warp; geometry is fixed and evaluated at pixel centres."""
    b, n = boxes.shape[:2]
    h, w = size
    xy = boxes.float()
    extent = (xy[..., 2:] - xy[..., :2]).clamp_min(1e-8)
    xs = (torch.arange(w, device=xy.device).float() + 0.5) / w
    ys = (torch.arange(h, device=xy.device).float() + 0.5) / h
    gx = (xs[None, None, None, :] - xy[..., 0, None, None]) / extent[..., 0, None, None]
    gy = (ys[None, None, :, None] - xy[..., 1, None, None]) / extent[..., 1, None, None]
    gx, gy = torch.broadcast_tensors(gx, gy)
    support = (gx >= 0) & (gx < 1) & (gy >= 0) & (gy < 1) & valid[..., None, None]
    grid = torch.stack((gx, gy), -1).reshape(b * n, h, w, 2) * 2 - 1
    field = F.grid_sample(
        patches.flatten(0, 1).float(), grid, padding_mode="border", align_corners=False
    )
    return field.reshape(b, n, h, w), support


def competition(scores, fields, support):
    """Soft categorical assignment to OCR slots plus a zero-logit background slot."""
    logits = (
        scores.float()[..., None, None]
        .expand_as(fields)
        .masked_fill(~support, -torch.inf)
    )
    weights = torch.cat((torch.zeros_like(logits[:, :1]), logits), 1).softmax(1)[:, 1:]
    return (weights * fields).sum(1, keepdim=True), weights


class EvidenceTransport(nn.Module):
    """No pyramid mask/semantic bypass inside the evidence branch; dimensions are inferred."""

    def __init__(self, text_dim, visual_dim, width=64):
        super().__init__()
        self.config = {"text_dim": text_dim, "visual_dim": visual_dim, "width": width}
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, width))
        self.visual = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, width)
        )
        self.geometry = nn.Linear(7, width)
        self.question_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.set_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.selection = nn.Linear(width, 1)
        self.scope = nn.Sequential(
            nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1)
        )
        self.rgb = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 24, 3, padding=1),
            nn.GroupNorm(4, 24),
            nn.GELU(),
        )
        self.condition = nn.Linear(width, 24)
        self.mask = nn.Sequential(
            nn.Conv2d(24, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 1)
        )

    def forward(self, batch):
        valid = batch["valid"]
        q = self.text(batch["question"].float())
        scene = self.visual(batch["scene"].float())[:, None]
        boxes = batch["boxes"].float()
        geometry = torch.cat(
            (boxes, boxes[..., 2:] - boxes[..., :2], batch["confidence"][..., None]), -1
        )
        words = (
            self.text(batch["words"].float())
            + self.visual(batch["visual"].float())
            + self.geometry(geometry)
        )
        attended, _ = self.question_attention(
            words, q, q, key_padding_mask=~batch["question_valid"], need_weights=False
        )
        words = self.norm1(words + attended)
        keys = torch.cat((words, scene), 1)
        available = torch.cat((valid, torch.ones_like(valid[:, :1])), 1)
        attended, _ = self.set_attention(
            words, keys, keys, key_padding_mask=~available, need_weights=False
        )
        words = self.norm2(words + attended)
        scores = self.selection(words).squeeze(-1)
        pooling = scores.float().masked_fill(~valid, -1e4).softmax(1) * valid
        pooled = (pooling[..., None] * words).sum(1)
        scope = self.scope(torch.cat((pooled, scene[:, 0]), -1)).squeeze(-1)
        gate = scope.float().sigmoid() * valid.any(1)
        b, n = valid.shape
        ids = valid.flatten().nonzero().flatten()
        crops = batch["crops"].flatten(0, 1)[ids].float() / 127.5 - 1
        # Only valid crops run the CNN. A one-slot dummy keeps the empty case finite.
        if len(ids):
            encoded = self.rgb(crops)
            conditioned = (
                encoded + self.condition(words.flatten(0, 1)[ids])[..., None, None]
            )
            local = self.mask(F.gelu(conditioned)).float()
            patches = local.new_zeros(b * n, 1, *local.shape[-2:]).index_copy(
                0, ids, local
            )
        else:
            anchor = sum(
                p.flatten()[0].float() * 0
                for module in (self.rgb, self.condition, self.mask)
                for p in module.parameters()
            )
            patches = crops.new_zeros(b * n, 1, *batch["crops"].shape[-2:]) + anchor
        patches = patches.reshape(b, n, 1, *patches.shape[-2:])
        fields, support = render_candidates(
            patches.sigmoid(), batch["contexts"], valid, batch["canvas_size"]
        )
        evidence, weights = competition(scores, fields, support)
        return {
            "scores": scores,
            "scope": scope,
            "gate": gate,
            "patches": patches,
            "fields": fields,
            "support": support,
            "weights": weights,
            "evidence": evidence,
        }


@torch.no_grad()
def compose_native(output, batch, index, size):
    """Inference-only native rasterization with stable per-pixel log-sum-exp."""
    scores = output["scores"][index].float()
    valid = batch["valid"][index]
    probability = scores.new_zeros(1, *size)
    log_denominator = scores.new_zeros(1, *size)  # Background logit is zero.
    for j in valid.nonzero().flatten().tolist():
        sample = local_window(
            output["patches"][index, j].float().sigmoid(),
            batch["contexts"][index, j].tolist(),
            size,
        )
        if sample is None:
            continue
        bounds, local = sample
        l, t, r, b = bounds
        previous = log_denominator[:, t:b, l:r]
        updated = torch.logaddexp(previous, scores[j])
        probability[:, t:b, l:r] = (
            probability[:, t:b, l:r] * (previous - updated).exp()
            + local * (scores[j] - updated).exp()
        )
        log_denominator[:, t:b, l:r] = updated
    return probability
