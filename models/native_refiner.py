"""Two lightweight native-coordinate refiners with a shared question/OCR context."""

import torch
from torch import nn
from torch.nn import functional as F

from models.ocr_evidence import block
from models.region_ops import sample_regions
from native_refinement_ops import (
    compose_native,
    crop_canvas,
    global_native,
    pixel_coordinates,
    pixel_window,
    query_grid,
    sample_native_regions,
)

QUERY_CHUNK = 65536
PIXEL_SAMPLES = 8192
CLASS_SAMPLES = 1024


def point_samples(
    global_logits,
    boxes,
    valid,
    truth,
    pixel_samples=PIXEL_SAMPLES,
    class_samples=CLASS_SAMPLES,
):
    """Known proposal probabilities plus explicit class-balanced supervision."""
    device = global_logits.device
    h, w = global_logits.shape[-2:]
    with torch.no_grad():
        uncertainty = (-global_logits.detach().double().abs()).exp().flatten()
        # exp may underflow for a confidently wrong mask; uniform component remains.
        uncertainty = (
            uncertainty / uncertainty.sum()
            if uncertainty.sum() > 0
            else torch.full_like(uncertainty, 1 / (h * w))
        )
        support = torch.zeros(h, w, device=device, dtype=torch.float64)
        available = valid.detach().cpu().tolist()
        for j, box in enumerate(boxes.detach().cpu().tolist()):
            if available[j]:
                l, t, r, b = pixel_window(box, (h, w))
                support[t:b, l:r] = 1
        support = support.flatten()
        support = (
            support / support.sum()
            if support.sum() > 0
            else torch.full_like(support, 1 / (h * w))
        )
        proposal = 0.5 / (h * w) + 0.25 * uncertainty + 0.25 * support
        proposal = proposal / proposal.sum()
        cumulative = proposal.cumsum(0)
        cumulative[-1] = 1
        sampled = torch.searchsorted(
            cumulative, torch.rand(pixel_samples, device=device, dtype=torch.float64)
        ).clamp_max(h * w - 1)
        pixel_weights = 1 / ((h * w) * proposal[sampled] * pixel_samples)
        indices, pixel, balanced = (
            [sampled],
            [pixel_weights],
            [torch.zeros_like(pixel_weights)],
        )
        y = truth.to(device, non_blocking=True).bool().flatten()
        classes = [torch.nonzero(y == label).flatten() for label in (False, True)]
        present = sum(len(c) > 0 for c in classes)
        for choices in classes:
            if len(choices):
                ids = choices[
                    torch.randint(len(choices), (class_samples,), device=device)
                ]
                indices.append(ids)
                pixel.append(
                    torch.zeros(class_samples, device=device, dtype=torch.float64)
                )
                balanced.append(
                    torch.full(
                        (class_samples,),
                        1 / (present * class_samples),
                        device=device,
                        dtype=torch.float64,
                    )
                )
    return torch.cat(indices), torch.cat(pixel), torch.cat(balanced)


def rgb_neighbours(rgb, indices):
    h, w = rgb.shape[-2:]
    x, y = indices.remainder(w), indices.div(w, rounding_mode="floor")
    samples = [
        rgb[:, (y + dy).clamp(0, h - 1), (x + dx).clamp(0, w - 1)].transpose(0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    ]
    return torch.cat(samples, -1).float() / 127.5 - 1


class NativeRefiner(nn.Module):
    def __init__(
        self,
        text_dim,
        detail_dim,
        semantic_dim,
        width=64,
        mode="native_crop",
        tiny=False,
    ):
        super().__init__()
        self.mode, self.tiny = mode, tiny
        self.text = nn.Linear(text_dim, width)
        self.word = nn.Linear(detail_dim + 3 + 7, width)
        self.question_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.context_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.scene = nn.Linear(detail_dim, width)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.condition = nn.Linear(width, 8)
        self.global_head = nn.Sequential(
            block(detail_dim + semantic_dim + 11, 32),
            block(32, 16),
            nn.Conv2d(16, 1, 1),
        )
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)
        if mode == "native_crop":
            self.encoder1 = block(3, 16)
            self.encoder2 = block(16, 32, 2)
            self.encoder3 = block(32, 48, 2)
            self.middle = block(48 + semantic_dim + 8, 48)
            self.decode2 = block(48 + 32, 32)
            self.decode1 = nn.Sequential(block(32 + 16, 16), nn.Conv2d(16, 1, 1))
        elif mode == "native_point":
            self.word_support = nn.Linear(width, 1)
            self.point_features = nn.Sequential(
                block(detail_dim + semantic_dim + 10, 16)
            )
            self.point_head = nn.Sequential(
                nn.Linear(46, 32),
                nn.GELU(),
                nn.Linear(32, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            nn.init.zeros_(self.point_head[-1].weight)
            nn.init.zeros_(self.point_head[-1].bias)
        else:
            raise ValueError("Invalid native refiner mode")

    def _context(
        self,
        detail,
        question_tokens,
        question_mask,
        recognized_tokens,
        patches,
        boxes,
        confidence,
        valid,
    ):
        b, _n = valid.shape
        local = sample_regions(detail, boxes, (4, 8)).mean((-1, -2))
        geometry = torch.cat(
            (boxes, boxes[..., 2:] - boxes[..., :2], confidence[..., None]), -1
        )
        words = self.text(recognized_tokens) + self.word(
            torch.cat((local, patches.mean((-1, -2)), geometry), -1)
        )
        scene = self.scene(detail.mean((-1, -2)))[:, None]
        tokens = torch.cat((words, scene), 1)
        q = self.text(question_tokens)
        attended, _ = self.question_attention(
            tokens, q, q, key_padding_mask=~question_mask, need_weights=False
        )
        tokens = self.norm1(tokens + attended)
        available = torch.cat(
            (valid, torch.ones(b, 1, device=valid.device, dtype=torch.bool)), 1
        )
        attended, _ = self.context_attention(
            tokens, tokens, tokens, key_padding_mask=~available, need_weights=False
        )
        return self.norm2(tokens + attended)

    def _crops(self, crops, semantic, boxes, tokens):
        b, n = crops.shape[:2]
        rgb = crops.flatten(0, 1) * 2 - 1
        a = self.encoder1(rgb)
        b2 = self.encoder2(a)
        c = self.encoder3(b2)
        sem = sample_native_regions(semantic, boxes, c.shape[-2:]).flatten(0, 1)
        condition = (
            self.condition(tokens[:, :n])
            .flatten(0, 1)[..., None, None]
            .expand(-1, -1, *c.shape[-2:])
        )
        c = self.middle(torch.cat((c, sem, condition), 1))
        c = self.decode2(
            torch.cat(
                (
                    F.interpolate(
                        c, b2.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    b2,
                ),
                1,
            )
        )
        c = self.decode1(
            torch.cat(
                (
                    F.interpolate(
                        c, a.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    a,
                ),
                1,
            )
        )
        return c.reshape(b, n, 1, *c.shape[-2:])

    def _points(self, features, global_logits, rgb, indices):
        xy = pixel_coordinates(indices, global_logits.shape[-2:])
        context = query_grid(features, xy)
        value = torch.cat(
            (
                context,
                rgb_neighbours(rgb, indices),
                xy.float() * 2 - 1,
                global_logits.flatten()[indices, None].float(),
            ),
            -1,
        )
        return (
            global_logits.flatten()[indices]
            + self.point_head(value).squeeze(-1).float()
        )

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
        refine_boxes,
        native_rgb,
        native_crops=None,
        truths=None,
        global_only=False,
    ):
        tokens = self._context(
            detail,
            question_tokens,
            question_mask,
            recognized_tokens,
            patches,
            boxes,
            confidence,
            valid,
        )
        shape = (96, 96) if not self.tiny else (16, 16)
        if self.mode == "native_crop":
            local = self._crops(native_crops, semantic, refine_boxes, tokens)
        else:
            local = self.word_support(tokens[:, : valid.shape[1]])[
                ..., None, None
            ].expand(-1, -1, 1, 8, 8)
        evidence = []
        coverage = []
        for i in range(len(detail)):
            field, count = crop_canvas(
                local[i].float().sigmoid(), refine_boxes[i], valid[i], shape
            )
            evidence.append(field)
            coverage.append((count > 0).float())
        evidence, coverage = torch.stack(evidence), torch.stack(coverage)
        condition = self.condition(tokens[:, -1])[..., None, None].expand(
            -1, -1, *shape
        )
        d = F.adaptive_avg_pool2d(detail, shape)
        s = F.adaptive_avg_pool2d(semantic, shape)
        coarse = F.interpolate(
            base_logits.float(), shape, mode="bilinear", align_corners=False
        )
        delta = self.global_head(
            torch.cat((d, s, coarse, evidence, coverage, condition), 1)
        )
        output = {
            "base_logits": base_logits,
            "global_delta": delta,
            "local_logits": local,
            "refine_boxes": refine_boxes,
            "candidate_valid": valid,
        }
        if self.mode == "native_point":
            features = self.point_features(
                torch.cat((d, s, evidence, coverage, condition), 1)
            )
        native = []
        sampled = []
        for i, (wrapped,) in enumerate(native_rgb):
            rgb = (
                wrapped.to(base_logits.device, non_blocking=True)
                if self.mode == "native_point"
                else wrapped
            )
            size = tuple(rgb.shape[-2:])
            global_logits = global_native(
                base_logits[i : i + 1], delta[i : i + 1], size
            )
            if self.mode == "native_crop":
                native.append(
                    global_logits
                    if global_only
                    else compose_native(
                        base_logits[i : i + 1],
                        delta[i : i + 1],
                        local[i],
                        refine_boxes[i],
                        valid[i],
                        size,
                    )
                )
            elif self.training:
                ids, pixel, balanced = point_samples(
                    global_logits,
                    refine_boxes[i],
                    valid[i],
                    truths[i],
                    256 if self.tiny else PIXEL_SAMPLES,
                    64 if self.tiny else CLASS_SAMPLES,
                )
                sampled.append(
                    {
                        "logits": self._points(
                            features[i : i + 1], global_logits, rgb, ids
                        ),
                        "indices": ids,
                        "pixel_weights": pixel,
                        "balanced_weights": balanced,
                    }
                )
            elif global_only:
                native.append(global_logits)
            else:
                values = []
                for begin in range(0, size[0] * size[1], QUERY_CHUNK):
                    ids = torch.arange(
                        begin,
                        min(begin + QUERY_CHUNK, size[0] * size[1]),
                        device=rgb.device,
                    )
                    values.append(
                        self._points(features[i : i + 1], global_logits, rgb, ids)
                    )
                native.append(torch.cat(values).reshape(1, *size))
        output.update(native_logits=native, point_samples=sampled)
        return output
