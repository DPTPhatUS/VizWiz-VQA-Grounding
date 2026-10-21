"""Question-guided native-image crops and local logit refinement.

Boxes are normalized xyxy edges in the upright original image. Sampling and
inverse placement both use pixel centers with align_corners=False. Proposal
selection is discrete and detached; the locator has its own mask supervision.
"""

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.mask_decoder import conv_block

CROP_FRACTIONS = (0.25, 0.5)  # Square side / native image's shorter side.
PROPOSAL_GRID = 16
UNCERTAINTY_WEIGHT = 0.25
OVERLAP_PENALTY = 0.5
EDGE_TAPER = 0.1  # Fraction of crop width/height used for the seam transition.


def sample_regions(
    image: Tensor,
    boxes: Tensor,
    size: int,
    *,
    mode: Literal["bilinear", "nearest"] = "bilinear",
) -> Tensor:
    """CHW image -> KCHW crops, sampled before any full-image resize."""
    with torch.autocast(image.device.type, enabled=False):
        positions = (torch.arange(size, device=boxes.device).float() + 0.5) / size
        x = boxes[:, 0, None] + positions * (boxes[:, 2] - boxes[:, 0])[:, None]
        y = boxes[:, 1, None] + positions * (boxes[:, 3] - boxes[:, 1])[:, None]
        grid = torch.stack(
            (x[:, None, :].expand(-1, size, -1), y[:, :, None].expand(-1, -1, size)),
            dim=-1,
        )
        return F.grid_sample(
            image[None].float().expand(len(boxes), -1, -1, -1),
            grid * 2 - 1,
            mode=mode,
            padding_mode="border",
            align_corners=False,
        )


def boxes_at_centers(
    centers: Tensor, fraction: float, height: int, width: int
) -> Tensor:
    """Square crops in native pixels; shift at borders without changing their size."""
    extent = centers.new_tensor(
        [fraction * min(height, width) / width, fraction * min(height, width) / height]
    )
    lower = torch.minimum((centers - extent / 2).clamp_min(0), 1 - extent)
    return torch.cat((lower, lower + extent), dim=-1)


def snap_boxes(boxes: Tensor, height: int, width: int) -> Tensor:
    """Use actual native pixel edges consistently for RGB, features, GT and pasting."""
    scale = boxes.new_tensor([width, height, width, height])
    edges = boxes * scale
    edges = torch.cat((edges[:, :2].floor(), edges[:, 2:].ceil()), dim=-1)
    return torch.minimum(edges.clamp_min(0), scale) / scale


def native_rgb_crops(image: Tensor, boxes: Tensor, size: int) -> Tensor:
    """Slice bytes first, then bicubic antialiased resize; no native FP32 RGB copy."""
    height, width = image.shape[-2:]
    scale = boxes.new_tensor([width, height, width, height])
    edges = (boxes * scale).round().long().tolist()
    crops = []
    with torch.autocast(image.device.type, enabled=False):
        for x0, y0, x1, y1 in edges:
            crop = image[None, :, y0:y1, x0:x1].float() / 255
            crops.append(
                F.interpolate(
                    crop,
                    size=(size, size),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                ).clamp(0, 1)
            )
    return torch.cat(crops)


def box_iou(boxes: Tensor, reference: Tensor) -> Tensor:
    intersection = (
        (
            torch.minimum(boxes[:, 2:], reference[2:])
            - torch.maximum(boxes[:, :2], reference[:2])
        )
        .clamp_min(0)
        .prod(-1)
    )
    area = (boxes[:, 2:] - boxes[:, :2]).prod(-1)
    other = (reference[2:] - reference[:2]).prod()
    return intersection / (area + other - intersection).clamp_min(1e-8)


def window_scores(utility: Tensor, boxes: Tensor) -> Tensor:
    """Exact rectangular means on the locator grid, without sparse-sampling aliasing."""
    height, width = utility.shape[-2:]
    scale = boxes.new_tensor([width, height])
    lower = (boxes[:, :2] * scale).floor().long()
    upper = (boxes[:, 2:] * scale).ceil().long()
    x0, y0 = lower[:, 0].clamp(0, width - 1), lower[:, 1].clamp(0, height - 1)
    x1, y1 = upper[:, 0].clamp(1, width), upper[:, 1].clamp(1, height)
    integral = F.pad(utility[0].cumsum(0).cumsum(1), (1, 0, 1, 0))
    sums = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    return sums / ((x1 - x0) * (y1 - y0)).float()


@torch.no_grad()
def select_regions(
    locator: Tensor, global_logits: Tensor, native_images: list[Tensor]
) -> Tensor:
    """One crop per scale; prefer relevant, uncertain regions and discourage overlap."""
    device = locator.device
    positions = (
        torch.arange(PROPOSAL_GRID, device=device).float() + 0.5
    ) / PROPOSAL_GRID
    yy, xx = torch.meshgrid(positions, positions, indexing="ij")
    centers = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
    probability = F.interpolate(
        global_logits.float(),
        size=locator.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).sigmoid()
    relevance = locator.float().sigmoid()
    # Confidently wrong masks are still eligible through the independent locator.
    utility = relevance * (1 + UNCERTAINTY_WEIGHT * 4 * probability * (1 - probability))
    selected = []
    for index, image in enumerate(native_images):
        height, width = image.shape[-2:]
        chosen = []
        for fraction in CROP_FRACTIONS:
            candidates = boxes_at_centers(centers, fraction, height, width)
            score = window_scores(utility[index], candidates)
            if chosen:
                score = score * (1 - OVERLAP_PENALTY * box_iou(candidates, chosen[0]))
            chosen.append(candidates[score.argmax()])
        selected.append(snap_boxes(torch.stack(chosen), height, width))
    return torch.stack(selected)


@torch.no_grad()
def training_region(mask: Tensor) -> Tensor:
    """One GT-only teaching crop; never used for the final prediction or at eval."""
    height, width = mask.shape[-2:]
    points = (mask[0] > 0).nonzero().float()
    if len(points):
        centroid = points.mean(0)
        point = points[(points - centroid).square().sum(-1).argmin()]
        center = torch.stack(((point[1] + 0.5) / width, (point[0] + 0.5) / height))
    else:
        center = mask.new_tensor([0.5, 0.5], dtype=torch.float32)
    return snap_boxes(
        boxes_at_centers(center[None], CROP_FRACTIONS[0], height, width), height, width
    )


def place_corrections(
    global_logits: Tensor,
    boxes: Tensor,
    corrections: Tensor,
    quality: Tensor,
    size: tuple[int, int],
) -> Tensor:
    """Blend local signed corrections at any output grid, including native resolution.

    The outer 10% tapers to zero; the interior permits full replacement by a
    reliable crop. Dividing by max(1,sum(weights)) prevents overlap amplification.
    """
    base = F.interpolate(
        global_logits.float(), size=size, mode="bilinear", align_corners=False
    )
    height, width = size
    y = (torch.arange(height, device=base.device).float() + 0.5) / height
    x = (torch.arange(width, device=base.device).float() + 0.5) / width
    total, weights = torch.zeros_like(base), torch.zeros_like(base)
    with torch.autocast(base.device.type, enabled=False):
        for box, correction, confidence in zip(boxes, corrections, quality):
            u = (x - box[0]) / (box[2] - box[0])
            v = (y - box[1]) / (box[3] - box[1])
            grid = torch.stack(
                (u[None, :].expand(height, -1), v[:, None].expand(-1, width)), dim=-1
            )
            delta = F.grid_sample(
                correction[None].float(),
                (grid[None] * 2 - 1),
                align_corners=False,
                padding_mode="border",
            )
            horizontal = (torch.minimum(u, 1 - u) / EDGE_TAPER).clamp(0, 1)
            vertical = (torch.minimum(v, 1 - v) / EDGE_TAPER).clamp(0, 1)
            window = horizontal[None, :] * vertical[:, None]
            weight = confidence.float().sigmoid() * window[None, None]
            total = total + weight * delta
            weights = weights + weight
    return base + total / weights.clamp_min(1)


def prediction_at_size(
    output: dict, index: int, size: tuple[int, int], *, global_only: bool = False
) -> Tensor:
    """Canonical train/eval restoration: paste native crops directly at the target grid."""
    zoom = output["zoom"]
    if global_only:
        return F.interpolate(
            zoom["global_logits"][index : index + 1].float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
    return place_corrections(
        zoom["global_logits"][index : index + 1],
        zoom["boxes"][index],
        zoom["corrections"][index],
        zoom["quality"][index],
        size,
    )


class CropReader(nn.Module):
    """Small shared U-Net; local pixels query full Q tokens with global spatial context."""

    def __init__(self, semantic_dim: int, text_dim: int, *, tiny: bool = False):
        super().__init__()
        a, b, c = (4, 8, 16) if tiny else (16, 32, 64)
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(conv_block(3, a), conv_block(a, a)),
                nn.Sequential(conv_block(a, b), conv_block(b, b)),
                nn.Sequential(conv_block(b, c), conv_block(c, c)),
            ]
        )
        self.context = nn.Conv2d(semantic_dim, c, 1)
        self.text_projection = nn.Linear(text_dim, c)
        self.query_norm, self.text_norm = nn.LayerNorm(c), nn.LayerNorm(c)
        self.attention = nn.MultiheadAttention(c, 4, batch_first=True)
        self.fusion = conv_block(2 * c, c)
        self.up128 = conv_block(c + b, b)
        self.up256 = conv_block(b + a, a)
        self.correction = nn.Conv2d(a, 1, 1)
        self.quality = nn.Linear(c, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        nn.init.zeros_(self.quality.weight)
        nn.init.constant_(self.quality.bias, -2)

    def forward(
        self, rgb: Tensor, context: Tensor, text: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        s1 = self.encoder[0](rgb)
        s2 = self.encoder[1](F.max_pool2d(s1, 2))
        x = self.encoder[2](F.max_pool2d(s2, 2))
        x = x + self.context(context)
        query = self.query_norm(x.flatten(2).transpose(1, 2))
        tokens = self.text_norm(self.text_projection(text))
        attended, _ = self.attention(
            query, tokens, tokens, key_padding_mask=~attention_mask, need_weights=False
        )
        attended = attended.transpose(1, 2).reshape_as(x)
        x = self.fusion(torch.cat((x, attended), dim=1))
        quality = self.quality(x.mean((2, 3))).squeeze(-1)
        x = self.up128(
            torch.cat(
                (
                    F.interpolate(
                        x, size=s2.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    s2,
                ),
                dim=1,
            )
        )
        x = self.up256(
            torch.cat(
                (
                    F.interpolate(
                        x, size=s1.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    s1,
                ),
                dim=1,
            )
        )
        return self.correction(x), quality


class QuestionGuidedZoom(nn.Module):
    def __init__(
        self, semantic_dim: int, text_dim: int, question_dim: int, *, tiny: bool = False
    ):
        super().__init__()
        hidden = 8 if tiny else 32
        self.locator = nn.Sequential(
            nn.Conv2d(semantic_dim + question_dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.reader = CropReader(semantic_dim, text_dim, tiny=tiny)
        self.crop_size = 32 if tiny else 256

    def forward(
        self,
        feature: Tensor,
        question: Tensor,
        text: Tensor,
        attention_mask: Tensor,
        global_logits: Tensor,
        batch: dict,
        mean: Tensor,
        std: Tensor,
        run_block,
    ) -> dict[str, Tensor]:
        images = batch["native_image"]
        if len(images) != len(feature) or any(
            im.ndim != 3
            or im.dtype != torch.uint8
            or im.shape[0] != 3
            or im.device != feature.device
            for im in images
        ):
            raise ValueError(
                "native_image must contain one upright uint8 RGB CHW tensor per sample"
            )
        qmap = question[:, :, None, None].expand(-1, -1, *feature.shape[-2:])
        locator = self.locator(torch.cat((feature, qmap), dim=1))
        boxes = select_regions(locator, global_logits, images)
        all_boxes = boxes
        if self.training:
            teachers = torch.stack(
                [training_region(mask) for mask in batch["native_mask"]]
            )
            all_boxes = torch.cat((boxes, teachers), dim=1)
        count = all_boxes.shape[1]
        rgb, context, coarse = [], [], []
        for i, native in enumerate(images):
            rgb.append(native_rgb_crops(native, all_boxes[i], self.crop_size))
            context.append(
                sample_regions(feature[i], all_boxes[i], self.crop_size // 4)
            )
            coarse.append(
                sample_regions(global_logits[i], all_boxes[i], self.crop_size)
            )
        rgb = (torch.cat(rgb) - mean) / std
        delta, quality = run_block(
            self.reader,
            rgb,
            torch.cat(context),
            text.repeat_interleave(count, 0),
            attention_mask.repeat_interleave(count, 0),
        )
        shape = (len(images), count, 1, self.crop_size, self.crop_size)
        output = dict(
            global_logits=global_logits,
            boxes=boxes,
            corrections=delta.reshape(shape)[:, : len(CROP_FRACTIONS)],
            quality=quality.reshape(len(images), count)[:, : len(CROP_FRACTIONS)],
        )
        if self.training:
            targets = torch.cat(
                [
                    sample_regions(mask, all_boxes[i], self.crop_size, mode="nearest")
                    for i, mask in enumerate(batch["native_mask"])
                ]
            )
            # Area resize starts from native GT, preserving tiny foreground coverage.
            locator_target = torch.stack(
                [
                    F.interpolate(
                        mask[None].float(), size=locator.shape[-2:], mode="area"
                    )[0]
                    for mask in batch["native_mask"]
                ]
            )
            output.update(
                locator_logits=locator,
                locator_target=locator_target,
                crop_logits=torch.cat(coarse) + delta.float(),
                crop_target=targets,
                crop_quality=quality,
            )
        return output
