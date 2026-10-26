"""Backbone-independent normalized region sampling and local correction rendering."""

import torch
from torch.nn import functional as F

PATCH_SIZE = (48, 128)


def sample_regions(feature, boxes, size=PATCH_SIZE):
    """Normalized xyxy boxes -> B,N,C,h,w using continuous pixel centers."""
    b, n = boxes.shape[:2]
    h, w = size
    xs = (torch.arange(w, device=feature.device, dtype=torch.float32) + 0.5) / w
    ys = (torch.arange(h, device=feature.device, dtype=torch.float32) + 0.5) / h
    x = (
        boxes[..., 0, None, None]
        + xs * (boxes[..., 2] - boxes[..., 0])[..., None, None]
    )
    y = (
        boxes[..., 1, None, None]
        + ys[:, None] * (boxes[..., 3] - boxes[..., 1])[..., None, None]
    )
    grid = torch.stack((x.expand(b, n, h, w), y.expand(b, n, h, w)), -1) * 2 - 1
    sampled = F.grid_sample(
        feature.float(),
        grid.reshape(b, n * h, w, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled.reshape(b, feature.shape[1], n, h, w).transpose(1, 2)


def render_regions(patches, boxes, weights, output_size):
    """Paste local corrections only inside their normalized xyxy support boxes.

    Geometry is fixed input, not learned. Candidate probabilities sum to <=1,
    including at overlapping regions; no hard argmax or validation-tuned gate.
    """
    b, n = boxes.shape[:2]
    h, w = output_size
    result = patches.new_zeros(b, 1, h, w, dtype=torch.float32)
    coordinates = boxes.detach().cpu().tolist()
    for i in range(b):
        for j in range(n):
            x1, y1, x2, y2 = coordinates[i][j]
            left, top = max(0, int(x1 * w)), max(0, int(y1 * h))
            right, bottom = (
                min(w, int(x2 * w + 0.999999)),
                min(h, int(y2 * h + 0.999999)),
            )
            if right <= left or bottom <= top:
                continue
            # Sampling inverse coordinates avoids stretching fractional box edges.
            xs = (torch.arange(left, right, device=patches.device) + 0.5) / w
            ys = (torch.arange(top, bottom, device=patches.device) + 0.5) / h
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            grid = (
                torch.stack(((xx - x1) / (x2 - x1), (yy - y1) / (y2 - y1)), -1) * 2 - 1
            )
            local = F.grid_sample(
                patches[i, j : j + 1].float(),
                grid[None],
                padding_mode="border",
                align_corners=False,
            )[0]
            inside = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
            result[i, :, top:bottom, left:right] = (
                result[i, :, top:bottom, left:right] + local * inside * weights[i, j]
            )
    return result
