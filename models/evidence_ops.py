"""Independent multi-region evidence composition, with no single-candidate softmax."""

import torch
from torch.nn import functional as F


def splat_regions(patches, boxes, output_size):
    """Sum B,N,C,h,w patches inside fixed normalized boxes, preserving gradients."""
    b, n, channels = patches.shape[:3]
    h, w = output_size
    result = (
        patches.new_zeros(b, channels, h, w, dtype=torch.float32)
        + patches.float().sum() * 0
    )
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
                result[i, :, top:bottom, left:right] + local * inside
            )
    return result


def compose_evidence(local_probability, boxes, inclusion, output_size):
    """Noisy-OR union in optical-density space: separate true words retain their mass."""
    probability = (local_probability.float() * inclusion[..., None, None, None]).clamp(
        0, 1 - 1e-6
    )
    density = -torch.log1p(-probability)
    total = splat_regions(density, boxes, output_size)
    return -torch.expm1(-total)


def rasterize_tokens(tokens, local_probability, boxes, inclusion, output_size):
    weights = local_probability.float() * inclusion[..., None, None, None]
    patch = torch.cat((tokens[..., None, None] * weights, weights), 2)
    field = splat_regions(patch, boxes, output_size)
    return field[:, :-1] / field[:, -1:].clamp_min(1e-6)
