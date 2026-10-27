"""Shared pixel-centre geometry and stable native-resolution objectives."""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def pixel_window(box, size):
    """Half-open support: x1 <= (x+.5)/W < x2. No epsilon/floor raster heuristic."""
    h, w = size
    x1, y1, x2, y2 = [float(v) for v in box]

    def lower_bound(edge, length):
        # Search the declared centres directly. Multiplying a rounded normalized
        # edge back by W can put an exact-centre boundary one ULP above an integer.
        lo, hi = 0, length
        while lo < hi:
            mid = (lo + hi) // 2
            if (mid + 0.5) / length < edge:
                lo = mid + 1
            else:
                hi = mid
        return lo

    return (
        lower_bound(x1, w),
        lower_bound(y1, h),
        lower_bound(x2, w),
        lower_bound(y2, h),
    )


def pixel_coordinates(indices, size):
    h, w = size
    return torch.stack(
        (
            (indices.remainder(w).double() + 0.5) / w,
            (indices.div(w, rounding_mode="floor").double() + 0.5) / h,
        ),
        -1,
    )


def query_grid(feature, xy, *, precise=False):
    """Feature interpolation is float32; native mask-logit mapping uses float64."""
    with torch.autocast(feature.device.type, enabled=False):
        dtype = torch.float64 if precise else torch.float32
        grid = (xy.double() * 2 - 1).to(dtype).reshape(1, 1, -1, 2)
        return (
            F.grid_sample(
                feature.to(dtype), grid, align_corners=False, padding_mode="border"
            )[0, :, 0]
            .transpose(0, 1)
            .float()
        )


def sample_native_regions(feature, boxes, size):
    rows = []
    h, w = size
    axis_x = (torch.arange(w, device=feature.device, dtype=torch.float64) + 0.5) / w
    axis_y = (torch.arange(h, device=feature.device, dtype=torch.float64) + 0.5) / h
    for i in range(len(feature)):
        box = boxes[i].double()
        x = box[:, 0, None, None] + axis_x * (box[:, 2] - box[:, 0])[:, None, None]
        y = (
            box[:, 1, None, None]
            + axis_y[:, None] * (box[:, 3] - box[:, 1])[:, None, None]
        )
        xy = torch.stack((x.expand(-1, h, w), y.expand(-1, h, w)), -1)
        values = query_grid(feature[i : i + 1], xy.reshape(-1, 2))
        rows.append(values.reshape(len(box), h, w, -1).permute(0, 3, 1, 2))
    return torch.stack(rows)


def local_window(patch, box, size):
    left, top, right, bottom = pixel_window(box, size)
    if right <= left or bottom <= top:
        return None
    h, w = size
    x1, y1, x2, y2 = [float(v) for v in box]
    width, height = right - left, bottom - top
    chunks = []
    for begin in range(0, width * height, 65536):
        end = min(begin + 65536, width * height)

        def query(patch, begin=begin, end=end):
            ids = torch.arange(begin, end, device=patch.device)
            xx = ids.remainder(width).double() + left + 0.5
            yy = ids.div(width, rounding_mode="floor").double() + top + 0.5
            xy = torch.stack(((xx / w - x1) / (x2 - x1), (yy / h - y1) / (y2 - y1)), -1)
            return query_grid(patch[None], xy, precise=True).transpose(0, 1)

        chunks.append(
            checkpoint(query, patch, use_reentrant=False)
            if patch.requires_grad
            else query(patch)
        )
    return (left, top, right, bottom), torch.cat(chunks, -1).reshape(1, height, width)


def crop_canvas(patches, boxes, valid, size):
    """Average overlapping absolute mask logits; deterministic geometry, no learned gate."""
    total = patches.new_zeros(1, *size, dtype=torch.float32)
    count = torch.zeros_like(total)
    available = valid.detach().cpu().tolist()
    for j, box in enumerate(boxes.detach().cpu().tolist()):
        if not available[j]:
            continue
        sample = local_window(patches[j], box, size)
        if sample is None:
            continue
        (l, t, r, b), values = sample
        # Out-of-place padding keeps overlap gradients independent of assignment order.
        total = total + F.pad(values, (l, size[1] - r, t, size[0] - b))
        count[:, t:b, l:r] += 1
    return total / count.clamp_min(1), count


def global_native(base, delta, size):
    with torch.autocast(base.device.type, enabled=False):
        return (
            F.interpolate(base.float(), size, mode="bilinear", align_corners=False)
            + F.interpolate(delta.float(), size, mode="bilinear", align_corners=False)
        )[0]


def compose_native(base, delta, patches, boxes, valid, size):
    global_logits = global_native(base, delta, size)
    local, count = crop_canvas(patches, boxes, valid, size)
    alpha = (count > 0).float()
    return global_logits * (1 - alpha) + local * alpha


def native_loss(logits, truth):
    """Full native pixel BCE + soft Dice; float32 nonlinearities, float64 reductions."""
    if logits.shape != truth.shape or logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("Expected matching 1xHxW native logits and binary GT")
    with torch.autocast(logits.device.type, enabled=False):
        z, y = logits.float(), truth.to(logits.device, non_blocking=True).float()
        bce = (
            F.binary_cross_entropy_with_logits(z, y, reduction="none").sum(
                dtype=torch.float64
            )
            / y.numel()
        )
        prob = z.sigmoid()
        intersection = (prob * y).sum(dtype=torch.float64)
        denominator = prob.sum(dtype=torch.float64) + y.sum(dtype=torch.float64)
        dice = 1 - (2 * intersection + 1e-6) / (denominator + 1e-6)
        return bce + dice


def native_metric_samples(output, batch, *, pixel_totals=None):
    from metrics import accumulate_pixel_counts

    for i, name in enumerate(batch["filename"]):
        prediction = output["native_logits"][i][0] > 0
        truth = batch["original_mask"][i][0].to(prediction.device).bool()
        if prediction.shape != truth.shape:
            raise ValueError("Native output/GT dimensions differ")
        intersection, union = (
            (prediction & truth).sum().item(),
            (prediction | truth).sum().item(),
        )
        accumulate_pixel_counts(prediction, truth, pixel_totals)
        yield name, intersection / union if union else 1.0, prediction


def native_base_output(base, batch):
    return {
        "native_logits": [
            global_native(base[i : i + 1], base[i : i + 1] * 0, tuple(truth.shape[-2:]))
            for i, truth in enumerate(batch["original_mask"])
        ]
    }
