"""Exact polygon edits, parent-unseen cohorts, and independent safety auditing."""

import functools
import hashlib
import math
import operator
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torch.utils.data import Dataset

from ocr_selection_data import hypothesis_members

VERSION = "utility-evidence-v1"
OPERATIONS = ("replace", "intersect", "add", "erase")
THRESHOLDS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, None)


def validate_manifest(manifest):
    if manifest.get("version") != VERSION or not manifest.get("complete"):
        raise ValueError("Incomplete or incompatible utility cache")
    cohorts = manifest["cohorts"]
    if set(cohorts) != {"fit", "tune", "safety"} or any(
        not names for names in cohorts.values()
    ):
        raise ValueError("Missing independent fit/tune/safety cohort")
    names = functools.reduce(operator.iadd, cohorts.values(), [])
    if len(names) != len(set(names)) or set(names) != set(
        manifest["splits"]["train"]["names"]
    ):
        raise ValueError(
            "Head cohorts overlap or do not partition the cached train images"
        )
    if set(names) & set(manifest["parent_training_images"]):
        raise ValueError("Head/safety images were exposed to parent training")


def split_head(names, seed=42):
    ordered = sorted(
        names,
        key=lambda n: hashlib.sha256(
            f"utility-cohorts-v1:{seed}:{n}".encode()
        ).hexdigest(),
    )
    if len(ordered) < 5:
        raise ValueError(
            "Need at least five parent-unseen images for fit/tune/safety cohorts"
        )
    a, b = max(1, int(0.6 * len(ordered))), max(2, int(0.8 * len(ordered)))
    return {"fit": ordered[:a], "tune": ordered[a:b], "safety": ordered[b:]}


def pack_mask(mask):
    return {
        "shape": tuple(mask.shape),
        "bits": torch.from_numpy(np.packbits(mask, bitorder="little").copy()),
    }


def unpack_mask(packed):
    return (
        np.unpackbits(
            packed["bits"].numpy(), count=math.prod(packed["shape"]), bitorder="little"
        )
        .reshape(packed["shape"])
        .astype(bool)
    )


def polygon_union(polygons, indices, size):
    canvas = Image.new("1", size)
    draw = ImageDraw.Draw(canvas)
    for i in indices:
        draw.polygon([tuple(p) for p in polygons[i]], fill=1)
    return np.asarray(canvas, dtype=bool)


def edited_mask(parent, polygon, operation):
    if operation == "replace":
        return polygon
    if operation == "intersect":
        return parent & polygon
    if operation == "add":
        return parent | polygon
    if operation == "erase":
        return parent & ~polygon
    raise ValueError("Unknown edit operation")


def mask_iou(mask, truth):
    intersection = int(np.count_nonzero(mask & truth))
    union = int(np.count_nonzero(mask | truth))
    return intersection / union if union else 1.0


def action_features(parent, regions, visual_map, truth=None):
    """Candidate construction is GT-free; optional labels never affect action membership/order.

    Masks are rendered in original coordinates. Only descriptors are downsampled.
    The action set is bounded by 4*(2*N+1)+1, with exact parent at slot zero.
    """
    h, w = parent.shape
    members = hypothesis_members(regions, max(1, len(regions)), (w, h))[1:]
    polygons = [r["polygon"] for r in regions]
    descriptions, membership, operations, pooled, quality, counts = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    parent_area = int(parent.sum())
    truth_quality = mask_iou(parent, truth) if truth is not None else None
    pixels = visual_map.detach().float().cpu().flatten(1).T

    def append(mask, group, op):
        area = int(mask.sum())
        intersection = int(np.count_nonzero(mask & parent))
        columns = mask.sum(0, dtype=np.int64)
        rows = mask.sum(1, dtype=np.int64)
        xs, ys = np.flatnonzero(columns), np.flatnonzero(rows)
        # Ratios use Python integers/float64, not float16 mask sums.
        cx = float((columns * (np.arange(w) + 0.5)).sum() / (area * w)) if area else 0.0
        cy = float((rows * (np.arange(h) + 0.5)).sum() / (area * h)) if area else 0.0
        changed = int(np.count_nonzero(mask ^ parent))
        descriptor = [
            area / (h * w),
            parent_area / (h * w),
            cx,
            cy,
            intersection / max(1, area),
            intersection / max(1, parent_area),
            changed / (h * w),
            math.log1p(area / max(1, parent_area)),
            float(xs.max() - xs.min() + 1) / w if area else 0.0,
            float(ys.max() - ys.min() + 1) / h if area else 0.0,
        ]
        grid = F.adaptive_avg_pool2d(
            torch.from_numpy(mask.copy()).float()[None, None], visual_map.shape[-2:]
        ).flatten()
        pooled.append((grid @ pixels / grid.sum().clamp_min(1e-8)).half())
        descriptions.append(descriptor)
        membership.append(group)
        operations.append(op)
        if truth is not None:
            numerator = int(np.count_nonzero(mask & truth))
            denominator = int(np.count_nonzero(mask | truth))
            counts.append((numerator, denominator) if denominator else (1, 1))
            quality.append(numerator / denominator if denominator else 1.0)

    zero = torch.zeros(len(regions), dtype=torch.bool)
    append(parent, zero, 0)
    # No learned geometry, score multiplication, or native CNN mask regeneration.
    for group in members:
        ids = group.nonzero().flatten().tolist()
        if not ids:
            continue
        polygon = polygon_union(polygons, ids, (w, h))
        for op, name in enumerate(OPERATIONS, 1):
            mask = edited_mask(parent, polygon, name)
            if np.array_equal(mask, parent):
                continue
            append(mask, group, op)
    features = {
        "members": torch.stack(membership),
        "operation": torch.tensor(operations),
        "descriptors": torch.tensor(descriptions, dtype=torch.float32),
        "action_visual": torch.stack(pooled),
    }
    labels = (
        None
        if truth is None
        else {
            "quality": torch.tensor(quality, dtype=torch.float64),
            "quality_counts": torch.tensor(counts, dtype=torch.int64),
            "gain": torch.tensor(quality, dtype=torch.float64) - truth_quality,
            "area_fraction": torch.tensor(float(truth.mean()), dtype=torch.float64),
        }
    )
    return features, labels


def render_action(record, index):
    parent = unpack_mask(record["parent"])
    if index == 0:
        return parent
    ids = record["features"]["members"][index].nonzero().flatten().tolist()
    polygon = polygon_union(record["polygons"], ids, (parent.shape[1], parent.shape[0]))
    return edited_mask(
        parent, polygon, OPERATIONS[int(record["features"]["operation"][index]) - 1]
    )


class UtilityDataset(Dataset):
    def __init__(self, root, names, training=False):
        self.root, self.names, self.training = Path(root), names, training

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        record = torch.load(
            self.root / f"{Path(self.names[index]).stem}.pt", weights_only=True
        )
        return record["features"], record.get("labels") if self.training else None


def collate_utility(records):
    inputs, labels = zip(*records)
    n = max(1, max(len(x["words"]) for x in inputs))
    k = max(len(x["operation"]) for x in inputs)
    l = max(len(x["question"]) for x in inputs)

    def padded(x, length):
        return F.pad(x, (0, 0) * (x.ndim - 1) + (0, length - len(x)))

    batch = {
        key: torch.stack([padded(x[key], n) for x in inputs])
        for key in ("words", "word_visual", "boxes", "confidence")
    }
    for key in ("operation", "descriptors", "action_visual"):
        batch[key] = torch.stack([padded(x[key], k) for x in inputs])
    batch["members"] = torch.stack(
        [
            F.pad(
                x["members"], (0, n - x["members"].shape[1], 0, k - len(x["members"]))
            )
            for x in inputs
        ]
    )
    batch["valid"] = (
        torch.arange(k)[None]
        < torch.tensor([len(x["operation"]) for x in inputs])[:, None]
    )
    batch["word_valid"] = (
        torch.arange(n)[None] < torch.tensor([len(x["words"]) for x in inputs])[:, None]
    )
    batch["question"] = torch.stack([padded(x["question"], l) for x in inputs])
    batch["question_valid"] = (
        torch.arange(l)[None]
        < torch.tensor([len(x["question"]) for x in inputs])[:, None]
    )
    batch["filename"] = [x["filename"] for x in inputs]
    target = (
        None
        if labels[0] is None
        else {
            "gain": torch.stack([padded(x["gain"], k) for x in labels]),
            "quality": torch.stack([padded(x["quality"], k) for x in labels]),
            "quality_counts": torch.stack(
                [padded(x["quality_counts"], k) for x in labels]
            ),
            "area_fraction": torch.stack([x["area_fraction"] for x in labels]),
        }
    )
    return batch, target


def choose_actions(scores, valid, threshold):
    if threshold is None:
        return torch.zeros(len(scores), device=scores.device, dtype=torch.long)
    values = scores.float().masked_fill(~valid, -torch.inf)
    best = values.argmax(1)
    return torch.where(values.gather(1, best[:, None])[:, 0] > threshold, best, 0)


def binomial_upper(errors, count, delta):
    """One-sided exact Clopper-Pearson upper bound, evaluated in float64/log space."""
    if not 0 < delta < 1 or not 0 <= errors <= count:
        raise ValueError("Invalid binomial arguments")
    if count == 0 or errors == count:
        return 1.0
    if errors == 0:
        return -math.expm1(math.log(delta) / count)

    def cdf(p):
        logs = [
            math.lgamma(count + 1)
            - math.lgamma(i + 1)
            - math.lgamma(count - i + 1)
            + i * math.log(p)
            + (count - i) * math.log1p(-p)
            for i in range(errors + 1)
        ]
        top = max(logs)
        return math.exp(top) * sum(math.exp(x - top) for x in logs)

    left, right = errors / count, 1.0 - 1e-15
    for _ in range(70):
        mid = (left + right) / 2
        if cdf(mid) > delta:
            left = mid
        else:
            right = mid
    return right


def certify(rows, alpha=0.05, delta=0.025):
    """Audit ONE already selected policy; do not search thresholds on these rows."""
    large = [r for r in rows if r["area_fraction"] >= 0.1]

    def harmed(row):
        if "counts" not in row:
            return row["iou"] < row["base_iou"] - 0.05
        i, u = row["counts"]
        bi, bu = row["base_counts"]
        # Strict drop > 1/20. Python integers avoid both rounding and int64 overflow.
        return 20 * (bi * u - i * bu) > u * bu

    errors = sum(harmed(r) for r in large)
    bound = binomial_upper(errors, len(large), delta)
    mean_large = sum(r["iou"] - r["base_iou"] for r in large) / max(1, len(large))
    mean_all = sum(r["iou"] - r["base_iou"] for r in rows) / max(1, len(rows))
    return {
        "accepted": bool(
            large and bound <= alpha and mean_large >= 0 and mean_all >= 0
        ),
        "n_large": len(large),
        "large_harms_gt5pp": errors,
        "harm_rate_upper": bound,
        "alpha": alpha,
        "delta": delta,
        "mean_large_delta": mean_large,
        "mean_delta": mean_all,
        "risk": "P(IoU loss > 0.05 | GT area >= 0.10), across all large images, not only switches",
        "assumptions": "Fixed policy independent of safety labels; IID images within group; distribution shift invalidates transfer of the bound",
    }
