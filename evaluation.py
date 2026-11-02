"""The one evaluation path, used by eval.py and by validation during training.

Protocol: logits are bilinearly resized to each image's native resolution, thresholded at 0, and
scored by per-image IoU against the native ground-truth mask (two empty masks score 1); the
dataset score is the mean over images. The forward pass runs in FP32.
"""

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Sampler

from metrics import iou_per_sample


class ShardSampler(Sampler[int]):
    """Rank r takes indices r, r + W, r + 2W, ...: every image exactly once across ranks, no padding."""

    def __init__(self, length: int, rank: int = 0, world_size: int = 1):
        self.indices = range(rank, length, world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def native_prediction(logits: torch.Tensor, size) -> torch.Tensor:
    """logits (1, 1, h, w) -> boolean mask (1, 1, H, W) at the native size (H, W)."""
    return F.interpolate(logits.float(), size=tuple(size), mode="bilinear", align_corners=False) > 0


@torch.no_grad()
def evaluate(model, loader, device, on_prediction=None) -> dict[str, float]:
    """Score every image in loader (batches from dataset.collate with native_mask=True).

    on_prediction(filename, mask) is called with each boolean (H, W) prediction, still on device,
    e.g. to save it. Scores stay on the device until the end: one GPU-to-CPU transfer per call.
    Returns {filename: IoU} for this process's images.
    """
    model.eval()
    names, scores = [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)  # (B, 3, S, S)
        logits = model(images, batch["text"])  # (B, 1, h, w)
        for i, (name, truth) in enumerate(zip(batch["filename"], batch["native_mask"], strict=True)):
            truth = truth.to(device, non_blocking=True).unsqueeze(0)  # (1, 1, H, W)
            prediction = native_prediction(logits[i:i + 1], truth.shape[-2:])  # (1, 1, H, W)
            scores.append(iou_per_sample(prediction, truth)[0])
            names.append(name)
            if on_prediction is not None:
                on_prediction(name, prediction[0, 0])
    values = torch.stack(scores).cpu().tolist() if scores else []
    return dict(zip(names, values, strict=True))


def gather_scores(scores: dict[str, float], world_size: int) -> dict[str, float]:
    """Merge the per-rank score dicts on every rank; fail if an image was scored twice."""
    if world_size == 1:
        return scores
    parts = [None] * world_size
    dist.all_gather_object(parts, scores)
    merged = {}
    for part in parts:
        overlap = merged.keys() & part.keys()
        if overlap:
            raise RuntimeError(f"Images scored on more than one rank: {sorted(overlap)[:3]}")
        merged.update(part)
    return merged


def mean_iou(scores: dict[str, float], expected: int) -> float:
    if len(scores) != expected:
        raise RuntimeError(f"Scored {len(scores)} of {expected} images")
    return math.fsum(scores.values()) / len(scores)
