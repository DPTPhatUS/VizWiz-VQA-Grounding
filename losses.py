import torch
from torch.nn import functional as F


def resize_logits(logits, target):
    if logits.shape[-2:] == target.shape[-2:]:
        return logits
    return F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)


def segmentation_loss(logits, target):
    logits = resize_logits(logits, target).float()
    return F.binary_cross_entropy_with_logits(logits, target.float())


def per_image_iou(logits, target):
    prediction = resize_logits(logits, target) > 0
    truth = target > .5
    return iou_from_counts(*intersection_union(prediction, truth))


def intersection_union(prediction, truth):
    """Per-image integer pixel counts for binary prediction and truth masks."""
    axes = tuple(range(1, truth.ndim))
    return (prediction & truth).sum(axes), (prediction | truth).sum(axes)


def iou_from_counts(intersection, union):
    intersection, union = intersection.float(), union.float()
    return torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union))
