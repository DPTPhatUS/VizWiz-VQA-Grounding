import torch
from torch.nn import functional as F


def resize_logits(logits, target):
    return F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)


def segmentation_loss(logits, target, dice_weight=0.0):
    logits = resize_logits(logits, target).float()
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    axes = tuple(range(1, target.ndim))
    dice = 1 - (2*(probability*target).sum(axes)+1) / (
        probability.sum(axes)+target.sum(axes)+1)
    return bce + dice_weight*dice.mean()


def per_image_iou(logits, target):
    prediction = resize_logits(logits, target) > 0
    truth = target > .5
    axes = tuple(range(1,target.ndim))
    intersection = (prediction & truth).sum(axes).float()
    union = (prediction | truth).sum(axes).float()
    return torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union))


class SupervisedObjective:
    def __init__(self, dice_weight=0.):
        self.dice_weight = dice_weight
    def __call__(self, model, batch):
        output = model(batch)
        loss = segmentation_loss(output["logits"], batch["mask"], self.dice_weight)
        return loss, {"segmentation": loss.detach()}
