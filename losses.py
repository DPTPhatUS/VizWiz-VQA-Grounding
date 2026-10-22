import torch
from torch.nn import functional as F


def resize_logits(logits, target):
    if logits.shape[-2:] == target.shape[-2:]:
        return logits
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


def membership_loss(output, mask):
    """Per-image class-balanced soft labels for directed foreground-source edges."""
    shape = output['membership_logits'].shape[-2:]
    occupancy = F.adaptive_avg_pool2d(mask.float(), shape).flatten(1)
    receiver = occupancy[:, :, None]
    sender = occupancy[:, output['edge_indices']]
    targets = torch.stack((receiver * sender, (1 - receiver) * sender, 1 - sender), -1)
    targets = targets * output['edge_valid'][None, :, :, None]
    log_probability = output['edge_logits'].float().log_softmax(-1)
    mass = targets.sum((1, 2))
    present = mass > 0
    class_losses = -(targets * log_probability).sum((1, 2)) / mass.clamp_min(1e-8)
    return ((class_losses * present).sum(-1) / present.sum(-1).clamp_min(1)).mean()


class SupervisedObjective:
    def __init__(self, edge_loss_weight=.1):
        self.edge_loss_weight = edge_loss_weight
    def __call__(self, model, batch):
        output = model(batch)
        segmentation = segmentation_loss(output['logits'], batch['mask'])
        edges = membership_loss(output, batch['mask'])
        return segmentation + self.edge_loss_weight * edges, {
            'segmentation': segmentation.detach(), 'membership': edges.detach(),
        }
