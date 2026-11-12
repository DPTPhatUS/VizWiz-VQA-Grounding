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


def edge_occupancies(output, mask):
    """Receiver and sender occupancies on the local refinement grid."""
    shape = output['membership_logits'].shape[-2:]
    occupancy = F.adaptive_avg_pool2d(mask.float(), shape).flatten(1)
    receiver = occupancy[:, :, None]
    sender = occupancy[:, output['edge_indices']]
    return receiver, sender


def membership_targets(output, mask):
    """Three-role soft targets, or source-weighted masses for two-role CM.

    Two-role masses sum to source occupancy y_j. Equivalently the conditional
    targets are (y_i, 1-y_i), weighted by y_j. Background sources have no auxiliary
    role target, but all valid edges still participate in the forward update.
    """
    receiver, sender = edge_occupancies(output, mask)
    targets = torch.stack((receiver * sender, (1 - receiver) * sender, 1 - sender), -1)
    roles = output['edge_logits'].shape[-1]
    if roles not in (2, 3):
        raise ValueError('Membership supervision requires two or three roles')
    targets = targets[..., :roles]
    if output.get('symmetric_targets', False):
        # Reciprocal averaging preserves total class mass on the valid graph.
        # A mixed binary pair has (include, exclude, neutral) = (0, .5, .5)
        # in both directions, rather than incompatible exclude/neutral labels.
        from models.connections import symmetric_logits
        targets = symmetric_logits(targets, output['edge_indices'])
    targets = targets * output['edge_valid'][None, :, :, None]
    return targets


def balanced_soft_loss(targets, log_probability):
    """Average loss over supported target masses per image, then over images."""
    mass = targets.sum((1, 2))
    present = mass > 0
    class_losses = -(targets * log_probability).sum((1, 2)) / mass.clamp_min(1e-8)
    return ((class_losses * present).sum(-1) / present.sum(-1).clamp_min(1)).mean()


def membership_loss(output, mask):
    """Per-image class-balanced soft labels for directed foreground-source edges."""
    targets = membership_targets(output, mask)
    log_probability = output['edge_logits'].float().log_softmax(-1)
    return balanced_soft_loss(targets, log_probability)


def affinity_target_masses(output, mask):
    """Symmetric foreground agreement, background agreement, and disagreement.

    The first two masses sum to the same-class target
    y_i*y_j + (1-y_i)*(1-y_j); the three masses sum to one on valid edges.
    """
    receiver, sender = edge_occupancies(output, mask)
    targets = torch.stack((receiver * sender, (1 - receiver) * (1 - sender),
                           receiver * (1 - sender) + (1 - receiver) * sender), -1)
    return targets * output['edge_valid'][None, :, :, None]


def affinity_loss(output, mask):
    """Balance the three edge strata so background agreements cannot dominate."""
    logits = output['affinity_logits'].float()
    agreement = F.logsigmoid(logits)
    log_probability = torch.stack((agreement, agreement, F.logsigmoid(-logits)), -1)
    return balanced_soft_loss(affinity_target_masses(output, mask), log_probability)


class SupervisedObjective:
    def __init__(self, edge_loss_weight=.1):
        self.edge_loss_weight = edge_loss_weight
    def __call__(self, model, batch):
        output = model(batch)
        segmentation = segmentation_loss(output['logits'], batch['mask'])
        relation_name = 'affinity' if 'affinity_logits' in output else 'membership'
        if relation_name == 'affinity':
            edges = affinity_loss(output, batch['mask'])
        elif 'edge_logits' in output:
            edges = membership_loss(output, batch['mask'])
        else:
            edges = segmentation.new_zeros(())
            if self.edge_loss_weight:
                raise ValueError('Positive relation loss requires local relation outputs')
        return segmentation + self.edge_loss_weight * edges, {
            'segmentation': segmentation.detach(), relation_name: edges.detach(),
        }
