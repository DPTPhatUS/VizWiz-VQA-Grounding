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


def balanced_weights(values, target):
    """Each nonempty foreground/background region gets equal mass per sample."""
    values = values.detach().float().clamp_min(0)
    result = torch.zeros_like(values)
    for region in (target > .5, target <= .5):
        masked = values * region
        mass = masked.flatten(1).sum(1).view(-1,1,1,1)
        result = result + masked / mass.clamp_min(1e-8)
    mass = result.flatten(1).sum(1).view(-1,1,1,1)
    return result / mass.clamp_min(1e-8)


def support_envelope(target):
    """Bounding rectangle of each annotated region; empty masks stay empty."""
    foreground = target > .5
    rows = foreground.any(dim=3, keepdim=True)
    columns = foreground.any(dim=2, keepdim=True)
    row_span = rows.cumsum(2).bool() & rows.flip([2]).cumsum(2).flip([2]).bool()
    col_span = columns.cumsum(3).bool() & columns.flip([3]).cumsum(3).flip([3]).bool()
    return (row_span & col_span).to(target.dtype)


def area_loss(logits, target):
    probability = resize_logits(logits, target).float().sigmoid()
    return F.mse_loss(probability.flatten(1).mean(1), target.float().flatten(1).mean(1))


def paired_losses(first, second, mask1, mask2, same):
    if same.any() and not torch.equal(mask1[same], mask2[same]):
        raise ValueError('Verified same-region pairs must have identical masks')
    probability1 = resize_logits(first, mask1).float().sigmoid()
    probability2 = resize_logits(second, mask2).float().sigmoid()
    difference = probability2 - probability1
    delta = F.mse_loss(difference, mask2.float() - mask1.float())
    consistency = difference[same].square().mean() if same.any() else difference.sum() * 0
    return delta, consistency


class ExtentObjective:
    def __init__(self, dice_weight=0., support_weight=.2, area_weight=.1,
                 pair_delta_weight=0., pair_consistency_weight=0.):
        self.dice_weight = dice_weight
        self.support_weight = support_weight
        self.area_weight = area_weight
        self.pair_delta_weight = pair_delta_weight
        self.pair_consistency_weight = pair_consistency_weight

    def __call__(self, model, batch):
        n = batch['image'].shape[0]
        questions = list(batch.get('question_text', batch['text']))
        image, target = batch['image'], batch['mask']
        available = batch.get('pair_available', torch.zeros(n, dtype=torch.bool, device=image.device))
        paired_indices = available.nonzero(as_tuple=True)[0]
        if paired_indices.numel():
            questions += [batch['paired_text'][i] for i in paired_indices.tolist()]
            image = torch.cat([image, image[available]])
            target = torch.cat([target, batch['paired_mask'][available]])
        # A single forward is important for DDP reduction and both labels are supervised.
        output = model({'image': image, 'text': questions, 'question_text': questions})
        # Sum each original view and its optional counterpart, normalized by
        # original images. Equal-sized DDP shards then average identically even
        # when they contain different numbers of annotated pairs.
        view_weight = target.shape[0] / n
        segmentation = segmentation_loss(output['logits'], target, self.dice_weight) * view_weight
        support = segmentation_loss(output['location_logits'], support_envelope(target)) * view_weight
        area = area_loss(output['logits'], target) * view_weight
        delta = consistency = output['logits'].sum() * 0
        if paired_indices.numel():
            delta, consistency = paired_losses(output['logits'][:n][available], output['logits'][n:],
                batch['mask'][available], batch['paired_mask'][available], batch['pair_same'][available])
            delta = delta * (paired_indices.numel() / n)
            consistency = consistency * (batch['pair_same'][available].sum() / n)
        loss = (segmentation + self.support_weight * support + self.area_weight * area
                + self.pair_delta_weight * delta + self.pair_consistency_weight * consistency)
        metrics = {'segmentation': segmentation, 'support': support, 'area': area,
                   'pair_delta': delta, 'pair_consistency': consistency}
        return loss, {key: value.detach() for key, value in metrics.items()}
