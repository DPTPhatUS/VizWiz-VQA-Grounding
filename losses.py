import math
import torch
from torch.nn import functional as F


def resize_logits(logits, target):
    if logits.shape[-2:] == target.shape[-2:]:
        return logits  # (B, 1, H, W)
    return F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)  # (B, 1, H, W)


def segmentation_loss(logits, target, dice_weight=0.0):
    logits = resize_logits(logits, target).float()  # (B, 1, H, W)
    target = target.float()  # (B, 1, H, W)
    bce = F.binary_cross_entropy_with_logits(logits, target)  # ()
    if not dice_weight:
        return bce
    probability = logits.sigmoid()  # (B, 1, H, W)
    axes = tuple(range(1, target.ndim))
    dice = 1 - (2*(probability*target).sum(axes)+1) / (
        probability.sum(axes)+target.sum(axes)+1)  # (B,)
    return bce + dice_weight*dice.mean()


def per_image_iou(logits, target):
    prediction = resize_logits(logits, target) > 0  # (B, 1, H, W)
    truth = target > .5  # (B, 1, H, W)
    axes = tuple(range(1,target.ndim))
    intersection = (prediction & truth).sum(axes).float()  # (B,)
    union = (prediction | truth).sum(axes).float()  # (B,)
    return torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union))  # (B,)


def balanced_weights(values, target):
    """Balance foreground/background focus while scaling each region by its effective gain."""
    values = values.detach().float().clamp(0, 1)  # (B, 1, H, W)
    result = torch.zeros_like(values)  # (B, 1, H, W)
    regions = (target > .5, target <= .5)  # 2 x (B, 1, H, W)
    nonempty = sum(r.flatten(1).any(1).float() for r in regions).view(-1, 1, 1, 1).clamp_min(1)  # (B, 1, 1, 1)
    for region in regions:
        masked = values * region  # (B, 1, H, W)
        mass = masked.flatten(1).sum(1).view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        dist = masked / mass.clamp_min(1e-8)  # (B, 1, H, W)
        scale = (dist * masked).flatten(1).sum(1).view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        result = result + scale * dist  # (B, 1, H, W)
    return result / nonempty  # (B, 1, H, W)


def distillation_weights(answer_logits, question_logits, target, *, probability=None, entropy=None):
    """Positive answer-induced BCE reduction on teacher-correct pixels, weighted by confidence."""
    with torch.no_grad():
        answer = resize_logits(answer_logits, target).float()  # (B, 1, H, W)
        question = resize_logits(question_logits, target).float()  # (B, 1, H, W)
        target_f = target.float()  # (B, 1, H, W)
        truth = target > .5  # (B, 1, H, W)
        correct = (answer > 0) == truth  # (B, 1, H, W)
        if probability is None:
            probability = answer.sigmoid()  # (B, 1, H, W)
        if entropy is None:
            entropy = F.binary_cross_entropy_with_logits(answer, probability, reduction="none")  # (B, 1, H, W)
        error = F.binary_cross_entropy_with_logits(answer, target_f, reduction="none")  # (B, 1, H, W)
        question_error = F.binary_cross_entropy_with_logits(question, target_f, reduction="none")  # (B, 1, H, W)
        confidence = (1 - entropy / math.log(2)).clamp(0, 1)  # (B, 1, H, W)
        gain = 1 - torch.exp(-(question_error - error).clamp_min(0))  # (B, 1, H, W)
        values = gain * confidence * correct  # (B, 1, H, W)
        return balanced_weights(values, target)  # (B, 1, H, W)


class DistillationObjective:
    def __init__(self, teacher):
        self.teacher = teacher.requires_grad_(False).eval()

    def __call__(self, model, batch):
        output = model(batch)
        target = batch["mask"]  # (B, 1, H, W)
        student = resize_logits(output["logits"], target).float()  # (B, 1, H, W)
        supervised = segmentation_loss(student, target)  # ()
        self.teacher.eval()
        with torch.no_grad():
            features = self.teacher.encode_image(batch["image"])  # 4 x (B, D_v, H_p, W_p)
            answer = resize_logits(self.teacher.decode(features, batch["answer_text"])["logits"], target).float()  # (B, 1, H, W)
            question = self.teacher.decode(features, batch["question_text"])["logits"]  # (B, 1, 16*H_p, 16*W_p)
            probability = answer.sigmoid()  # (B, 1, H, W)
            entropy = F.binary_cross_entropy_with_logits(answer, probability, reduction="none")  # (B, 1, H, W)
            weights = distillation_weights(answer, question, target, probability=probability, entropy=entropy)  # (B, 1, H, W)
            weights *= batch["answer_available"].to(weights).view(-1,1,1,1)  # (B, 1, 1, 1) -> (B, 1, H, W)
            active = weights.flatten(1).sum(1).mean()  # ()
        divergence = F.binary_cross_entropy_with_logits(student, probability, reduction="none") - entropy  # (B, 1, H, W)
        # Batch mean: no-answer/no-gain examples contribute zero KD.
        kd = (weights * divergence).flatten(1).sum(1).mean()  # ()
        return supervised + kd, {"segmentation": supervised.detach(),
                "distillation": kd.detach(), "active_fraction": active}


class SupervisedObjective:
    def __init__(self, dice_weight=0.):
        self.dice_weight = dice_weight
    def __call__(self, model, batch):
        output = model(batch)
        loss = segmentation_loss(output["logits"], batch["mask"], self.dice_weight)  # ()
        return loss, {"segmentation": loss.detach()}
