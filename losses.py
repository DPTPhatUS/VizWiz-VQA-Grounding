import math
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


def distillation_weights(answer_logits, question_logits, target, mode):
    """Detached weights balanced over foreground/background in each image.

    Incremental = positive answer-induced BCE reduction times teacher confidence.
    Error = exp(-answer BCE); confidence ignores ground-truth correctness.
    """
    with torch.no_grad():
        answer = resize_logits(answer_logits, target).float()
        probability = answer.sigmoid()
        error = F.binary_cross_entropy_with_logits(answer, target.float(), reduction="none")
        entropy = F.binary_cross_entropy_with_logits(answer, probability, reduction="none")
        confidence = (1 - entropy / math.log(2)).clamp(0, 1)
        if mode == "ordinary":
            values = torch.ones_like(target)
        elif mode == "confidence":
            values = confidence
        elif mode == "error":
            values = (-error).exp()
        elif mode == "incremental":
            question = resize_logits(question_logits, target).float()
            question_error = F.binary_cross_entropy_with_logits(question, target.float(), reduction="none")
            values = (question_error - error).clamp_min(0) * confidence
        else:
            raise ValueError(f"Unknown distillation mode: {mode}")
        return balanced_weights(values, target)


class DistillationObjective:
    def __init__(self, teacher, mode, weight, dice_weight):
        self.teacher, self.mode, self.weight, self.dice_weight = teacher, mode, weight, dice_weight
        if teacher is not None:
            teacher.requires_grad_(False).eval()

    def __call__(self, model, batch):
        output = model(batch)
        target = batch["mask"]
        supervised = segmentation_loss(output["logits"], target, self.dice_weight)
        kd = supervised * 0
        active = supervised.detach() * 0
        if self.teacher is not None and self.mode != "none":
            self.teacher.eval()
            with torch.no_grad():
                features = self.teacher.encode_image(batch["image"])
                answer = self.teacher.decode(features, batch["answer_text"])["logits"]
                question = (self.teacher.decode(features, batch["question_text"])["logits"]
                            if self.mode == "incremental" else answer)
                weights = distillation_weights(answer, question, target, self.mode)
                weights *= batch["answer_available"].to(weights).view(-1,1,1,1)
                answer = resize_logits(answer, target).float()
                probability = answer.sigmoid()
                entropy = F.binary_cross_entropy_with_logits(answer, probability, reduction="none")
                active = (weights.flatten(1).sum(1) > 0).float().mean()
            student = resize_logits(output["logits"], target).float()
            divergence = F.binary_cross_entropy_with_logits(student, probability, reduction="none") - entropy
            # Batch mean: no-answer/no-gain examples contribute zero KD.
            kd = (weights * divergence).flatten(1).sum(1).mean()
        return supervised + self.weight * kd, {"segmentation": supervised.detach(),
                "distillation": kd.detach(), "active_fraction": active}


class SupervisedObjective:
    def __init__(self, dice_weight=0.):
        self.dice_weight = dice_weight
    def __call__(self, model, batch):
        output = model(batch)
        loss = segmentation_loss(output["logits"], batch["mask"], self.dice_weight)
        return loss, {"segmentation": loss.detach()}
