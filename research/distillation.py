"""Training-only privileged answers; student inference is exactly compact CEUD."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from research.model import ResearchGrounder
from research.losses import balanced_weights, resize_logits, segmentation_loss


class AnswerStudent(nn.Module):
    def __init__(self, tiny=False):
        super().__init__()
        self.coarse = ResearchGrounder("compact", tiny)
        self.experiment_config = dict(self.coarse.experiment_config, experiment="answer-value-distillation")

    def forward(self, batch):
        return self.coarse(dict(batch, text=batch.get("question_text", batch["text"])))


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
