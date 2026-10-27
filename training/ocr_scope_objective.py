"""Quality supervision and decision-focused scope ranking, normalized per image."""

import torch
from torch.nn import functional as F

from training.ocr_selection_objective import native_base_quality

SCOPE_TARGET_VERSION = "independent-native-quality-v1"


def quality_loss(logits, target, valid):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss, count = torch.zeros_like(logits[:, 0]), torch.zeros_like(logits[:, 0])
    for mask in (valid & (target > 0), valid & (target == 0)):
        n = mask.sum(1)
        loss = loss + (bce * mask).sum(1) / n.clamp_min(1)
        count = count + (n > 0)
    return loss / count.clamp_min(1)


def pairwise_loss(logits, target, valid):
    gap = target[:, :, None] - target[:, None, :]
    weights = gap.clamp_min(0) * (gap > 0.05) * valid[:, :, None] * valid[:, None, :]
    return (F.softplus(logits[:, None, :] - logits[:, :, None]) * weights).sum(
        (1, 2)
    ) / weights.sum((1, 2)).clamp_min(1e-6)


def choice_regret_loss(logits, target, valid, chosen, *, minimum_quality=0.0):
    best = target.masked_fill(~valid, -1).argmax(1)
    best_quality = target.gather(1, best[:, None]).squeeze(1)
    chosen_quality = target.gather(1, chosen[:, None]).squeeze(1)
    regret = (best_quality - chosen_quality).clamp_min(0)
    best_score = logits.gather(1, best[:, None]).squeeze(1)
    chosen_score = logits.gather(1, chosen[:, None]).squeeze(1)
    active = (regret > 0.05) & (best_quality >= minimum_quality) & valid.any(1)
    active = active & valid.gather(1, chosen[:, None]).squeeze(1)
    return regret * active * F.softplus(chosen_score - best_score + 0.1)


class ScopeObjective:
    def __init__(self, recipe="ranking"):
        if recipe not in {"ranking", "scope"}:
            raise ValueError("Invalid selector recipe")
        self.recipe = recipe

    def __call__(self, output, batch):
        logits, valid = output["quality_logits"], output["hypothesis_valid"]
        target = batch["hypothesis_quality"].float().clone()
        target[:, 0] = native_base_quality(
            output["base_logits"], batch["original_mask"]
        )
        quality = quality_loss(logits, target, valid)
        ranking = (
            pairwise_loss(logits, target, valid)
            if self.recipe == "ranking"
            else logits.sum(1) * 0
        )
        full = choice_regret_loss(logits, target, valid, output["selected_index"])
        ocr_valid = valid.clone()
        ocr_valid[:, 0] = False
        ocr = choice_regret_loss(
            logits, target, ocr_valid, output["evidence_index"], minimum_quality=0.1
        )
        if self.recipe == "scope":
            loss = (quality + full + 2 * ocr).mean()
        else:
            loss = (quality + ranking).mean()
        return loss, {
            "quality": quality.mean().detach(),
            "ranking": ranking.mean().detach(),
            "decision_regret": full.mean().detach(),
            "ocr_regret": ocr.mean().detach(),
            "quality_mae": (
                ((logits.sigmoid() - target).abs() * valid).sum(1) / valid.sum(1)
            )
            .mean()
            .detach(),
            "ocr_available": output["candidate_valid"].any(1).float().mean().detach(),
        }
