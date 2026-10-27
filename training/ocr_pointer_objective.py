"""Supervise sufficiency and phrase scope without observing the parent's accuracy."""

import torch
from torch.nn import functional as F

SUFFICIENT_IOU = 0.5
SCOPE_TOLERANCE = 0.05
RETRIEVAL_WEIGHT = 0.25
TOKEN_WEIGHT = 0.25


def pointer_targets(quality, valid):
    quality = quality.float().detach().clone()
    quality[:, 0] = 0  # Slot zero is defer, not the parent mask quality.
    ocr_valid = valid.clone()
    ocr_valid[:, 0] = False
    best = quality.masked_fill(~ocr_valid, 0).max(1).values
    positive = (
        ocr_valid
        & (quality >= SUFFICIENT_IOU)
        & (quality >= best[:, None] - SCOPE_TOLERANCE)
    )
    target = positive.float() * quality
    target[:, 0] = (~positive.any(1)).float()
    target = target / target.sum(1, keepdim=True).clamp_min(1e-6)
    retrieval = (
        ocr_valid & (quality >= best[:, None] - SCOPE_TOLERANCE)
    ).float() * quality
    retrieval = retrieval / retrieval.sum(1, keepdim=True).clamp_min(1e-6)
    return target, retrieval, ocr_valid, best


def per_image_pointer_losses(output, batch, recipe):
    logits = output["selection_logits"].float()
    target, retrieval, ocr_valid, best = pointer_targets(
        batch["hypothesis_quality"], output["hypothesis_valid"]
    )
    choice = -(target * logits.log_softmax(1)).sum(1)
    ocr_logits = logits.masked_fill(~ocr_valid, -1e4)
    rank = -(retrieval * ocr_logits.log_softmax(1)).sum(1) * (best >= 0.1)
    # Keep every adapter parameter in DDP even for recipe A or missing OCR.
    token = output["token_logits"].float().sum(1) * 0
    if recipe == "structured":
        purity = batch["ocr_word_purity"].float()
        valid = output["candidate_valid"]
        bce = F.binary_cross_entropy_with_logits(
            output["token_logits"].float(), purity, reduction="none"
        )
        count = torch.zeros_like(token)
        for group in (valid & (purity >= 0.5), valid & (purity < 0.5)):
            number = group.sum(1)
            token = token + (bce * group).sum(1) / number.clamp_min(1)
            count = count + (number > 0)
        token = token / count.clamp_min(1)
    return choice, rank, token, target


class PointerObjective:
    def __init__(self, recipe="categorical"):
        if recipe not in {"categorical", "structured"}:
            raise ValueError("Invalid pointer recipe")
        self.recipe = recipe

    def __call__(self, output, batch):
        choice, rank, token, target = per_image_pointer_losses(
            output, batch, self.recipe
        )
        loss = choice + RETRIEVAL_WEIGHT * rank + TOKEN_WEIGHT * token
        selected = output["selected_index"]
        return loss.mean(), {
            "choice_ce": choice.mean().detach(),
            "retrieval_ce": rank.mean().detach(),
            "token_bce": token.mean().detach(),
            "null_target": target[:, 0].mean().detach(),
            "null_probability": output["null_probability"].mean().detach(),
            "ocr_selected": (selected > 0).float().mean().detach(),
            "ocr_available": output["candidate_valid"].any(1).float().mean().detach(),
        }
