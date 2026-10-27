"""Native IoU supervision and within-image quality ranking, independent of routing."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def native_base_quality(logits, truth):
    scores = []
    for i, mask in enumerate(truth):
        resized = (
            F.interpolate(
                logits[i : i + 1].float(),
                mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            > 0
        )
        target = mask[0].to(resized.device, non_blocking=True).bool()
        union = (resized | target).sum()
        scores.append(
            torch.where(union > 0, (resized & target).sum() / union.clamp_min(1), 1.0)
        )
    return torch.stack(scores)


class SelectionObjective:
    def __init__(self, answer_weight=0.0):
        self.answer_weight = answer_weight

    def __call__(self, output, batch):
        logits, valid = output["quality_logits"], output["hypothesis_valid"]
        target = batch["hypothesis_quality"].float().clone()
        target[:, 0] = native_base_quality(
            output["base_logits"], batch["original_mask"]
        )
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        # Balance overlapping and disjoint hypotheses per image, not across ranks.
        positive, negative = valid & (target > 0), valid & (target == 0)
        terms, exists = [], []
        for selected in (positive, negative):
            count = selected.sum(1)
            terms.append((bce * selected).sum(1) / count.clamp_min(1))
            exists.append(count > 0)
        quality = (terms[0] + terms[1]) / (exists[0].float() + exists[1]).clamp_min(1)
        difference = target[:, :, None] - target[:, None, :]
        pairs = valid[:, :, None] & valid[:, None, :] & (difference > 0.05)
        gap = difference.clamp_min(0) * pairs
        ranking = (F.softplus(logits[:, None, :] - logits[:, :, None]) * gap).sum(
            (1, 2)
        ) / gap.sum((1, 2)).clamp_min(1e-6)
        # OCR-vs-OCR pairs are always trained even if the parent wins.
        candidate_valid = output["candidate_valid"]
        answer_target = batch["answer_candidate_target"].float() * candidate_valid
        support = answer_target.sum(1) > 0
        answer_logits = output["candidate_logits"].masked_fill(~candidate_valid, -1e4)
        distribution = answer_target / answer_target.sum(1, keepdim=True).clamp_min(1)
        answer = -(F.log_softmax(answer_logits, -1) * distribution).sum(1) * support
        loss = (quality + ranking + self.answer_weight * answer).mean()
        return loss, {
            "quality": quality.mean().detach(),
            "ranking": ranking.mean().detach(),
            "answer": answer.mean().detach(),
            "answer_support": support.float().mean().detach(),
            "quality_mae": (
                ((logits.sigmoid() - target).abs() * valid).sum(1) / valid.sum(1)
            )
            .mean()
            .detach(),
            "ocr_available": candidate_valid.any(1).float().mean().detach(),
        }
