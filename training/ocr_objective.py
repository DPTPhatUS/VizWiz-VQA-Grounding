"""Evidence relevance and segmentation losses; all GT access stays outside the model."""

import torch

from losses import regional_loss_per_image


class OCREvidenceObjective:
    def __init__(self, selection_weight=0.1, local_weight=0.1):
        self.selection_weight, self.local_weight = selection_weight, local_weight

    def __call__(self, output, batch):
        main = regional_loss_per_image(
            output["logits"], batch["mask"], overlap_loss="dice"
        )
        valid = output["candidate_valid"]
        quality = batch["ocr_quality"].float() * valid
        useful = quality.amax(1) >= 0.1
        # Soft multi-region target; no hard-coded answer vocabulary or GT-size gate.
        relevance = quality / quality.sum(1, keepdim=True).clamp_min(1e-8)
        target = torch.cat(
            (relevance * useful[:, None], (~useful)[:, None].float()), -1
        )
        selection = -(target * output["candidate_logits"].float().log_softmax(-1)).sum(
            1
        )
        local = regional_loss_per_image(
            output["region_logits"].flatten(0, 1),
            batch["ocr_patch_target"].flatten(0, 1),
            overlap_loss="dice",
        ).reshape_as(quality)
        # Supervise useful local regions, not unrelated candidates on non-text images.
        local = (local * quality).sum(1) / quality.sum(1).clamp_min(1e-8)
        loss = main + self.selection_weight * selection + self.local_weight * local
        return loss.mean(), {
            "segmentation": main.mean().detach(),
            "selection": selection.mean().detach(),
            "local": local.mean().detach(),
            "null_probability": output["null_probability"].mean().detach(),
            "ocr_available": valid.any(1).float().mean().detach(),
        }
