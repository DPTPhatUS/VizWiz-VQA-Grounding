"""Train evidence independently of the frozen parent's errors; supervise the route from GT geometry."""

from torch.nn import functional as F

from losses import regional_loss_per_image


class RoutingObjective:
    def __init__(
        self,
        selection_weight=0.1,
        local_weight=0.1,
        route_weight=0.2,
        evidence_weight=0.5,
    ):
        self.selection_weight, self.local_weight = selection_weight, local_weight
        self.route_weight, self.evidence_weight = route_weight, evidence_weight

    def __call__(self, output, batch):
        valid = output["candidate_valid"].float()
        n = valid.sum(1).clamp_min(1)
        route_target = batch["ocr_route_target"].float() * valid.any(1)
        main = regional_loss_per_image(
            output["logits"], batch["mask"], overlap_loss="dice"
        )
        selection = F.binary_cross_entropy_with_logits(
            output["candidate_logits"].float(),
            batch["ocr_inclusion_target"].float(),
            reduction="none",
        )
        selection = (selection * valid).sum(1) / n
        local = regional_loss_per_image(
            output["region_logits"].flatten(0, 1),
            batch["ocr_patch_target"].flatten(0, 1),
            overlap_loss="dice",
        ).reshape_as(valid)
        local = (local * valid).sum(1) / n
        route = F.binary_cross_entropy_with_logits(
            output["route_logits"], route_target, reduction="none"
        )
        evidence = regional_loss_per_image(
            output["evidence_logits"], batch["mask"], overlap_loss="dice"
        )
        # Full GT mask, including errors OUTSIDE the inherited base foreground.
        evidence = (evidence * route_target).mean()
        loss = (
            main
            + self.selection_weight * selection
            + self.local_weight * local
            + self.route_weight * route
        ).mean()
        loss = loss + self.evidence_weight * evidence
        return loss, {
            "segmentation": main.mean().detach(),
            "selection": selection.mean().detach(),
            "local": local.mean().detach(),
            "routing": route.mean().detach(),
            "evidence": evidence.detach(),
            "route_probability": output["route_probability"].mean().detach(),
            "route_target_rate": route_target.mean().detach(),
            "ocr_available": valid.any(1).float().mean().detach(),
        }
