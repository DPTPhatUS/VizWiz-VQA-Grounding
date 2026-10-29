"""Direct native-IoU advantage and asymmetric downside-sensitive decision losses."""

import torch
from torch.nn import functional as F


def utility_objective(scores, target, valid, recipe):
    if recipe not in ("gain", "risk"):
        raise ValueError("Unknown utility recipe")
    with torch.autocast(scores.device.type, enabled=False):
        gain = target["gain"].float()
        large = target["area_fraction"] >= 0.1
        utility = (
            gain
            if recipe == "gain"
            else gain - 3 * large[:, None] * (-gain).clamp_min(0)
        )
        # Regression makes the units of the decision threshold explicit: IoU advantage.
        regression = F.smooth_l1_loss(
            scores.float() / 0.1, utility / 0.1, reduction="none"
        )
        edit_valid = valid.clone()
        edit_valid[:, 0] = False
        regression = (regression * edit_valid).sum(1) / edit_valid.sum(1).clamp_min(1)
        probability = (scores.float() / 0.1).masked_fill(~valid, -torch.inf).softmax(1)
        best = utility.masked_fill(~valid, -torch.inf).max(1).values
        regret = (probability * (best[:, None] - utility)).sum(1) / 0.1
        weight = (
            torch.where(large, 1.0, 2.0) if recipe == "risk" else torch.ones_like(best)
        )
        loss = ((regression + regret) * weight).sum() / weight.sum()
        return loss, {
            "regression": regression.mean().detach(),
            "regret": regret.mean().detach(),
        }
