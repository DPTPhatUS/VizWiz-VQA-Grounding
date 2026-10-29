"""Selection, evidence-only segmentation, scope and optional leave-one-out utility."""

import torch
from torch.nn import functional as F


def balanced_mean(values, positive, valid):
    """Equal weight to present positive/negative groups, then equal weight to images."""
    pieces, counts = [], []
    for group in (positive & valid, ~positive & valid):
        count = group.sum(1)
        pieces.append((values * group).sum(1) / count.clamp_min(1))
        counts.append(count > 0)
    return (
        (pieces[0] + pieces[1]) / (counts[0].float() + counts[1]).clamp_min(1)
    ).mean()


def intervention_loss(output, target, valid):
    """Exact leave-one-slot-out expectation; no extra CNN/attention forward passes.

    Supervise contribution only where a positive polygon owns unique GT mass.
    Duplicated/overlapping positives are not incorrectly asked to be indispensable.
    Background evidence must not contribute foreground in its background pixels.
    """
    with torch.autocast(output["evidence"].device.type, enabled=False):
        fields = output["fields"].float()
        full = output["evidence"].float()
        # Stable complement from all OTHER slots plus background (rather than 1-w).
        logits = output["scores"].float()[..., None, None].expand_as(fields)
        logits = logits.masked_fill(~output["support"], -1e4)

        def exclude_each(values):
            left = F.pad(values.logcumsumexp(1)[:, :-1], (0, 0, 0, 0, 1, 0), value=-1e4)
            right = F.pad(
                values.flip(1).logcumsumexp(1).flip(1)[:, 1:],
                (0, 0, 0, 0, 0, 1),
                value=-1e4,
            )
            return torch.logaddexp(left, right)

        # O(B*N*H*W), no N separate forwards or quadratic N-by-N intermediates.
        denominator = torch.logaddexp(exclude_each(logits), torch.zeros_like(logits))
        numerator = exclude_each(logits + fields.clamp_min(1e-30).log())
        other_fg = (numerator - denominator).exp()
        contribution = full - other_fg
        polygons = target["polygons"].float()
        positive = valid & (target["purity"] >= 0.5)
        support = polygons * positive[..., None, None]
        # Soft area coverage; tolerate only rounding in otherwise unique support.
        unique = (
            (support.sum(1, keepdim=True) - support <= 1e-6) * support * target["truth"]
        )
        unique = unique * target["scope"][:, None, None, None]
        bg = polygons * (1 - target["truth"]) * (~positive & valid)[..., None, None]
        pos = ((1 - contribution).square() * unique).sum((2, 3)) / unique.sum(
            (2, 3)
        ).clamp_min(1e-6)
        neg = (F.relu(contribution).square() * bg).sum((2, 3)) / bg.sum(
            (2, 3)
        ).clamp_min(1e-6)
        active = (unique.sum((2, 3)) + bg.sum((2, 3))) > 0
        return ((pos + neg).sum(1) / active.sum(1).clamp_min(1)).mean()


def transport_objective(output, target, batch, recipe):
    with torch.autocast(output["evidence"].device.type, enabled=False):
        valid = batch["valid"]
        purity = target["purity"].float()
        selection = balanced_mean(
            F.binary_cross_entropy_with_logits(
                output["scores"].float(), purity, reduction="none"
            ),
            purity >= 0.5,
            valid,
        )
        scope_raw = F.binary_cross_entropy_with_logits(
            output["scope"].float(), target["scope"].float(), reduction="none"
        )
        # Missing-OCR scope is deterministically zero; keep the graph without training a spurious target.
        scope = (scope_raw * valid.any(1)).sum() / valid.any(1).sum().clamp_min(1)
        local = F.binary_cross_entropy_with_logits(
            output["patches"].float(), target["local"].float(), reduction="none"
        ).mean((2, 3, 4))
        local = balanced_mean(local, purity >= 0.5, valid)
        prediction = output["evidence"].float().clamp(1e-6, 1 - 1e-6)
        truth = target["truth"].float()
        bce = F.binary_cross_entropy(prediction, truth, reduction="none").mean(
            (1, 2, 3)
        )
        dice = 1 - (
            2 * (prediction * truth).sum((1, 2, 3), dtype=torch.float64) + 1e-6
        ) / (
            prediction.sum((1, 2, 3), dtype=torch.float64)
            + truth.sum((1, 2, 3), dtype=torch.float64)
            + 1e-6
        )
        evidence = ((bce + dice) * target["scope"]).sum() / target[
            "scope"
        ].sum().clamp_min(1)
        counterfactual = (
            intervention_loss(output, target, valid)
            if recipe == "intervention"
            else output["scores"].float().sum() * 0
        )
        loss = selection + scope + local + evidence + 0.5 * counterfactual
        return loss, {
            k: v.detach()
            for k, v in {
                "selection": selection,
                "scope": scope,
                "local": local,
                "evidence": evidence,
                "intervention": counterfactual,
            }.items()
        }
