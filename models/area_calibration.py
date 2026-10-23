"""Backbone-independent separation of spatial scores and foreground mass."""

import torch
from torch import nn
from torch.nn import functional as F


class _MassProjection(torch.autograd.Function):
    """Solve mean(sigmoid(scores + bias)) = area; differentiate implicitly.

    Saves only the sigmoid slope, not an unrolled bisection graph. The output
    preserves all within-image score ordering. It constrains soft mass, not the
    number of thresholded foreground pixels.
    """

    @staticmethod
    def forward(ctx, scores, area):
        axes = tuple(range(1, scores.ndim))
        shape = (-1,) + (1,) * (scores.ndim - 1)
        prior = torch.logit(area).reshape(shape)
        lo = prior - scores.amax(axes, keepdim=True)
        hi = prior - scores.amin(axes, keepdim=True)
        for _ in range(40 if scores.dtype == torch.float64 else 26):
            middle = (lo + hi) * 0.5
            mass = (scores + middle).sigmoid().mean(axes, keepdim=True)
            lower = mass < area.reshape(shape)
            lo, hi = torch.where(lower, middle, lo), torch.where(lower, hi, middle)
        result = scores + (lo + hi) * 0.5
        probability = result.sigmoid()
        ctx.save_for_backward(probability * (1 - probability))
        ctx.area_shape = area.shape
        return result

    @staticmethod
    def backward(ctx, grad):
        (slope,) = ctx.saved_tensors
        axes = tuple(range(1, grad.ndim))
        total_slope = slope.sum(axes, keepdim=True).clamp_min(1e-12)
        scale = grad.sum(axes, keepdim=True) / total_slope
        score_grad = grad - scale * slope
        area_grad = scale.flatten(1).squeeze(1) * slope[0].numel()
        return score_grad, area_grad.reshape(ctx.area_shape)


def project_foreground_mass(scores, area):
    """FP32 solve under AMP; double precision retained for numerical gradchecks."""
    if scores.ndim != 4 or scores.shape[1] != 1 or area.shape != (scores.shape[0],):
        raise ValueError("Expected Bx1xHxW scores and B foreground fractions")
    dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
    return _MassProjection.apply(scores.to(dtype), area.to(dtype).clamp(1e-6, 1 - 1e-6))


class AreaCalibration(nn.Module):
    """Predict image-specific foreground mass from any decoder feature + Q.

    Constructor dimensions are the only backbone adapters. No CLIP/ViT names,
    patch counts, square-grid assumptions, or fixed input resolution occur here.
    """

    def __init__(self, feature_channels, question_dim, *, width=16):
        super().__init__()
        self.visual = nn.Sequential(nn.Conv2d(feature_channels, width, 1), nn.GELU())
        self.question = nn.Sequential(nn.Linear(question_dim, width), nn.GELU())
        self.area = nn.Sequential(
            nn.Linear(width * 17, width * 4), nn.GELU(), nn.Linear(width * 4, 1)
        )
        nn.init.zeros_(self.area[-1].weight)
        nn.init.zeros_(self.area[-1].bias)
        self.register_buffer("format_version", torch.tensor(1))

    def forward(self, logits, feature, question):
        # Predict extent independently: do not let near-perfect train masks
        # teach a trivial identity correction that cannot transfer to validation.
        visual = F.adaptive_avg_pool2d(self.visual(feature), (4, 4)).flatten(1)
        descriptor = torch.cat((visual, self.question(question)), 1)
        area_logit = self.area(descriptor).float().squeeze(1)
        predicted_area = area_logit.sigmoid().clamp(1e-6, 1 - 1e-6)
        calibrated = project_foreground_mass(logits, predicted_area)
        return {
            "logits": calibrated,
            "predicted_area": predicted_area,
            "area_logit": area_logit,
            "coarse_logits": logits,
        }
