"""Small RGB detail branch with semantic gating and residual logits."""
import torch
from torch import nn
from torch.nn import functional as F


def separable(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.GroupNorm(4, out_channels), nn.GELU(),
    )


class DetailRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 16), nn.GELU(),
            separable(16, 32, stride=2),
        )
        self.gate = nn.Conv2d(1, 32, 1)
        self.refine = separable(33, 32)
        self.output = nn.Conv2d(32, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, image, logits):
        detail = self.rgb(image)
        coarse = F.interpolate(logits, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        detail = detail * torch.sigmoid(self.gate(coarse))
        delta = self.output(self.refine(torch.cat([detail, coarse], dim=1)))
        delta = F.interpolate(delta, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        return logits + delta
