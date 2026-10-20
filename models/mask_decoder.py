"""Fixed question-conditioned compact decoder with local RGB detail fusion."""

import torch
from torch import nn
from torch.nn import functional as F


class QuestionSkipConditioner(nn.Module):
    def __init__(self, text_dim, channels, reduction=64):
        super().__init__()
        self.reduction = nn.Linear(text_dim, reduction)
        self.affine = nn.ModuleList(
            [nn.Linear(reduction, 2 * channels) for _ in range(3)]
        )
        for head in self.affine:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, skips, question):
        condition = F.gelu(self.reduction(question))
        result = []
        for feature, head in zip(skips, self.affine):
            gamma, beta = head(condition).chunk(2, dim=-1)
            result.append(
                feature * (1 + gamma.to(feature.dtype)[:, :, None, None])
                + beta.to(feature.dtype)[:, :, None, None]
            )
        return result


def separable(in_channels, out_channels):
    groups = 4 if out_channels % 4 == 0 else 1
    return nn.Sequential(
        nn.Conv2d(
            in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False
        ),
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.GELU(),
    )


class PixelDetailStem(nn.Module):
    """Learn local detail from the same normalized RGB input used by CLIP."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )

    def forward(self, image):
        return self.stem(image)


class RegionPixelFuser(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.guide_proj = nn.Conv2d(channels, 1, 1)
        self.guide_scale = nn.Parameter(torch.tensor(1.0))
        self.guide_bias = nn.Parameter(torch.zeros(1))
        self.refine = nn.Sequential(
            nn.Conv2d(
                channels + 16,
                channels + 16,
                3,
                padding=1,
                groups=channels + 16,
                bias=False,
            ),
            nn.Conv2d(channels + 16, channels, 1, bias=False),
            nn.GroupNorm(4, channels),
            nn.GELU(),
        )
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, visual, detail):
        detail = F.interpolate(
            detail, size=visual.shape[-2:], mode="bilinear", align_corners=False
        )
        guide = torch.sigmoid(
            self.guide_scale.to(visual.dtype) * self.guide_proj(visual)
            + self.guide_bias.to(visual.dtype)
        )
        delta = self.refine(torch.cat([visual, detail.to(visual.dtype) * guide], dim=1))
        return visual + self.residual_scale.to(delta.dtype) * delta


class CompactDecoder(nn.Module):
    """Three Q-conditioned skips and detail fusion at upsampling stages 2/3."""

    def __init__(self, in_channels=1024, text_dim=768, width=128):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(in_channels, width, 1) for _ in range(4)]
        )
        self.conditioner = QuestionSkipConditioner(text_dim, width)
        self.fusion = nn.ModuleList([separable(width, width) for _ in range(3)])
        self.upsample = nn.ModuleList(
            [
                separable(width, 64),
                separable(64, 32),
                separable(32, 16),
                separable(16, 16),
            ]
        )
        self.output = nn.Conv2d(16, 1, 1)
        self.detail_stem = PixelDetailStem()
        self.detail_upsample = nn.ModuleDict(
            {str(i): RegionPixelFuser() for i in (2, 3)}
        )

    def forward(self, bottleneck, skip3, skip2, skip1, question, *, detail_features):
        if detail_features.ndim != 4 or detail_features.shape[:2] != (
            bottleneck.shape[0],
            16,
        ):
            raise ValueError("Detail features must have shape (B, 16, H, W)")
        x = self.projections[0](bottleneck)
        skips = [
            projection(feature)
            for projection, feature in zip(self.projections[1:], (skip3, skip2, skip1))
        ]
        for block, skip in zip(self.fusion, self.conditioner(skips, question)):
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(
                    skip, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            x = x + block(x + skip)
        for index, block in enumerate(self.upsample):
            x = block(
                F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            )
            if str(index) in self.detail_upsample:
                x = self.detail_upsample[str(index)](x, detail_features)
        return self.output(x)
