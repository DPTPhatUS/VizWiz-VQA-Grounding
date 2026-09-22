"""Low-rank text conditioning for grounding features."""
import torch
from torch import nn
from torch.nn import functional as F


def masked_mean(tokens, mask):
    # Accumulate in float32 under mixed precision; empty spans map to zero.
    weights = mask.unsqueeze(-1).to(torch.float32)
    pooled = (tokens.float() * weights).sum(1) / weights.sum(1).clamp_min(1)
    return pooled.to(tokens.dtype)


class TextConditioner(nn.Module):
    def __init__(self, text_dim, separate=False, reduction=64):
        super().__init__()
        self.separate = separate
        if separate:
            self.gate = nn.Sequential(
                nn.Linear(2 * text_dim, reduction), nn.GELU(),
                nn.Linear(reduction, text_dim), nn.Sigmoid(),
            )

    def forward(self, text):
        if not self.separate:
            return masked_mean(text.tokens, text.valid_mask)
        question = masked_mean(text.tokens, text.question_mask)
        answer = masked_mean(text.tokens, text.answer_mask)
        joint = masked_mean(text.tokens, text.valid_mask)
        question = torch.where(text.question_mask.any(1, keepdim=True), question, joint)
        answer = torch.where(text.answer_mask.any(1, keepdim=True), answer, question)
        gate = self.gate(torch.cat([question, answer], dim=-1))
        return gate * question + (1 - gate) * answer


class SkipConditioner(nn.Module):
    def __init__(self, text_dim, channels, reduction=64, separate=False):
        super().__init__()
        self.pool = TextConditioner(text_dim, separate=separate, reduction=reduction)
        self.reduction = nn.Linear(text_dim, reduction)
        self.affine = nn.ModuleList([nn.Linear(reduction, 2 * channels) for _ in range(3)])
        for head in self.affine:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, skips, text):
        condition = F.gelu(self.reduction(self.pool(text)))
        output = []
        for feature, head in zip(skips, self.affine):
            gamma, beta = head(condition).chunk(2, dim=-1)
            output.append(feature * (1 + gamma[:, :, None, None]) + beta[:, :, None, None])
        return output


def separable(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.GroupNorm(4, out_channels), nn.GELU(),
    )


class CompactDecoder(nn.Module):
    def __init__(self, in_channels=1024, text_dim=768, width=128, separate=False):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, width, 1) for _ in range(4)])
        self.conditioner = SkipConditioner(text_dim, width, separate=separate)
        self.fusion = nn.ModuleList([separable(width, width) for _ in range(3)])
        self.upsample = nn.ModuleList([
            separable(width, 64), separable(64, 32),
            separable(32, 16), separable(16, 16),
        ])
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, bottleneck, skip3, skip2, skip1, text):
        features = [projection(feature) for projection, feature in zip(
            self.projections, [bottleneck, skip3, skip2, skip1]
        )]
        x = features[0]
        skips = self.conditioner(features[1:], text)
        for block, skip in zip(self.fusion, skips):
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = x + block(x + skip)
        for block in self.upsample:
            x = block(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return self.output(x)
