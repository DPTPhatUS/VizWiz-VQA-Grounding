"""Text-conditioned compact mask decoding."""
import torch
from torch import nn
from torch.nn import functional as F
from models.text_encoder import masked_mean


class SkipConditioner(nn.Module):
    def __init__(self, text_dim, channels, reduction=64):
        super().__init__()
        self.qa_gate = nn.Linear(2 * text_dim, text_dim)
        self.reduction = nn.Linear(text_dim, reduction)
        self.affine = nn.ModuleList([nn.Linear(reduction, 2 * channels) for _ in range(3)])
        for head in self.affine:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, skips, text, answer_vec=None):
        q = masked_mean(text.tokens, text.valid_mask) if hasattr(text, "tokens") else text  # (B, D_t)
        a = q if answer_vec is None else answer_vec.to(q.dtype)  # (B, D_t)
        gate = torch.sigmoid(self.qa_gate(torch.cat([q, a], dim=-1)))  # (B, D_t)
        blended = gate * q + (1 - gate) * a  # (B, D_t)
        condition = F.gelu(self.reduction(blended))  # (B, R)
        output = []
        for feature, head in zip(skips, self.affine):
            gamma, beta = head(condition).chunk(2, dim=-1)  # (B, C), (B, C)
            gamma = gamma.to(feature.dtype)
            beta = beta.to(feature.dtype)
            output.append(feature * (1 + gamma[:, :, None, None]) + beta[:, :, None, None])  # (B, C, H_p, W_p)
        return output


def separable(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.GroupNorm(4, out_channels), nn.GELU(),
    )


class CompactDecoder(nn.Module):
    def __init__(self, in_channels=1024, text_dim=768, width=128):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, width, 1) for _ in range(4)])
        self.conditioner = SkipConditioner(text_dim, width)
        self.fusion = nn.ModuleList([separable(width, width) for _ in range(3)])
        self.upsample = nn.ModuleList([
            separable(width, 64), separable(64, 32),
            separable(32, 16), separable(16, 16),
        ])
        self.output = nn.Conv2d(16, 1, 1)

    def project_skips(self, skip3, skip2, skip1):
        return [proj(feat) for proj, feat in zip(self.projections[1:], (skip3, skip2, skip1))]  # 3 x (B, C, H_p, W_p)

    def forward(self, bottleneck, skip3, skip2, skip1, text, answer_vec=None, *, projected_skips=None):
        x = self.projections[0](bottleneck)  # (B, C, H_p, W_p)
        skips = projected_skips if projected_skips is not None else self.project_skips(skip3, skip2, skip1)  # 3 x (B, C, H_p, W_p)
        skips = self.conditioner(skips, text, answer_vec=answer_vec)  # 3 x (B, C, H_p, W_p)
        for block, skip in zip(self.fusion, skips):
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = x + block(x + skip)  # (B, C, H_p, W_p)
        for block in self.upsample:
            x = block(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))  # 2x spatial per stage
        return self.output(x)  # (B, 1, 16*H_p, 16*W_p)


class UNetDecoder(nn.Module):
    def __init__(self, in_channels=1024, mid_channels=(1024, 512, 256, 128), out_channels=1):
        """
        U-Net based decoder (matched to ImageEncoder output structure)
        in_channels: number of bottleneck and skip channels (e.g. 1024 for ViT-L)
        mid_channels: decoder output channels per stage (1024, 512, 256, 128)
        """
        super().__init__()
        c0, c1, c2, c3 = mid_channels

        # upsample layers
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(in_channels, c0, kernel_size=2, stride=2),
            nn.ConvTranspose2d(c0, c1, kernel_size=2, stride=2),
            nn.ConvTranspose2d(c1, c2, kernel_size=2, stride=2),
            nn.ConvTranspose2d(c2, c3, kernel_size=2, stride=2),
        ])

        # decoder blocks
        self.dec_blocks = nn.ModuleList([
            # stage 1 (c0 + in_channels -> c0)
            nn.Sequential(
                nn.Conv2d(c0 + in_channels, c0, kernel_size=3, padding=1),
                nn.BatchNorm2d(c0),
                nn.ReLU(inplace=True),
                nn.Conv2d(c0, c0, kernel_size=3, padding=1),
                nn.BatchNorm2d(c0),
                nn.ReLU(inplace=True),
            ),
            # stage 2 (c1 + in_channels -> c1)
            nn.Sequential(
                nn.Conv2d(c1 + in_channels, c1, kernel_size=3, padding=1),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c1, c1, kernel_size=3, padding=1),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
            ),
            # stage 3 (c2 + in_channels -> c2)
            nn.Sequential(
                nn.Conv2d(c2 + in_channels, c2, kernel_size=3, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.Conv2d(c2, c2, kernel_size=3, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
            ),
            # stage 4 (no skip connection)
            nn.Sequential(
                nn.Conv2d(c3, c3, kernel_size=3, padding=1),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
                nn.Conv2d(c3, c3, kernel_size=3, padding=1),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
            ),
        ])

        # output layer
        self.final_conv = nn.Conv2d(c3, out_channels, kernel_size=1)

    def forward(self, x, enc_feat3, enc_feat2, enc_feat1):
        # stage 1: x(768) -> 1024 upsample + enc_feat3(1024) concat
        x = self.upconvs[0](x)  # (B, c0, 2*H_p, 2*W_p)
        if x.shape[2:] != enc_feat3.shape[2:]:
            enc_feat3 = F.interpolate(enc_feat3, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, enc_feat3], dim=1)  # (B, c0 + D_v, 2*H_p, 2*W_p)
        x = self.dec_blocks[0](x)  # (B, c0, 2*H_p, 2*W_p)

        # stage 2
        x = self.upconvs[1](x)  # (B, c1, 4*H_p, 4*W_p)
        if x.shape[2:] != enc_feat2.shape[2:]:
            enc_feat2 = F.interpolate(enc_feat2, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, enc_feat2], dim=1)  # (B, c1 + D_v, 4*H_p, 4*W_p)
        x = self.dec_blocks[1](x)  # (B, c1, 4*H_p, 4*W_p)

        # stage 3
        x = self.upconvs[2](x)  # (B, c2, 8*H_p, 8*W_p)
        if x.shape[2:] != enc_feat1.shape[2:]:
            enc_feat1 = F.interpolate(enc_feat1, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, enc_feat1], dim=1)  # (B, c2 + D_v, 8*H_p, 8*W_p)
        x = self.dec_blocks[2](x)  # (B, c2, 8*H_p, 8*W_p)

        # stage 4 (no skip)
        x = self.upconvs[3](x)  # (B, c3, 16*H_p, 16*W_p)
        x = self.dec_blocks[3](x)  # (B, c3, 16*H_p, 16*W_p)

        return self.final_conv(x)  # (B, out_channels, 16*H_p, 16*W_p)
