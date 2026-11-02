import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        mid_channels=(1024, 512, 256, 128),
        out_channels=1,
    ):
        """
        U-Net based decoder (matched to ImageEncoder output structure)
        in_channels: number of bottleneck and skip channels (e.g. 1024 for ViT-L)
        mid_channels: decoder output channels per stage (1024, 512, 256, 128)
        """
        super().__init__()
        c0, c1, c2, c3 = mid_channels

        # upsample layers
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(in_channels, c0, kernel_size=2, stride=2),
                nn.ConvTranspose2d(c0, c1, kernel_size=2, stride=2),
                nn.ConvTranspose2d(c1, c2, kernel_size=2, stride=2),
                nn.ConvTranspose2d(c2, c3, kernel_size=2, stride=2),
            ]
        )

        # decoder blocks
        self.dec_blocks = nn.ModuleList(
            [
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
            ]
        )

        # output layer
        self.final_conv = nn.Conv2d(c3, out_channels, kernel_size=1)

    def forward(self, x, enc_feat3, enc_feat2, enc_feat1):
        # stage 1
        x = self.upconvs[0](x)  # (B, 1024, 2*H_p, 2*W_p)
        if x.shape[2:] != enc_feat3.shape[2:]:
            enc_feat3 = F.interpolate(
                enc_feat3,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )  # (B, 1024, 2*H_p, 2*W_p)
        x = torch.cat([x, enc_feat3], dim=1)  # (B, 2048, 2*H_p, 2*W_p)
        x = self.dec_blocks[0](x)  # (B, 1024, 2*H_p, 2*W_p)

        # stage 2
        x = self.upconvs[1](x)  # (B, 512, 4*H_p, 4*W_p)
        if x.shape[2:] != enc_feat2.shape[2:]:
            enc_feat2 = F.interpolate(
                enc_feat2,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )  # (B, 1024, 4*H_p, 4*W_p)
        x = torch.cat([x, enc_feat2], dim=1)  # (B, 1536, 4*H_p, 4*W_p)
        x = self.dec_blocks[1](x)  # (B, 512, 4*H_p, 4*W_p)

        # stage 3
        x = self.upconvs[2](x)  # (B, 256, 8*H_p, 8*W_p)
        if x.shape[2:] != enc_feat1.shape[2:]:
            enc_feat1 = F.interpolate(
                enc_feat1,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )  # (B, 1024, 8*H_p, 8*W_p)
        x = torch.cat([x, enc_feat1], dim=1)  # (B, 1280, 8*H_p, 8*W_p)
        x = self.dec_blocks[2](x)  # (B, 256, 8*H_p, 8*W_p)

        # stage 4 (no skip)
        x = self.upconvs[3](x)  # (B, 128, 16*H_p, 16*W_p)
        x = self.dec_blocks[3](x)  # (B, 128, 16*H_p, 16*W_p)

        return self.final_conv(x)  # (B, 1, 16*H_p, 16*W_p)


def _groups(channels: int) -> int:
    """GroupNorm groups: 8 channels per group (a documented choice; GroupNorm does not depend on
    the per-GPU batch, which is 2 images in the reference recipe)."""
    return max(1, channels // 8)


def conv_gn_gelu(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.GroupNorm(_groups(out_channels), out_channels),
        nn.GELU(),
    )


class Bottleneck(nn.Module):
    """Residual branch 1x1 (C -> C/4) -> 3x3 -> 1x1 (C/4 -> C) with GroupNorm after each conv.

    The last GroupNorm starts with zero scale, so the branch outputs exactly zero at initialisation
    and every skip fusion x <- x + branch(x + s) starts as the identity.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = channels // reduction
        self.branch = nn.Sequential(
            conv_gn_gelu(channels, mid, 1),
            conv_gn_gelu(mid, mid, 3),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )
        nn.init.zeros_(self.branch[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class FusionDecoder(nn.Module):
    """B0 decoder (ocrg D017): fuse the ViT skips at the patch grid, then upsample.

    All ViT features share one resolution (24x24 for ViT-L/14-336), so the skips are fused there:
        x = P(F);  x <- x + Bottleneck_j(x + P_j(s_j))  for j = 3, 2, 1
    followed by four stages of bilinear x2 upsampling and a double 3x3 conv block, and a 1x1 head:
        24 -> 48 -> 96 -> 192 -> 384 pixels, width 256 -> 128 -> 64 -> 32 -> 16 -> 1 logit.
    The upsampling stages carry no ViT skips; they are where higher-resolution evidence can enter later.
    """

    def __init__(self, in_channels: int = 1024, width: int = 256, stage_channels=(128, 64, 32, 16)):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, width, 1)
        self.skip_proj = nn.ModuleList(nn.Conv2d(in_channels, width, 1) for _ in range(3))
        self.fusion = nn.ModuleList(Bottleneck(width) for _ in range(3))
        channels = (width, *stage_channels)
        self.stages = nn.ModuleList(
            nn.Sequential(conv_gn_gelu(c_in, c_out, 3), conv_gn_gelu(c_out, c_out, 3))
            for c_in, c_out in zip(channels[:-1], channels[1:], strict=True)
        )
        self.head = nn.Conv2d(channels[-1], 1, 1)

    def forward(self, fused: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor, s3: torch.Tensor) -> torch.Tensor:
        """fused, s1, s2, s3: (B, D_v, H_p, W_p) -> logits (B, 1, 16*H_p, 16*W_p)."""
        x = self.input_proj(fused)  # (B, W, H_p, W_p)
        for proj, block, skip in zip(self.skip_proj, self.fusion, (s3, s2, s1), strict=True):  # deepest first
            x = x + block(x + proj(skip))  # (B, W, H_p, W_p)
        for stage in self.stages:
            x = stage(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))  # 2x per stage
        return self.head(x)  # (B, 1, 16*H_p, 16*W_p)
