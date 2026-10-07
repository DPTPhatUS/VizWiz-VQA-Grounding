"""Text-conditioned compact mask decoding."""
import torch
from torch import nn
from torch.nn import functional as F
from models.text_encoder import masked_mean


class SkipConditioner(nn.Module):
    def __init__(self, text_dim, channels, reduction=64):
        super().__init__()
        self.reduction = nn.Linear(text_dim, reduction)
        self.affine = nn.ModuleList([nn.Linear(reduction, 2 * channels) for _ in range(3)])
        for head in self.affine:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, skips, text):
        condition = F.gelu(self.reduction(masked_mean(text.tokens, text.valid_mask)))
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


class UNetDecoder(nn.Module):
    def __init__(self, in_channels=1024, mid_channels=[1024, 512, 256, 128], out_channels=1):
        """
        U-Net based decoder (matched to ImageEncoder output structure)
        in_channels: number of bottleneck channels (e.g. 768)
        mid_channels: decoder output channels per stage [1024, 512, 256, 128]
        """
        super(UNetDecoder, self).__init__()

        # upsample layers
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        ])

        # decoder blocks
        self.dec_blocks = nn.ModuleList([
            # stage 1 (1024 + 1024 = 2048 -> 1024)
            nn.Sequential(
                nn.Conv2d(mid_channels[0] + 1024, mid_channels[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True)
            ),
            # stage 2 (1024 + 512 = 1536 -> 512)
            nn.Sequential(
                nn.Conv2d(1536, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            # stage 3 (512 + 256 + skip 512 = 1280 -> 256)
            nn.Sequential(
                nn.Conv2d(1280, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            # stage 4 (no skip connection)
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
        ])

        # output layer
        self.final_conv = nn.Conv2d(mid_channels[3], out_channels, kernel_size=1)

    def forward(self, x, enc_feat3, enc_feat2, enc_feat1):
        # stage 1: x(768) -> 1024 upsample + enc_feat3(1024) concat
        x = self.upconvs[0](x)
        if x.shape[2:] != enc_feat3.shape[2:]:
            enc_feat3 = F.interpolate(enc_feat3, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, enc_feat3], dim=1)
        x = self.dec_blocks[0](x)

        # stage 2
        x = self.upconvs[1](x)
        if x.shape[2:] != enc_feat2.shape[2:]:
            enc_feat2 = F.interpolate(enc_feat2, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, enc_feat2], dim=1)
        x = self.dec_blocks[1](x)

        # stage 3
        x = self.upconvs[2](x)
        if x.shape[2:] != enc_feat1.shape[2:]:
            enc_feat1 = F.interpolate(enc_feat1, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, enc_feat1], dim=1)
        x = self.dec_blocks[2](x)

        # stage 4 (no skip)
        x = self.upconvs[3](x)
        x = self.dec_blocks[3](x)

        return self.final_conv(x)
