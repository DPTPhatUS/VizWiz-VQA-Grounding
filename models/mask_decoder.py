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
        x = self.upconvs[0](x)
        if x.shape[2:] != enc_feat3.shape[2:]:
            enc_feat3 = F.interpolate(
                enc_feat3,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        x = torch.cat([x, enc_feat3], dim=1)
        x = self.dec_blocks[0](x)

        # stage 2
        x = self.upconvs[1](x)
        if x.shape[2:] != enc_feat2.shape[2:]:
            enc_feat2 = F.interpolate(
                enc_feat2,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        x = torch.cat([x, enc_feat2], dim=1)
        x = self.dec_blocks[1](x)

        # stage 3
        x = self.upconvs[2](x)
        if x.shape[2:] != enc_feat1.shape[2:]:
            enc_feat1 = F.interpolate(
                enc_feat1,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        x = torch.cat([x, enc_feat1], dim=1)
        x = self.dec_blocks[2](x)

        # stage 4 (no skip)
        x = self.upconvs[3](x)
        x = self.dec_blocks[3](x)

        return self.final_conv(x)
