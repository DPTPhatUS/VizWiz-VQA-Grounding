"""Benchmark-only adapter of the replicated dense U-Net; no training preset."""
import torch
import torch.nn as nn
import torch.nn.functional as F

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



from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder


class ReplicatedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder=ImageEncoder()
        self.text_encoder=TextEncoder()
        self.text_proj=nn.Linear(768,1024)
        self.cross_attn=nn.MultiheadAttention(1024,8,batch_first=True)
        self.decoder=UNetDecoder()
        self.experiment_config={'experiment':'replicated-unet-benchmark','input_normalization':'unscaled-rgb',
                                'fusion':'unmasked-direct-attention','trainable_text':True}

    def encode_image(self,image):
        return self.image_encoder(image)

    def prepare_decoder(self,features,text):
        s1,s2,s3,vision=features
        tokens=self.text_proj(text.tokens)
        attended,_=self.cross_attn(vision.flatten(2).transpose(1,2),tokens,tokens,need_weights=False)
        return (attended.transpose(1,2).reshape_as(vision),s3,s2,s1)

    def forward(self,batch):
        features=self.encode_image(batch['image']);text=self.text_encoder(batch['text'])
        return {'logits':self.decoder(*self.prepare_decoder(features,text))}
