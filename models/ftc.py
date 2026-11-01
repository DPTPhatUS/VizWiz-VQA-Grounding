"""Exact frozen-text compact control, with an optional convolutional capacity control."""
import torch
from torch import nn
from torch.nn import functional as F
from models.mask_decoder import SkipConditioner, separable
from models.cm import GroundingModel
from models.image_encoder import ImageEncoder, TinyVision
from models.text_encoder import TextEncoder


class CapacityBlock(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(width, hidden, 1, bias=False), nn.GroupNorm(4, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(4, hidden), nn.GELU(),
            nn.Conv2d(hidden, width, 1, bias=False), nn.GroupNorm(4, width))

    def forward(self, x):
        return x + self.layers(x)


class FTCDecoder(nn.Module):
    def __init__(self, in_channels=1024, text_dim=768, width=128, capacity=False):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, width, 1) for _ in range(4)])
        self.conditioner = SkipConditioner(text_dim, width)
        self.fusion = nn.ModuleList([separable(width, width) for _ in range(3)])
        self.upsample = nn.ModuleList([
            separable(width, 64), separable(64, 32),
            separable(32, 16), separable(16, 16),
        ])
        self.output = nn.Conv2d(16, 1, 1)
        self.capacity = nn.ModuleList([CapacityBlock(width, 532 if width == 128 else 32) for _ in range(3)]) if capacity else nn.ModuleList()

    def forward(self, bottleneck, skip3, skip2, skip1, text):
        features = [projection(feature) for projection, feature in zip(
            self.projections, [bottleneck, skip3, skip2, skip1]
        )]
        x = features[0]
        skips = self.conditioner(features[1:], text)
        for block, skip in zip(self.fusion, skips):
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = x + block(x + skip)
        for block in self.capacity:
            x = block(x)
        for block in self.upsample:
            x = block(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return self.output(x)


class FTCModel(GroundingModel):
    def __init__(self, tiny=False, capacity=False, pretrained=True):
        nn.Module.__init__(self)
        self.image_encoder = TinyVision() if tiny else ImageEncoder(pretrained=pretrained)
        self.text_encoder = TextEncoder(tiny, pretrained=pretrained).requires_grad_(False)
        self.visual_dim = self.image_encoder.out_channels
        self.text_dim = self.text_encoder.output_dim
        self.text_proj = nn.Linear(self.text_dim, self.visual_dim)
        self.cross_attn = nn.MultiheadAttention(self.visual_dim, 2 if tiny else 8, batch_first=True)
        self.decoder = FTCDecoder(self.visual_dim, self.text_dim, width=16 if tiny else 128, capacity=capacity)
        self.register_buffer("image_mean", torch.tensor([.48145466, .4578275, .40821073]).view(1,3,1,1))
        self.register_buffer("image_std", torch.tensor([.26862954, .26130258, .27577711]).view(1,3,1,1))
        self.experiment_config = {"experiment": "ftc", "model": "ftc", "architecture": "compact", "tiny": tiny,
                                  "protocol": "normalized-masked-question-v1"}

        if capacity:
            self.experiment_config.update(experiment='ftc-capacity', model='ftc-capacity', capacity_blocks=3,
                                          capacity_hidden=32 if tiny else 532)

    def decode(self, features, texts):
        result = super().decode(features, texts)
        return {'logits': result}
