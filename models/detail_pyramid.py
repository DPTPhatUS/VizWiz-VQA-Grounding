"""Small U-Net-style encoder and top-down detail pyramid on high-resolution RGB."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.mask_decoder import conv_block


class DetailPyramid(nn.Module):
    """Return features ordered from the coarsest grid to full image resolution."""

    def __init__(self, *, tiny: bool = False):
        super().__init__()
        channels = (4, 8, 16, 32, 64) if tiny else (8, 16, 32, 64, 128)
        width = 8 if tiny else 16
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(conv_block(a, b), conv_block(b, b))
                for a, b in zip((3, *channels[:-1]), channels)
            ]
        )
        self.lateral = nn.ModuleList([nn.Conv2d(c, width, 1) for c in channels])
        self.smooth = nn.ModuleList([conv_block(width, width) for _ in channels])

    def forward(self, image: Tensor, run_block) -> tuple[Tensor, ...]:
        levels = []
        x = image
        for index, block in enumerate(self.encoder):
            if index:
                x = F.max_pool2d(x, 2)
            x = run_block(block, x)
            levels.append(x)
        pyramid = []
        top = None
        for feature, lateral, smooth in zip(
            reversed(levels), reversed(self.lateral), reversed(self.smooth)
        ):
            x = lateral(feature)
            if top is not None:
                x = x + F.interpolate(
                    top, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            top = run_block(smooth, x)
            pyramid.append(top)
        return tuple(pyramid)


class HighResolutionRefiner(nn.Module):
    """Predict signed logit corrections everywhere, including outside the coarse mask."""

    def __init__(self, semantic_channels: int, detail_dim: int, *, tiny: bool = False):
        super().__init__()
        semantic_dim, hidden = (4, 8) if tiny else (8, 16)
        self.semantic = nn.Conv2d(semantic_channels, semantic_dim, 1)
        self.fusion = conv_block(1 + semantic_dim + 2 * detail_dim, hidden)
        self.correction = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(
        self, logits: Tensor, semantic: Tensor, detail: Tensor, question: Tensor
    ) -> Tensor:
        size = detail.shape[-2:]
        up = F.interpolate(
            logits.float(), size=size, mode="bilinear", align_corners=False
        )
        meaning = F.interpolate(
            self.semantic(semantic), size=size, mode="bilinear", align_corners=False
        )
        qmap = question[:, :, None, None].expand(-1, -1, *size)
        feature = torch.cat((up.to(detail.dtype), meaning, detail, qmap), dim=1)
        delta = self.correction(self.fusion(feature))
        return up + delta.float()
