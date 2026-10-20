"""Question-conditioned skip compression and four-stage detail-guided decoding."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def conv_block(
    in_channels: int, out_channels: int, *, separable: bool = False
) -> nn.Sequential:
    """Convolution, affine GroupNorm and GELU; no bias before normalization."""
    layers = []
    if separable:
        layers.append(
            nn.Conv2d(
                in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False
            )
        )
        layers.append(nn.Conv2d(in_channels, out_channels, 1, bias=False))
    else:
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False))
    layers.extend((nn.GroupNorm(4, out_channels), nn.GELU()))
    return nn.Sequential(*layers)


class SkipConditioner(nn.Module):
    """Nonlinear skip compression, Q-FiLM, then reduction to decoder width."""

    def __init__(
        self, visual_dim: int, text_dim: int, width: int, hidden: int, reduction: int
    ):
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(visual_dim, hidden, 1, bias=False),
                    nn.GroupNorm(8, hidden),
                    nn.GELU(),
                )
                for _ in range(3)
            ]
        )
        self.question = nn.Sequential(nn.Linear(text_dim, reduction), nn.GELU())
        self.affine = nn.ModuleList(
            [nn.Linear(reduction, 2 * hidden) for _ in range(3)]
        )
        self.outputs = nn.ModuleList([nn.Conv2d(hidden, width, 1) for _ in range(3)])
        for head in self.affine:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, skips: tuple[Tensor, ...], question: Tensor) -> list[Tensor]:
        if len(skips) != 3:
            raise ValueError("Exactly three CLIP skips are required")
        condition = self.question(question)
        result = []
        for skip, project, head, output in zip(
            skips, self.projections, self.affine, self.outputs
        ):
            feature = project(skip)
            gamma, beta = head(condition).chunk(2, dim=-1)
            feature = feature * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
            result.append(output(feature))
        return result


class QuestionDetailFusion(nn.Module):
    """A nonlinear spatial guide combines semantic, local and question features."""

    def __init__(self, channels: int, detail_dim: int, question_dim: int):
        super().__init__()
        self.guide = nn.Sequential(
            nn.Conv2d(channels + detail_dim + question_dim, detail_dim, 1),
            nn.GELU(),
            nn.Conv2d(detail_dim, 1, 1),
            nn.Sigmoid(),
        )
        self.refine = conv_block(channels + detail_dim, channels, separable=True)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, visual: Tensor, detail: Tensor, question: Tensor) -> Tensor:
        if detail.shape[-2:] != visual.shape[-2:]:
            raise ValueError(
                "Detail pyramid and decoder stage must have matching grids"
            )
        qmap = question[:, :, None, None].expand(-1, -1, *visual.shape[-2:])
        guide = self.guide(torch.cat((visual, detail, qmap), dim=1))
        delta = self.refine(torch.cat((visual, detail * guide), dim=1))
        return visual + self.residual_scale * delta


class DetailDecoder(nn.Module):
    """Fuse three CLIP skips, then upsample through four detail-supported stages."""

    def __init__(self, visual_dim: int, text_dim: int, *, tiny: bool = False):
        super().__init__()
        width, hidden, reduction = (16, 32, 16) if tiny else (128, 256, 64)
        detail_dim = 8 if tiny else 16
        channels = (8, 8, 4, 4) if tiny else (64, 32, 16, 16)
        self.bottleneck = nn.Conv2d(visual_dim, width, 1)
        self.skips = SkipConditioner(visual_dim, text_dim, width, hidden, reduction)
        self.skip_fusion = nn.ModuleList(
            [conv_block(width, width, separable=True) for _ in range(3)]
        )
        self.upsample = nn.ModuleList(
            [conv_block(a, b) for a, b in zip((width, *channels[:-1]), channels)]
        )
        self.detail_fusion = nn.ModuleList(
            [QuestionDetailFusion(c, detail_dim, detail_dim) for c in channels]
        )
        self.auxiliary_heads = nn.ModuleDict(
            {
                "96": nn.Conv2d(channels[1], 1, 1),
                "192": nn.Conv2d(channels[2], 1, 1),
            }
        )
        self.mask_head = nn.Conv2d(channels[-1], 1, 1)
        self.output_channels = channels[-1]

    def forward(
        self,
        features: tuple[Tensor, ...],
        question: Tensor,
        detail: tuple[Tensor, ...],
        detail_question: Tensor,
        run_block,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        s1, s2, s3, bottleneck = features
        x = self.bottleneck(bottleneck)
        for skip, block in zip(self.skips((s3, s2, s1), question), self.skip_fusion):
            x = x + block(x + skip)
        auxiliary = {}
        for index, (block, fusion, local) in enumerate(
            zip(self.upsample, self.detail_fusion, detail[:4])
        ):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = run_block(block, x)
            x = run_block(fusion, x, local, detail_question)
            if self.training and index in (1, 2):
                name = "96" if index == 1 else "192"
                auxiliary[name] = self.auxiliary_heads[name](x)
        logits = self.mask_head(x)
        if self.training:
            auxiliary["384"] = logits
        return logits, x, auxiliary
