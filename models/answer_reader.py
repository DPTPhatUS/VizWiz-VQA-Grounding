"""Local RGB reader: no globally contextualized visual features or spatial normalization."""

import torch
from torch import nn
from torch.nn import functional as F


class MaskedAnswerReader(nn.Module):
    """Read a soft region with a five-pixel receptive field before masked pooling.

    GT-region supervision trains the reader. The predicted-region path uses detached
    reader weights/features, keeping its gradient only through the predicted mask.
    Question features modulate visual channels; there is no question-only output path.
    """

    def __init__(self, text_dim, width=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width // 2, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(width // 2, width, 3, padding=1, bias=False),
            nn.GELU(),
        )
        self.question_gate = nn.Linear(text_dim, width, bias=False)
        self.projection = nn.Sequential(
            nn.Linear(width, width, bias=False),
            nn.GELU(),
            nn.Linear(width, text_dim, bias=False),
        )

    def encode(self, image):
        return self.encoder(image)  # (B, C, H, W), RGB in [0, 1]

    def read(self, local, mask, question, *, frozen=False):
        if (
            mask.ndim != 4
            or mask.shape[:2] != (local.shape[0], 1)
            or mask.shape[-2:] != local.shape[-2:]
        ):
            raise ValueError(
                "Reader mask must have shape (B, 1, H, W) matching local features"
            )
        features = local.detach() if frozen else local
        weights = mask.float()
        pooled = (features.float() * weights).sum((-2, -1)) / weights.sum(
            (-2, -1)
        ).clamp_min(1e-6)

        def weight(layer):
            return layer.weight.detach() if frozen else layer.weight

        gate = 1 + torch.tanh(F.linear(question.detach(), weight(self.question_gate)))
        hidden = F.gelu(F.linear(pooled * gate.float(), weight(self.projection[0])))
        return F.linear(hidden, weight(self.projection[2]))  # (B, D_t)
