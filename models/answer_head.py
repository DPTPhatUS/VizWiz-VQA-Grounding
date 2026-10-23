"""Small training-only autoregressive reader of ordered GT-region features."""

import torch
from torch import nn
from torch.nn import functional as F

from answer_data import BOS, EOS, MAX_ANSWER_BYTES, VOCAB_SIZE, answer_bytes


def region_memory(feature, mask, grid_size):
    """Crop the GT envelope and suppress non-evidence; preserve ordered spatial tokens."""
    _, _, height, width = mask.shape
    locations, coverages, nonempty = [], [], []
    steps = (torch.arange(grid_size, device=feature.device).float() + 0.5) / grid_size
    for truth in mask.detach()[:, 0] > 0.5:
        points = truth.nonzero()
        nonempty.append(bool(len(points)))
        if len(points):
            top, left = points.amin(0)
            bottom, right = points.amax(0) + 1
        else:
            top = left = 0
            bottom, right = height, width
        y = (top + steps * (bottom - top)) * (2 / height) - 1
        x = (left + steps * (right - left)) * (2 / width) - 1
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        locations.append(torch.stack((xx, yy), -1))
        # Area coverage cannot miss a thin/single-pixel target between grid centers.
        cropped = truth[int(top) : int(bottom), int(left) : int(right)].float()[
            None, None
        ]
        coverages.append(F.adaptive_avg_pool2d(cropped, grid_size))
    coordinates = torch.stack(locations)
    visual = F.grid_sample(feature.float(), coordinates, align_corners=False)
    coverage = torch.cat(coverages)
    visual = visual * coverage
    padding = coverage.flatten(1) <= 0
    padding[padding.all(1), 0] = False  # Avoid all-masked attention on invalid samples.
    return (
        visual.flatten(2).transpose(1, 2),
        padding,
        torch.tensor(nonempty, device=feature.device),
    )


class AnswerHead(nn.Module):
    """Teacher forcing is confined to this auxiliary head, never the mask decoder."""

    def __init__(self, channels, *, tiny=False):
        super().__init__()
        width = 16 if tiny else 128
        self.grid_size = 4 if tiny else 16
        self.visual = nn.Linear(channels, width)
        self.spatial_position = nn.Parameter(torch.empty(1, self.grid_size**2, width))
        self.embedding = nn.Embedding(VOCAB_SIZE, width, padding_idx=0)
        self.answer_position = nn.Parameter(torch.empty(1, MAX_ANSWER_BYTES + 1, width))
        layer = nn.TransformerDecoderLayer(
            width,
            2 if tiny else 4,
            4 * width,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, 1 if tiny else 2, norm=nn.LayerNorm(width)
        )
        self.output = nn.Linear(width, VOCAB_SIZE)
        nn.init.normal_(self.spatial_position, std=0.02)
        nn.init.normal_(self.answer_position, std=0.02)

    def forward(self, feature, mask, answers):
        device, batch = feature.device, len(feature)
        encoded = [answer_bytes(text) for text in answers]
        length = max(len(row) + 1 if row is not None else 1 for row in encoded)
        inputs = torch.zeros(batch, length, dtype=torch.long, device=device)
        targets = torch.full_like(inputs, -100)
        for index, row in enumerate(encoded):
            inputs[index, 0] = BOS
            if row is not None:
                inputs[index, : len(row) + 1] = torch.tensor([BOS, *row], device=device)
                targets[index, : len(row) + 1] = torch.tensor(
                    [*row, EOS], device=device
                )
        memory, padding, nonempty = region_memory(feature, mask, self.grid_size)
        targets[~nonempty] = -100
        memory = (
            self.visual(memory.to(self.visual.weight.dtype)) + self.spatial_position
        )
        tokens = self.embedding(inputs) + self.answer_position[:, :length]
        causal = torch.ones(length, length, dtype=torch.bool, device=device).triu(1)
        decoded = self.decoder(
            tokens, memory, tgt_mask=causal, memory_key_padding_mask=padding
        )
        return {"answer_logits": self.output(decoded), "answer_targets": targets}
