"""Training-only byte reader with differentiable spatial attention priors."""

import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F

from answer_data import BOS, EOS, MAX_ANSWER_BYTES, VOCAB_SIZE, answer_bytes


class AnswerHead(nn.Module):
    """Read a fixed full-image grid; the mask biases attention, never crops tokens.

    GT and predicted views share geometry and weights. Mask probabilities remain
    differentiable through area pooling and log attention priors. Reader math is
    float32 under AMP to protect small coverage values and mask gradients.
    """

    def __init__(self, channels, *, tiny=False):
        super().__init__()
        width = 16 if tiny else 128
        self.register_buffer("mask_attention_version", torch.tensor(1))
        self.grid_size = 4 if tiny else 32
        self.num_heads = 2 if tiny else 4
        self.visual = nn.Linear(channels, width)
        self.spatial_position = nn.Parameter(torch.empty(1, self.grid_size**2, width))
        self.embedding = nn.Embedding(VOCAB_SIZE, width, padding_idx=0)
        self.answer_position = nn.Parameter(torch.empty(1, MAX_ANSWER_BYTES + 1, width))
        layer = nn.TransformerDecoderLayer(
            width,
            self.num_heads,
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
        with torch.autocast(feature.device.type, enabled=False):
            return self._read(feature.float(), mask.float(), answers)

    def _read(self, feature, mask, answers):
        device, batch = feature.device, len(feature)
        encoded = [answer_bytes(text) for text in answers]
        length = max(len(row) + 1 if row is not None else 1 for row in encoded)
        inputs = torch.zeros(batch, length, dtype=torch.long, device=device)
        targets = torch.full_like(inputs, -100)
        nonempty = mask.detach().flatten(1).sum(1) > 0
        for index, row in enumerate(encoded):
            inputs[index, 0] = BOS
            if row is not None:
                inputs[index, : len(row) + 1] = torch.tensor([BOS, *row], device=device)
                targets[index, : len(row) + 1] = torch.tensor(
                    [*row, EOS], device=device
                )
        targets[~nonempty] = -100
        spatial = F.adaptive_avg_pool2d(feature, self.grid_size)
        memory = self.visual(spatial.flatten(2).transpose(1, 2)) + self.spatial_position
        coverage = F.adaptive_avg_pool2d(mask, self.grid_size).flatten(1)
        # Normalize before clamping: a one-pixel GT region must dominate empty cells.
        coverage = coverage / coverage.amax(1, keepdim=True).clamp_min(1e-8)
        coverage = torch.where(nonempty[:, None], coverage, torch.ones_like(coverage))
        prior = coverage.clamp_min(1e-8).log()
        prior = prior[:, None, None, :].expand(batch, self.num_heads, length, -1)
        prior = prior.reshape(batch * self.num_heads, length, -1)
        tokens = self.embedding(inputs) + self.answer_position[:, :length]
        causal = torch.ones(length, length, dtype=torch.bool, device=device).triu(1)
        decoded = self.decoder(tokens, memory, tgt_mask=causal, memory_mask=prior)
        return {"answer_logits": self.output(decoded), "answer_targets": targets}

    def supervise_mask(self, feature, probability, answers):
        """Fixed reader + detached features; only the probability input receives gradients.

        This is NOT no_grad: functional detached weights preserve the derivative
        with respect to the mask without mutating DDP parameter flags or copying a model.
        """
        state = {name: value.detach() for name, value in self.named_parameters()}
        state.update({name: value.detach() for name, value in self.named_buffers()})
        return functional_call(
            self, state, (feature.detach(), probability, answers), strict=True
        )
