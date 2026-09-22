"""Low-rank text conditioning for grounding features."""
import torch
from torch import nn
from torch.nn import functional as F


def masked_mean(tokens, mask):
    # Accumulate in float32 under mixed precision; empty spans map to zero.
    weights = mask.unsqueeze(-1).to(torch.float32)
    pooled = (tokens.float() * weights).sum(1) / weights.sum(1).clamp_min(1)
    return pooled.to(tokens.dtype)


class TextConditioner(nn.Module):
    def __init__(self, text_dim, separate=False, reduction=64):
        super().__init__()
        self.separate = separate
        if separate:
            self.gate = nn.Sequential(
                nn.Linear(2 * text_dim, reduction), nn.GELU(),
                nn.Linear(reduction, text_dim), nn.Sigmoid(),
            )

    def forward(self, text):
        if not self.separate:
            return masked_mean(text.tokens, text.valid_mask)
        question = masked_mean(text.tokens, text.question_mask)
        answer = masked_mean(text.tokens, text.answer_mask)
        joint = masked_mean(text.tokens, text.valid_mask)
        question = torch.where(text.question_mask.any(1, keepdim=True), question, joint)
        answer = torch.where(text.answer_mask.any(1, keepdim=True), answer, question)
        gate = self.gate(torch.cat([question, answer], dim=-1))
        return gate * question + (1 - gate) * answer


class SkipConditioner(nn.Module):
    def __init__(self, text_dim, channels, reduction=64, separate=False):
        super().__init__()
        self.pool = TextConditioner(text_dim, separate=separate, reduction=reduction)
        self.reduction = nn.Linear(text_dim, reduction)
        self.affine = nn.ModuleList([nn.Linear(reduction, 2 * channels) for _ in range(3)])
        for head in self.affine:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, skips, text):
        condition = F.gelu(self.reduction(self.pool(text)))
        output = []
        for feature, head in zip(skips, self.affine):
            gamma, beta = head(condition).chunk(2, dim=-1)
            output.append(feature * (1 + gamma[:, :, None, None]) + beta[:, :, None, None])
        return output
