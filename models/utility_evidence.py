"""A small question-conditioned advantage scorer over exact mask edits."""

import torch
from torch import nn


class UtilityEvidence(nn.Module):
    def __init__(self, text_dim, visual_dim, width=64):
        super().__init__()
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, width))
        self.visual = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, width)
        )
        self.geometry = nn.Linear(5, width)
        self.question_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.context_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.operation = nn.Embedding(5, 16)
        self.score = nn.Sequential(
            nn.Linear(4 * width + 10 + 16, width), nn.GELU(), nn.Linear(width, 1)
        )

    def forward(self, batch):
        q = self.text(batch["question"].float())
        words = (
            self.text(batch["words"].float())
            + self.visual(batch["word_visual"].float())
            + self.geometry(
                torch.cat((batch["boxes"].float(), batch["confidence"][..., None]), -1)
            )
        )
        a, _ = self.question_attention(
            words, q, q, key_padding_mask=~batch["question_valid"], need_weights=False
        )
        words = self.norm1(words + a)
        parent = self.visual(batch["action_visual"][:, 0].float())
        keys = torch.cat((words, parent[:, None]), 1)
        available = torch.cat(
            (batch["word_valid"], torch.ones_like(batch["word_valid"][:, :1])), 1
        )
        a, _ = self.context_attention(
            words, keys, keys, key_padding_mask=~available, need_weights=False
        )
        words = self.norm2(words + a)
        members = batch["members"].float()
        pooled = torch.bmm(members, words.float()) / members.sum(
            -1, keepdim=True
        ).clamp_min(1)
        visual = self.visual(batch["action_visual"].float())
        parent_q, _ = self.question_attention(
            parent[:, None],
            q,
            q,
            key_padding_mask=~batch["question_valid"],
            need_weights=False,
        )
        reference = (parent + parent_q[:, 0])[:, None].expand_as(visual)
        score = (
            self.score(
                torch.cat(
                    (
                        pooled,
                        visual,
                        reference,
                        visual - reference,
                        batch["descriptors"].float(),
                        self.operation(batch["operation"]),
                    ),
                    -1,
                )
            )
            .squeeze(-1)
            .float()
        )
        # Identity is an exact zero-advantage reference, not a learned null probability.
        return torch.cat((score[:, :1] * 0, score[:, 1:]), 1)
