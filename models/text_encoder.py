"""CLIP token features and the small offline encoder used by tests."""
from typing import NamedTuple

import torch
from torch import nn


class TextFeatures(NamedTuple):
    tokens: torch.Tensor
    attention_mask: torch.Tensor
    valid_mask: torch.Tensor


def masked_mean(tokens, mask):
    # Accumulate in float32 under mixed precision; empty spans map to zero.
    weights = mask.unsqueeze(-1).to(torch.float32)
    pooled = (tokens.float() * weights).sum(1) / weights.sum(1).clamp_min(1)
    return pooled.to(tokens.dtype)


class TextEncoder(nn.Module):
    def __init__(self, tiny=False, model_name="openai/clip-vit-large-patch14-336", pretrained=True):
        super().__init__()
        self.tiny = tiny
        if tiny:
            self.embedding = nn.Embedding(259, 16)
            self.output_dim = 16
        else:
            from transformers import CLIPTextModel
            # Keep the encoder.model path used by existing checkpoints.
            self.encoder = nn.Module()
            if pretrained:
                self.encoder.model = CLIPTextModel.from_pretrained(model_name)
            else:
                from transformers import CLIPTextConfig
                self.encoder.model = CLIPTextModel(CLIPTextConfig(
                    hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                    num_attention_heads=12, projection_dim=768))
            self.model_name = model_name
            self.tokenizer = None
            self.output_dim = self.encoder.model.config.hidden_size

    def forward(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if self.tiny:
            rows = [[1] + [b + 3 for b in text.encode("utf8")[:75]] + [2] for text in texts]
            ids = torch.zeros(len(rows), max(map(len, rows)), dtype=torch.long, device=self.embedding.weight.device)
            for i, row in enumerate(rows):
                ids[i, :len(row)] = torch.tensor(row, device=ids.device)
            attention = ids != 0
            tokens = self.embedding(ids)
        else:
            if self.tokenizer is None:
                from transformers import CLIPTokenizer
                self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True,
                truncation=True, max_length=77).to(self.encoder.model.device)
            attention = inputs.attention_mask.bool()
            tokens = self.encoder.model(**inputs).last_hidden_state
        valid = attention.clone()
        valid[:, 0] = False
        valid[torch.arange(len(texts), device=valid.device), attention.sum(1) - 1] = False
        return TextFeatures(tokens, attention, valid)
