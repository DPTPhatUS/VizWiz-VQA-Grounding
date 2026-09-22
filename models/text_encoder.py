"""CLIP text tokens with optional masks for lightweight conditioning."""
from typing import NamedTuple

import torch
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel


class TextFeatures(NamedTuple):
    tokens: torch.Tensor
    valid_mask: torch.Tensor
    question_mask: torch.Tensor
    answer_mask: torch.Tensor


class TextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14-336"):
        super().__init__()
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.output_dim = self.model.config.hidden_size

    def forward(self, texts, return_features=False):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=self.model.config.max_position_embeddings,
        ).to(self.model.device)
        tokens = self.model(**inputs).last_hidden_state
        if not return_features:
            return tokens

        # EOS is also CLIP's padding token; the attention mask alone is not
        # sufficient to exclude special tokens from pooled content.
        valid = inputs.attention_mask.bool().clone()
        valid[:, 0] = False
        lengths = inputs.attention_mask.sum(1)
        valid[torch.arange(len(texts), device=valid.device), lengths - 1] = False
        question_mask = torch.zeros_like(valid)
        answer_mask = torch.zeros_like(valid)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        for row, text in enumerate(texts):
            before, marker, _ = text.rpartition(" A: ")
            question_text = before if marker else text
            prefix = "Q: " if question_text.startswith("Q: ") else ""
            start = 1 + len(self.tokenizer.encode(prefix, add_special_tokens=False, verbose=False))
            end = 1 + len(self.tokenizer.encode(question_text, add_special_tokens=False, verbose=False))
            question_mask[row] = (positions >= start) & (positions < end) & valid[row]
            if marker:
                answer_start = 1 + len(self.tokenizer.encode(before + marker, add_special_tokens=False, verbose=False))
                answer_mask[row] = (positions >= answer_start) & valid[row]
        return TextFeatures(tokens, valid, question_mask, answer_mask)
