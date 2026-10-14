"""CLIP text by default; optional text tower matching this branch's SigLIP vision."""
from torch import nn
from transformers import (CLIPTextConfig, CLIPTextModel, CLIPTokenizer,
                          Siglip2TextConfig, Siglip2TextModel, Siglip2Tokenizer)

from .image_encoder import VISION_MODEL

CLIP_TEXT_MODEL = "openai/clip-vit-large-patch14-336"


class TextEncoder(nn.Module):
    def __init__(self, backbone="clip", pretrained=True, config=None):
        super().__init__()
        self.backbone = backbone
        if backbone == "clip":
            name, model_cls, config_cls, tokenizer_cls = CLIP_TEXT_MODEL, CLIPTextModel, CLIPTextConfig, CLIPTokenizer
        elif backbone == "siglip2":
            name, model_cls, config_cls, tokenizer_cls = VISION_MODEL, Siglip2TextModel, Siglip2TextConfig, Siglip2Tokenizer
        else:
            raise ValueError("text encoder must be clip or siglip2")
        self.model = (model_cls.from_pretrained(name) if pretrained else
                      model_cls(config_cls(**config) if config is not None else config_cls.from_pretrained(name)))
        self.tokenizer = tokenizer_cls.from_pretrained(name)
        self.output_dim = self.model.config.hidden_size
        if backbone == "siglip2":
            self.model.head.requires_grad_(False)  # pooled projection is unused; use token features

    def forward(self, texts, return_mask=False):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, return_tensors="pt", return_attention_mask=True,
                                padding="max_length" if self.backbone == "siglip2" else True,
                                truncation=True, max_length=self.model.config.max_position_embeddings)
        inputs = inputs.to(self.model.device)
        outputs = self.model(**inputs)
        if return_mask:
            return outputs.last_hidden_state, ~inputs["attention_mask"].bool()
        return outputs.last_hidden_state
