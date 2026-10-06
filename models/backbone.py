"""Shared CLIP grounding backbone used by this branch and its frozen references."""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.mask_decoder import UNetDecoder
from models.experiment import CompactDecoder, SkipConditioner, masked_mean


class TinyVision(nn.Module):
    out_channels = 16
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
    def forward(self, image):
        feature = F.adaptive_avg_pool2d(self.conv(image), (2, 2))
        return feature, feature, feature, feature


class GroundingTextEncoder(nn.Module):
    def __init__(self, tiny=False):
        super().__init__()
        self.tiny = tiny
        if tiny:
            self.embedding = nn.Embedding(259, 16)
            self.output_dim = 16
        else:
            self.encoder = TextEncoder()
            self.output_dim = self.encoder.output_dim

    def forward(self, texts):
        if self.tiny:
            rows = [[1] + [b + 3 for b in text.encode("utf8")[:75]] + [2] for text in texts]
            ids = torch.zeros(len(rows), max(map(len, rows)), dtype=torch.long, device=self.embedding.weight.device)
            for i, row in enumerate(rows):
                ids[i, :len(row)] = torch.tensor(row, device=ids.device)
            attention = ids != 0
            tokens = self.embedding(ids)
        else:
            inputs = self.encoder.tokenizer(texts, return_tensors="pt", padding=True,
                truncation=True, max_length=77).to(self.encoder.model.device)
            attention = inputs.attention_mask.bool()
            tokens = self.encoder.model(**inputs).last_hidden_state
        valid = attention.clone()
        valid[:, 0] = False
        valid[torch.arange(len(texts), device=valid.device), attention.sum(1) - 1] = False
        return SimpleNamespace(tokens=tokens, attention_mask=attention, valid_mask=valid)


class BaseGroundingModel(nn.Module):
    def __init__(self, architecture="compact", tiny=False):
        super().__init__()
        if architecture not in {"compact", "joint"}:
            raise ValueError("Unknown architecture")
        if tiny and architecture != "compact":
            raise ValueError("Tiny offline model supports compact architecture only")
        self.image_encoder = TinyVision() if tiny else ImageEncoder()
        self.text_encoder = GroundingTextEncoder(tiny)
        self.visual_dim = self.image_encoder.out_channels
        self.text_dim = self.text_encoder.output_dim
        self.architecture = architecture
        self.text_proj = nn.Linear(self.text_dim, self.visual_dim)
        self.cross_attn = nn.MultiheadAttention(self.visual_dim, 2 if tiny else 8, batch_first=True)
        self.residual_scale = nn.Parameter(torch.tensor(.01))
        if architecture == "compact":
            self.decoder = CompactDecoder(self.visual_dim, self.text_dim, width=16 if tiny else 128)
        else:
            self.decoder = UNetDecoder(self.visual_dim)
        self.skip_conditioner = SkipConditioner(self.text_dim, self.visual_dim) if architecture == "joint" else None
        self.register_buffer("image_mean", torch.tensor([.48145466, .4578275, .40821073]).view(1,3,1,1))
        self.register_buffer("image_std", torch.tensor([.26862954, .26130258, .27577711]).view(1,3,1,1))
        self.experiment_config = {"experiment": "controls", "architecture": architecture, "tiny": tiny,
                                  "protocol": "normalized-masked-question-v1"}

    def encode_image(self, image):
        return self.image_encoder((image - self.image_mean) / self.image_std)

    def decode(self, features, texts):
        s1, s2, s3, vision = features
        text = self.text_encoder(texts)
        query = vision.flatten(2).transpose(1, 2)
        projected = self.text_proj(text.tokens)
        attended, _ = self.cross_attn(query, projected, projected,
            key_padding_mask=~text.attention_mask, need_weights=False)
        fused = query + self.residual_scale * attended
        fused = fused.transpose(1,2).reshape_as(vision)
        if self.architecture == "compact":
            logits = self.decoder(fused, s3, s2, s1, text)
        else:
            skips = [s3,s2,s1]
            if self.skip_conditioner is not None:
                skips = self.skip_conditioner(skips, text)
            logits = self.decoder(fused, *skips)
        return {"logits": logits, "visual": fused, "text_tokens": text.tokens,
                "text_mask": text.attention_mask, "pooled_text": masked_mean(text.tokens, text.valid_mask)}

    def forward(self, batch):
        return self.decode(self.encode_image(batch["image"]), batch["text"])
