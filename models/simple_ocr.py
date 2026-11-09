"""Optional simple OCR branch; original B0 modules and checkpoints are unchanged."""

import math
from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import nn

from models.model import GroundingModel, ModelConfig

ARMS = ("b0_continued", "b1_ocr_48", "b1_ocr_96")


class OCRInputs(TypedDict):
    questions: list[str]
    embedding: torch.Tensor  # B,N,D frozen text features
    geometry: torch.Tensor  # B,N,5 normalized xyxy + missing-text flag
    valid: torch.Tensor  # B,N bool; no gold targets/answers


class OCRAttention(nn.Module):
    def __init__(self, channels: int, text_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(max(1, channels // 8), channels)
        self.text = nn.Linear(text_dim, channels)
        self.geometry = nn.Linear(5, channels)
        self.attention = nn.MultiheadAttention(
            channels, channels // 32, batch_first=True
        )
        self.beta = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x: torch.Tensor,
        embedding: torch.Tensor,
        geometry: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        # Same normalized frequency range in both widths; no high-frequency
        # aliasing introduced solely by the wider 48px branch.
        freq = torch.linspace(1, 16, c // 4, device=x.device)
        yy, xx = torch.meshgrid(
            (torch.arange(h, device=x.device) + 0.5) / h,
            (torch.arange(w, device=x.device) + 0.5) / w,
            indexing="ij",
        )
        position = torch.cat(
            [
                f(t[..., None] * freq * (2 * math.pi))
                for t in (xx, yy)
                for f in (torch.sin, torch.cos)
            ],
            -1,
        ).reshape(1, h * w, c)
        query = self.norm(x).flatten(2).transpose(1, 2)
        query = query + position.to(query.dtype)
        keys = self.text(embedding) + self.geometry(geometry)
        # A finite dummy key prevents all-masked softmax. Final explicit gate
        # removes projection biases, and keeps empty ranks connected to DDP.
        mask = valid.clone()
        present = valid.any(1)
        mask[:, 0] |= ~present
        attended, _ = self.attention(
            query.to(keys.dtype), keys, keys, key_padding_mask=~mask, need_weights=False
        )
        delta = attended.transpose(1, 2).reshape(b, c, h, w)
        return x + self.beta.to(delta.dtype) * present[:, None, None, None] * delta


class SimpleOCRModel(nn.Module):
    def __init__(self, arm: str, config: ModelConfig | None = None):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.base = GroundingModel(config)
        self.base.text_encoder.requires_grad_(False)
        # Preserve the common RNG stream when constructing extra random layers.
        with torch.random.fork_rng(devices=[]):
            self.ocr = (
                None
                if arm == "b0_continued"
                else OCRAttention(
                    128 if arm == "b1_ocr_48" else 64, self.base.text_encoder.output_dim
                )
            )

    def forward(self, images: torch.Tensor, inputs: OCRInputs) -> torch.Tensor:
        if self.ocr is None:
            return self.base(images, inputs["questions"])
        s1, s2, s3, vision = self.base.image_encoder(images)
        tokens, padding = self.base.text_encoder(inputs["questions"])
        fused = self.base.fuse(vision, tokens, padding)
        decoder = self.base.decoder
        x = decoder.input_proj(fused)
        for proj, block, skip in zip(
            decoder.skip_proj, decoder.fusion, (s3, s2, s1), strict=True
        ):
            x = x + block(x + proj(skip))
        for i, stage in enumerate(decoder.stages):
            x = stage(
                F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            )
            if decoder.stage_attention is not None and i < len(decoder.stage_attention):
                x = decoder.stage_attention[i](x, tokens, padding)
            if i == (0 if self.arm == "b1_ocr_48" else 1):
                x = self.ocr(
                    x,
                    inputs["embedding"].to(x.device),
                    inputs["geometry"].to(x.device),
                    inputs["valid"].to(x.device),
                )
        return decoder.head(x)
