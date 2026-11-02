"""Question-only grounding model: CLIP encoders, question cross-attention, mask decoder."""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPTextConfig, CLIPVisionConfig

from dataset import IMAGE_SIZE, load_image, model_text, preprocess_image
from models.image_encoder import ImageEncoder
from models.mask_decoder import FusionDecoder, UNetDecoder
from models.text_encoder import TextEncoder
from utils import align_state_dict

CLIP_NAME = "openai/clip-vit-large-patch14-336"


@dataclass(frozen=True)
class ModelConfig:
    """Architecture; saved in every checkpoint so evaluation rebuilds exactly the trained model.

    Defaults are B0 (ocrg decision D017). decoder="unet" with attention_residual=False and
    stage_attention=False is the upstream baseline-fixes architecture.
    """

    decoder: str = "fusion"  # "fusion" (FusionDecoder) or "unet" (upstream UNetDecoder)
    attention_residual: bool = True  # F = V + alpha * MHA(V, T); False: F = MHA(V, T) (upstream)
    attention_residual_init: float = 0.01  # initial alpha (learnable), as in the ACIIDS models
    stage_attention: bool = True  # question attention in decoder stages 48/96/192 px (gamma init 0); fusion only
    clip_name: str = CLIP_NAME
    tiny: bool = False  # randomly initialised tiny encoders, for offline tests only

    def __post_init__(self):
        if self.decoder not in {"fusion", "unet"}:
            raise ValueError(f"Unknown decoder {self.decoder!r}")
        if self.stage_attention and self.decoder != "fusion":
            raise ValueError("stage_attention exists only in the fusion decoder; use stage_attention=False with unet")

    def to_dict(self) -> dict:
        return asdict(self)


def _tiny_configs():
    vision = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=10,
                              num_attention_heads=2, image_size=IMAGE_SIZE, patch_size=14)
    text = CLIPTextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2)
    return vision, text


class GroundingModel(nn.Module):
    """(images (B, 3, 336, 336) CLIP-normalised, questions list[str]) -> mask logits (B, 1, 384, 384)."""

    def __init__(self, config: ModelConfig | None = None, n_heads: int = 8):
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        vision_config, text_config = _tiny_configs() if config.tiny else (None, None)
        self.image_encoder = ImageEncoder(config.clip_name, vision_config)
        self.text_encoder = TextEncoder(config.clip_name, text_config)
        self.hidden_dim = self.image_encoder.out_channels  # 1024 for ViT-L
        heads = 2 if config.tiny else n_heads
        self.text_proj = nn.Linear(self.text_encoder.output_dim, self.hidden_dim)
        self.cross_attn = nn.MultiheadAttention(self.hidden_dim, heads, batch_first=True)
        self.attention_scale = (nn.Parameter(torch.tensor(float(config.attention_residual_init)))
                                if config.attention_residual else None)
        self.decoder = (FusionDecoder(self.hidden_dim, text_dim=self.text_encoder.output_dim,
                                      stage_attention=config.stage_attention)
                        if config.decoder == "fusion" else UNetDecoder(in_channels=self.hidden_dim))

    def fuse(self, vision: torch.Tensor, tokens: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        """Patches attend to question tokens. vision (B, D_v, H_p, W_p), tokens (B, L, D_t), padding (B, L)."""
        query = vision.flatten(2).transpose(1, 2)  # (B, N, D_v)
        keys = self.text_proj(tokens)  # (B, L, D_v)
        query = query.to(keys.dtype)
        attended, _ = self.cross_attn(query, keys, keys, key_padding_mask=padding, need_weights=False)  # (B, N, D_v)
        fused = query + self.attention_scale.to(attended.dtype) * attended if self.attention_scale is not None else attended
        return fused.transpose(1, 2).reshape(vision.shape)  # (B, D_v, H_p, W_p)

    def forward(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        s1, s2, s3, vision = self.image_encoder(images)  # each (B, D_v, H_p, W_p)
        tokens, padding = self.text_encoder(texts)  # (B, L, D_t), (B, L)
        fused = self.fuse(vision, tokens, padding)  # (B, D_v, H_p, W_p)
        if isinstance(self.decoder, FusionDecoder):
            return self.decoder(fused, s1, s2, s3, tokens, padding)  # (B, 1, 16*H_p, 16*W_p)
        return self.decoder(fused, s3, s2, s1)  # upstream argument order

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        """Rebuild the trained architecture from the checkpoint's model_config and load its weights.

        Returns (model in eval mode on device, metadata: the checkpoint without weights or optimizer state).
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model_config" not in checkpoint:
            raise ValueError(f"{path} has no model_config; it was not written by this branch's train.py")
        model = cls(ModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(align_state_dict(checkpoint["model_state_dict"], model))
        metadata = {k: v for k, v in checkpoint.items()
                    if k not in {"model_state_dict", "optimizer_state_dict", "scaler_state_dict"}}
        return model.to(device).eval(), metadata

    @torch.no_grad()
    def predict(self, image, question: str, answer: str | None = None) -> torch.Tensor:
        """Single image (path or PIL) and question -> boolean mask (H, W) at the image's own resolution.
        Pass answer only for a model trained with text_mode="qa"."""
        self.eval()
        pil = load_image(image) if not hasattr(image, "size") else image.convert("RGB")
        device = next(self.parameters()).device
        logits = self(preprocess_image(pil).unsqueeze(0).to(device), [model_text({"question": question, "most_common_answer": answer}, "qa" if answer else "question")])  # (1, 1, h, w)
        logits = F.interpolate(logits.float(), size=(pil.height, pil.width), mode="bilinear", align_corners=False)
        return logits[0, 0] > 0  # (H, W)
