"""Question-only grounding model: CLIP encoders, question cross-attention, mask decoder."""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPTextConfig, CLIPVisionConfig

from dataset import (
    CONVNEXT_IMAGE_SIZE,
    IMAGE_SIZE,
    load_image,
    model_text,
    preprocess_image,
)
from models.image_encoder import ImageEncoder, TinyConvNeXtVisionConfig
from models.mask_decoder import FusionDecoder, UNetDecoder
from models.text_encoder import TextEncoder
from utils import align_state_dict

CLIP_NAME = "openai/clip-vit-large-patch14-336"
CONVNEXT_NAME = "laion/CLIP-convnext_base-laion400M-s13B-b51K"
CONVNEXT_REVISION = "4c478c07a1511dafb7331443cd20bf7867a0f1b1"
CONVNEXT_TOKENIZER_NAME = "openai/clip-vit-base-patch16"
CONVNEXT_TOKENIZER_REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
TEXT_REVISION = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
VISION_BACKBONES = ("clip-vit-l14-336", "convnext-base")


@dataclass(frozen=True)
class ModelConfig:
    """Architecture; saved in every checkpoint so evaluation rebuilds exactly the trained model.

    Defaults are B0 (ocrg decision D017). decoder="unet" with attention_residual=False and
    stage_attention=False is the upstream baseline-fixes architecture.
    """

    decoder: str = "fusion"  # "fusion" (FusionDecoder) or "unet" (upstream UNetDecoder)
    attention_residual: bool = True  # F = V + alpha * MHA(V, T); False: F = MHA(V, T) (upstream)
    attention_residual_init: float = 0.01  # initial alpha (learnable), as in the ACIIDS models
    stage_attention: bool = True  # question attention in the first three upsampling stages; gamma init 0
    clip_name: str = CLIP_NAME
    tiny: bool = False  # randomly initialised tiny encoders, for offline tests only
    vision_backbone: str = "clip-vit-l14-336"
    vision_revision: str | None = None
    text_revision: str | None = None

    def __post_init__(self):
        if self.vision_backbone not in VISION_BACKBONES:
            raise ValueError(f"Unknown vision backbone {self.vision_backbone!r}")
        if self.vision_backbone == "convnext-base":
            if self.clip_name not in (CLIP_NAME, CONVNEXT_NAME):
                raise ValueError("ConvNeXt-Base requires its paired CLIP text encoder")
            object.__setattr__(self, "clip_name", CONVNEXT_NAME)
            object.__setattr__(self, "vision_revision", self.vision_revision or CONVNEXT_REVISION)
            object.__setattr__(self, "text_revision", self.text_revision or self.vision_revision)
        if self.vision_revision != self.text_revision:
            raise ValueError("Vision/text encoder pair must share the same pretrained revision")
        for revision in (self.vision_revision, self.text_revision):
            if revision is not None and (len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision)):
                raise ValueError("Pretrained revisions must be immutable 40-character commit hashes")
        if self.decoder not in {"fusion", "unet"}:
            raise ValueError(f"Unknown decoder {self.decoder!r}")
        if self.stage_attention and self.decoder != "fusion":
            raise ValueError("stage_attention exists only in the fusion decoder; use stage_attention=False with unet")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def vision_name(self) -> str:
        return CONVNEXT_NAME if self.vision_backbone == "convnext-base" else self.clip_name

    @property
    def image_size(self) -> int:
        return CONVNEXT_IMAGE_SIZE if self.vision_backbone == "convnext-base" else IMAGE_SIZE


def _tiny_configs(convnext: bool = False):
    vision = (
        TinyConvNeXtVisionConfig()
        if convnext
        else CLIPVisionConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=10, num_attention_heads=2,
            image_size=IMAGE_SIZE, patch_size=14,
        )
    )
    text = CLIPTextConfig(
        hidden_size=16 if convnext else 32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2
    )
    return vision, text


class GroundingModel(nn.Module):
    """CLIP-normalised images (336px for L/14, 384px for ConvNeXt-Base) + questions -> logits (384px)."""

    def __init__(self, config: ModelConfig | None = None, n_heads: int = 8):
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        convnext = config.vision_backbone == "convnext-base"
        vision_config, text_config = _tiny_configs(convnext) if config.tiny else (None, None)
        self.image_encoder = ImageEncoder(
            config.vision_name, vision_config, revision=config.vision_revision, backbone=config.vision_backbone
        )
        self.text_encoder = TextEncoder(
            config.clip_name,
            text_config,
            revision=config.text_revision,
            tokenizer_name=CONVNEXT_TOKENIZER_NAME if convnext else None,
            tokenizer_revision=CONVNEXT_TOKENIZER_REVISION if convnext else None,
        )
        self.hidden_dim = self.image_encoder.out_channels  # 1024 for ViT-L and ConvNeXt-Base
        heads = 2 if config.tiny else n_heads
        self.text_proj = nn.Linear(self.text_encoder.output_dim, self.hidden_dim)
        self.cross_attn = nn.MultiheadAttention(self.hidden_dim, heads, batch_first=True)
        self.attention_scale = (nn.Parameter(torch.tensor(float(config.attention_residual_init)))
                                if config.attention_residual else None)
        self.decoder = (FusionDecoder(self.hidden_dim, text_dim=self.text_encoder.output_dim,
                                      stage_attention=config.stage_attention,
                                      skip_channels=self.image_encoder.skip_channels)
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
        s1, s2, s3, vision = self.image_encoder(images)  # ViT maps or CNN pyramid
        tokens, padding = self.text_encoder(texts)  # (B, L, D_t), (B, L)
        fused = self.fuse(vision, tokens, padding)  # (B, D_v, H_p, W_p)
        if isinstance(self.decoder, FusionDecoder):
            return self.decoder(fused, s1, s2, s3, tokens, padding)
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
        image_tensor = preprocess_image(pil, self.config.image_size)
        logits = self(image_tensor.unsqueeze(0).to(device), [model_text({"question": question, "most_common_answer": answer}, "qa" if answer else "question")])  # (1, 1, h, w)
        logits = F.interpolate(logits.float(), size=(pil.height, pil.width), mode="bilinear", align_corners=False)
        return logits[0, 0] > 0  # (H, W)
