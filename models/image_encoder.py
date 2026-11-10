from dataclasses import dataclass

import torch
from torch import nn
from transformers import CLIPVisionConfig, CLIPVisionModel

# hidden_states[0] is the patch embedding; hidden_states[k] is the output of transformer block k.
SKIP_LAYERS = (4, 7, 9)
CONVNEXT_SKIP_CHANNELS = (128, 256, 512)
CONVNEXT_OUT_CHANNELS = 1024


@dataclass(frozen=True)
class TinyConvNeXtVisionConfig:
    stage_channels: tuple[int, int, int, int] = (8, 16, 24, 32)


class _TinyConvNeXtVision(nn.Module):
    """Offline tiny substitute matching OpenCLIP/timm ConvNeXt trunk.stem + trunk.stages[0..3] spatial strides."""

    def __init__(self, config: TinyConvNeXtVisionConfig):
        super().__init__()
        c1, c2, c3, c4 = config.stage_channels
        self.trunk = nn.Module()
        self.trunk.stem = nn.Sequential(nn.Conv2d(3, c1, kernel_size=4, stride=4), nn.GroupNorm(1, c1))
        self.trunk.stages = nn.Sequential(
            nn.Sequential(nn.Conv2d(c1, c1, kernel_size=3, padding=1), nn.GroupNorm(1, c1), nn.GELU()),
            nn.Sequential(
                nn.GroupNorm(1, c1),
                nn.Conv2d(c1, c2, kernel_size=2, stride=2),
                nn.Conv2d(c2, c2, kernel_size=3, padding=1),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.GroupNorm(1, c2),
                nn.Conv2d(c2, c3, kernel_size=2, stride=2),
                nn.Conv2d(c3, c3, kernel_size=3, padding=1),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.GroupNorm(1, c3),
                nn.Conv2d(c3, c4, kernel_size=2, stride=2),
                nn.Conv2d(c4, c4, kernel_size=3, padding=1),
                nn.GELU(),
            ),
        )
        self.trunk.norm_pre = nn.LayerNorm(c4)
        self.trunk.head = nn.Linear(c4, c4)
        self.head = nn.Linear(c4, c4)


def _load_convnext_vision(model_name: str, *, revision: str | None = None) -> nn.Module:
    import open_clip
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model_name, "open_clip_pytorch_model.bin", revision=revision)
    return open_clip.create_model("convnext_base", pretrained=path).visual


class ImageEncoder(nn.Module):
    """CLIP vision encoder returning three skip maps and the final stage map.

    config: build a randomly initialised encoder from this config instead of loading pretrained
    weights (offline tests only).
    """

    def __init__(self, model_name="openai/clip-vit-large-patch14-336",
                 config: CLIPVisionConfig | TinyConvNeXtVisionConfig | None = None, *,
                 revision: str | None = None, backbone: str = "clip-vit-l14-336"):
        super().__init__()
        self.backbone = backbone
        if backbone == "convnext-base":
            self.vision_encoder = (
                _TinyConvNeXtVision(config)
                if config is not None
                else _load_convnext_vision(model_name, revision=revision)
            )
            # Global pooling and projection heads are never used by dense grounding. Replacing them with
            # Identity removes their parameters so TrainingControls(--no-freeze-vision-encoder) and DDP
            # see no unused parameters.
            self.vision_encoder.trunk.norm_pre = nn.Identity()
            self.vision_encoder.trunk.head = nn.Identity()
            self.vision_encoder.head = nn.Identity()
            self.skip_channels = config.stage_channels[:3] if config is not None else CONVNEXT_SKIP_CHANNELS
            self.out_channels = config.stage_channels[3] if config is not None else CONVNEXT_OUT_CHANNELS
        else:
            self.vision_encoder = (
                CLIPVisionModel(config)
                if config is not None
                else CLIPVisionModel.from_pretrained(model_name, **({"revision": revision} if revision else {}))
            )
            # post_layernorm only normalises the pooled CLS output, which is never used. Freezing it keeps
            # every trainable parameter in the graph, so DDP can run with find_unused_parameters=False.
            # Matched by name: transformers 4.x nests it under vision_model, 5.x does not.
            for name, module in self.vision_encoder.named_modules():
                if name.endswith("post_layernorm"):
                    module.requires_grad_(False)
            self.skip_channels = None
            self.out_channels = self.vision_encoder.config.hidden_size

    @staticmethod
    def _to_map(tokens: torch.Tensor) -> torch.Tensor:
        """(B, 1 + N, D) tokens with a leading CLS token -> (B, D, H_p, W_p) patch map."""
        patches = tokens[:, 1:]  # (B, N, D)
        side = int(patches.shape[1] ** 0.5)
        if side * side != patches.shape[1]:
            raise ValueError(f"Expected a square patch grid, got {patches.shape[1]} patches")
        return patches.unflatten(1, (side, side)).permute(0, 3, 1, 2).contiguous()  # (B, D, H_p, W_p)

    def forward(self, x: torch.Tensor):
        """x: (B, 3, H, W) CLIP-normalised images -> (s1, s2, s3, last)."""
        if self.backbone == "convnext-base":
            trunk = self.vision_encoder.trunk
            x = trunk.stem(x.to(trunk.stem[0].weight.dtype))
            s1 = trunk.stages[0](x)
            s2 = trunk.stages[1](s1)
            s3 = trunk.stages[2](s2)
            return s1, s2, s3, trunk.stages[3](s3)
        outputs = self.vision_encoder(x, output_hidden_states=True)
        skips = tuple(self._to_map(outputs.hidden_states[i]) for i in SKIP_LAYERS)
        return skips + (self._to_map(outputs.last_hidden_state),)
