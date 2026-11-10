from dataclasses import dataclass

import torch
from torch import nn
from transformers import CLIPVisionConfig, CLIPVisionModel

# hidden_states[0] is the patch embedding; hidden_states[k] is the output of transformer block k.
SKIP_LAYERS = (4, 7, 9)
RN50_SKIP_CHANNELS = (256, 512, 1024)
RN50_OUT_CHANNELS = 2048


@dataclass(frozen=True)
class TinyRN50VisionConfig:
    stage_channels: tuple[int, int, int, int] = (8, 16, 24, 32)


class _TinyRN50Vision(nn.Module):
    """Offline tiny substitute matching OpenAI CLIP ModifiedResNet stem + layer1..4 spatial strides."""

    def __init__(self, config: TinyRN50VisionConfig):
        super().__init__()
        c1, c2, c3, c4 = config.stage_channels
        w = max(4, c1 // 2)
        self.conv1 = nn.Conv2d(3, w, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(w, w, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(w)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(w, c1, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c1)
        self.act3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.AvgPool2d(2), nn.Conv2d(c1, c2, 3, padding=1, bias=False), nn.BatchNorm2d(c2), nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.AvgPool2d(2), nn.Conv2d(c2, c3, 3, padding=1, bias=False), nn.BatchNorm2d(c3), nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.AvgPool2d(2), nn.Conv2d(c3, c4, 3, padding=1, bias=False), nn.BatchNorm2d(c4), nn.ReLU(inplace=True)
        )
        self.attnpool = nn.Linear(c4, c4)

    def stem(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        return self.avgpool(x)


def _load_rn50_vision(model_name: str, *, revision: str | None = None) -> nn.Module:
    import open_clip
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model_name, "open_clip_model.safetensors", revision=revision)
    return open_clip.create_model("RN50-quickgelu", pretrained=path).visual


class ImageEncoder(nn.Module):
    """CLIP vision encoder returning three skip maps and the final stage map.

    config: build a randomly initialised encoder from this config instead of loading pretrained
    weights (offline tests only).
    """

    def __init__(self, model_name="openai/clip-vit-large-patch14-336",
                 config: CLIPVisionConfig | TinyRN50VisionConfig | None = None, *,
                 revision: str | None = None, backbone: str = "clip-vit-l14-336"):
        super().__init__()
        self.backbone = backbone
        if backbone == "clip-rn50":
            self.vision_encoder = (
                _TinyRN50Vision(config) if config is not None else _load_rn50_vision(model_name, revision=revision)
            )
            # attnpool only produces the global pooled vector, which is never used. Replacing it with Identity
            # removes its parameters so TrainingControls(--no-freeze-vision-encoder) and DDP see no unused params.
            self.vision_encoder.attnpool = nn.Identity()
            self.skip_channels = config.stage_channels[:3] if config is not None else RN50_SKIP_CHANNELS
            self.out_channels = config.stage_channels[3] if config is not None else RN50_OUT_CHANNELS
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
        if self.backbone == "clip-rn50":
            x = self.vision_encoder.stem(x.to(self.vision_encoder.conv1.weight.dtype))
            s1 = self.vision_encoder.layer1(x)
            s2 = self.vision_encoder.layer2(s1)
            s3 = self.vision_encoder.layer3(s2)
            return s1, s2, s3, self.vision_encoder.layer4(s3)
        outputs = self.vision_encoder(x, output_hidden_states=True)
        skips = tuple(self._to_map(outputs.hidden_states[i]) for i in SKIP_LAYERS)
        return skips + (self._to_map(outputs.last_hidden_state),)
