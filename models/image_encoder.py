import torch
from torch import nn
from transformers import (
    CLIPVisionConfig,
    CLIPVisionModel,
    SiglipVisionConfig,
    SiglipVisionModel,
)

# hidden_states[0] is the patch embedding; hidden_states[k] is the output of transformer block k.
SKIP_LAYERS = (4, 7, 9)


class ImageEncoder(nn.Module):
    """Vision transformer returning skip features (blocks 4, 7, 9) and the last block, as maps.

    config: build a randomly initialised encoder from this config instead of loading pretrained
    weights (offline tests only).
    """

    def __init__(self, model_name="openai/clip-vit-large-patch14-336",
                 config: CLIPVisionConfig | SiglipVisionConfig | None = None, *, revision: str | None = None):
        super().__init__()
        options = {"revision": revision} if revision else {}
        is_siglip = isinstance(config, SiglipVisionConfig) or model_name.startswith("google/siglip")
        if is_siglip:
            self.vision_encoder = (SiglipVisionModel(config) if config is not None else
                                   SiglipVisionModel.from_pretrained(model_name, **options))
            # SigLIP's multihead attention pooling head only computes pooler_output, which is never
            # used for dense grounding maps. Replacing it with Identity keeps DDP unused-param free.
            self.vision_encoder.use_head = False
            self.vision_encoder.head = nn.Identity()
            self.has_cls_token = False
        else:
            self.vision_encoder = (CLIPVisionModel(config) if config is not None else
                                   CLIPVisionModel.from_pretrained(model_name, **options))
            self.has_cls_token = True
        # post_layernorm is frozen by TrainingControls across vision backbones so every parameter with
        # requires_grad=True participates in the loss graph (DDP find_unused_parameters=False).
        for name, module in self.vision_encoder.named_modules():
            if name.endswith("post_layernorm"):
                module.requires_grad_(False)
        self.out_channels = self.vision_encoder.config.hidden_size

    def _to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, [1 +] N, D) tokens -> (B, D, H_p, W_p) patch map."""
        patches = tokens[:, 1:] if self.has_cls_token else tokens  # (B, N, D)
        side = int(patches.shape[1] ** 0.5)
        if side * side != patches.shape[1]:
            raise ValueError(f"Expected a square patch grid, got {patches.shape[1]} patches")
        return patches.unflatten(1, (side, side)).permute(0, 3, 1, 2).contiguous()  # (B, D, H_p, W_p)

    def forward(self, x: torch.Tensor):
        """x: (B, 3, H, W) normalised images -> (s1, s2, s3, last), each (B, D_v, H_p, W_p)."""
        outputs = self.vision_encoder(x, output_hidden_states=True)
        skips = tuple(self._to_map(outputs.hidden_states[i]) for i in SKIP_LAYERS)
        return skips + (self._to_map(outputs.last_hidden_state),)
