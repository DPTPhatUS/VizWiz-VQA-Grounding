import torch
from torch import nn
from transformers import CLIPVisionConfig, CLIPVisionModel

# hidden_states[0] is the patch embedding; hidden_states[k] is the output of transformer block k.
SKIP_LAYERS = (4, 7, 9)


class ImageEncoder(nn.Module):
    """CLIP vision transformer returning skip features (blocks 4, 7, 9) and the last block, as maps.

    config: build a randomly initialised encoder from this config instead of loading pretrained
    weights (offline tests only).
    """

    def __init__(self, model_name="openai/clip-vit-large-patch14-336", config: CLIPVisionConfig | None = None):
        super().__init__()
        self.vision_encoder = CLIPVisionModel(config) if config is not None else CLIPVisionModel.from_pretrained(model_name)
        # post_layernorm only normalises the pooled CLS output, which is never used. Freezing it keeps
        # every trainable parameter in the graph, so DDP can run with find_unused_parameters=False.
        # Matched by name: transformers 4.x nests it under vision_model, 5.x does not.
        for name, module in self.vision_encoder.named_modules():
            if name.endswith("post_layernorm"):
                module.requires_grad_(False)
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
        """x: (B, 3, H, W) CLIP-normalised images -> (s1, s2, s3, last), each (B, D_v, H_p, W_p)."""
        outputs = self.vision_encoder(x, output_hidden_states=True)
        skips = tuple(self._to_map(outputs.hidden_states[i]) for i in SKIP_LAYERS)
        return skips + (self._to_map(outputs.last_hidden_state),)
