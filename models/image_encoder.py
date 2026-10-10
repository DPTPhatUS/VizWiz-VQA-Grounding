import torch
from torch import nn
from transformers import CLIPVisionConfig, CLIPVisionModel


class ImageEncoder(nn.Module):
    def __init__(
        self, model_name="openai/clip-vit-large-patch14-336", pretrained=True
    ):
        super().__init__()
        if pretrained:
            self.vision_encoder = CLIPVisionModel.from_pretrained(model_name)
        else:
            config = CLIPVisionConfig.from_pretrained(model_name)
            self.vision_encoder = CLIPVisionModel(config)

        # post_layernorm only applies to pooled_output ([CLS]), which is discarded below.
        # Freezing it prevents unused-parameter errors in DistributedDataParallel.
        self.vision_encoder.vision_model.post_layernorm.requires_grad_(False)
        self.out_channels = self.vision_encoder.config.hidden_size

    def forward(self, x):
        outputs = self.vision_encoder(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # intermediate layers + final bottleneck
        enc_feat1 = hidden_states[4]  # (B, seq, D)
        enc_feat2 = hidden_states[7]
        enc_feat3 = hidden_states[9]
        bottleneck = outputs.last_hidden_state

        def reshape_feat(feat):
            B, _, D = feat.shape
            feat = feat[:, 1:, :]
            num_patches = feat.shape[1]
            grid_size = int(num_patches**0.5)
            if grid_size * grid_size != num_patches:
                raise ValueError(
                    f"Expected square patch grid, got {num_patches} patches"
                )
            return (
                feat.view(B, grid_size, grid_size, D)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        return tuple(map(reshape_feat, [enc_feat1, enc_feat2, enc_feat3])) + (
            reshape_feat(bottleneck),
        )
