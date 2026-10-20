import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel


class ImageEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14-336"):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(model_name)
        # This layer normalizes pooled CLS only, not the patch tokens used below.
        # Keep its checkpoint keys but exclude it from optimization and DDP reduction.
        vision = getattr(self.vision_encoder, "vision_model", self.vision_encoder)
        vision.post_layernorm.requires_grad_(False)
        self.out_channels = self.vision_encoder.config.hidden_size

    def forward(self, x):
        outputs = self.vision_encoder(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # intermediate layers + final bottleneck
        enc_feat1 = hidden_states[4]  # (B, 1 + N, D_v)
        enc_feat2 = hidden_states[7]  # (B, 1 + N, D_v)
        enc_feat3 = hidden_states[9]  # (B, 1 + N, D_v)
        bottleneck = outputs.last_hidden_state  # (B, 1 + N, D_v)

        def reshape_feat(feat):
            B, _, D = feat.shape
            feat = feat[:, 1:, :]  # (B, N, D_v), drop CLS token
            num_patches = feat.shape[1]
            grid_size = int(num_patches**0.5)
            if grid_size * grid_size != num_patches:
                raise ValueError(
                    f"Expected square patch grid, got {num_patches} patches"
                )
            return (
                feat.view(B, grid_size, grid_size, D).permute(0, 3, 1, 2).contiguous()
            )  # (B, D_v, H_p, W_p)

        return tuple(map(reshape_feat, [enc_feat1, enc_feat2, enc_feat3])) + (
            reshape_feat(bottleneck),
        )


class TinyVision(nn.Module):
    out_channels = 16

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)

    def forward(self, image):
        feature = F.adaptive_avg_pool2d(self.conv(image), (2, 2))  # (B, 16, 2, 2)
        return feature, feature, feature, feature
