import torch.nn.functional as F
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14-336", pretrained=True):
        super().__init__()
        from transformers import CLIPVisionModel
        if pretrained:
            self.vision_encoder = CLIPVisionModel.from_pretrained(model_name)
        else:
            from transformers import CLIPVisionConfig
            self.vision_encoder = CLIPVisionModel(CLIPVisionConfig(
                hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
                num_attention_heads=16, image_size=336, patch_size=14, projection_dim=768))
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
            feat = feat[:, 1:, :].transpose(1, 2)  # (B, D, seq_len)
            patch_size = int((feat.shape[-1]) ** 0.5)
            return feat.view(feat.shape[0], feat.shape[1], patch_size, patch_size)

        return tuple(map(reshape_feat, [enc_feat1, enc_feat2, enc_feat3])) + (reshape_feat(bottleneck),)


class TinyVision(nn.Module):
    out_channels = 16
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
    def forward(self, image):
        feature = F.adaptive_avg_pool2d(self.conv(image), (2, 2))
        return feature, feature, feature, feature
