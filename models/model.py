import torch
import torch.nn as nn
from models import ImageEncoder, TextEncoder, UNetDecoder
from models.experiment import SkipConditioner

class GroundingModel(nn.Module):
    def __init__(self, n_heads=8):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

        # fixed based on image hidden dim
        self.hidden_dim = self.image_encoder.out_channels  # typically 768

        # project text dim to match hidden dim
        self.text_proj = nn.Linear(self.text_encoder.output_dim, self.hidden_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=n_heads, batch_first=True)
        self.decoder = UNetDecoder(in_channels=self.hidden_dim)
        self.residual_scale = nn.Parameter(torch.tensor(0.01))
        self.experiment_config = {"architecture": "separate-qa-skips"}
        self.skip_conditioner = SkipConditioner(self.text_encoder.output_dim, self.hidden_dim, separate=True)

    def forward(self, image, text):
        enc_feat1, enc_feat2, enc_feat3, bottleneck = self.image_encoder(image)
        B, D, H, W = bottleneck.shape
        img_tokens = bottleneck.flatten(2).permute(0, 2, 1)  # (B, N, D)

        text_features = self.text_encoder(text, return_features=True)
        text_tokens = text_features.tokens
        text_tokens = self.text_proj(text_tokens)          # align to (B, L, D)

        attn_output, _ = self.cross_attn(query=img_tokens, key=text_tokens, value=text_tokens)
        fused_tokens = img_tokens + self.residual_scale * attn_output
        fused = fused_tokens.permute(0, 2, 1).view(B, D, H, W)

        enc_feat3, enc_feat2, enc_feat1 = self.skip_conditioner(
            [enc_feat3, enc_feat2, enc_feat1], text_features
        )
        output = self.decoder(fused, enc_feat3, enc_feat2, enc_feat1)
        return output
