import torch
import torch.nn as nn
from models import ImageEncoder, TextEncoder, UNetDecoder

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

    def forward(self, image, text):
        img_feat = self.image_encoder(image)  # (B, D, H, W)
        B, D, H, W = img_feat.shape
        img_tokens = img_feat.flatten(2).permute(0, 2, 1)  # (B, N, D)

        text_tokens = self.text_encoder(text)              # (B, L, D_text)
        text_tokens = self.text_proj(text_tokens)          # align to (B, L, D)

        attn_output, _ = self.cross_attn(query=img_tokens, key=text_tokens, value=text_tokens)
        fused = attn_output.permute(0, 2, 1).view(B, D, H, W)

        output = self.decoder(fused)
        return output
