"""Question-conditioned grounding models."""
import torch
from torch import nn
from models.image_encoder import ImageEncoder, TinyVision
from models.text_encoder import TextEncoder
from models.mask_decoder import CompactDecoder

EXPERIMENT = 'context-membership'


class GroundingModel(nn.Module):
    def __init__(self, tiny=False, connection_mode='shared'):
        super().__init__()
        self.image_encoder = TinyVision() if tiny else ImageEncoder()
        self.text_encoder = TextEncoder(tiny).requires_grad_(False)
        self.visual_dim = self.image_encoder.out_channels
        self.text_dim = self.text_encoder.output_dim
        self.text_proj = nn.Linear(self.text_dim, self.visual_dim)
        self.cross_attn = nn.MultiheadAttention(self.visual_dim, 2 if tiny else 8, batch_first=True)
        self.decoder = CompactDecoder(self.visual_dim, self.text_dim, width=16 if tiny else 128,
                                      connection_mode=connection_mode, tiny=tiny)
        self.register_buffer("image_mean", torch.tensor([.48145466, .4578275, .40821073]).view(1,3,1,1))
        self.register_buffer("image_std", torch.tensor([.26862954, .26130258, .27577711]).view(1,3,1,1))
        self.experiment_config = {
            'experiment': EXPERIMENT, 'architecture': 'compact', 'tiny': tiny,
            'connection_mode': connection_mode, 'connection_width': 16 if tiny else 128,
            'pair_width': 16, 'membership_steps': 2, 'edge_neighborhood': 8,
            'freeze_text_encoder': True, 'skip_film': True, 'residual_fusion': False,
            'protocol': 'context-membership-v1',
        }

    def encode_image(self, image):
        return self.image_encoder((image - self.image_mean) / self.image_std)

    def decode(self, features, texts):
        s1, s2, s3, vision = features
        text = self.text_encoder(texts)
        query = vision.flatten(2).transpose(1, 2)
        projected = self.text_proj(text.tokens)
        attended, _ = self.cross_attn(query, projected, projected,
            key_padding_mask=~text.attention_mask, need_weights=False)
        fused = attended.transpose(1,2).reshape_as(vision)
        return self.decoder(fused, s3, s2, s1, text)

    def forward(self, batch):
        return self.decode(self.encode_image(batch["image"]), batch["text"])
