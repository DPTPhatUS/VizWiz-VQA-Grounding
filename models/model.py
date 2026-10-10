"""Question-conditioned grounding models."""
import math
import torch
from torch import nn
from models.image_encoder import ImageEncoder, TinyVision
from models.text_encoder import TextEncoder
from models.mask_decoder import CompactDecoder

from losses import resize_logits

class CompactGroundingModel(nn.Module):
    def __init__(self, tiny=False):
        super().__init__()
        self.image_encoder = TinyVision() if tiny else ImageEncoder()
        self.text_encoder = TextEncoder(tiny)
        self.visual_dim = self.image_encoder.out_channels
        self.text_dim = self.text_encoder.output_dim
        self.text_proj = nn.Linear(self.text_dim, self.visual_dim)
        self.cross_attn = nn.MultiheadAttention(self.visual_dim, 2 if tiny else 8, batch_first=True)
        self.residual_scale = nn.Parameter(torch.tensor(.01))
        self.decoder = CompactDecoder(self.visual_dim, self.text_dim, width=16 if tiny else 128)
        self.register_buffer("image_mean", torch.tensor([.48145466, .4578275, .40821073]).view(1,3,1,1))
        self.register_buffer("image_std", torch.tensor([.26862954, .26130258, .27577711]).view(1,3,1,1))
        self.experiment_config = {"experiment": "controls", "architecture": "compact", "tiny": tiny,
                                  "protocol": "normalized-masked-question-v1"}

    def encode_image(self, image):
        return self.image_encoder((image - self.image_mean) / self.image_std)

    def decode(self, features, texts):
        s1, s2, s3, vision = features
        text = self.text_encoder(texts)
        query = vision.flatten(2).transpose(1, 2)
        projected = self.text_proj(text.tokens)
        attended, _ = self.cross_attn(query, projected, projected,
            key_padding_mask=~text.attention_mask, need_weights=False)
        fused = query + self.residual_scale * attended
        fused = fused.transpose(1,2).reshape_as(vision)
        logits = self.decoder(fused, s3, s2, s1, text)
        return {"logits": logits, "visual": fused, "text_tokens": text.tokens,
                "text_mask": text.attention_mask}

    def forward(self, batch):
        return self.decode(self.encode_image(batch["image"]), batch["text"])


EXPERIMENT = 'evidence-extent'


class GroundingModel(nn.Module):
    def __init__(self, tiny=False, width=None):
        super().__init__()
        width = (16 if tiny else 64) if width is None else width
        if width <= 0 or width % (2 if tiny else 4):
            raise ValueError('Evidence width must be positive and divisible by the attention head count')
        self.coarse = CompactGroundingModel(tiny=tiny)
        self.query_tokens = nn.Parameter(torch.randn(2, width) * .02)
        self.text_projection = nn.Linear(self.coarse.text_dim, width)
        self.visual_projection = nn.Conv2d(self.coarse.visual_dim, width, 1)
        self.question_attention = nn.MultiheadAttention(width, 2 if tiny else 4, batch_first=True)
        self.visual_attention = nn.MultiheadAttention(width, 2 if tiny else 4, batch_first=True)
        self.question_norm = nn.LayerNorm(width)
        self.visual_norm = nn.LayerNorm(width)
        self.token_bias = nn.Linear(width, 1)
        self.experiment_config = dict(self.coarse.experiment_config, experiment='evidence-extent',
                                      token_width=width, envelope='bounding-rectangle-v1')

    def forward(self, batch):
        # Explicit question_text takes precedence over any privileged training view.
        questions = batch.get('question_text', batch.get('text'))
        output = self.coarse({'image': batch['image'], 'text': questions})
        visual = self.visual_projection(output['visual'])
        visual_tokens = visual.flatten(2).transpose(1, 2)
        text = self.text_projection(output['text_tokens'])
        tokens = self.query_tokens.unsqueeze(0).expand(visual.shape[0], -1, -1)
        attended, _ = self.question_attention(tokens, text, text,
            key_padding_mask=~output['text_mask'], need_weights=False)
        tokens = self.question_norm(tokens + attended)
        attended, _ = self.visual_attention(tokens, visual_tokens, visual_tokens, need_weights=False)
        tokens = self.visual_norm(tokens + attended)
        maps = torch.einsum('bkc,bchw->bkhw', tokens, visual) / math.sqrt(visual.shape[1])
        maps = maps + self.token_bias(tokens).unsqueeze(-1)
        location, extent = maps[:, :1], maps[:, 1:]
        logits = output['logits'] + resize_logits(location + extent, output['logits'])
        return {'logits': logits, 'location_logits': location, 'extent_residual': extent}
