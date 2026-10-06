"""Branch architecture and checkpoint-driven model construction."""
import math
import torch
from torch import nn
from models.backbone import BaseGroundingModel
from losses import resize_logits

EXPERIMENT = 'evidence-extent'


class GroundingModel(nn.Module):
    def __init__(self, tiny=False, width=None):
        super().__init__()
        width = (16 if tiny else 64) if width is None else width
        if width <= 0 or width % (2 if tiny else 4):
            raise ValueError('Evidence width must be positive and divisible by the attention head count')
        self.coarse = BaseGroundingModel(tiny=tiny)
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


def build_model(args=None):
    return GroundingModel(tiny=getattr(args, 'tiny', False))
