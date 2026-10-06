"""Question-only location and extent tokens over the compact grounding control."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from research.model import ResearchGrounder
from research.losses import resize_logits, segmentation_loss


class EvidenceExtentGrounder(nn.Module):
    def __init__(self, tiny=False, width=None):
        super().__init__()
        width = (16 if tiny else 64) if width is None else width
        if width <= 0 or width % (2 if tiny else 4):
            raise ValueError('Evidence width must be positive and divisible by the attention head count')
        self.coarse = ResearchGrounder('compact', tiny)
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


def support_envelope(target):
    """Bounding rectangle of each annotated region; empty masks stay empty."""
    foreground = target > .5
    rows = foreground.any(dim=3, keepdim=True)
    columns = foreground.any(dim=2, keepdim=True)
    row_span = rows.cumsum(2).bool() & rows.flip([2]).cumsum(2).flip([2]).bool()
    col_span = columns.cumsum(3).bool() & columns.flip([3]).cumsum(3).flip([3]).bool()
    return (row_span & col_span).to(target.dtype)


def area_loss(logits, target):
    probability = resize_logits(logits, target).float().sigmoid()
    return F.mse_loss(probability.flatten(1).mean(1), target.float().flatten(1).mean(1))


def paired_losses(first, second, mask1, mask2, same):
    if same.any() and not torch.equal(mask1[same], mask2[same]):
        raise ValueError('Verified same-region pairs must have identical masks')
    probability1 = resize_logits(first, mask1).float().sigmoid()
    probability2 = resize_logits(second, mask2).float().sigmoid()
    difference = probability2 - probability1
    delta = F.mse_loss(difference, mask2.float() - mask1.float())
    consistency = difference[same].square().mean() if same.any() else difference.sum() * 0
    return delta, consistency


class ExtentObjective:
    def __init__(self, dice_weight=0., support_weight=.2, area_weight=.1,
                 pair_delta_weight=0., pair_consistency_weight=0.):
        self.dice_weight = dice_weight
        self.support_weight = support_weight
        self.area_weight = area_weight
        self.pair_delta_weight = pair_delta_weight
        self.pair_consistency_weight = pair_consistency_weight

    def __call__(self, model, batch):
        n = batch['image'].shape[0]
        questions = list(batch.get('question_text', batch['text']))
        image, target = batch['image'], batch['mask']
        available = batch.get('pair_available', torch.zeros(n, dtype=torch.bool, device=image.device))
        paired_indices = available.nonzero(as_tuple=True)[0]
        if paired_indices.numel():
            questions += [batch['paired_text'][i] for i in paired_indices.tolist()]
            image = torch.cat([image, image[available]])
            target = torch.cat([target, batch['paired_mask'][available]])
        # A single forward is important for DDP reduction and both labels are supervised.
        output = model({'image': image, 'text': questions, 'question_text': questions})
        # Sum each original view and its optional counterpart, normalized by
        # original images. Equal-sized DDP shards then average identically even
        # when they contain different numbers of annotated pairs.
        view_weight = target.shape[0] / n
        segmentation = segmentation_loss(output['logits'], target, self.dice_weight) * view_weight
        support = segmentation_loss(output['location_logits'], support_envelope(target)) * view_weight
        area = area_loss(output['logits'], target) * view_weight
        delta = consistency = output['logits'].sum() * 0
        if paired_indices.numel():
            delta, consistency = paired_losses(output['logits'][:n][available], output['logits'][n:],
                batch['mask'][available], batch['paired_mask'][available], batch['pair_same'][available])
            delta = delta * (paired_indices.numel() / n)
            consistency = consistency * (batch['pair_same'][available].sum() / n)
        loss = (segmentation + self.support_weight * support + self.area_weight * area
                + self.pair_delta_weight * delta + self.pair_consistency_weight * consistency)
        metrics = {'segmentation': segmentation, 'support': support, 'area': area,
                   'pair_delta': delta, 'pair_consistency': consistency}
        return loss, {key: value.detach() for key, value in metrics.items()}
