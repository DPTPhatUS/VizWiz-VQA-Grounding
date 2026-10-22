"""Question-conditioned context exchange and directed mask membership."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F


def coordinates(height, width, device, dtype):
    y, x = torch.meshgrid(torch.linspace(-1, 1, height, device=device, dtype=dtype),
                          torch.linspace(-1, 1, width, device=device, dtype=dtype), indexing='ij')
    return torch.stack((y, x), -1).reshape(-1, 2)


def neighborhood(height, width, device):
    """Eight directed neighbor edges per receiver, with explicit border validity."""
    y, x = torch.meshgrid(torch.arange(height, device=device),
                          torch.arange(width, device=device), indexing='ij')
    offsets = torch.tensor([(-1, -1), (-1, 0), (-1, 1), (0, -1),
                            (0, 1), (1, -1), (1, 0), (1, 1)], device=device)
    sy = y.flatten()[:, None] + offsets[:, 0]
    sx = x.flatten()[:, None] + offsets[:, 1]
    valid = (sy >= 0) & (sy < height) & (sx >= 0) & (sx < width)
    indices = sy.clamp(0, height - 1) * width + sx.clamp(0, width - 1)
    return indices, valid


class ConnectionNetwork(nn.Module):
    """The entire token-attention and pair-feature network is shared or untied."""
    def __init__(self, channels, text_dim, width, pair_width):
        super().__init__()
        self.visual = nn.Linear(channels, width)
        self.position = nn.Linear(2, width)
        self.text = nn.Linear(text_dim, width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.receiver = nn.Linear(width, pair_width)
        self.sender = nn.Linear(width, pair_width)
        self.relative = nn.Linear(3, pair_width)

    def condition(self, feature, text):
        height, width = feature.shape[-2:]
        positions = coordinates(height, width, feature.device, feature.dtype)
        query = self.visual(feature.flatten(2).transpose(1, 2)) + self.position(positions)
        tokens = self.text(text.tokens)
        attended, _ = self.attention(query, tokens, tokens,
                                    key_padding_mask=~text.attention_mask, need_weights=False)
        return self.norm(query + attended), positions

    def pairs(self, receiver, sender, delta):
        geometry = torch.cat((delta, delta.square().sum(-1, keepdim=True)), -1)
        return F.gelu(receiver + sender + self.relative(geometry))

    def global_pairs(self, conditioned, positions):
        # Receiver i reads from sender j.
        delta = positions[None, :, :] - positions[:, None, :]
        return self.pairs(self.receiver(conditioned)[:, :, None],
                          self.sender(conditioned)[:, None, :], delta)

    def local_pairs(self, conditioned, positions, indices):
        delta = positions[indices] - positions[:, None]
        return self.pairs(self.receiver(conditioned)[:, :, None],
                          self.sender(conditioned)[:, indices], delta)


class ContextMembership(nn.Module):
    def __init__(self, channels, text_dim, mode, width=128, pair_width=16, steps=2):
        super().__init__()
        if mode not in ('shared', 'separate'):
            raise ValueError('connection mode must be shared or separate')
        self.mode, self.steps = mode, steps
        self.context_network = ConnectionNetwork(channels, text_dim, width, pair_width)
        # No duplicate registered alias in shared mode: one actual parameter set.
        # Independent parameters start equal, without consuming additional RNG.
        # With the same seed, both modes begin with identical common weights/output.
        self.membership_network = deepcopy(self.context_network) if mode == 'separate' else None
        self.context_head = nn.Linear(pair_width, 1)
        self.membership_head = nn.Linear(pair_width, 3)
        self.context_value = nn.Conv2d(channels, channels, 1)
        self.context_gate = nn.Conv2d(channels, 1, 1)
        self.unary_text = nn.Linear(text_dim, channels)
        self.unary_attention = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.unary_norm = nn.LayerNorm(channels)
        self.unary_head = nn.Conv2d(channels, 1, 1)
        self.update_strength = nn.Parameter(torch.zeros(2))

    def forward(self, feature, text):
        batch, channels, height, width = feature.shape
        # Full-image context at the CLIP grid; membership at twice its resolution.
        context_feature = F.avg_pool2d(feature, 2)
        conditioned, positions = self.context_network.condition(context_feature, text)
        pairs = self.context_network.global_pairs(conditioned, positions)
        read_weights = self.context_head(pairs).squeeze(-1).float().softmax(-1)
        values = self.context_value(context_feature).flatten(2).transpose(1, 2)
        context = (read_weights.to(values.dtype) @ values).transpose(1, 2).reshape_as(context_feature)
        context = F.interpolate(context, size=(height, width), mode='bilinear', align_corners=False)
        contextual = feature + self.context_gate(feature).sigmoid() * context

        network = self.context_network if self.mode == 'shared' else self.membership_network
        conditioned, positions = network.condition(feature, text)
        indices, valid = neighborhood(height, width, feature.device)
        pairs = network.local_pairs(conditioned, positions, indices)
        edge_logits = self.membership_head(pairs)
        weights = edge_logits.float().softmax(-1) * valid[None, :, :, None]

        query = contextual.flatten(2).transpose(1, 2)
        tokens = self.unary_text(text.tokens)
        attended, _ = self.unary_attention(query, tokens, tokens,
                                          key_padding_mask=~text.attention_mask, need_weights=False)
        unary_features = self.unary_norm(query + attended).transpose(1, 2).reshape_as(feature)
        unary = self.unary_head(unary_features)
        logits = unary.float().flatten(1)
        plus, minus = 2 * self.update_strength.sigmoid()
        degree = valid.sum(-1).clamp_min(1)
        for _ in range(self.steps):
            senders = logits.sigmoid()[:, indices]
            include = (weights[..., 0] * senders).sum(-1) / degree
            exclude = (weights[..., 1] * senders).sum(-1) / degree
            logits = unary.float().flatten(1) + plus * include - minus * exclude
        return {
            'features': unary_features,
            'membership_logits': logits.reshape(batch, 1, height, width),
            'unary_logits': unary,
            'edge_logits': edge_logits,
            'edge_indices': indices,
            'edge_valid': valid,
        }
