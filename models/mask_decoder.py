"""Text-conditioned compact mask decoding."""
import torch
from numbers import Integral
from torch import nn
from torch.nn import functional as F
from models.text_encoder import masked_mean


class SkipConditioner(nn.Module):
    def __init__(self, text_dim, channels, reduction=64):
        super().__init__()
        self.qa_gate = nn.Linear(2 * text_dim, text_dim)
        self.reduction = nn.Linear(text_dim, reduction)
        self.affine = nn.ModuleList([nn.Linear(reduction, 2 * channels) for _ in range(3)])
        for head in self.affine:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, skips, text, answer_vec=None):
        q = masked_mean(text.tokens, text.valid_mask) if hasattr(text, "tokens") else text  # (B, D_t)
        if answer_vec is None:
            a = q
        elif answer_vec.dim() == 3:
            a = answer_vec.mean(dim=1).to(q.dtype)
        else:
            a = answer_vec.to(q.dtype)
        gate = torch.sigmoid(self.qa_gate(torch.cat([q, a], dim=-1)))  # (B, D_t)
        blended = gate * q + (1 - gate) * a  # (B, D_t)
        condition = F.gelu(self.reduction(blended))  # (B, R)
        output = []
        for feature, head in zip(skips, self.affine):
            gamma, beta = head(condition).chunk(2, dim=-1)  # (B, C), (B, C)
            gamma = gamma.to(feature.dtype)
            beta = beta.to(feature.dtype)
            output.append(feature * (1 + gamma[:, :, None, None]) + beta[:, :, None, None])  # (B, C, H_p, W_p)
        return output


def separable(in_channels, out_channels):
    num_groups = 4 if out_channels % 4 == 0 else 1
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.GroupNorm(num_groups, out_channels), nn.GELU(),
    )


class SpatialEvidenceInteraction(nn.Module):
    """Spatial evidence-decoder interaction inspired by DaVI (Pan et al., CVPR 2022 Workshop).

    Computes per-location cosine compatibility between visual features and answer evidence,
    generating a continuous 2D spatial gate without single-key softmax degeneracy:
        S(b, 1, h, w) = sum_c ( norm(V_proj(b, c, h, w)) * norm(A_proj(b, c)) )
        Gate(b, 1, h, w) = sigmoid(gate_scale * S + gate_bias)

    Modulated visual features are combined with the spatial gate and refined via
    depthwise-separable convolution:
        Delta_V = Refine([V * Gate, Gate])
        V_out = V + residual_scale * Delta_V

    Properties:
    1. residual_scale is initialized to 0 -> strictly identity mapping at step 0.
    2. When answer_vec is None (Q-only view), returns visual_feat untouched.
    3. Position-wise sigmoid gating avoids softmax collapse over a single answer token.
    4. Supports single pooled vectors (B, D_t) and multi-token evidence (B, K, D_t).
    """
    def __init__(self, visual_channels, text_dim, hidden_dim=None):
        super().__init__()
        self.visual_channels = visual_channels
        self.text_dim = text_dim
        hidden_dim = hidden_dim or visual_channels
        self.visual_proj = nn.Conv2d(visual_channels, hidden_dim, kernel_size=1, bias=False)
        self.evidence_proj = nn.Linear(text_dim, hidden_dim, bias=False)
        self.gate_scale = nn.Parameter(torch.tensor(1.0))
        self.gate_bias = nn.Parameter(torch.zeros(1))

        num_groups = 4 if visual_channels % 4 == 0 else 1
        self.refine = nn.Sequential(
            nn.Conv2d(visual_channels + 1, visual_channels + 1, kernel_size=3, padding=1,
                      groups=visual_channels + 1, bias=False),
            nn.Conv2d(visual_channels + 1, visual_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, visual_channels),
            nn.GELU(),
        )
        self.residual_scale = nn.Parameter(torch.zeros(1))
        self.last_gate = None

    def compute_gate(self, visual_feat, answer_vec):
        """Compute continuous 2D spatial evidence gate in (0, 1)."""
        v_proj = self.visual_proj(visual_feat)  # (B, H_dim, H, W)
        v_norm = F.normalize(v_proj, p=2, dim=1, eps=1e-6)

        a_proj = self.evidence_proj(answer_vec.to(dtype=visual_feat.dtype, device=visual_feat.device))
        a_norm = F.normalize(a_proj, p=2, dim=-1, eps=1e-6)

        if a_norm.dim() == 2:
            sim = (v_norm * a_norm[:, :, None, None]).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        elif a_norm.dim() == 3:
            sim = torch.einsum("bchw,bkc->bkhw", v_norm, a_norm).max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
        else:
            raise ValueError(f"Expected answer_vec with 2 or 3 dims, got {answer_vec.shape}")

        gate_logits = self.gate_scale.to(sim.dtype) * sim + self.gate_bias.to(sim.dtype)
        return torch.sigmoid(gate_logits)

    def forward(self, visual_feat, answer_vec=None):
        if answer_vec is None:
            self.last_gate = None
            return visual_feat

        gate = self.compute_gate(visual_feat, answer_vec).to(visual_feat.dtype)
        self.last_gate = gate.detach()

        gated = visual_feat * gate
        cat_feat = torch.cat([gated, gate], dim=1)  # (B, C + 1, H, W)
        delta = self.refine(cat_feat)
        return visual_feat + self.residual_scale.to(delta.dtype) * delta


UPSAMPLE_STAGE_CHANNELS = (64, 32, 16, 16)


def resolve_spatial_upsample_stages(stages):
    if stages is None:
        return ()
    if isinstance(stages, bool):
        raise ValueError("Spatial stages must be integer IDs, not booleans")
    if isinstance(stages, Integral):
        if not 0 <= stages <= len(UPSAMPLE_STAGE_CHANNELS):
            raise ValueError("Spatial stage count must be between 0 and 4")
        stages = range(stages)
    try:
        stages = tuple(stages)
    except TypeError as error:
        raise ValueError("Spatial stages must be a sequence of integer IDs") from error
    if any(isinstance(s, bool) or not isinstance(s, Integral) for s in stages):
        raise ValueError("Spatial stages must contain only integer IDs")
    stages = tuple(int(s) for s in stages)
    for s in stages:
        if s < 0 or s >= len(UPSAMPLE_STAGE_CHANNELS):
            raise ValueError(f"Invalid spatial upsample stage: {s}. Must be in range(0, {len(UPSAMPLE_STAGE_CHANNELS)})")
    if len(set(stages)) != len(stages):
        raise ValueError(f"Duplicate spatial upsample stages: {stages}")
    return tuple(sorted(stages))


class CompactDecoder(nn.Module):
    def __init__(self, in_channels=1024, text_dim=768, width=128, spatial_evidence=True,
                 spatial_upsample_stages=(0, 1)):
        super().__init__()
        self.spatial_evidence_enabled = spatial_evidence
        self.spatial_upsample_stages = resolve_spatial_upsample_stages(spatial_upsample_stages) if spatial_evidence else ()
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, width, 1) for _ in range(4)])
        self.conditioner = SkipConditioner(text_dim, width)
        self.fusion = nn.ModuleList([separable(width, width) for _ in range(3)])
        self.upsample = nn.ModuleList([
            separable(width, 64), separable(64, 32),
            separable(32, 16), separable(16, 16),
        ])
        self.output = nn.Conv2d(16, 1, 1)
        self.register_buffer("spatial_stage_ids", torch.tensor(self.spatial_upsample_stages, dtype=torch.long)
                             if spatial_evidence else None)

        if spatial_evidence and self.spatial_upsample_stages:
            self.spatial_bottleneck = SpatialEvidenceInteraction(width, text_dim)
            self.spatial_upsample = nn.ModuleDict({
                str(i): SpatialEvidenceInteraction(UPSAMPLE_STAGE_CHANNELS[i], text_dim)
                for i in self.spatial_upsample_stages
            })
        elif spatial_evidence:
            self.spatial_bottleneck = SpatialEvidenceInteraction(width, text_dim)
            self.spatial_upsample = None
        else:
            self.spatial_bottleneck = None
            self.spatial_upsample = None

    def project_skips(self, skip3, skip2, skip1):
        return [proj(feat) for proj, feat in zip(self.projections[1:], (skip3, skip2, skip1))]  # 3 x (B, C, H_p, W_p)

    def get_spatial_gates(self):
        """Retrieve recent spatial evidence gate maps for inspection or visualization."""
        if not self.spatial_evidence_enabled or self.spatial_bottleneck is None:
            return []
        gates = []
        if self.spatial_bottleneck.last_gate is not None:
            gates.append(self.spatial_bottleneck.last_gate)
        if self.spatial_upsample is not None:
            for module in self.spatial_upsample.values():
                if module.last_gate is not None:
                    gates.append(module.last_gate)
        return gates

    def forward(self, bottleneck, skip3, skip2, skip1, text, answer_vec=None, *, projected_skips=None):
        x = self.projections[0](bottleneck)  # (B, C, H_p, W_p)
        skips = projected_skips if projected_skips is not None else self.project_skips(skip3, skip2, skip1)  # 3 x (B, C, H_p, W_p)
        skips = self.conditioner(skips, text, answer_vec=answer_vec)  # 3 x (B, C, H_p, W_p)
        for block, skip in zip(self.fusion, skips):
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = x + block(x + skip)  # (B, C, H_p, W_p)

        # Bottleneck spatial evidence interaction
        if self.spatial_bottleneck is not None:
            x = self.spatial_bottleneck(x, answer_vec)

        # Progressive upsampling with multi-level spatial evidence guidance
        for i, block in enumerate(self.upsample):
            x = block(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))  # 2x spatial per stage
            if self.spatial_upsample is not None and i in self.spatial_upsample_stages:
                x = self.spatial_upsample[str(i)](x, answer_vec)

        return self.output(x)  # (B, 1, 16*H_p, 16*W_p)


class UNetDecoder(nn.Module):
    def __init__(self, in_channels=1024, mid_channels=(1024, 512, 256, 128), out_channels=1):
        """
        U-Net based decoder (matched to ImageEncoder output structure)
        in_channels: number of bottleneck and skip channels (e.g. 1024 for ViT-L)
        mid_channels: decoder output channels per stage (1024, 512, 256, 128)
        """
        super().__init__()
        c0, c1, c2, c3 = mid_channels

        # upsample layers
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(in_channels, c0, kernel_size=2, stride=2),
            nn.ConvTranspose2d(c0, c1, kernel_size=2, stride=2),
            nn.ConvTranspose2d(c1, c2, kernel_size=2, stride=2),
            nn.ConvTranspose2d(c2, c3, kernel_size=2, stride=2),
        ])

        # decoder blocks
        self.dec_blocks = nn.ModuleList([
            # stage 1 (c0 + in_channels -> c0)
            nn.Sequential(
                nn.Conv2d(c0 + in_channels, c0, kernel_size=3, padding=1),
                nn.BatchNorm2d(c0),
                nn.ReLU(inplace=True),
                nn.Conv2d(c0, c0, kernel_size=3, padding=1),
                nn.BatchNorm2d(c0),
                nn.ReLU(inplace=True),
            ),
            # stage 2 (c1 + in_channels -> c1)
            nn.Sequential(
                nn.Conv2d(c1 + in_channels, c1, kernel_size=3, padding=1),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c1, c1, kernel_size=3, padding=1),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
            ),
            # stage 3 (c2 + in_channels -> c2)
            nn.Sequential(
                nn.Conv2d(c2 + in_channels, c2, kernel_size=3, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.Conv2d(c2, c2, kernel_size=3, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
            ),
            # stage 4 (no skip connection)
            nn.Sequential(
                nn.Conv2d(c3, c3, kernel_size=3, padding=1),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
                nn.Conv2d(c3, c3, kernel_size=3, padding=1),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
            ),
        ])

        # output layer
        self.final_conv = nn.Conv2d(c3, out_channels, kernel_size=1)

    def forward(self, x, enc_feat3, enc_feat2, enc_feat1):
        # stage 1: x(768) -> 1024 upsample + enc_feat3(1024) concat
        x = self.upconvs[0](x)  # (B, c0, 2*H_p, 2*W_p)
        if x.shape[2:] != enc_feat3.shape[2:]:
            enc_feat3 = F.interpolate(enc_feat3, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, enc_feat3], dim=1)  # (B, c0 + D_v, 2*H_p, 2*W_p)
        x = self.dec_blocks[0](x)  # (B, c0, 2*H_p, 2*W_p)

        # stage 2
        x = self.upconvs[1](x)  # (B, c1, 4*H_p, 4*W_p)
        if x.shape[2:] != enc_feat2.shape[2:]:
            enc_feat2 = F.interpolate(enc_feat2, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, enc_feat2], dim=1)  # (B, c1 + D_v, 4*H_p, 4*W_p)
        x = self.dec_blocks[1](x)  # (B, c1, 4*H_p, 4*W_p)

        # stage 3
        x = self.upconvs[2](x)  # (B, c2, 8*H_p, 8*W_p)
        if x.shape[2:] != enc_feat1.shape[2:]:
            enc_feat1 = F.interpolate(enc_feat1, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, enc_feat1], dim=1)  # (B, c2 + D_v, 8*H_p, 8*W_p)
        x = self.dec_blocks[2](x)  # (B, c2, 8*H_p, 8*W_p)

        # stage 4 (no skip)
        x = self.upconvs[3](x)  # (B, c3, 16*H_p, 16*W_p)
        x = self.dec_blocks[3](x)  # (B, c3, 16*H_p, 16*W_p)

        return self.final_conv(x)  # (B, out_channels, 16*H_p, 16*W_p)
