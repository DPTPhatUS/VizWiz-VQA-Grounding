"""Question-only grounding with a high-resolution detail pyramid."""

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from models.detail_pyramid import DetailPyramid, HighResolutionRefiner
from models.image_encoder import ImageEncoder, TinyVision
from models.mask_decoder import DetailDecoder
from models.region_grounding import RegionRanker, TokenPixelInteraction
from models.text_encoder import TextEncoder, masked_mean

EXPERIMENT = "pyramid-token-region-ranking-v1"


class GroundingModel(nn.Module):
    """336 CLIP view + 768 CNN view + Q -> 768 logits; never consumes an answer."""

    def __init__(
        self,
        tiny: bool = False,
        gradient_checkpointing: bool = False,
        token_interaction: bool = True,
        region_ranking: bool = True,
    ):
        super().__init__()
        self.image_encoder = TinyVision() if tiny else ImageEncoder()
        self.text_encoder = TextEncoder(tiny)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        visual_dim, text_dim = (
            self.image_encoder.out_channels,
            self.text_encoder.output_dim,
        )
        self.text_proj = nn.Linear(text_dim, visual_dim)
        self.cross_attn = nn.MultiheadAttention(
            visual_dim, 2 if tiny else 8, batch_first=True
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.01))
        self.question_projection = nn.Sequential(
            nn.Linear(text_dim, 8 if tiny else 16), nn.GELU()
        )
        self.detail_pyramid = DetailPyramid(tiny=tiny)
        self.decoder = DetailDecoder(visual_dim, text_dim, tiny=tiny)
        self.refiner = HighResolutionRefiner(
            self.decoder.output_channels, 8 if tiny else 16, tiny=tiny
        )
        self.register_buffer(
            "image_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )
        self.gradient_checkpointing = gradient_checkpointing
        self.image_size, self.detail_size = (32, 64) if tiny else (336, 768)
        # Add modules after all parent modules: common initial weights stay matched across ablations.
        if token_interaction:
            self.decoder.token_interaction = TokenPixelInteraction(
                self.decoder.stage96_channels, text_dim, 16 if tiny else 64
            )
        self.region_ranker = (
            RegionRanker(self.decoder.stage96_channels, text_dim, 16 if tiny else 64)
            if region_ranking
            else None
        )
        self.register_buffer("token_region_version", torch.tensor(1, dtype=torch.int64))
        if gradient_checkpointing and not tiny:
            self.image_encoder.vision_encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.experiment_config = {
            "experiment": EXPERIMENT,
            "tiny": tiny,
            "image_size": self.image_size,
            "detail_size": self.detail_size,
            "text_mode": "question",
            "freeze_text_encoder": True,
            "decoder_width": 16 if tiny else 128,
            "skip_hidden": 32 if tiny else 256,
            "detail_width": 8 if tiny else 16,
            "upsample_convolution": "standard-3x3",
            "detail_stages": [0, 1, 2, 3],
            "auxiliary_heads": [96, 192, 384],
            "refiner": "signed-logit-residual",
            "normalization": "clip-rgb-both-views",
            "token_interaction": token_interaction,
            "token_width": (16 if tiny else 64) if token_interaction else None,
            "token_stage": 96 if token_interaction else None,
            "region_ranking": region_ranking,
            "region_mining": "translated-gt-soft-support-halo1-grid5-peaks8-top3"
            if region_ranking
            else None,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_encoder.eval()
        return self

    def _run(self, block: nn.Module, *inputs: Tensor) -> Tensor:
        if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(block, *inputs, use_reentrant=False)
        return block(*inputs)

    def forward(self, batch: dict) -> dict:
        image, detail_image = batch["image"], batch["detail_image"]
        if image.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(
                f"CLIP view must have shape (B, 3, {self.image_size}, {self.image_size})"
            )
        if detail_image.shape != (
            image.shape[0],
            3,
            self.detail_size,
            self.detail_size,
        ):
            raise ValueError(
                f"Detail view must have shape (B, 3, {self.detail_size}, {self.detail_size})"
            )
        features = self.image_encoder((image - self.image_mean) / self.image_std)
        questions = (
            batch["question_text"] if "question_text" in batch else batch["text"]
        )
        text = self.text_encoder(questions)
        question = masked_mean(text.tokens, text.valid_mask)
        projected = self.text_proj(text.tokens)
        tokens = features[-1].flatten(2).transpose(1, 2).to(projected.dtype)
        attended, _ = self.cross_attn(
            tokens,
            projected,
            projected,
            key_padding_mask=~text.attention_mask,
            need_weights=False,
        )
        fused = (
            (tokens + self.residual_scale * attended)
            .transpose(1, 2)
            .reshape_as(features[-1])
        )
        detail = self.detail_pyramid(
            (detail_image - self.image_mean) / self.image_std, self._run
        )
        detail_question = self.question_projection(question)
        logits, semantic, auxiliary, region_feature = self.decoder(
            (*features[:3], fused),
            question,
            detail,
            detail_question,
            self._run,
            text_tokens=text.tokens,
            attention_mask=text.attention_mask,
        )
        final = self._run(self.refiner, logits, semantic, detail[-1], detail_question)
        output = {"logits": final, "auxiliary_logits": auxiliary}
        if self.training and self.region_ranker is not None:
            output["region_scores"] = self.region_ranker(
                region_feature, question, final, batch["mask"]
            )
        return output

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        from utils import checkpoint_model_kwargs, load_model_weights, read_checkpoint

        saved = read_checkpoint(path)
        model = cls(**checkpoint_model_kwargs(saved))
        load_model_weights(model, saved)
        return model.to(device).eval()
