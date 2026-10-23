"""Question-only grounding with a high-resolution detail pyramid."""

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from models.answer_head import AnswerHead
from models.detail_pyramid import DetailPyramid, HighResolutionRefiner
from models.image_encoder import ImageEncoder, TinyVision
from models.mask_decoder import DetailDecoder
from models.text_encoder import TextEncoder, masked_mean

EXPERIMENT = "answer-token-supervision-v1"


class GroundingModel(nn.Module):
    """336 CLIP view + 768 CNN view + Q -> 768 logits; never consumes an answer."""

    def __init__(self, tiny: bool = False, gradient_checkpointing: bool = False):
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
            "answer_supervision": "gt-region-autoregressive-utf8",
            "answer_feature_scale": 192,
            "answer_max_bytes": 64,
        }
        # Construct after the parent to preserve its seeded initialization.
        self.answer_head = AnswerHead(12 if tiny else 32, tiny=tiny)

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
        logits, semantic, auxiliary, answer_feature = self.decoder(
            (*features[:3], fused), question, detail, detail_question, self._run
        )
        final = self._run(self.refiner, logits, semantic, detail[-1], detail_question)
        output = {"logits": final, "auxiliary_logits": auxiliary}
        if self.training and batch.get("answer_supervision", False):
            output.update(
                self.answer_head(answer_feature, batch["mask"], batch["answer_text"])
            )
        if batch.get("inspect_answer", False):
            output["answer_features"] = answer_feature
        return output

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        from utils import checkpoint_model_kwargs, load_model_weights, read_checkpoint

        saved = read_checkpoint(path)
        model = cls(**checkpoint_model_kwargs(saved))
        load_model_weights(model, saved)
        return model.to(device).eval()
