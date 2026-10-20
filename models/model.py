"""Grounding first, with answer supervision through a local masked RGB reader."""

import torch
from torch import nn
from torch.nn import functional as F

from models.answer_reader import MaskedAnswerReader
from models.image_encoder import ImageEncoder, TinyVision
from models.mask_decoder import CompactDecoder
from models.text_encoder import TextEncoder, masked_mean

EXPERIMENT = "grounding-first-answer-v1"


class QuestionGrounder(nn.Module):
    """Question-conditioned compact/detail decoder with frozen CLIP text features."""

    def __init__(self, tiny=False):
        super().__init__()
        self.image_encoder = TinyVision() if tiny else ImageEncoder()
        self.text_encoder = TextEncoder(tiny)
        visual_dim, text_dim = (
            self.image_encoder.out_channels,
            self.text_encoder.output_dim,
        )
        self.text_proj = nn.Linear(text_dim, visual_dim)
        self.cross_attn = nn.MultiheadAttention(
            visual_dim, 2 if tiny else 8, batch_first=True
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.01))
        self.decoder = CompactDecoder(visual_dim, text_dim, width=16 if tiny else 128)
        self.register_buffer(
            "image_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.text_encoder.eval()
        return self

    def forward(self, image, questions):
        normalized = (image - self.image_mean) / self.image_std
        s1, s2, s3, vision = self.image_encoder(normalized)
        text = self.text_encoder(questions)
        question = masked_mean(text.tokens, text.valid_mask)
        detail = self.decoder.detail_stem(normalized)
        query = vision.flatten(2).transpose(1, 2)
        projected = self.text_proj(text.tokens)
        query = query.to(projected.dtype)
        attended, _ = self.cross_attn(
            query,
            projected,
            projected,
            key_padding_mask=~text.attention_mask,
            need_weights=False,
        )
        fused = (
            (query + self.residual_scale.to(query.dtype) * attended)
            .transpose(1, 2)
            .reshape_as(vision)
        )
        logits = self.decoder(fused, s3, s2, s1, question, detail_features=detail)
        return logits, question


class GroundingModel(nn.Module):
    def __init__(self, tiny=False):
        super().__init__()
        self.coarse = QuestionGrounder(tiny)
        self.reader = MaskedAnswerReader(
            self.coarse.text_encoder.output_dim, width=16 if tiny else 64
        )
        self.experiment_config = {
            "experiment": EXPERIMENT,
            "tiny": tiny,
            "architecture": "compact-detail",
            "text_mode": "question",
            "freeze_text_encoder": True,
            "spatial_evidence": False,
            "detail_upsample_stages": [2, 3],
            "detail_dim": 16,
            "reader_width": 16 if tiny else 64,
            "reader_receptive_field": 5,
            "reader_protocol": "gt-trained-local-rgb-detached-predicted-read-v1",
        }

    def forward(self, batch):
        questions = (
            batch["question_text"] if "question_text" in batch else batch["text"]
        )
        logits, question = self.coarse(batch["image"], questions)
        output = {"logits": logits}
        if not self.training and not batch.get("read_answer", False):
            return output
        local = self.reader.encode(batch["image"])
        output["question_vec"] = question.detach()
        probability = F.interpolate(
            logits.float(), size=local.shape[-2:], mode="bilinear", align_corners=False
        ).sigmoid()
        output["predicted_answer"] = self.reader.read(
            local, probability, question, frozen=True
        )
        if "mask" in batch:
            truth = F.interpolate(
                batch["mask"].float(), size=local.shape[-2:], mode="nearest"
            )
            output["gt_region_answer"] = self.reader.read(local, truth, question)
            output["region_nonempty"] = truth.flatten(1).sum(1) > 0
        if "answer_only_text" in batch:
            with torch.no_grad():
                text = self.coarse.text_encoder(batch["answer_only_text"])
                output["target_answer"] = masked_mean(text.tokens, text.valid_mask)
        return output

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        from utils import checkpoint_model_kwargs, load_model_weights, read_checkpoint

        saved = read_checkpoint(path)
        model = cls(**checkpoint_model_kwargs(saved))
        load_model_weights(model, saved)
        return model.to(device).eval()
