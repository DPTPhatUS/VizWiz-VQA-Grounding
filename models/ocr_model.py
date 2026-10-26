"""Frozen pyramid context plus a portable OCR evidence adapter; no answer inputs."""

import torch
from torch import nn

from models.model import GroundingModel
from models.ocr_evidence import OCREvidenceAdapter
from models.region_ops import PATCH_SIZE
from models.text_encoder import masked_mean
from utils import checkpoint_model_kwargs, load_model_weights

EXPERIMENT = "ocr-evidence-pyramid-v1"


class OCRGroundingModel(nn.Module):
    def __init__(
        self, parent, *, evidence_mode="full", ocr_dropout=0.2, text_dropout=0.2
    ):
        super().__init__()
        if evidence_mode not in {"full", "geometry"}:
            raise ValueError("Evidence mode must be full or geometry")
        if not 0 <= ocr_dropout < 1 or not 0 <= text_dropout < 1:
            raise ValueError("Dropout probabilities must be in [0,1)")
        self.parent = parent.requires_grad_(False).eval()
        tiny = parent.experiment_config["tiny"]
        self.adapter = self._make_adapter(
            parent.text_encoder.output_dim, 8 if tiny else 16, 32 if tiny else 128
        )
        self.evidence_mode, self.ocr_dropout, self.text_dropout = (
            evidence_mode,
            ocr_dropout,
            text_dropout,
        )
        self.image_size, self.detail_size = parent.image_size, parent.detail_size
        self.experiment_config = {
            "experiment": EXPERIMENT,
            "tiny": tiny,
            "evidence_mode": evidence_mode,
            "ocr_dropout": ocr_dropout,
            "text_dropout": text_dropout,
            "parent_experiment": parent.experiment_config,
            "adapter_width": 32 if tiny else 128,
            "patch_size": list(PATCH_SIZE),
            "support_context": 1.25,
        }
        self.inference_view = "full"

    def _make_adapter(self, text_dim, detail_dim, width):
        return OCREvidenceAdapter(text_dim, detail_dim, width)

    def _decode_parent(self, batch, features, question):
        return self.parent.decode_features(
            batch, features, question, return_detail=True
        )

    def _adapter_features(self, base, batch):
        return {"detail": base["detail_features"]}

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()
        return self

    def forward(self, batch):
        with torch.no_grad(), torch.autocast(batch["image"].device.type, enabled=False):
            features, question = self.parent.encode_inputs(batch)
            base = self._decode_parent(batch, features, question)
        if not self.training and self.inference_view == "base":
            return {"logits": base["logits"]}
        valid = batch["ocr_valid"].clone()
        b, n = valid.shape
        if self.training:
            valid &= torch.rand(b, 1, device=valid.device) >= self.ocr_dropout
        elif self.inference_view == "missing":
            valid.zero_()
        boxes = batch["ocr_boxes"] * valid[..., None]
        recognized = question.tokens.new_zeros(b * n, question.tokens.shape[-1])
        texts = [text for row in batch["ocr_texts"] for text in row]
        keep = valid.flatten().clone()
        if self.evidence_mode == "geometry" or (
            not self.training and self.inference_view == "geometry"
        ):
            keep.zero_()
        if self.training:
            keep &= torch.rand_like(keep, dtype=torch.float32) >= self.text_dropout
        ids = keep.nonzero().flatten()
        if len(ids):
            active_texts = [texts[i] for i in ids.tolist()]
            with torch.no_grad():
                # Same frozen text encoder as Q; no duplicated language backbone.
                encoded = self.parent.text_encoder(active_texts)
                pooled = masked_mean(encoded.tokens, encoded.valid_mask)
            recognized[ids] = pooled.to(recognized.dtype)
        # Adapter parameters remain in the graph even when a rank has no OCR.
        return self.adapter(
            **self._adapter_features(base, batch),
            base_logits=base["logits"],
            question_tokens=question.tokens,
            question_mask=question.attention_mask,
            recognized_tokens=recognized.reshape(b, n, -1),
            patches=batch["ocr_patches"],
            boxes=boxes,
            confidence=batch["ocr_confidence"],
            valid=valid,
        )

    @classmethod
    def from_parent(cls, checkpoint, **options):
        parent = GroundingModel(**checkpoint_model_kwargs(checkpoint))
        load_model_weights(parent, checkpoint)
        return cls(parent, **options)
