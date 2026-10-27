"""Frozen pyramid wrapper for native OCR hypothesis selection."""

import torch

from models.ocr_model import OCRGroundingModel
from models.ocr_selection import HypothesisSelector, augment_quality_masks
from ocr_selection_data import HYPOTHESIS_VERSION, SCORE_SIZE

EXPERIMENT = "ocr-evidence-selection-v1"


class OCRSelectionModel(OCRGroundingModel):
    def __init__(
        self,
        parent,
        *,
        evidence_mode="full",
        ocr_dropout=0.2,
        text_dropout=0.2,
        min_gain=0.02,
    ):
        if not 0 <= min_gain < 1:
            raise ValueError("min_gain must be in [0,1)")
        self.min_gain = min_gain
        super().__init__(
            parent,
            evidence_mode=evidence_mode,
            ocr_dropout=ocr_dropout,
            text_dropout=text_dropout,
        )
        self.experiment_config.update(
            experiment=EXPERIMENT,
            architecture="hypotheses",
            min_gain=min_gain,
            hypothesis_protocol=HYPOTHESIS_VERSION,
            score_size=SCORE_SIZE,
            output_protocol="native-polygon-selection-v1",
            quality_augmentation="half-base-shift-or-morphology-v1",
        )

    def _make_adapter(self, text_dim, detail_dim, width):
        return HypothesisSelector(
            text_dim,
            detail_dim,
            width,
            self.parent.decoder.output_channels,
            self.min_gain,
        )

    def _decode_parent(self, batch, features, question):
        output = self.parent.decode_features(
            batch, features, question, return_detail=True
        )
        if self.training:
            output["logits"] = augment_quality_masks(output["logits"])
        return output

    def _adapter_features(self, base, batch):
        return {
            "detail": base["detail_features"],
            "semantic": base["semantic_features"],
            "polygons": batch["ocr_polygon_patches"],
            "members": batch["hypothesis_members"],
            "masks": batch["hypothesis_masks"],
        }

    def forward(self, batch):
        output = super().forward(batch)
        if "quality_logits" not in output:  # Explicit base view.
            return {
                "base_logits": output["logits"],
                "selected_index": torch.zeros(
                    len(batch["image"]), dtype=torch.long, device=batch["image"].device
                ),
            }
        if not self.training and self.inference_view == "evidence":
            output["selected_index"] = torch.where(
                output["candidate_valid"].any(1), output["evidence_index"], 0
            )
        return output
