"""OCR sufficiency without parent-quality labels; frozen pyramid context/fallback."""

import torch

from models.ocr_model import OCRGroundingModel
from models.ocr_pointer import EvidencePointer
from ocr_selection_data import HYPOTHESIS_VERSION, SCORE_SIZE

EXPERIMENT = "ocr-sufficiency-pointer-v1"


class OCRPointerModel(OCRGroundingModel):
    def __init__(
        self,
        parent,
        *,
        evidence_mode="full",
        ocr_dropout=0.2,
        text_dropout=0.2,
        pointer_recipe="categorical",
    ):
        self.pointer_recipe = pointer_recipe
        super().__init__(
            parent,
            evidence_mode=evidence_mode,
            ocr_dropout=ocr_dropout,
            text_dropout=text_dropout,
        )
        self.experiment_config.update(
            experiment=EXPERIMENT,
            architecture="pointer",
            pointer_recipe=pointer_recipe,
            hypothesis_protocol=HYPOTHESIS_VERSION,
            score_size=SCORE_SIZE,
            output_protocol="native-polygon-selection-v1",
            parent_quality_target=False,
            null_protocol="no-OCR-mask-IoU-ge-0.5",
            candidate_tolerance=0.05,
            decision_protocol="OCR-total-mass-gt-defer-v1",
            member_pooling="question-conditioned-masked-attention-v1",
        )

    def _make_adapter(self, text_dim, detail_dim, width):
        return EvidencePointer(
            text_dim,
            detail_dim,
            self.parent.decoder.output_channels,
            width,
            self.pointer_recipe,
        )

    def _adapter_features(self, base, batch):
        return {
            "detail": base["detail_features"],
            "semantic": base["semantic_features"],
            "image": batch["detail_image"],
            "polygons": batch["ocr_polygon_patches"],
            "members": batch["hypothesis_members"],
            "masks": batch["hypothesis_masks"],
        }

    def forward(self, batch):
        output = super().forward(batch)
        if "selection_logits" not in output:  # Explicit base view.
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
