"""Routing experiment wrapper; the original residual model/checkpoints remain separate."""

from models.ocr_model import OCRGroundingModel
from models.ocr_routing import EvidenceRoutingAdapter
from models.region_ops import PATCH_SIZE

EXPERIMENT = "ocr-evidence-routing-v1"


class OCRRoutingModel(OCRGroundingModel):
    def __init__(
        self,
        parent,
        *,
        architecture="proposal",
        evidence_mode="full",
        ocr_dropout=0.2,
        text_dropout=0.2,
    ):
        self.architecture = architecture
        super().__init__(
            parent,
            evidence_mode=evidence_mode,
            ocr_dropout=ocr_dropout,
            text_dropout=text_dropout,
        )
        self.experiment_config.update(
            experiment=EXPERIMENT,
            architecture=architecture,
            mask_composition="independent-noisy-or-probability-route",
            multi_label_selection=True,
            uses_parent_semantics=True,
            semantic_dim=parent.decoder.output_channels,
            patch_size=list(PATCH_SIZE),
        )

    def _make_adapter(self, text_dim, detail_dim, width):
        return EvidenceRoutingAdapter(
            text_dim,
            detail_dim,
            width,
            self.architecture,
            semantic_dim=self.parent.decoder.output_channels,
        )

    def _decode_parent(self, batch, features, question):
        return self.parent.decode_features(
            batch,
            features,
            question,
            return_detail=True,
            return_evidence=self.architecture == "pyramid",
        )

    def _adapter_features(self, base, batch):
        return {
            "detail": base["detail_features"],
            "semantic": base["semantic_features"],
            "polygons": batch["ocr_polygon_patches"],
            "detail_levels": base.get("evidence_levels"),
        }

    def forward(self, batch):
        output = super().forward(batch)
        if not self.training and self.inference_view == "evidence":
            output["logits"] = output["evidence_logits"]
        return output
