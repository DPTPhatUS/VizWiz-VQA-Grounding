"""Independent mask scoring with optional scope-focused training."""

from models.ocr_scope import IndependentEvidenceScorer, quality_training_masks
from models.ocr_selection_model import OCRSelectionModel

EXPERIMENT = "ocr-scope-ranking-v1"


class OCRScopeModel(OCRSelectionModel):
    def __init__(
        self,
        parent,
        *,
        evidence_mode="full",
        ocr_dropout=0.2,
        text_dropout=0.2,
        min_gain=0.02,
        selector_recipe="ranking",
    ):
        if selector_recipe not in {"ranking", "scope"}:
            raise ValueError("selector_recipe must be ranking or scope")
        self.selector_recipe = selector_recipe
        super().__init__(
            parent,
            evidence_mode=evidence_mode,
            ocr_dropout=ocr_dropout,
            text_dropout=text_dropout,
            min_gain=min_gain,
        )
        self.experiment_config.update(
            experiment=EXPERIMENT,
            architecture="independent",
            selector_recipe=selector_recipe,
            mask_scorer="shared-own-mask-v1",
            quality_augmentation="matched-geometric-v1"
            if selector_recipe == "ranking"
            else "matched-scope-v1",
            mask_feature_protocol="binary-supersampled-own-mask-v1",
            reference_features=False,
            base_identity=False,
        )

    def _make_adapter(self, text_dim, detail_dim, width):
        return IndependentEvidenceScorer(
            text_dim,
            detail_dim,
            self.parent.decoder.output_channels,
            width,
            self.min_gain,
        )

    def _decode_parent(self, batch, features, question):
        output = self.parent.decode_features(
            batch, features, question, return_detail=True
        )
        if self.training:
            output["logits"] = quality_training_masks(
                output["logits"],
                batch["hypothesis_masks"],
                batch["ocr_valid"],
                self.selector_recipe,
            )
        return output

    def _adapter_features(self, base, batch):
        features = super()._adapter_features(base, batch)
        n = batch["ocr_valid"].shape[1]
        features.update(
            image=batch["detail_image"],
            word_masks=batch["hypothesis_masks"][:, 1 : n + 1],
        )
        return features
