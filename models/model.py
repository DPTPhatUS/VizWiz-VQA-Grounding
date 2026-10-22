"""Frozen pyramid anchor with a small question-conditioned scope corrector."""

import torch
from torch import Tensor, nn

from models.pyramid import PyramidFeatures, PyramidModel
from models.scope_correction import (
    CORRUPTION_PROTOCOL,
    SELECTION_DEFAULTS,
    EditEvaluator,
    ScopePrediction,
    ScopeVerifier,
    expanded_training_logits,
    suppress_logits,
    validate_selection,
)

EXPERIMENT = "question-guided-scope-correction-v1"


class GroundingModel(nn.Module):
    fixed_reference = True

    def __init__(
        self,
        tiny=False,
        gradient_checkpointing=False,
        gain_margin=SELECTION_DEFAULTS["gain_margin"],
        max_damage=SELECTION_DEFAULTS["max_damage"],
    ):
        super().__init__()
        validate_selection(gain_margin, max_damage)
        self.parent = PyramidModel(tiny=tiny).requires_grad_(False).eval()
        text_dim = self.parent.text_encoder.output_dim
        self.verifier = ScopeVerifier(
            self.parent.decoder.stage96_channels,
            8 if tiny else 16,
            text_dim,
            tiny=tiny,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.evaluator = EditEvaluator(self.verifier.evidence_dim, text_dim, tiny=tiny)
        self.register_buffer("scope_version", torch.tensor(1, dtype=torch.int64))
        self.register_buffer(
            "selection_thresholds",
            torch.tensor([gain_margin, max_damage], dtype=torch.float64),
        )
        self.image_size, self.detail_size = (
            self.parent.image_size,
            self.parent.detail_size,
        )
        self.experiment_config = {
            "experiment": EXPERIMENT,
            "tiny": tiny,
            "parent": self.parent.experiment_config,
            "image_size": self.image_size,
            "detail_size": self.detail_size,
            "text_mode": "question",
            "parent_frozen": True,
            "gain_margin": gain_margin,
            "max_damage": max_damage,
            "suppression": "probability-product-stable-logits",
            "scope_width": 8 if tiny else 32,
            "evaluator_grid": 8,
            "evaluator_detached": True,
            "corruption": CORRUPTION_PROTOCOL.copy(),
        }

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()
        return self

    def correct(self, features: PyramidFeatures, source: Tensor) -> ScopePrediction:
        removal, evidence = self.verifier(
            features.region, features.detail, features.text, source
        )
        candidate = suppress_logits(source, removal)
        gain, damage = self.evaluator(evidence, features.text, source, candidate)
        thresholds = self.selection_thresholds.to(dtype=gain.dtype)
        selected = (gain > thresholds[0]) & (damage.sigmoid() < thresholds[1])
        return ScopePrediction(source, candidate, removal, gain, damage, selected)

    def forward(self, batch):
        # Frozen features feed trainable layers, so use no_grad, not inference_mode.
        with torch.no_grad():
            features = self.parent(batch)
        predicted = self.correct(features, features.logits)
        logits = torch.where(
            predicted.selected[:, None, None, None],
            predicted.candidate_logits,
            features.logits.float(),
        )
        auxiliary = None
        if self.training and "scope_training" in batch:
            corrupted = expanded_training_logits(
                features.logits,
                batch["mask"],
                batch["scope_training"]["corruption_probability"],
            )
            auxiliary = self.correct(features, corrupted)
        return {
            "logits": logits,
            "parent_logits": features.logits.float(),
            "candidate_logits": predicted.candidate_logits,
            "scope": predicted,
            "training_scope": auxiliary,
        }

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        from utils import checkpoint_model_kwargs, load_model_weights, read_checkpoint

        saved = read_checkpoint(path)
        model = cls(**checkpoint_model_kwargs(saved))
        load_model_weights(model, saved)
        return model.to(device).eval()
