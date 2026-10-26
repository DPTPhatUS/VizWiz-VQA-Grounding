"""Small adapter checkpoints tied to an exact, separately retained parent checkpoint."""

import torch

from models.ocr_model import EXPERIMENT, OCRGroundingModel
from ocr_cache import file_hash
from utils import read_checkpoint

VERSION = "ocr-adapter-checkpoint-v1"


def load_ocr_model(parent_path, adapter_path=None, **options):
    digest = file_hash(parent_path)
    saved = (
        torch.load(adapter_path, map_location="cpu", weights_only=True)
        if adapter_path
        else None
    )
    if saved:
        if (
            saved.get("version") != VERSION
            or saved["experiment_config"]["experiment"] != EXPERIMENT
        ):
            raise ValueError("Expected this experiment's adapter checkpoint")
        if saved["parent_sha256"] != digest:
            raise ValueError(
                "Parent checkpoint SHA256 mismatch; use the exact parent used for training"
            )
        config = saved["experiment_config"]
        options = {
            key: config[key] for key in ("evidence_mode", "ocr_dropout", "text_dropout")
        }
    parent = read_checkpoint(parent_path)
    model = OCRGroundingModel.from_parent(parent, **options)
    del parent
    if saved:
        if saved["experiment_config"] != model.experiment_config:
            raise ValueError("Adapter architecture metadata mismatch")
        model.adapter.load_state_dict(saved["adapter_state_dict"], strict=True)
    return model, saved, digest
