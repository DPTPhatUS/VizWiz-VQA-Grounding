"""Strict small adapter checkpoints, with separate versions for residual and routing architectures."""

import torch

from models.ocr_model import EXPERIMENT as RESIDUAL_EXPERIMENT
from models.ocr_model import OCRGroundingModel
from models.ocr_routing_model import EXPERIMENT as ROUTING_EXPERIMENT
from models.ocr_routing_model import OCRRoutingModel
from ocr_cache import file_hash
from utils import read_checkpoint

VERSION = "ocr-adapter-checkpoint-v1"
ROUTING_VERSION = "ocr-routing-checkpoint-v1"


def checkpoint_version(model):
    return (
        ROUTING_VERSION
        if model.experiment_config["experiment"] == ROUTING_EXPERIMENT
        else VERSION
    )


def load_ocr_model(parent_path, adapter_path=None, **options):
    digest = file_hash(parent_path)
    saved = (
        torch.load(adapter_path, map_location="cpu", weights_only=True)
        if adapter_path
        else None
    )
    architecture = options.pop("architecture", "residual")
    model_class = OCRGroundingModel if architecture == "residual" else OCRRoutingModel
    if saved:
        config = saved["experiment_config"]
        pair = (saved.get("version"), config.get("experiment"))
        if pair == (VERSION, RESIDUAL_EXPERIMENT):
            model_class = OCRGroundingModel
        elif pair == (ROUTING_VERSION, ROUTING_EXPERIMENT):
            model_class = OCRRoutingModel
            architecture = config["architecture"]
        else:
            raise ValueError("Unsupported OCR adapter checkpoint/version")
        if saved["parent_sha256"] != digest:
            raise ValueError("Parent SHA256 mismatch; retain the exact original parent")
        options = {
            key: config[key] for key in ("evidence_mode", "ocr_dropout", "text_dropout")
        }
    if model_class is OCRRoutingModel:
        options["architecture"] = architecture
    parent = read_checkpoint(parent_path)
    model = model_class.from_parent(parent, **options)
    del parent
    if saved:
        if saved["experiment_config"] != model.experiment_config:
            raise ValueError("Adapter architecture metadata mismatch")
        model.adapter.load_state_dict(saved["adapter_state_dict"], strict=True)
    return model, saved, digest
