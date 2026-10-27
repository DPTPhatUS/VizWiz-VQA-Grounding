"""Strict small adapter checkpoints, with separate versions for residual and routing architectures."""

import torch

from models.native_refinement_model import EXPERIMENT as NATIVE_EXPERIMENT
from models.native_refinement_model import NativeRefinementModel
from models.ocr_model import EXPERIMENT as RESIDUAL_EXPERIMENT
from models.ocr_model import OCRGroundingModel
from models.ocr_pointer_model import EXPERIMENT as POINTER_EXPERIMENT
from models.ocr_pointer_model import OCRPointerModel
from models.ocr_routing_model import EXPERIMENT as ROUTING_EXPERIMENT
from models.ocr_routing_model import OCRRoutingModel
from models.ocr_scope_model import EXPERIMENT as SCOPE_EXPERIMENT
from models.ocr_scope_model import OCRScopeModel
from models.ocr_selection_model import EXPERIMENT as SELECTION_EXPERIMENT
from models.ocr_selection_model import OCRSelectionModel
from ocr_cache import file_hash
from utils import read_checkpoint

VERSION = "ocr-adapter-checkpoint-v1"
ROUTING_VERSION = "ocr-routing-checkpoint-v1"
SELECTION_VERSION = "ocr-selection-checkpoint-v1"
SCOPE_VERSION = "ocr-scope-checkpoint-v1"
POINTER_VERSION = "ocr-pointer-checkpoint-v1"
NATIVE_VERSION = "ocr-native-refinement-checkpoint-v1"


def checkpoint_version(model):
    versions = {
        RESIDUAL_EXPERIMENT: VERSION,
        ROUTING_EXPERIMENT: ROUTING_VERSION,
        SELECTION_EXPERIMENT: SELECTION_VERSION,
        SCOPE_EXPERIMENT: SCOPE_VERSION,
        POINTER_EXPERIMENT: POINTER_VERSION,
        NATIVE_EXPERIMENT: NATIVE_VERSION,
    }
    return versions[model.experiment_config["experiment"]]


def load_ocr_model(parent_path, adapter_path=None, **options):
    digest = file_hash(parent_path)
    saved = (
        torch.load(adapter_path, map_location="cpu", weights_only=True)
        if adapter_path
        else None
    )
    architecture = options.pop("architecture", "residual")
    model_class = {
        "residual": OCRGroundingModel,
        "proposal": OCRRoutingModel,
        "pyramid": OCRRoutingModel,
        "hypotheses": OCRSelectionModel,
        "independent": OCRScopeModel,
        "pointer": OCRPointerModel,
        "native_crop": NativeRefinementModel,
        "native_point": NativeRefinementModel,
    }[architecture]
    if saved:
        config = saved["experiment_config"]
        pair = (saved.get("version"), config.get("experiment"))
        if pair == (VERSION, RESIDUAL_EXPERIMENT):
            model_class = OCRGroundingModel
        elif pair == (ROUTING_VERSION, ROUTING_EXPERIMENT):
            model_class = OCRRoutingModel
            architecture = config["architecture"]
        elif pair == (SELECTION_VERSION, SELECTION_EXPERIMENT):
            model_class = OCRSelectionModel
            architecture = "hypotheses"
        elif pair == (SCOPE_VERSION, SCOPE_EXPERIMENT):
            model_class = OCRScopeModel
            architecture = "independent"
        elif pair == (POINTER_VERSION, POINTER_EXPERIMENT):
            model_class = OCRPointerModel
            architecture = "pointer"
        elif pair == (NATIVE_VERSION, NATIVE_EXPERIMENT):
            model_class = NativeRefinementModel
            architecture = config["architecture"]
        else:
            raise ValueError("Unsupported OCR adapter checkpoint/version")
        if saved["parent_sha256"] != digest:
            raise ValueError("Parent SHA256 mismatch; retain the exact original parent")
        options = {
            key: config[key] for key in ("evidence_mode", "ocr_dropout", "text_dropout")
        }
        if model_class in (OCRSelectionModel, OCRScopeModel):
            options["min_gain"] = config["min_gain"]
        if model_class is OCRScopeModel:
            options["selector_recipe"] = config["selector_recipe"]
        if model_class is OCRPointerModel:
            options["pointer_recipe"] = config["pointer_recipe"]
    if model_class in (OCRRoutingModel, NativeRefinementModel):
        options["architecture"] = architecture
    parent = read_checkpoint(parent_path)
    model = model_class.from_parent(parent, **options)
    del parent
    if saved:
        if saved["experiment_config"] != model.experiment_config:
            raise ValueError("Adapter architecture metadata mismatch")
        model.adapter.load_state_dict(saved["adapter_state_dict"], strict=True)
    return model, saved, digest
