"""Batch transfer and plain PyTorch weight loading."""
import torch
from models.model import GroundingModel


def to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def read_checkpoint(path):
    """Accept a raw state_dict or a checkpoint; run settings are optional."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in saved:
        saved = {"model_state_dict": saved}
    saved.setdefault("run_config", {})
    return saved


def load_model_weights(model, saved):
    """Strictly load tensors, checking model identity when the file records it."""
    config = saved.get("experiment_config")
    if config is not None and config != model.experiment_config:
        raise ValueError("Checkpoint configuration does not match model configuration")
    model.load_state_dict(saved.get("model_state_dict", saved), strict=True)


def initialize_weights(saved, model):
    state = saved["model_state_dict"]
    target = model
    if set(state) == set(model.coarse.state_dict()):
        target = model.coarse
    load_model_weights(target, saved)


def load_inference_model(path, device, stage=None, budget=2):
    """Restore the saved architecture, with optional inference-stage/budget overrides."""
    saved = read_checkpoint(path)
    config = saved["run_config"]
    model = GroundingModel(stage=stage or config.get("stage", "router"), budget=budget,
                           tiny=config.get("tiny", False), crop_size=config.get("crop_size", 336))
    load_model_weights(model, saved)
    return model.to(device).eval(), config
