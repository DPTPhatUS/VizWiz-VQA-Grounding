"""Batch transfer and strict checkpoints for question-guided native zoom."""

import os
from pathlib import Path

import torch


def to_device(batch, device):
    def transfer(value):
        if torch.is_tensor(value):
            return value.to(device, non_blocking=True)
        if isinstance(value, list) and value and torch.is_tensor(value[0]):
            return [transfer(item) for item in value]
        return value

    return {key: transfer(value) for key, value in batch.items()}


def read_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" not in saved:
        saved = {"model_state_dict": saved}
    saved.setdefault("run_config", {})
    return saved


def checkpoint_model_kwargs(saved):
    from models.model import EXPERIMENT

    state = saved["model_state_dict"]
    key = "refiner.correction.weight"
    if key not in state or "zoom_version" not in state:
        raise ValueError(
            "Use a question-guided-zoom checkpoint; parent checkpoints belong to their original branch"
        )
    if state["zoom_version"].item() != 1:
        raise ValueError("Unknown zoom checkpoint version")
    tiny = "image_encoder.conv.weight" in state
    if state[key].shape != (1, 8 if tiny else 16, 1, 1):
        raise ValueError("Unexpected detail-pyramid refiner architecture")
    for config in (
        saved.get("experiment_config", {}),
        saved.get("run_config", {}).get("experiment_config", {}),
    ):
        if config and config.get("experiment") != EXPERIMENT:
            raise ValueError("Checkpoint belongs to a different experiment")
        if "tiny" in config and config["tiny"] != tiny:
            raise ValueError("Checkpoint tiny metadata disagrees with tensors")
    return {"tiny": tiny}


def load_model_weights(model, saved):
    checkpoint_model_kwargs(saved)
    for config in (
        saved.get("experiment_config"),
        saved.get("run_config", {}).get("experiment_config"),
    ):
        if config is not None and config != model.experiment_config:
            raise ValueError(
                "Checkpoint configuration does not match this question-guided-zoom model"
            )
    model.load_state_dict(saved["model_state_dict"], strict=True)


def save_checkpoint(saved, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(saved, temporary, _use_new_zipfile_serialization=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
