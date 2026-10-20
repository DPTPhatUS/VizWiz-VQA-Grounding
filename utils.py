"""Batch transfer and strict checkpoints for this grounding-first experiment."""

import os
from pathlib import Path

import torch


def to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def read_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" not in saved:
        saved = {"model_state_dict": saved}
    saved.setdefault("run_config", {})
    return saved


def checkpoint_model_kwargs(saved):
    from models.model import EXPERIMENT

    state = saved["model_state_dict"]
    key = "reader.encoder.0.weight"
    if key not in state or "coarse.answer_query" in state:
        raise ValueError(
            "Use a grounding-first checkpoint; latent-answer checkpoints belong to their original branch"
        )
    tiny = "coarse.image_encoder.conv.weight" in state
    width = 8 if tiny else 32
    if state[key].shape != (width, 3, 3, 3):
        raise ValueError("Unexpected grounding-first reader architecture")
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
                "Checkpoint configuration does not match this grounding-first model"
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
