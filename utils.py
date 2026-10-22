"""Batch transfer, strict scope checkpoints and explicit parent initialization."""

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


def parent_model_kwargs(saved):
    from models.pyramid import EXPERIMENT

    state = saved["model_state_dict"]
    key = "refiner.correction.weight"
    if key not in state or "detail_pyramid.encoder.0.0.0.weight" not in state:
        raise ValueError(
            "Use a question-guided-detail-pyramid checkpoint; older experiments belong to their original branches"
        )
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


def checkpoint_model_kwargs(saved):
    from models.model import EXPERIMENT

    state = saved["model_state_dict"]
    if "scope_version" not in state or "selection_thresholds" not in state:
        raise ValueError(
            "Use a scope-correction checkpoint; initialize a pyramid via --init-parent-checkpoint"
        )
    if state["scope_version"].item() != 1 or state["selection_thresholds"].shape != (
        2,
    ):
        raise ValueError("Unsupported scope-correction checkpoint")
    parent = {
        key.removeprefix("parent."): value
        for key, value in state.items()
        if key.startswith("parent.")
    }
    options = parent_model_kwargs({"model_state_dict": parent})
    gain, damage = state["selection_thresholds"].tolist()
    for config in (
        saved.get("experiment_config", {}),
        saved.get("run_config", {}).get("experiment_config", {}),
    ):
        if config and config.get("experiment") != EXPERIMENT:
            raise ValueError("Checkpoint belongs to a different experiment")
    return dict(**options, gain_margin=gain, max_damage=damage)


def initialize_parent(model, saved):
    """Load only a validated pyramid checkpoint; new modules retain seeded init."""
    options = parent_model_kwargs(saved)
    if options["tiny"] != model.experiment_config["tiny"]:
        raise ValueError("Parent checkpoint architecture disagrees with model")
    for config in (
        saved.get("experiment_config"),
        saved.get("run_config", {}).get("experiment_config"),
    ):
        if config is not None and config != model.parent.experiment_config:
            raise ValueError("Parent checkpoint configuration mismatch")
    model.parent.load_state_dict(saved["model_state_dict"], strict=True)
    model.parent.requires_grad_(False).eval()


def load_model_weights(model, saved):
    checkpoint_model_kwargs(saved)
    for config in (
        saved.get("experiment_config"),
        saved.get("run_config", {}).get("experiment_config"),
    ):
        if config is not None and config != model.experiment_config:
            raise ValueError(
                "Checkpoint configuration does not match this scope-correction model"
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
