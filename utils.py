"""Batch transfer and strict checkpoints for the question-guided detail pyramid."""

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
    key = "refiner.correction.weight"
    if "answer_head.embedding.weight" not in state:
        raise ValueError(
            "Use a mask-conditioned-answer checkpoint; other experiments are not resumes"
        )
    version = state.get("answer_head.mask_attention_version")
    if version is None or version.numel() != 1 or version.item() != 1:
        raise ValueError("Use this branch's mask-conditioned-answer checkpoint")
    grid = state.get("answer_head.spatial_position")
    expected_grid = 16 if "image_encoder.conv.weight" in state else 1024
    if grid is None or grid.shape[1] != expected_grid:
        raise ValueError("Checkpoint reader grid belongs to another experiment")
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


def load_model_weights(model, saved):
    checkpoint_model_kwargs(saved)
    for config in (
        saved.get("experiment_config"),
        saved.get("run_config", {}).get("experiment_config"),
    ):
        if config is not None and config != model.experiment_config:
            raise ValueError(
                "Checkpoint configuration does not match this detail-pyramid model"
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
