"""Batch transfer and strict checkpoints for the question-guided detail pyramid."""

import errno
import os
import shutil
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
    if key not in state or "detail_pyramid.encoder.0.0.0.weight" not in state:
        raise ValueError(
            "Use a question-guided-detail-pyramid checkpoint; older experiments belong to their original branches"
        )
    tiny = "image_encoder.conv.weight" in state
    if state[key].shape != (1, 8 if tiny else 16, 1, 1):
        raise ValueError("Unexpected detail-pyramid refiner architecture")
    detail_inputs = set()
    for config in (
        saved.get("experiment_config", {}),
        saved.get("run_config", {}).get("experiment_config", {}),
    ):
        if config and config.get("experiment") != EXPERIMENT:
            raise ValueError("Checkpoint belongs to a different experiment")
        if "tiny" in config and config["tiny"] != tiny:
            raise ValueError("Checkpoint tiny metadata disagrees with tensors")
        if config:
            detail_inputs.add(config.get("detail_input", "image"))
    if len(detail_inputs) > 1:
        raise ValueError("Checkpoint detail_input metadata is inconsistent")
    return {"tiny": tiny, "detail_input": detail_inputs.pop() if detail_inputs else "image"}


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


def checkpoint_storage_bytes(value):
    """Count unique tensor storages, including optimizer moments and shared views."""
    storages = {}

    def visit(item):
        if torch.is_tensor(item):
            storage = item.untyped_storage()
            storages[(str(storage.device), storage.data_ptr(), storage.nbytes())] = (
                storage.nbytes()
            )
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return sum(storages.values())


def save_checkpoint(saved, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload_bytes = checkpoint_storage_bytes(saved)
    required = payload_bytes + max(64 * 1024**2, payload_bytes // 20)
    available = shutil.disk_usage(path.parent).free
    if available < required:
        raise OSError(
            errno.ENOSPC,
            f"Not enough free disk space for {path.name}: "
            f"{available / 1024**3:.2f} GiB free, approximately "
            f"{required / 1024**3:.2f} GiB required for the new atomic write. "
            "Existing checkpoints are unchanged. Free disk space or use a fresh session; "
            "no checkpoints are deleted automatically.",
            str(path),
        )
    try:
        torch.save(saved, temporary, _use_new_zipfile_serialization=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
