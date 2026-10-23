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
    from answer_data import vocabulary_digest
    from models.model import EXPERIMENT

    state = saved["model_state_dict"]
    key = "refiner.correction.weight"
    if "answer_head.classifier.weight" not in state:
        raise ValueError(
            "Use a dense-answer-supervision checkpoint; other experiments are not valid resumes"
        )
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
    config = saved.get("experiment_config") or saved.get("run_config", {}).get(
        "experiment_config", {}
    )
    vocabulary = config.get("answer_vocabulary")
    if (
        not isinstance(vocabulary, list)
        or not vocabulary
        or not all(isinstance(x, str) for x in vocabulary)
    ):
        raise ValueError(
            "Dense checkpoints require saved train-answer vocabulary metadata"
        )
    digest = state.get("answer_head.vocabulary_hash")
    if digest is None or bytes(digest.tolist()) != vocabulary_digest(vocabulary):
        raise ValueError("Answer vocabulary metadata does not match checkpoint tensors")
    if state["answer_head.classifier.weight"].shape[0] != len(vocabulary) + 1:
        raise ValueError("Classifier size does not match answer vocabulary")
    return {"tiny": tiny, "answer_vocabulary": vocabulary}


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
