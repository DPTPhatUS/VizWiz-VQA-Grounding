"""Strict weight loading with experiment identity checks for training checkpoints."""
def load_model_weights(model, checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        saved = checkpoint.get("experiment_config")
        if saved is not None and saved != model.experiment_config:
            raise ValueError(
                f"Checkpoint configuration {saved} does not match model configuration "
                f"{model.experiment_config}. Use the matching experiment branch/options."
            )
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(state, strict=True)


"""Versioned research checkpoints; only load checkpoints from trusted sources."""
import os
import random
from pathlib import Path
import torch


def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, scaler, epoch, best_iou, run_config,
                    rng_states=None, provenance=None):
    path = Path(path)
    payload = {"format_version": 1, "experiment_config": model.experiment_config,
               "model_state_dict": model.state_dict(), "epoch": epoch, "best_iou": best_iou,
               "run_config": run_config, "rng_states": rng_states or [rng_state()],
               "provenance": provenance or {}}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def read_checkpoint(path):
    # Python RNG tuples are deliberately included; never use untrusted checkpoint files.
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or value.get("format_version") != 1:
        raise ValueError("Expected a versioned research checkpoint; legacy weights need explicit migration")
    return value


def load_checkpoint(path, model):
    value = read_checkpoint(path)
    if value["experiment_config"] != model.experiment_config:
        raise ValueError("Checkpoint configuration does not match model configuration")
    model.load_state_dict(value["model_state_dict"], strict=True)
    return value


def initialize_weights(path, model):
    """Exact architecture or a controls checkpoint into a variant's coarse model."""
    value = read_checkpoint(path)
    if value["experiment_config"] == model.experiment_config:
        model.load_state_dict(value["model_state_dict"], strict=True)
    elif hasattr(model, "coarse") and value["experiment_config"] == model.coarse.experiment_config:
        model.coarse.load_state_dict(value["model_state_dict"], strict=True)
    else:
        raise ValueError("Initialization requires matching architecture or matching controls coarse model")
    return value
