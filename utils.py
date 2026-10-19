"""Batch transfer and plain PyTorch weight loading."""
import os
from pathlib import Path
import warnings
import torch


BASE_LATENT_ANSWER_KEYS = {
    "visual_to_text.weight", "visual_to_text.bias",
    "answer_query",
    "answer_attn.in_proj_weight", "answer_attn.in_proj_bias",
    "answer_attn.out_proj.weight", "answer_attn.out_proj.bias",
    "answer_norm.weight", "answer_norm.bias",
    "answer_mlp.0.weight", "answer_mlp.0.bias",
    "answer_mlp.2.weight", "answer_mlp.2.bias",
    "decoder.conditioner.qa_gate.weight", "decoder.conditioner.qa_gate.bias",
}

SPATIAL_EVIDENCE_KEYS = {
    "decoder.spatial_bottleneck.visual_proj.weight",
    "decoder.spatial_bottleneck.evidence_proj.weight",
    "decoder.spatial_bottleneck.gate_scale",
    "decoder.spatial_bottleneck.gate_bias",
    "decoder.spatial_bottleneck.refine.0.weight",
    "decoder.spatial_bottleneck.refine.1.weight",
    "decoder.spatial_bottleneck.refine.2.weight",
    "decoder.spatial_bottleneck.refine.2.bias",
    "decoder.spatial_bottleneck.residual_scale",
    "decoder.spatial_upsample.0.visual_proj.weight",
    "decoder.spatial_upsample.0.evidence_proj.weight",
    "decoder.spatial_upsample.0.gate_scale",
    "decoder.spatial_upsample.0.gate_bias",
    "decoder.spatial_upsample.0.refine.0.weight",
    "decoder.spatial_upsample.0.refine.1.weight",
    "decoder.spatial_upsample.0.refine.2.weight",
    "decoder.spatial_upsample.0.refine.2.bias",
    "decoder.spatial_upsample.0.residual_scale",
    "decoder.spatial_upsample.1.visual_proj.weight",
    "decoder.spatial_upsample.1.evidence_proj.weight",
    "decoder.spatial_upsample.1.gate_scale",
    "decoder.spatial_upsample.1.gate_bias",
    "decoder.spatial_upsample.1.refine.0.weight",
    "decoder.spatial_upsample.1.refine.1.weight",
    "decoder.spatial_upsample.1.refine.2.weight",
    "decoder.spatial_upsample.1.refine.2.bias",
    "decoder.spatial_upsample.1.residual_scale",
}

LATENT_ANSWER_KEYS = BASE_LATENT_ANSWER_KEYS | SPATIAL_EVIDENCE_KEYS
COARSENED_LATENT_ANSWER_KEYS = {f"coarse.{k}" for k in LATENT_ANSWER_KEYS}


def to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def is_tiny_state(state):
    return isinstance(state, dict) and (
        "image_encoder.conv.weight" in state or "coarse.image_encoder.conv.weight" in state
    )


def read_checkpoint(path):
    """Accept a raw state_dict or a checkpoint; run settings are optional."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in saved:
        saved = {"model_state_dict": saved}
    saved.setdefault("run_config", {})
    return saved


def checkpoint_model_kwargs(saved):
    """Resolve architecture from tensors; verify any recorded spatial setting."""
    state = saved["model_state_dict"]
    spatial = any(k.startswith(("decoder.spatial_", "coarse.decoder.spatial_")) for k in state)
    run = saved.get("run_config", {})
    for config in (saved.get("experiment_config", {}), run.get("experiment_config", {}), run):
        if "spatial_evidence" in config and config["spatial_evidence"] != spatial:
            raise ValueError("Checkpoint spatial_evidence metadata disagrees with model tensors")
    return {"tiny": run.get("tiny", is_tiny_state(state)), "spatial_evidence": spatial}


def save_checkpoint(saved, path):
    """Replace a checkpoint only after its temporary file was fully written."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(saved, temporary, _use_new_zipfile_serialization=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_model_weights(model, saved, *, allow_latent_initialization=False):
    """Strict for inference/resume/teachers; only explicit student initialization may omit latent tensors."""
    state = saved.get("model_state_dict", saved)
    target_keys = set(model.state_dict())
    missing = target_keys - set(state)
    allowed = (BASE_LATENT_ANSWER_KEYS, SPATIAL_EVIDENCE_KEYS, LATENT_ANSWER_KEYS)
    allowed += tuple({f"coarse.{k}" for k in group} for group in allowed)
    partial = allow_latent_initialization and set(state) < target_keys and missing in allowed
    config = saved.get("experiment_config")
    if config is not None:
        config = dict(config)
        config.setdefault("spatial_evidence", any("decoder.spatial_" in k for k in state))
        expected = dict(model.experiment_config)
        if partial:
            # Explicit weight initialization can add whole modules, never arbitrary tensors.
            config["spatial_evidence"] = expected["spatial_evidence"]
        if config != expected:
            raise ValueError("Checkpoint configuration does not match model configuration")
    if partial:
        incompatible = model.load_state_dict(state, strict=False)
        if set(incompatible.missing_keys) != missing or incompatible.unexpected_keys:
            raise ValueError("Unexpected partial checkpoint tensors")
        warnings.warn(f"Initialized new module tensors: {', '.join(sorted(missing))}", stacklevel=2)
    else:
        model.load_state_dict(state, strict=True)
    run = saved.get("run_config", {})
    core = model.coarse if hasattr(model, "coarse") else model
    core.question_only = run.get("training_config", {}).get("question_only", run.get("question_only", False))


def initialize_weights(path, model):
    saved = path if isinstance(path, dict) else read_checkpoint(path)
    state = saved["model_state_dict"]
    target = model
    if hasattr(model, "coarse") and state and not any(k.startswith("coarse.") for k in state):
        target = model.coarse
    load_model_weights(target, saved, allow_latent_initialization=True)
