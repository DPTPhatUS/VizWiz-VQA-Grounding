"""Batch transfer and plain PyTorch weight loading."""
import os
from collections import OrderedDict
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

def _spatial_module_keys(prefix):
    return {
        f"{prefix}.visual_proj.weight",
        f"{prefix}.evidence_proj.weight",
        f"{prefix}.gate_scale",
        f"{prefix}.gate_bias",
        f"{prefix}.refine.0.weight",
        f"{prefix}.refine.1.weight",
        f"{prefix}.refine.2.weight",
        f"{prefix}.refine.2.bias",
        f"{prefix}.residual_scale",
    }


def spatial_evidence_keys(stages=(0, 1)):
    keys = _spatial_module_keys("decoder.spatial_bottleneck")
    keys.add("decoder.spatial_stage_ids")
    if stages:
        for idx in stages:
            keys.update(_spatial_module_keys(f"decoder.spatial_upsample.{idx}"))
    return keys


SPATIAL_EVIDENCE_KEYS = spatial_evidence_keys((0, 1))
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
    configs = (saved.get("experiment_config", {}), run.get("experiment_config", {}), run)
    for config in configs:
        if "spatial_evidence" in config and config["spatial_evidence"] != spatial:
            raise ValueError("Checkpoint spatial_evidence metadata disagrees with model tensors")
    kwargs = {"tiny": run.get("tiny", is_tiny_state(state)), "spatial_evidence": spatial}
    from models.mask_decoder import resolve_spatial_upsample_stages
    recorded = []
    for config in configs:
        if "spatial_upsample_stages" in config:
            value = config["spatial_upsample_stages"]
            if not isinstance(value, (list, tuple)):
                raise ValueError("Checkpoint spatial stages must be a list of integer IDs")
            recorded.append(resolve_spatial_upsample_stages(value))
    markers = [v for k, v in state.items() if k.endswith("decoder.spatial_stage_ids")]
    for marker in markers:
        if marker.ndim != 1 or marker.dtype != torch.long:
            raise ValueError("Invalid spatial stage buffer")
        stages = resolve_spatial_upsample_stages(marker.tolist())
        if list(stages) != marker.tolist():
            raise ValueError("Spatial stage buffer must be sorted")
        recorded.append(stages)
    if recorded and any(value != recorded[0] for value in recorded):
        raise ValueError("Conflicting checkpoint spatial_upsample_stages metadata")
    indices = sorted({int(k.split("decoder.spatial_upsample.")[1].split(".")[0])
                      for k in state if "decoder.spatial_upsample." in k})
    if spatial:
        if recorded:
            stages = recorded[0]
        elif not indices:
            stages = ()
        else:
            # Positional legacy weights cannot identify stages 2 versus 3.
            projections = {int(k.split("decoder.spatial_upsample.")[1].split(".")[0]): v
                           for k, v in state.items()
                           if "decoder.spatial_upsample." in k and k.endswith(".visual_proj.weight")}
            if set(projections) != set(indices):
                raise ValueError("Incomplete legacy spatial weights: missing visual projection")
            channels = [projections[i].shape[0] for i in indices]
            if indices != list(range(len(indices))) or channels not in ([64], [64, 32], [64, 32, 16, 16]):
                raise ValueError("Ambiguous legacy raw spatial weights; use the full checkpoint with stage metadata")
            stages = tuple(indices)
        expected_indices = list(stages) if markers else list(range(len(stages)))
        if indices != expected_indices:
            raise ValueError("Checkpoint spatial stage IDs disagree with module tensors")
        kwargs["spatial_upsample_stages"] = stages
    elif recorded and recorded[0]:
        raise ValueError("Non-spatial checkpoint contains spatial stage metadata")
    return kwargs


def _normalize_spatial_state(saved):
    """Migrate legacy positional names in memory; never modify checkpoint tensors."""
    state = saved["model_state_dict"]
    options = checkpoint_model_kwargs(saved)
    if not options["spatial_evidence"] or any(k.endswith("decoder.spatial_stage_ids") for k in state):
        return state, options
    stages = options["spatial_upsample_stages"]
    prefix = "coarse.decoder." if any(k.startswith("coarse.decoder.spatial_") for k in state) else "decoder."

    def rename(key):
        start = prefix + "spatial_upsample."
        if key.startswith(start):
            index, dot, tail = key[len(start):].partition(".")
            return start + str(stages[int(index)]) + dot + tail
        return key

    migrated = OrderedDict((rename(k), v) for k, v in state.items())
    migrated[prefix + "spatial_stage_ids"] = torch.tensor(stages, dtype=torch.long)
    if hasattr(state, "_metadata"):
        migrated._metadata = OrderedDict((rename(k), v) for k, v in state._metadata.items())
    return migrated, options


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
    payload = saved if "model_state_dict" in saved else {"model_state_dict": saved}
    state, options = _normalize_spatial_state(payload)
    target_keys = set(model.state_dict())
    missing = target_keys - set(state)
    model_spatial = {k[7:] if k.startswith("coarse.") else k for k in target_keys if "decoder.spatial_" in k}
    model_latent = BASE_LATENT_ANSWER_KEYS | model_spatial
    allowed_groups = {
        frozenset(BASE_LATENT_ANSWER_KEYS),
        frozenset(SPATIAL_EVIDENCE_KEYS),
        frozenset(LATENT_ANSWER_KEYS),
        frozenset(model_spatial),
        frozenset(model_latent),
    }
    allowed = tuple(set(g) for g in allowed_groups if g)
    allowed += tuple({f"coarse.{k}" for k in group} for group in allowed)
    partial = allow_latent_initialization and set(state) < target_keys and missing in allowed
    config = saved.get("experiment_config")
    if config is not None:
        config = dict(config)
        config.setdefault("spatial_evidence", any("decoder.spatial_" in k for k in state))
        expected = dict(model.experiment_config)
        if "spatial_upsample_stages" in config:
            config["spatial_upsample_stages"] = list(config["spatial_upsample_stages"])
        if "spatial_upsample_stages" in expected:
            expected["spatial_upsample_stages"] = list(expected["spatial_upsample_stages"])
        if partial:
            # Explicit weight initialization can add whole modules, never arbitrary tensors.
            config["spatial_evidence"] = expected.get("spatial_evidence", False)
            if "spatial_upsample_stages" in expected:
                config["spatial_upsample_stages"] = expected["spatial_upsample_stages"]
        else:
            if "spatial_upsample_stages" not in config and "spatial_upsample_stages" in expected:
                config["spatial_upsample_stages"] = list(options.get("spatial_upsample_stages", ()))
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
