"""Batch transfer and plain PyTorch weight loading."""
import torch


LATENT_ANSWER_KEYS = {
    "visual_to_text.weight", "visual_to_text.bias",
    "answer_query",
    "answer_attn.in_proj_weight", "answer_attn.in_proj_bias",
    "answer_attn.out_proj.weight", "answer_attn.out_proj.bias",
    "answer_norm.weight", "answer_norm.bias",
    "answer_mlp.0.weight", "answer_mlp.0.bias",
    "answer_mlp.2.weight", "answer_mlp.2.bias",
    "decoder.conditioner.qa_gate.weight", "decoder.conditioner.qa_gate.bias",
}
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


def load_model_weights(model, saved, *, allow_latent_initialization=False):
    """Strict for inference/resume/teachers; only explicit student initialization may omit latent tensors."""
    config = saved.get("experiment_config")
    if config is not None and config != model.experiment_config:
        raise ValueError("Checkpoint configuration does not match model configuration")
    state = saved.get("model_state_dict", saved)
    target_keys = set(model.state_dict())
    missing = target_keys - set(state)
    if allow_latent_initialization and set(state) < target_keys and missing in (LATENT_ANSWER_KEYS, COARSENED_LATENT_ANSWER_KEYS):
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=True)
    # Full checkpoints retain the Q-only control's inference path. Old/raw
    # checkpoints keep the historical predicted-answer default.
    run = saved.get("run_config", {})
    core = model.coarse if hasattr(model, "coarse") else model
    core.question_only = run.get("training_config", {}).get("question_only", run.get("question_only", False))


def initialize_weights(path, model):
    saved = path if isinstance(path, dict) else read_checkpoint(path)
    state = saved["model_state_dict"]
    target = model
    if hasattr(model, "coarse"):
        coarse_keys = set(model.coarse.state_dict())
        if set(state) == coarse_keys or (set(state) < coarse_keys and coarse_keys - set(state) == LATENT_ANSWER_KEYS):
            target = model.coarse
    load_model_weights(target, saved, allow_latent_initialization=True)
