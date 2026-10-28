"""Compact checkpoints omit only verified frozen pretrained encoder tensors."""

import hashlib

import torch

from models.joint_grounding import EXPERIMENT, JointGroundingModel

VERSION = "joint-native-checkpoint-v1"


def tensor_digest(state):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(f"{key}:{value.dtype}:{list(value.shape)}".encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def frozen_layout(model):
    omitted = {}
    if not model.experiment_config["tiny"]:
        for name in ("image_encoder", "text_encoder"):
            module = getattr(model.pyramid, name)
            if all(not p.requires_grad for p in module.parameters()):
                prefix = f"pyramid.{name}."
                omitted[prefix] = tensor_digest(module.state_dict())
    return omitted


def compact_state(model, omitted):
    return {
        k: v
        for k, v in model.state_dict().items()
        if not any(k.startswith(prefix) for prefix in omitted)
    }


def load_state(model, saved):
    if (
        saved.get("version") != VERSION
        or saved["experiment_config"].get("experiment") != EXPERIMENT
    ):
        raise ValueError("Not a joint/native control checkpoint")
    if model.experiment_config != saved["experiment_config"]:
        raise ValueError("Joint architecture metadata mismatch")
    current = model.state_dict()
    omitted = saved["omitted_frozen_encoders"]
    if model.experiment_config["tiny"] and omitted:
        raise ValueError("Tiny random encoders must be stored in full")
    if set(omitted) - {"pyramid.image_encoder.", "pyramid.text_encoder."}:
        raise ValueError("Invalid omitted encoder")
    expected = {k for k in current if any(k.startswith(prefix) for prefix in omitted)}
    for prefix, digest in omitted.items():
        part = {k[len(prefix) :]: v for k, v in current.items() if k.startswith(prefix)}
        if tensor_digest(part) != digest:
            raise ValueError(
                "Frozen pretrained encoder fingerprint mismatch; retain original HF cache/version"
            )
    state = saved["model_state_dict"]
    if set(state) != (set(current) - expected):
        raise ValueError("Checkpoint state keys differ from the exact compact layout")
    # Strict merge, after validating both omitted values and the full key set.
    model.load_state_dict({**current, **state}, strict=True)


def read_joint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("version") != VERSION:
        raise ValueError("Unsupported joint checkpoint")
    c = saved["experiment_config"]
    if c.get("experiment") != EXPERIMENT:
        raise ValueError("Unsupported joint experiment")
    model = JointGroundingModel(
        c["variant"],
        tiny=c["tiny"],
        gradient_checkpointing=c["gradient_checkpointing"],
        ocr_dropout=c["ocr_dropout"],
        text_dropout=c["text_dropout"],
    )
    load_state(model, saved)
    return model, saved


def storage_estimate(model, parameters, omitted, snapshots):
    state_bytes = sum(
        t.numel() * t.element_size() for t in compact_state(model, omitted).values()
    )
    train_bytes = sum(p.numel() * p.element_size() for p in parameters)
    # Four weight snapshots plus old/new full Adam resumes during atomic replacement.
    return {
        "snapshot_bytes": state_bytes,
        "resume_bytes": state_bytes + 2 * train_bytes,
        "peak_checkpoint_bytes": (snapshots + 2) * state_bytes + 4 * train_bytes,
        "note": "Tensor-byte estimate for Adam/AdamW, excluding serializer/config overhead and result files",
    }
