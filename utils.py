"""Small shared helpers: device transfer, state-dict key alignment, distributed setup."""

import os

import torch
import torch.distributed as dist


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def setup_distributed():
    """Initialise the process group from torchrun's environment, if present.

    Returns (rank, world_size, local_rank, device). NCCL with one GPU per process; gloo on CPU (tests).
    """
    if "RANK" not in os.environ:
        return 0, 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank, world_size, local_rank = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", device_id=device)
    else:
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo")
    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def align_state_dict(state_dict, model):
    """Normalize DDP prefix and CLIPVisionModel/CLIPTextModel key paths across transformers versions."""
    target_state = model.state_dict()
    target_keys = set(target_state.keys())
    replacements = (
        ("vision_encoder.", "vision_encoder.vision_model."),
        ("vision_encoder.vision_model.", "vision_encoder."),
        ("text_encoder.model.", "text_encoder.model.text_model."),
        ("text_encoder.model.text_model.", "text_encoder.model."),
    )
    remapped = {}
    for k, v in state_dict.items():
        k_clean = k.removeprefix("module.")
        if k_clean in target_keys:
            remapped[k_clean] = v
            continue
        mapped = k_clean
        for old, new in replacements:
            if old in k_clean:
                candidate = k_clean.replace(old, new, 1)
                if candidate in target_keys:
                    mapped = candidate
                    break
        if mapped not in target_keys and mapped.endswith(".position_ids"):
            continue
        remapped[mapped] = v
    for k in target_keys:
        if k not in remapped and k.endswith(".position_ids"):
            remapped[k] = target_state[k]
    return remapped
