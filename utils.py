import torch


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


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


def compute_iou(pred, gt, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * gt).sum()
    union = ((pred_bin + gt) >= 1).float().sum()
    return (intersection / union).item() if union > 0 else 0.0