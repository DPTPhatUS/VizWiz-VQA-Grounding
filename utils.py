import torch


def to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(to_device(v, device) for v in batch)
    return batch


def grounding_loss(predictions, masks, loss_fn):
    """Equal image weighting, preserving baseline BCE for fixed-size batches."""
    if torch.is_tensor(predictions):
        predictions = torch.nn.functional.interpolate(
            predictions, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        return loss_fn(predictions, masks)
    if len(predictions) != len(masks) or not predictions:
        raise ValueError("Prediction/mask batch sizes differ or are empty")
    losses = []
    for prediction, mask in zip(predictions, masks):
        prediction = torch.nn.functional.interpolate(
            prediction, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        losses.append(loss_fn(prediction, mask.unsqueeze(0)))
    return torch.stack(losses).mean()


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
