"""Original-resolution per-image IoU and mask export."""
import torch
from torch.nn import functional as F
from pathlib import Path
import numpy as np
from PIL import Image
from utils import to_device
from losses import per_image_iou


def metric_samples(logits, batch):
    for i, name in enumerate(batch["filename"]):
        truth = batch["original_mask"][i].unsqueeze(0).to(logits.device)
        resized = F.interpolate(logits[i:i+1].float(), size=truth.shape[-2:],
                                mode="bilinear", align_corners=False)
        score = per_image_iou(resized, truth).item()
        yield name, score, resized[0,0] > 0


@torch.no_grad()
def evaluate(model, loader, device, output=None):
    model.eval()
    scores = {}
    for batch in loader:
        batch = to_device(batch, device)
        # Dataset validation/test always supplies question-only text.
        logits = model(batch)["logits"]
        for name, score, prediction in metric_samples(logits, batch):
            scores[name] = score
            if output is not None:
                path = Path(output) / Path(name).with_suffix(".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((prediction.cpu().numpy().astype(np.uint8)*255)).save(path)
    return scores
