"""Original-resolution region/boundary overlap and mask export."""
import math
import torch
from torch.nn import functional as F
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from runtime import to_device
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
    for batch in tqdm(loader, desc="Evaluating", unit="batch", dynamic_ncols=True):
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


def native_logits(logits, batch):
    for i, name in enumerate(batch['filename']):
        truth = batch['original_mask'][i][0].to(logits.device) > .5
        resized = F.interpolate(logits[i:i+1].float(), size=truth.shape,
                                mode='bilinear', align_corners=False)[0, 0]
        yield name, resized, truth


def overlap_counts(prediction, truth):
    return {'intersection': int((prediction & truth).sum()), 'predicted': int(prediction.sum()),
            'ground_truth': int(truth.sum()), 'pixels': truth.numel()}


def boundary_band(mask, ratio=.02):
    """Inner band from square erosion, including the image border.

    Equivalent to r iterations of zero-padded 3x3 erosion, where r is 2% of
    the native image diagonal (at least one). Integral images avoid an extra
    OpenCV dependency and make the cost independent of the erosion radius.
    """
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not all(mask.shape):
        raise ValueError('Boundary masks must be nonempty 2D arrays')
    radius = max(1, round(ratio * math.hypot(*mask.shape)))
    width = 2 * radius + 1
    padded = np.pad(mask, radius)
    integral = np.pad(padded.cumsum(0, dtype=np.int64).cumsum(1), ((1, 0), (1, 0)))
    total = (integral[width:, width:] - integral[:-width, width:]
             - integral[width:, :-width] + integral[:-width, :-width])
    return mask & (total != width * width)


def boundary_counts(prediction, truth):
    predicted_band, truth_band = boundary_band(prediction), boundary_band(truth)
    return {'boundary_intersection': int((predicted_band & truth_band).sum()),
            'boundary_union': int((predicted_band | truth_band).sum())}


def sample_metrics(row):
    i, p, g = (row[k] for k in ('intersection', 'predicted', 'ground_truth'))
    union = p + g - i
    result = {'iou': i / union if union else 1., 'precision': i / p if p else 0.,
            'recall': i / g if g else 0., 'area_ratio': p / g if g else None,
            'relative_area_error': abs(p-g) / g if g else None}
    if 'boundary_union' in row:
        result['boundary_iou'] = (row['boundary_intersection'] / row['boundary_union']
                                  if row['boundary_union'] else 1.)
    return result


def summarize_counts(records):
    rows = [sample_metrics(r) for r in records.values()]
    if not rows:
        raise ValueError('Cannot summarize an empty split')
    means = {}
    for key in ('iou', 'precision', 'recall', 'area_ratio', 'relative_area_error'):
        values = [r[key] for r in rows if r[key] is not None]
        means[key] = sum(values) / len(values) if values else None
    result = {'num_samples': len(rows), 'mean_iou': means['iou'],
            'macro_precision': means['precision'], 'macro_recall': means['recall'],
            'acc50': sum(r['iou'] >= .5 for r in rows) / len(rows),
            'severe_count': sum(r['iou'] < .1 for r in rows),
            'mean_area_ratio': means['area_ratio'], 'mean_relative_area_error': means['relative_area_error'],
            'area_ratio_samples': sum(r['area_ratio'] is not None for r in rows)}
    if any('boundary_iou' in row for row in rows):
        if not all('boundary_iou' in row for row in rows):
            raise ValueError('Boundary counts must cover every image in the split')
        result['boundary_miou'] = sum(r['boundary_iou'] for r in rows) / len(rows)
    return result


@torch.inference_mode()
def evaluate_counts(model, loader, device, output=None, threshold=.5, logit_writer=None, boundary=False):
    import math
    if not 0 < threshold < 1:
        raise ValueError('Probability threshold must be in (0, 1)')
    cutoff = math.log(threshold / (1-threshold))
    model.eval(); records = {}
    for batch in loader:
        batch = to_device(batch, device)
        raw_logits = model(batch)['logits']
        for i, (name, logits, truth) in enumerate(native_logits(raw_logits, batch)):
            if name in records:
                raise ValueError(f'Duplicate sample: {name}')
            pred = logits > cutoff
            records[name] = overlap_counts(pred, truth)
            if boundary:
                records[name].update(boundary_counts(pred, truth))
            if logit_writer is not None:
                logit_writer.save(name, raw_logits[i, 0], truth.shape)
            if output is not None:
                path = Path(output) / Path(name).with_suffix('.png')
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pred.cpu().numpy().astype(np.uint8)*255).save(path)
    return records
