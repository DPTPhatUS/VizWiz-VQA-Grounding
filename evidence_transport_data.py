"""Ragged, backbone-independent feature records; labels live in a separate payload."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torch.utils.data import Dataset

from models.region_ops import sample_regions
from native_refinement_ops import pixel_window, sample_native_regions

VERSION = "evidence-transport-cache-v1"
CANVAS = 192
CROP = 64


def encoded_features(
    name, visual_map, question, words, boxes, contexts, confidence, rgb
):
    """Shared cache/deployment conversion, including the declared storage rounding."""
    n = len(boxes)
    if n:
        crops = sample_native_regions(
            rgb[None].float() / 255, contexts[None], (CROP, CROP)
        )[0]
        crops = crops.mul(255).round().clamp(0, 255).byte().cpu()
        visual = (
            sample_regions(visual_map[None], boxes[None].float(), (4, 4))[0]
            .mean((-1, -2))
            .half()
            .cpu()
        )
    else:
        crops = torch.zeros(0, 3, CROP, CROP, dtype=torch.uint8)
        visual = torch.zeros(0, visual_map.shape[0], dtype=torch.float16)
    return {
        "filename": name,
        "boxes": boxes.double().cpu(),
        "contexts": contexts.double().cpu(),
        "confidence": confidence.cpu().clone(),
        "words": words.half().cpu().clone(),
        "visual": visual,
        "crops": crops,
        "question": question.half().cpu().clone(),
        "scene": visual_map.mean((-1, -2)).half().cpu().clone(),
        "native_size": tuple(rgb.shape[-2:]),
    }


def targets(mask, regions, contexts):
    """Exact native polygon purity/coverage; BOX-resampled masks are explicit surrogates."""
    truth = np.asarray(mask) > 127
    polygons, purity = [], []
    union = np.zeros_like(truth)
    local = []
    source = torch.from_numpy(truth.copy()).float()[None, None]
    for region, box in zip(regions, contexts):
        canvas = Image.new("1", mask.size)
        ImageDraw.Draw(canvas).polygon([tuple(p) for p in region["polygon"]], fill=1)
        polygon = np.asarray(canvas, dtype=bool)
        value = float((truth & polygon).sum() / max(1, polygon.sum()))
        purity.append(value)
        if value >= 0.5:
            union |= polygon
        polygons.append(
            torch.from_numpy(
                np.array(
                    Image.fromarray(polygon.astype("float32")).resize(
                        (CANVAS, CANVAS), Image.Resampling.BOX
                    ),
                    copy=True,
                )
            )
        )
        l, t, r, b = pixel_window(box, truth.shape)
        if r > l and b > t:
            sampled = sample_native_regions(
                source,
                torch.tensor(box, dtype=torch.float64)[None, None],
                (CROP * 2, CROP * 2),
            )[0]
            local.append(F.avg_pool2d(sampled, 2)[0])
        else:
            local.append(torch.zeros(1, CROP, CROP))
    recall = float((truth & union).sum() / max(1, truth.sum()))
    precision = float((truth & union).sum() / max(1, union.sum()))
    sparse = (
        (torch.stack(polygons) if polygons else torch.zeros(0, CANVAS, CANVAS))
        .to_sparse()
        .coalesce()
    )
    return {
        "truth": torch.from_numpy(
            np.array(
                Image.fromarray(truth.astype("float32")).resize(
                    (CANVAS, CANVAS), Image.Resampling.BOX
                ),
                copy=True,
            )
        )[None],
        "purity": torch.tensor(purity),
        "local": torch.stack(local) if local else torch.zeros(0, 1, CROP, CROP),
        "polygons": {
            "indices": sparse.indices().short(),
            "values": sparse.values(),
            "size": tuple(sparse.shape),
        },
        "scope": torch.tensor(
            float(recall >= 0.8 and precision >= 0.5 and truth.any())
        ),
        "oracle_coverage": torch.tensor(recall),
    }


class FeatureDataset(Dataset):
    def __init__(self, root, names, training=False):
        self.root, self.names, self.training = Path(root), names, training

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        record = torch.load(self.root / f"{Path(name).stem}.pt", weights_only=True)
        features = record["features"]
        # Large parent logits stay on CPU and are never loaded during training.
        labels = record.get("targets") if self.training else None
        if labels is not None:
            sparse = labels["polygons"]
            polygons = torch.zeros(sparse["size"], dtype=sparse["values"].dtype)
            polygons[tuple(sparse["indices"].long())] = sparse["values"]
            labels["polygons"] = polygons
        return features, labels


def collate_features(records):
    features, labels = zip(*records)
    n = max(1, max(len(x["boxes"]) for x in features))
    length = max(len(x["question"]) for x in features)
    batch = {
        "canvas_size": (CANVAS, CANVAS),
        "filename": [x["filename"] for x in features],
        "native_size": [x["native_size"] for x in features],
    }
    candidate_keys = ("boxes", "contexts", "confidence", "words", "visual", "crops")
    for key in candidate_keys:
        batch[key] = torch.stack(
            [
                F.pad(x[key], (0, 0) * (x[key].ndim - 1) + (0, n - len(x[key])))
                for x in features
            ]
        )
    batch["valid"] = (
        torch.arange(n)[None]
        < torch.tensor([len(x["boxes"]) for x in features])[:, None]
    )
    batch["question"] = torch.stack(
        [F.pad(x["question"], (0, 0, 0, length - len(x["question"]))) for x in features]
    )
    batch["question_valid"] = (
        torch.arange(length)[None]
        < torch.tensor([len(x["question"]) for x in features])[:, None]
    )
    batch["scene"] = torch.stack([x["scene"] for x in features])
    if labels[0] is None:
        return batch, None
    target = {}
    for key in labels[0]:
        if key in ("purity", "local", "polygons"):
            target[key] = torch.stack(
                [
                    F.pad(x[key], (0, 0) * (x[key].ndim - 1) + (0, n - len(x[key])))
                    for x in labels
                ]
            )
        else:
            target[key] = torch.stack([x[key] for x in labels])
    return batch, target
