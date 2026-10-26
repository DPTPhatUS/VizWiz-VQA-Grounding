"""Native OCR candidates and a dataset wrapper; model inputs never require GT or answers."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor

from dataset import VizWizGroundingDataset, image_views
from models.region_ops import PATCH_SIZE
from ocr_cache import CACHE_VERSION, candidates, file_hash, pixel_hash, read_json


def native_candidates(image, record, max_candidates=24):
    """GT-free candidate inputs, also usable with arbitrary images outside VizWiz."""
    if (
        list(image.size) != record["size"]
        or pixel_hash(image) != record["pixel_sha256"]
    ):
        raise ValueError("OCR coordinates/image hash mismatch")
    regions = candidates(record, max_candidates)
    boxes = torch.zeros(max_candidates, 4)
    patches = torch.zeros(max_candidates, 3, *PATCH_SIZE)
    confidence = torch.zeros(max_candidates)
    valid = torch.zeros(max_candidates, dtype=torch.bool)
    texts = [""] * max_candidates
    for i, region in enumerate(regions):
        box = region["box"]
        crop = image.crop(tuple(x * scale for x, scale in zip(box, image.size * 2)))
        patches[i] = to_tensor(crop.resize(PATCH_SIZE[::-1], Image.Resampling.BICUBIC))
        boxes[i] = torch.tensor(box)
        confidence[i], valid[i], texts[i] = region["confidence"], True, region["text"]
    return {
        "ocr_boxes": boxes,
        "ocr_patches": patches,
        "ocr_confidence": confidence,
        "ocr_valid": valid,
        "ocr_texts": texts,
    }


def inference_sample(
    image, question, record, image_size=336, detail_size=768, max_candidates=24
):
    """Image must already be upright RGB, matching the OCR record."""
    return {
        **image_views(image, image_size, detail_size),
        **native_candidates(image, record, max_candidates),
        "question_text": f"Q: {question.strip()}",
    }


def evidence_targets(mask, boxes, valid):
    """Native GT quality and crop targets; training supervision only."""
    n = len(boxes)
    truth = np.asarray(mask) > 127
    integral = np.pad(truth.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    total = int(truth.sum())
    quality, patch_target = torch.zeros(n), torch.zeros(n, 1, *PATCH_SIZE)
    for i in valid.nonzero().flatten().tolist():
        xy = tuple(x * scale for x, scale in zip(boxes[i].tolist(), mask.size * 2))
        patch_target[i] = to_tensor(
            mask.crop(xy).resize(PATCH_SIZE[::-1], Image.Resampling.BOX)
        )
        x1, y1, x2, y2 = [round(v) for v in xy]
        intersection = (
            integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
        )
        quality[i] = 2 * intersection / max(1, (x2 - x1) * (y2 - y1) + total)
    return {"ocr_quality": quality, "ocr_patch_target": patch_target}


class OCRGroundingDataset(VizWizGroundingDataset):
    def __init__(
        self,
        root,
        split,
        cache_root,
        image_size=336,
        *,
        detail_size=768,
        max_candidates=24,
    ):
        super().__init__(root, split, image_size, detail_size=detail_size)
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        self.cache_root = Path(cache_root)
        manifest = read_json(self.cache_root / "manifest.json")
        if manifest.get("version") != CACHE_VERSION:
            raise ValueError("Invalid OCR cache manifest")
        self.cache_protocol = manifest["protocol"]
        self.max_candidates = max_candidates
        record_hashes = {}
        for name in self.names:
            path = self.cache_root / split / Path(name).with_suffix(".json")
            if not path.is_file():
                raise ValueError(
                    f"Missing OCR cache for {split}/{name}; generate the cache first"
                )
            record_hashes[name] = file_hash(path)
        self.cache_digest = hashlib.sha256(
            json.dumps(record_hashes, sort_keys=True).encode()
        ).hexdigest()

    def shuffle_recognized_text(self, seed):
        """Stable cross-image permutation without Q, answers or masks."""
        locations, texts = [], []
        for name in self.names:
            record = read_json(
                self.cache_root / self.split / Path(name).with_suffix(".json")
            )
            for i, region in enumerate(candidates(record, self.max_candidates)):
                locations.append((name, i))
                texts.append(region["text"])
        shuffled = texts.copy()
        random.Random(seed).shuffle(shuffled)
        self.text_override = dict(zip(locations, shuffled))
        return sum(a != b for a, b in zip(texts, shuffled))

    def __getitem__(self, index):
        result = super().__getitem__(index)
        name = self.names[index]
        record = read_json(
            self.cache_root / self.split / Path(name).with_suffix(".json")
        )
        if record["filename"] != name:
            raise ValueError(f"OCR filename mismatch: {name}")
        with Image.open(self.root / self.split / name) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        inputs = native_candidates(image, record, self.max_candidates)
        for i, text in enumerate(inputs["ocr_texts"]):
            inputs["ocr_texts"][i] = getattr(self, "text_override", {}).get(
                (name, i), text
            )
        result.update(inputs)
        if self.split == "train":
            with Image.open(
                self.root
                / "binary_masks_png"
                / self.split
                / Path(name).with_suffix(".png")
            ) as source:
                mask = source.convert("L")
            result.update(
                evidence_targets(mask, inputs["ocr_boxes"], inputs["ocr_valid"])
            )
        return result
