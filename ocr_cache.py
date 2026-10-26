"""OCR-provider-neutral cache schema and geometry, with no deep-learning dependencies."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CACHE_VERSION = "upright-ocr-v1"


def pixel_hash(image):
    return hashlib.sha256(np.asarray(image).tobytes()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def validate_record(record):
    if record.get("version") != CACHE_VERSION:
        raise ValueError("Unsupported OCR record schema")
    width, height = record["size"]
    if min(width, height) <= 0:
        raise ValueError("Invalid OCR image size")
    for item in record["regions"]:
        polygon = np.asarray(item["polygon"], dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValueError("OCR polygons require at least three xy points")
        if (
            not np.isfinite(polygon).all()
            or not np.isfinite(item["confidence"])
            or not 0 <= item["confidence"] <= 1
        ):
            raise ValueError("Nonfinite geometry or invalid OCR confidence")
        if not isinstance(item["text"], str):
            raise TypeError("Recognized OCR text must be a string")


def candidates(record, max_candidates=24, context=1.25):
    """Deterministic confidence order, independent of questions, answers and GT."""
    validate_record(record)
    width, height = record["size"]
    result, seen = [], set()
    for item in sorted(record["regions"], key=lambda r: -r["confidence"]):
        points = np.asarray(item["polygon"], dtype=np.float32)
        lo, hi = points.min(0), points.max(0)
        if (hi - lo).min() < 2:
            continue
        center, half = (lo + hi) / 2, (hi - lo) * context / 2
        box = np.concatenate(
            (np.maximum(center - half, 0), np.minimum(center + half, [width, height]))
        )
        if min(box[2:] - box[:2]) < 2:
            continue
        key = tuple(np.round(box).astype(int))
        if key in seen:
            continue
        seen.add(key)
        result.append({**item, "box": (box / [width, height, width, height]).tolist()})
        if len(result) == max_candidates:
            break
    return result


def polygon_mask(record, region):
    mask = Image.new("L", tuple(record["size"]), 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in region["polygon"]], fill=255)
    return np.asarray(mask) > 0
