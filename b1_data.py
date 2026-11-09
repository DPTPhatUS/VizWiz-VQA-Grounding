"""Label-blind OCR inputs and immutable feature bank for all continuation arms."""

import hashlib
import json
import unicodedata
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from dataset import VizWizGroundingDataset, collate, load_image


def text_permutation(name: str, count: int):
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    return torch.from_numpy(np.random.default_rng(seed).permutation(count))


def candidates(record: dict, cap: int = 64) -> list:
    w, h = record["size"]
    if w <= 0 or h <= 0 or record["version"] != "upright-ocr-v1":
        raise ValueError("Invalid OCR dimensions/schema")
    for r in record["regions"]:
        p = np.asarray(r["polygon"], dtype=float)
        if (
            p.ndim != 2
            or p.shape[1] != 2
            or len(p) < 3
            or not np.isfinite(p).all()
            or (p < 0).any()
            or (p > [w, h]).any()
            or not isinstance(r["text"], str)
            or not np.isfinite(r["confidence"])
            or not 0 <= r["confidence"] <= 1
        ):
            raise ValueError("Invalid OCR region")
        area = (
            abs(
                np.dot(p[:, 0], np.roll(p[:, 1], -1))
                - np.dot(p[:, 1], np.roll(p[:, 0], -1))
            )
            / 2
        )
        if area == 0 or (np.ptp(p, axis=0) <= 0).any():
            raise ValueError("Degenerate OCR polygon")
    return sorted(
        enumerate(record["regions"]), key=lambda pair: (-pair[1]["confidence"], pair[0])
    )[:cap]


@torch.no_grad()
def build_features(data: Path, cache: Path, encoder, device, counts=None) -> dict:
    """No masks/answers influence OCR features; verify every source RGB hash."""
    encoder.eval()
    bank = {}
    counts = counts or {"train": 6494, "val": 1131}
    with ZipFile(cache) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError("Duplicate OCR archive members")
        for split, count in counts.items():
            records = json.loads((data / f"{split}_grounding.json").read_text())
            if len(records) != count or any(Path(n).name != n for n in records):
                raise ValueError("Unexpected split membership")
            expected = {f"ocr_cache/{split}/{Path(n).stem}.json" for n in records}
            actual = {
                n
                for n in z.namelist()
                if n.startswith(f"ocr_cache/{split}/") and n.endswith(".json")
            }
            if expected != actual or len(expected) != count:
                raise ValueError("Cache membership differs")
            bank[split] = {}
            pending, locations = [], []
            for image_index, name in enumerate(sorted(records)):
                if image_index % 500 == 0:
                    print(
                        f"Verifying OCR identity: {split} {image_index}/{len(records)}",
                        flush=True,
                    )
                r = json.loads(z.read(f"ocr_cache/{split}/{Path(name).stem}.json"))
                image = load_image(data / split / name)
                if (
                    r["filename"] != name
                    or r["size"] != list(image.size)
                    or hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
                    != r["pixel_sha256"]
                ):
                    raise ValueError(f"OCR pixel identity mismatch: {split}/{name}")
                items = candidates(r)
                embedding = torch.zeros(len(items), encoder.output_dim)
                geometry = torch.zeros(len(items), 5)
                texts = []
                for j, (_, item) in enumerate(items):
                    text = unicodedata.normalize("NFC", item["text"]).strip()
                    texts.append(text)
                    pts = np.asarray(item["polygon"], dtype=float)
                    lo, hi = pts.min(0) / r["size"], pts.max(0) / r["size"]
                    geometry[j] = torch.tensor([*lo, *hi, float(not text)])
                    if text:
                        pending.append(text)
                        locations.append((name, j))
                bank[split][name] = {
                    "embedding": embedding,
                    "geometry": geometry,
                    "texts": texts,
                    "indices": [i for i, _ in items],
                    "raw_count": len(r["regions"]),
                    "pixel_sha256": r["pixel_sha256"],
                    "native_wh": r["size"],
                    "truncated_text": [False] * len(items),
                }
            for start in range(0, len(pending), 128):
                if start % 2048 == 0:
                    print(
                        f"Encoding frozen OCR text: {split} {start}/{len(pending)}",
                        flush=True,
                    )
                strings = pending[start : start + 128]
                lengths = [
                    len(ids)
                    for ids in encoder.tokenizer(strings, truncation=False)["input_ids"]
                ]
                tokens, padding = encoder(strings)
                end = (~padding).sum(1) - 1
                pooled = (
                    tokens[torch.arange(len(strings), device=device), end].float().cpu()
                )
                if not torch.isfinite(pooled).all():
                    raise ValueError("Nonfinite frozen OCR embedding")
                for value, length, (name, j) in zip(
                    pooled, lengths, locations[start : start + 128], strict=True
                ):
                    bank[split][name]["embedding"][j] = value
                    bank[split][name]["truncated_text"][j] = (
                        length > encoder.model.config.max_position_embeddings
                    )
            print(
                f"OCR features ready: {split}, {len(records)} images, {len(pending)} nonempty strings",
                flush=True,
            )
    return bank


class OCRDataset(VizWizGroundingDataset):
    def __init__(self, root, split, bank, *, view="clean", native_mask=False):
        super().__init__(root, split, native_mask=native_mask, text_mode="question")
        if set(bank) != set(self.filenames):
            raise ValueError("Feature bank IDs mismatch")
        if view not in {"clean", "empty", "cross_image", "text_shuffle", "geometry"}:
            raise ValueError(view)
        self.bank, self.view = bank, view
        names = sorted(self.filenames)
        order = np.random.default_rng(42).permutation(len(names))
        self.donors = {
            names[order[i]]: names[order[(i + 1) % len(names)]]
            for i in range(len(names))
        }

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        name = sample["filename"]
        record = self.bank[self.donors[name] if self.view == "cross_image" else name]
        e, g = record["embedding"].clone(), record["geometry"].clone()
        if self.view == "empty":
            e, g = e[:0], g[:0]
        if self.view == "geometry":
            e.zero_()
            g[:, 4] = 1
        if self.view == "text_shuffle" and len(e) > 1:
            # Deterministic within-image permutation; preserve geometry/count.
            order = text_permutation(name, len(e))
            e, g[:, 4] = e[order], g[order, 4]
        sample["ocr"] = (e, g)
        return sample


def ocr_collate(samples):
    batch = collate(samples)
    n = max(1, max(len(s["ocr"][0]) for s in samples))
    dim = samples[0]["ocr"][0].shape[1]
    e, g, valid = (
        torch.zeros(len(samples), n, dim),
        torch.zeros(len(samples), n, 5),
        torch.zeros(len(samples), n, dtype=torch.bool),
    )
    for i, sample in enumerate(samples):
        k = len(sample["ocr"][0])
        e[i, :k], g[i, :k], valid[i, :k] = *sample["ocr"], True
    batch["text"] = {
        "questions": batch["text"],
        "embedding": e,
        "geometry": g,
        "valid": valid,
    }
    return batch
