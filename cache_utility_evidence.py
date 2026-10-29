"""One frozen-parent pass, exact native edit utilities, and compact bit-packed masks."""

import argparse
import shutil
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor
from tqdm.auto import tqdm

from cache_evidence_transport import EvidenceSource
from dataset import make_loader
from models.region_ops import sample_regions
from models.text_encoder import masked_mean
from ocr_cache import atomic_json, candidates, file_hash, read_json
from training.joint_checkpoint import read_joint
from training_image_protocol import image_names
from utility_evidence import VERSION, action_features, pack_mask, split_head
from utils import checkpoint_storage_bytes, save_checkpoint


class UtilitySource(EvidenceSource):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        if "detail_image" not in item:
            image = to_pil_image(item["native_rgb"][0])
            item["detail_image"] = to_tensor(
                image.resize(
                    (self.detail_size, self.detail_size), Image.Resampling.BICUBIC
                )
            )
        return item


def encoded_actions(
    parent, visual, text, base, batch, index, regions, word_embeddings, truth=None
):
    """GT-free builder shared by cache and online prediction."""
    rgb = batch["native_rgb"][index][0]
    size = tuple(rgb.shape[-2:])
    mask = (
        (
            F.interpolate(
                base[index : index + 1].float(),
                size,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            > 0
        )
        .cpu()
        .numpy()
    )
    if truth is not None and truth.shape != mask.shape:
        raise ValueError("Upright mask/image size differs")
    actions, labels = action_features(mask, regions, visual[-1][index], truth)
    boxes = torch.tensor(
        [r["box"] for r in regions], device=base.device, dtype=torch.float32
    ).reshape(-1, 4)
    word_visual = (
        sample_regions(visual[-1][index : index + 1], boxes[None], (4, 4))[0]
        .mean((-1, -2))
        .half()
        .cpu()
        if len(boxes)
        else torch.zeros(0, visual[-1].shape[1], dtype=torch.float16)
    )
    features = {
        **actions,
        "filename": batch["filename"][index],
        "question": text.tokens[index, text.attention_mask[index]].half().cpu().clone(),
        "words": word_embeddings.half().cpu().clone(),
        "word_visual": word_visual,
        "boxes": boxes.cpu(),
        "confidence": torch.tensor([r["confidence"] for r in regions]),
    }
    record = {
        "features": features,
        "parent": pack_mask(mask),
        "polygons": [r["polygon"] for r in regions],
    }
    if labels is not None:
        record["labels"] = labels
    return record, mask


def prepare(args):
    started = time.perf_counter()
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Cache output must be new/empty")
    model, saved = read_joint(args.checkpoint)
    if model.variant != "pyramid":
        raise ValueError("Use a pyramid control parent")
    seen = saved.get("run_config", {}).get("protocol", {}).get("training_images")
    if not isinstance(seen, list) or not seen:
        raise ValueError("Known parent training-image provenance is required")
    parent = model.pyramid.to(args.device).eval().requires_grad_(False)
    datasets = {
        s: UtilitySource(
            args.data_root,
            s,
            args.cache_root,
            parent.image_size,
            detail_size=parent.detail_size,
            max_candidates=args.max_candidates,
            mode="native_point",
        )
        for s in ("train", "val", "test")
    }
    unseen = image_names(args.head_image_list, datasets["train"].names)
    if set(seen) & set(unseen):
        raise ValueError(
            "Head image list overlaps parent training images; train a new disjoint parent first"
        )
    if not set(seen) <= set(datasets["train"].names):
        raise ValueError("Parent provenance names differ from this dataset")

    def pixel_id(name):
        return read_json(
            Path(args.cache_root) / "train" / Path(name).with_suffix(".json")
        )["pixel_sha256"]

    parent_pixels = {pixel_id(n) for n in seen}
    groups = {}
    for name in sorted(unseen):
        identity = pixel_id(name)
        if identity in parent_pixels:
            raise ValueError(
                "Head pixels overlap parent training under another filename"
            )
        groups.setdefault(identity, []).append(name)
    duplicates = [n for group in groups.values() for n in group[1:]]
    unseen = sorted(group[0] for group in groups.values())
    datasets["train"].names = unseen
    cohorts = split_head(unseen, args.seed)
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < 3 * 1024**3:
        raise OSError("Need at least 3 GiB free for compact cache and notebook reserve")
    root.mkdir(parents=True)
    manifest = {
        "version": VERSION,
        "complete": False,
        "parent_sha256": file_hash(args.checkpoint),
        "parent_epoch": saved["epoch"],
        "parent_training_images": sorted(seen),
        "excluded_head_pixel_duplicates": duplicates,
        "cohorts": cohorts,
        "seed": args.seed,
        "max_candidates": args.max_candidates,
        "dimensions": {
            "text_dim": parent.text_encoder.output_dim,
            "visual_dim": parent.image_encoder.out_channels,
        },
        "protocol": datasets["train"].cache_protocol,
        "splits": {},
        "precision": "FP32 extraction; FP16 descriptors; native binary masks packed losslessly",
        "source_sha256": {
            p: file_hash(p)
            for p in (
                "cache_utility_evidence.py",
                "utility_evidence.py",
                "models/model.py",
                "training_image_protocol.py",
            )
        },
    }
    atomic_json(manifest, root / "manifest.json")
    words = OrderedDict()
    with torch.inference_mode():
        for split, data in datasets.items():
            folder = root / split
            folder.mkdir()
            for batch in tqdm(
                make_loader(data, args, args.batch_size), desc=f"Utility cache {split}"
            ):
                inputs = {
                    k: v.to(args.device, non_blocking=True)
                    if k in ("image", "detail_image")
                    else v
                    for k, v in batch.items()
                }
                visual, text = parent.encode_inputs(inputs)
                base = parent.decode_features(inputs, visual, text)["logits"]
                for i, name in enumerate(batch["filename"]):
                    ocr = read_json(
                        Path(args.cache_root) / split / Path(name).with_suffix(".json")
                    )
                    regions = candidates(ocr, args.max_candidates)
                    strings = [r["text"] for r in regions]
                    missing = list(dict.fromkeys(t for t in strings if t not in words))
                    if missing:
                        encoded = parent.text_encoder(missing)
                        for t, v in zip(
                            missing,
                            masked_mean(encoded.tokens, encoded.valid_mask).cpu(),
                        ):
                            words[t] = v.half().clone()
                    embeddings = (
                        torch.stack([words[t] for t in strings])
                        if strings
                        else torch.zeros(0, parent.text_encoder.output_dim)
                    )
                    # Training-cohort labels are a separate payload. No val/test labels are cached.
                    truth = None
                    if split == "train":
                        with Image.open(
                            Path(args.data_root)
                            / "binary_masks_png"
                            / split
                            / Path(name).with_suffix(".png")
                        ) as source:
                            truth = np.asarray(source.convert("L")) > 127
                    record, _ = encoded_actions(
                        parent, visual, text, base, batch, i, regions, embeddings, truth
                    )
                    if (
                        shutil.disk_usage(root).free
                        < checkpoint_storage_bytes(record) * 1.1 + 2 * 1024**3
                    ):
                        raise OSError(
                            "Cache would consume the 2 GiB notebook reserve; no file written"
                        )
                    save_checkpoint(record, folder / f"{Path(name).stem}.pt")
                    for t in strings:
                        words.move_to_end(t)
                    while len(words) > 16384:
                        words.popitem(last=False)
            manifest["splits"][split] = {
                "names": data.names,
                "annotation_sha256": file_hash(
                    Path(args.data_root) / f"{split}_grounding.json"
                ),
                "ocr_digest": data.cache_digest,
                "supervision_digest": data.supervision_digest
                if split == "train"
                else None,
            }
            atomic_json(manifest, root / "manifest.json")
    manifest.update(complete=True, extraction_seconds=time.perf_counter() - started)
    atomic_json(manifest, root / "manifest.json")


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in (
        "checkpoint",
        "head-image-list",
        "data-root",
        "cache-root",
        "output-dir",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--max-candidates", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if (
        args.batch_size < 1
        or args.num_workers < 0
        or not 1 <= args.max_candidates <= 128
    ):
        p.error("Invalid batch/worker/candidate setting")
    prepare(args)


if __name__ == "__main__":
    main()
