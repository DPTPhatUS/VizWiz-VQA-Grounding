"""Aligned low/high-resolution RGB views, one mask and question-only conditioning."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor


def image_views(
    image: Image.Image, image_size: int = 336, detail_size: int = 768
) -> dict:
    """Both views come directly from the same upright RGB image, in [0, 1]."""
    return {
        "image": to_tensor(
            image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        ),
        "detail_image": to_tensor(
            image.resize((detail_size, detail_size), Image.Resampling.BICUBIC)
        ),
    }


class VizWizGroundingDataset(Dataset):
    def __init__(
        self, root, split, image_size=336, *, detail_size=768, train_transform=None
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("Expected train, val or test split")
        if train_transform is not None and split != "train":
            raise ValueError("Training augmentation cannot be used on val/test")
        self.train_transform = train_transform
        self.root, self.split = Path(root), split
        self.image_size, self.detail_size = image_size, detail_size
        self.records = json.loads((self.root / f"{split}_grounding.json").read_text())
        self.names = list(self.records)
        if not self.names:
            raise ValueError("Dataset is empty")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(self.root / self.split / name) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        with Image.open(
            self.root / "binary_masks_png" / self.split / Path(name).with_suffix(".png")
        ) as source:
            mask = source.convert("L")
        # Masks are annotations in the upright coordinate system, not EXIF-tagged photos.
        if image.size != mask.size:
            raise ValueError(f"Upright image/mask dimensions differ for {name}")
        original_size = (image.height, image.width)
        augmentation = {}
        if self.train_transform is not None:
            source_mask = mask.resize(
                (self.detail_size, self.detail_size), Image.Resampling.NEAREST
            )
            histogram = source_mask.histogram()
            source_area = sum(histogram[128:]) / self.detail_size**2
            image, mask, augmentation = self.train_transform(image, mask)
            augmentation = {
                key: torch.tensor(value, dtype=torch.float64)
                for key, value in augmentation.items()
            }
            augmentation["source_area_fraction"] = torch.tensor(
                source_area, dtype=torch.float64
            )
        question = str(self.records[name].get("question") or "").strip()
        result = image_views(image, self.image_size, self.detail_size)
        result.update(
            mask=(
                to_tensor(
                    mask.resize(
                        (self.detail_size, self.detail_size), Image.Resampling.NEAREST
                    )
                )
                > 0.5
            ).float(),
            question_text=f"Q: {question}",
            filename=name,
            original_size=original_size,
        )
        result.update(augmentation)
        if self.split != "train":
            result["original_mask"] = (to_tensor(mask) > 0.5).float()
        return result


def collate_samples(samples):
    return {
        key: (
            torch.stack([s[key] for s in samples])
            if torch.is_tensor(samples[0][key]) and key != "original_mask"
            else [s[key] for s in samples]
        )
        for key in samples[0]
    }


GroundingDataset = VizWizGroundingDataset


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, args, batch_size, sampler=None, shuffle=False, generator=None):
    kwargs = dict(
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
        pin_memory=str(args.device).startswith("cuda"),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if args.num_workers:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)
