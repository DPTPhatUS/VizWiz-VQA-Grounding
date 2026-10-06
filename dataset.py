"""Aligned question-only images and masks at coarse and crop resolutions."""
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import to_device
from torchvision.transforms.functional import to_tensor


class VizWizGroundingDataset(Dataset):
    def __init__(self, root, split, image_size=336, detail_size=672):
        self.root, self.split = Path(root), split
        self.image_size, self.detail_size = image_size, detail_size
        self.records = json.loads((self.root / f"{split}_grounding.json").read_text())
        self.names = list(self.records)
        if not self.names:
            raise ValueError("Dataset is empty")

    def __len__(self):
        return len(self.names)

    @staticmethod
    def mask_tensor(mask, size):
        return (to_tensor(mask.resize((size, size), Image.Resampling.BILINEAR)) > 0.5).float()

    def __getitem__(self, index):
        name = self.names[index]
        meta = self.records[name]
        with Image.open(self.root / self.split / name) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        with Image.open(self.root / "binary_masks_png" / self.split / Path(name).with_suffix(".png")) as source:
            mask = source.convert("L")
        if image.size != mask.size:
            raise ValueError(f"Image/mask dimensions differ for {name}")
        question = str(meta["question"])
        question_text = f"Q: {question}"
        result = {
            "image": to_tensor(image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)),
            "mask": self.mask_tensor(mask, self.image_size),
            "text": question_text,
            "question_text": question_text, "filename": name,
            "original_size": (image.height, image.width),
        }
        if self.split != "train":
            result["original_mask"] = (to_tensor(mask) > .5).float()
        if self.detail_size:
            result["detail_mask"] = self.mask_tensor(mask, self.detail_size)
            # Independent resize from the original image: never enlarge the CLIP input.
            result["detail_image"] = to_tensor(image.resize(
                (self.detail_size, self.detail_size), Image.Resampling.BICUBIC))
        return result


def collate_samples(samples):
    result = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        if key == "original_mask":
            result[key] = values
        elif isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], bool):
            result[key] = torch.tensor(values, dtype=torch.bool)
        else:
            result[key] = values
    return result


GroundingDataset = VizWizGroundingDataset


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, args, batch_size, sampler=None, shuffle=False, generator=None):
    kwargs = {"batch_size":batch_size,"sampler":sampler,"shuffle":shuffle,
              "num_workers":args.num_workers,"collate_fn":collate_samples,
              "pin_memory":str(args.device).startswith("cuda"), "worker_init_fn":seed_worker,
              "generator":generator}
    if args.num_workers:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)
