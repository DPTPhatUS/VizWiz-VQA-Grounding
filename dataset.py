"""Question-only images with detail training masks and original evaluation masks."""
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms.functional import to_tensor


def load_image(path):
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def image_inputs(image, image_size, detail_size):
    """Resize both inputs independently from the original RGB image."""
    result = {"image": to_tensor(image.resize(
        (image_size, image_size), Image.Resampling.BICUBIC))}
    if detail_size:
        result["detail_image"] = to_tensor(image.resize(
            (detail_size, detail_size), Image.Resampling.BICUBIC))
    return result


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
        image = load_image(self.root / self.split / name)
        with Image.open(self.root / "binary_masks_png" / self.split / Path(name).with_suffix(".png")) as source:
            mask = source.convert("L")
        if image.size != mask.size:
            raise ValueError(f"Image/mask dimensions differ for {name}")
        question = str(meta["question"])
        question_text = f"Q: {question}"
        result = image_inputs(image, self.image_size, self.detail_size)
        result.update(question_text=question_text, filename=name)
        if self.split != "train":
            result["original_mask"] = (to_tensor(mask) > .5).float()
        if self.detail_size and self.split == "train":
            result["detail_mask"] = self.mask_tensor(mask, self.detail_size)
        return result


def collate_samples(samples):
    result = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        if key == "original_mask":
            result[key] = values
        elif isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
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
