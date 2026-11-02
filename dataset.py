"""VizWiz-VQA-Grounding data: question-only inputs (the answer field is never read)."""

import json
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from torch.utils.data import Dataset

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_SIZE = 336  # CLIP ViT-L/14-336 input; the full frame is resized without keeping the aspect ratio


def load_image(path) -> Image.Image:
    """Upright RGB image: EXIF orientation applied, as the annotators saw it."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def preprocess_image(image: Image.Image, size: int = IMAGE_SIZE) -> torch.Tensor:
    """PIL RGB -> (3, size, size) CLIP-normalised float tensor (bicubic, antialiased)."""
    image = TF.resize(image, [size, size], interpolation=T.InterpolationMode.BICUBIC, antialias=True)
    return TF.normalize(TF.to_tensor(image), CLIP_MEAN, CLIP_STD)  # (3, size, size)


def question_text(question: str) -> str:
    return f"Q: {str(question).strip()}"


class VizWizGroundingDataset(Dataset):
    """One split of VizWiz-VQA-Grounding under root: {split}_grounding.json, {split}/, binary_masks_png/{split}/.

    Each sample: image (3, S, S) normalised, mask (1, S, S) float {0, 1} (nearest resize), text
    "Q: <question>", filename; with native_mask=True also native_mask (1, H, W) bool at the original
    image resolution, used by the official native-resolution metric.
    """

    def __init__(self, root, split: str, *, image_size: int = IMAGE_SIZE, native_mask: bool = False):
        self.root, self.split = Path(root), split
        self.records = json.loads((self.root / f"{split}_grounding.json").read_text(encoding="utf-8"))
        self.filenames = list(self.records)
        self.image_size = image_size
        self.native_mask = native_mask

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict:
        filename = self.filenames[index]
        image = load_image(self.root / self.split / filename)
        mask_path = self.root / "binary_masks_png" / self.split / f"{Path(filename).stem}.png"
        with Image.open(mask_path) as m:  # FileNotFoundError if missing: never train on a made-up mask
            mask = m.convert("L")
        if mask.size != image.size:
            raise ValueError(f"Image and mask sizes differ for {filename}: {image.size} vs {mask.size}")
        size = [self.image_size, self.image_size]
        sample = {
            "image": preprocess_image(image, self.image_size),  # (3, S, S)
            "mask": (TF.to_tensor(TF.resize(mask, size, interpolation=T.InterpolationMode.NEAREST)) > 0.5).float(),  # (1, S, S)
            "text": question_text(self.records[filename]["question"]),
            "filename": filename,
        }
        if self.native_mask:
            sample["native_mask"] = TF.to_tensor(mask) > 0.5  # (1, H, W) bool
        return sample


def collate(samples: list[dict]) -> dict:
    """Stack fixed-size tensors; keep text, filenames and variable-size native masks as lists."""
    batch = {
        "image": torch.stack([s["image"] for s in samples]),  # (B, 3, S, S)
        "mask": torch.stack([s["mask"] for s in samples]),  # (B, 1, S, S)
        "text": [s["text"] for s in samples],
        "filename": [s["filename"] for s in samples],
    }
    if "native_mask" in samples[0]:
        batch["native_mask"] = [s["native_mask"] for s in samples]
    return batch
