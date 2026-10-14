import json
import os

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, default_collate
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class VizWizGroundingDataset(Dataset):
    def __init__(self, json_path, image_root, mask_root, image_processor,
                 is_test=False, use_aug=False, text_mode="qa", return_original_mask=False):
        if text_mode not in {"qa", "question"}:
            raise ValueError("text_mode must be qa or question")
        if return_original_mask and use_aug:
            raise ValueError("Native evaluation cannot use augmentation")
        with open(json_path, encoding="utf-8") as f:
            self.entries = list(json.load(f).items())
        self.image_root, self.mask_root = image_root, mask_root
        self.image_processor = image_processor
        self.is_test, self.use_aug = is_test, use_aug
        self.text_mode, self.return_original_mask = text_mode, return_original_mask

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        filename, meta = self.entries[index]
        with Image.open(os.path.join(self.image_root, filename)) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        mask_path = os.path.join(self.mask_root, os.path.splitext(filename)[0] + ".png")
        with Image.open(mask_path) as m:
            mask = m.convert("L")
        if image.size != mask.size:
            raise ValueError(f"Image/mask dimensions differ for {filename}; check orientation")
        q, a = str(meta.get("question", "")).strip(), str(meta.get("most_common_answer", "")).strip()
        text = f"Q: {q} A: {a}" if a and self.text_mode == "qa" else f"Q: {q}"
        original_mask = (TF.to_tensor(mask) > 0.5).float() if self.return_original_mask else None
        if self.use_aug and not self.is_test:
            angle = (0, 90, 180, 270)[int(torch.randint(0, 4, (1,)).item())]
            image, mask = image.rotate(angle, expand=True), mask.rotate(angle, expand=True)
            if torch.rand(1).item() > 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
        # Raw RGB goes directly to SigLIP: rescale and normalize exactly once.
        inputs = self.image_processor(images=image, return_tensors="pt")
        h, w = inputs["spatial_shapes"][0].tolist()
        size = (h * self.image_processor.patch_size, w * self.image_processor.patch_size)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)
        sample = {"image": {k: v[0] for k, v in inputs.items()}, "text": text,
                  "mask": (TF.to_tensor(mask) > 0.5).float(), "filename": filename}
        if original_mask is not None:
            sample["original_mask"] = original_mask
        return sample


def collate_eval_samples(samples):
    """Batch padded patch sequences; retain rectangular/native masks as lists."""
    batch = default_collate([{k: v for k, v in sample.items() if k not in {"mask", "original_mask"}}
                             for sample in samples])
    batch["mask"] = [sample["mask"] for sample in samples]
    if "original_mask" in samples[0]:
        batch["original_mask"] = [sample["original_mask"] for sample in samples]
    return batch
