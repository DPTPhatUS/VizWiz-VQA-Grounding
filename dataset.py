import json
import os
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class VizWizGroundingDataset(Dataset):
    def __init__(self, json_path, image_root, mask_root=None, image_size=(336, 336), is_test=False):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.mask_root = mask_root
        self.is_test = is_test
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        self.entries = list(self.data.items())

        self.normalize = T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        filename, meta = self.entries[idx]

        # 1. Load image and apply EXIF orientation safely (closing file handle)
        image_path = os.path.join(self.image_root, filename)
        with Image.open(image_path) as img:
            image = ImageOps.exif_transpose(img).convert("RGB")

        # 2. Sanitize question and answer strings
        question = str(meta.get("question", "")).strip()
        answer = str(meta.get("most_common_answer", "")).strip()
        text = f"Q: {question} A: {answer}" if answer else f"Q: {question}"

        # 3. Load ground-truth mask safely
        if self.mask_root:
            mask_path = os.path.join(self.mask_root, os.path.splitext(filename)[0] + ".png")
            if os.path.exists(mask_path):
                with Image.open(mask_path) as m:
                    mask = m.convert("L")
            else:
                mask = Image.new("L", image.size, 0)
        else:
            mask = Image.new("L", image.size, 0)

        # 4. Resize: BICUBIC for RGB image (CLIP standard), NEAREST for binary mask
        image = TF.resize(image, self.image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True)
        mask = TF.resize(mask, self.image_size, interpolation=T.InterpolationMode.NEAREST)

        # 5. Convert to tensor and apply CLIP normalization
        image_tensor = self.normalize(TF.to_tensor(image))
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()

        return {
            "image": image_tensor,
            "text": text,
            "mask": mask_tensor,
            "filename": filename,
        }