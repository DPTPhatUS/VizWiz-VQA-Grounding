import json
import os
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, default_collate
import torchvision.transforms as T
import torchvision.transforms.functional as TF

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class VizWizGroundingDataset(Dataset):
    def __init__(
        self,
        json_path,
        image_root,
        mask_root=None,
        image_size=(336, 336),
        is_test=False,
        use_aug=False,
        text_mode="qa",
        return_original_mask=False,
        naflex_processor=None,
    ):
        if text_mode not in {"qa", "question"}:
            raise ValueError("text_mode must be qa or question")
        if return_original_mask and use_aug:
            raise ValueError("Original-mask evaluation cannot use augmentation")
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.mask_root = mask_root
        self.is_test = is_test
        self.use_aug = use_aug
        self.text_mode = text_mode
        self.return_original_mask = return_original_mask
        self.naflex_processor = naflex_processor
        self.image_size = (
            (image_size, image_size)
            if isinstance(image_size, int)
            else tuple(image_size)
        )
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
        text = f"Q: {question} A: {answer}" if answer and self.text_mode == "qa" else f"Q: {question}"

        # 3. Load ground-truth mask safely
        if self.mask_root:
            mask_path = os.path.join(
                self.mask_root, os.path.splitext(filename)[0] + ".png"
            )
            if os.path.exists(mask_path):
                with Image.open(mask_path) as m:
                    mask = m.convert("L")
            else:
                if self.return_original_mask or self.naflex_processor is not None:
                    raise FileNotFoundError(f"Evaluation mask is missing: {mask_path}")
                mask = Image.new("L", image.size, 0)
        else:
            if self.return_original_mask:
                raise ValueError("Original-mask evaluation requires mask_root")
            mask = Image.new("L", image.size, 0)

        if self.naflex_processor is not None and image.size != mask.size:
            raise ValueError(f"NaFlex image/mask dimensions differ for {filename}; check orientation")
        original_mask = None
        if self.return_original_mask:
            if image.size != mask.size:
                raise ValueError(f"Image/mask dimensions differ for {filename}")
            original_mask = (TF.to_tensor(mask) > 0.5).float()

        if self.naflex_processor is not None:
            if not self.is_test and self.use_aug:
                angle = (0, 90, 180, 270)[int(torch.randint(0, 4, (1,)).item())]
                image, mask = image.rotate(angle, expand=True), mask.rotate(angle, expand=True)
                if torch.rand(1).item() > 0.5:
                    image, mask = TF.hflip(image), TF.hflip(mask)
            inputs = self.naflex_processor(images=image, return_tensors="pt")
            h, w = inputs["spatial_shapes"][0].tolist()
            size = (h * self.naflex_processor.patch_size, w * self.naflex_processor.patch_size)
            mask = TF.resize(mask, size, interpolation=T.InterpolationMode.NEAREST)
            sample = {"image": {k: v[0] for k, v in inputs.items()}, "text": text,
                      "mask": (TF.to_tensor(mask) > 0.5).float(), "filename": filename}
            if original_mask is not None:
                sample["original_mask"] = original_mask
            return sample

        # 4. Resize: BICUBIC for RGB image (CLIP standard), NEAREST for binary mask
        image = TF.resize(
            image,
            self.image_size,
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        )
        mask = TF.resize(
            mask, self.image_size, interpolation=T.InterpolationMode.NEAREST
        )

        # 5. Apply same augmentation (rotation + flip) — currently disabled by default (use_aug=False)
        if not self.is_test and self.use_aug:
            # rotation
            angle_idx = int(torch.randint(0, 4, (1,)).item())
            angle = (0, 90, 180, 270)[angle_idx]
            if angle != 0:
                image = image.rotate(angle)
                mask = mask.rotate(angle)

            # horizontal flip
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        # 6. Convert to tensor and apply CLIP normalization
        image_tensor = self.normalize(TF.to_tensor(image))  # (3, H, W)
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()  # (1, H, W)

        sample = {
            "image": image_tensor,
            "text": text,
            "mask": mask_tensor,
            "filename": filename,
        }
        if original_mask is not None:
            sample["original_mask"] = original_mask
        return sample


def collate_eval_samples(samples):
    """Stack CLIP tensors or NaFlex patches; keep variable-size masks as lists."""
    variable_grid = isinstance(samples[0]["image"], dict)
    excluded = {"original_mask", "mask"} if variable_grid else {"original_mask"}
    batch = default_collate([{k: v for k, v in sample.items() if k not in excluded}
                             for sample in samples])
    if variable_grid:
        batch["mask"] = [sample["mask"] for sample in samples]
    if "original_mask" in samples[0]:
        batch["original_mask"] = [sample["original_mask"] for sample in samples]
    return batch
