"""Aligned raw RGB inputs and explicitly separated question/answer views."""
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
    def __init__(self, root, split, image_size=336, detail_size=None,
                 text_mode="question", answer_dropout=0.5, pairs=None):
        self.root, self.split = Path(root), split
        if text_mode not in {"question", "answer", "dropout"}:
            raise ValueError("Unknown text mode")
        if split != "train" and text_mode != "question":
            raise ValueError("Research validation/test must be question-only")
        if not 0 <= answer_dropout <= 1:
            raise ValueError("answer_dropout must be in [0,1]")
        self.image_size, self.detail_size = image_size, detail_size
        self.text_mode, self.answer_dropout = text_mode, answer_dropout
        self.records = json.loads((self.root / f"{split}_grounding.json").read_text())
        self.names = list(self.records)
        if not self.names:
            raise ValueError("Dataset is empty")
        self.pairs = {}
        if pairs:
            if split != "train":
                raise ValueError("Paired annotations are training-only")
            manifest = Path(pairs)
            for pair in json.loads(manifest.read_text()):
                required = {"filename", "question1", "question2", "mask1", "mask2", "relation"}
                if not required <= pair.keys() or pair["relation"] not in {"change", "same"}:
                    raise ValueError("Pairs require filename, question1/2, mask1/2, relation=change|same")
                if pair["filename"] not in self.records:
                    raise ValueError("Pair image must belong to this training split")
                if not all(isinstance(pair[k], str) and pair[k].strip() for k in ("question1", "question2")):
                    raise ValueError("Pair questions cannot be empty")
                pair = dict(pair)
                for key in ("mask1", "mask2"):
                    pair[key] = str((manifest.parent / pair[key]).resolve())
                    if not Path(pair[key]).is_file():
                        raise ValueError(f"Pair mask does not exist: {pair[key]}")
                self.pairs.setdefault(pair["filename"], []).append(pair)
            if not self.pairs:
                raise ValueError("Pair manifest contains no valid pairs")

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
        answer = str(meta.get("most_common_answer") or "")
        question_text = f"Q: {question}"
        answer_text = f"{question_text} A: {answer}" if answer else question_text
        use_answer = self.text_mode == "answer" or (
            self.text_mode == "dropout" and random.random() >= self.answer_dropout
        )
        result = {
            "image": to_tensor(image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)),
            "mask": self.mask_tensor(mask, self.image_size),
            "text": answer_text if use_answer else question_text,
            "question_text": question_text, "answer_text": answer_text,
            "answer_available": bool(answer), "filename": name,
            "original_size": (image.height, image.width),
        }
        if self.split != "train":
            result["original_mask"] = (to_tensor(mask) > .5).float()
        if self.detail_size:
            # Independent resize from the original image: never enlarge the CLIP input.
            result["detail_image"] = to_tensor(image.resize(
                (self.detail_size, self.detail_size), Image.Resampling.BICUBIC))
        if self.pairs:
            pair = random.choice(self.pairs[name]) if name in self.pairs else None
            result.update(pair_available=pair is not None, pair_same=False,
                          paired_text=question_text, paired_mask=torch.zeros_like(result["mask"]))
            if pair:
                pair_masks, original_pair_masks = [], []
                for key in ("mask1", "mask2"):
                    with Image.open(pair[key]) as source:
                        pm = source.convert("L")
                    if pm.size != image.size:
                        raise ValueError(f"Pair mask dimensions differ for {name}")
                    original_pair_masks.append(to_tensor(pm) > .5)
                    pair_masks.append(self.mask_tensor(pm, self.image_size))
                if pair["relation"] == "same" and not torch.equal(*original_pair_masks):
                    raise ValueError("Verified same-region pairs must have identical masks")
                result.update(text=f"Q: {pair['question1']}", question_text=f"Q: {pair['question1']}",
                              mask=pair_masks[0], paired_text=f"Q: {pair['question2']}",
                              paired_mask=pair_masks[1], pair_same=pair["relation"] == "same",
                              answer_available=False, answer_text=f"Q: {pair['question1']}")
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
