"""Native RGB context crops; GT masks are never resized for the new losses."""

import hashlib
import json

import numpy as np
import torch

from native_refinement_ops import sample_native_regions
from ocr_cache import candidates, file_hash
from ocr_data import OCRGroundingDataset
from ocr_routing_data import routing_inputs

CROP_SIZE = (192, 192)
TARGET_VERSION = "native-refinement-pixel-centres-v1"


def refinement_inputs(image, record, max_candidates=24, mode="native_crop"):
    result = routing_inputs(image, record, max_candidates)
    boxes = torch.zeros(max_candidates, 4, dtype=torch.float64)
    for j, region in enumerate(candidates(record, max_candidates)):
        x1, y1, x2, y2 = region["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # Context twice the OCR box; at least 32 native pixels in each dimension.
        rw, rh = (
            max(2 * (x2 - x1), 32 / image.width),
            max(2 * (y2 - y1), 32 / image.height),
        )
        boxes[j] = torch.tensor(
            [
                max(0, cx - rw / 2),
                max(0, cy - rh / 2),
                min(1, cx + rw / 2),
                min(1, cy + rh / 2),
            ],
            dtype=torch.float64,
        )
    rgb = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
    result.update(native_rgb=(rgb,), refine_boxes=boxes)
    if mode == "native_crop":
        result["native_crops"] = sample_native_regions(
            rgb[None].float() / 255, boxes[None], CROP_SIZE
        )[0]
    return result


class NativeRefinementDataset(OCRGroundingDataset):
    def __init__(self, *args, mode="native_crop", **kwargs):
        self.mode = mode
        super().__init__(*args, **kwargs)
        self.supervision_digest = None
        if self.split == "train":
            hashes = {
                n: file_hash(
                    self.root / "binary_masks_png/train" / f"{n.rsplit('.', 1)[0]}.png"
                )
                for n in self.names
            }
            hashes["annotations"] = file_hash(self.root / "train_grounding.json")
            self.supervision_digest = hashlib.sha256(
                json.dumps(hashes, sort_keys=True).encode()
            ).hexdigest()

    def _native_inputs(self, image, record):
        return refinement_inputs(image, record, self.max_candidates, self.mode)

    def _training_targets(self, mask, inputs):
        return {
            "original_mask": torch.from_numpy((np.asarray(mask) > 127).copy())[None]
        }


def check_parent_exposure(model, names, required=False):
    run = getattr(model, "parent_training_provenance", {})
    known = run.get("training_lineage_known", False)
    trained = set(run.get("training_images", []))
    overlap = sorted(trained.intersection(names)) if known else None
    if required and (not known or overlap):
        raise ValueError(
            "Unseen-parent training requires known parent lineage and zero image overlap"
        )
    return {
        "lineage_known": known,
        "overlap_images": overlap,
        "parent_unseen": known and not overlap,
    }
