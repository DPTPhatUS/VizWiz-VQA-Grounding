"""Train-only context crops; all GT pixels stay inside a jittered original-RGB window."""

import math
import random


class ContextCrop:
    """Keep source aspect ratio and full GT extent; never invent labels for an empty crop."""

    def __init__(self, probability=0.25):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Crop probability must be finite and in [0, 1]")
        self.probability = probability

    @property
    def config(self):
        return {
            "version": "context-crop-v1",
            "probability": self.probability,
            "min_side_fraction": 0.35,
            "max_side_fraction": 0.9,
            "context_bbox_scales": [1.5, 3.0],
            "aspect_ratio": "source",
            "target_policy": "retain_all_gt",
            "group_assignment": "uncropped_training_mask",
        }

    def __call__(self, image, mask):
        if image.size != mask.size:
            raise ValueError("Context crop needs aligned image and mask")
        width, height = image.size
        box = (0, 0, width, height)
        requested = self.probability > 0 and random.random() < self.probability
        if requested:
            foreground = mask.point(lambda value: 255 if value > 127 else 0)
            bounds = foreground.getbbox()
            if bounds is not None:
                left, top, right, bottom = bounds
                bw, bh = right - left, bottom - top
                # A margin on every available side avoids an object-tight or centered crop.
                padded = (
                    max(0, math.floor(left - bw * 0.25)),
                    max(0, math.floor(top - bh * 0.25)),
                    min(width, math.ceil(right + bw * 0.25)),
                    min(height, math.ceil(bottom + bh * 0.25)),
                )
                relative = max(bw / width, bh / height)
                lower = max(
                    0.35,
                    relative * 1.5,
                    (padded[2] - padded[0]) / width,
                    (padded[3] - padded[1]) / height,
                )
                upper = min(0.9, max(0.5, relative * 3.0))
                if lower <= upper:
                    scale = random.uniform(lower, upper)
                    cw = min(width, max(1, math.ceil(width * scale)))
                    ch = min(height, max(1, math.ceil(height * scale)))
                    x = random.randint(
                        max(0, padded[2] - cw), min(padded[0], width - cw)
                    )
                    y = random.randint(
                        max(0, padded[3] - ch), min(padded[1], height - ch)
                    )
                    box = (x, y, x + cw, y + ch)
        applied = box != (0, 0, width, height)
        info = {
            "crop_box": box,
            "crop_requested": float(requested),
            "crop_applied": float(applied),
            "crop_input_area_fraction": ((box[2] - box[0]) * (box[3] - box[1]))
            / (width * height),
        }
        if not applied:
            return image, mask, info
        return image.crop(box), mask.crop(box), info
