"""Deterministic OCR mask hypotheses; answers and masks are supervision only."""

import hashlib
import json
import re

import numpy as np
import torch
from PIL import Image, ImageDraw

from ocr_cache import candidates, file_hash
from ocr_data import OCRGroundingDataset
from ocr_routing_data import routing_inputs

HYPOTHESIS_VERSION = "native-ocr-hypotheses-v1"
SCORE_SIZE = 96


def normalize_text(text):
    return " ".join(re.findall(r"\w+", str(text).casefold()))


def hypothesis_members(regions, max_candidates, image_size):
    """Singles, at most N local groups, then all words. No Q/GT-dependent proposals."""
    groups = [(i,) for i in range(len(regions))]
    local = set()
    # Nearby similarly sized words on one line; at most two neighbours per anchor.
    for i, region in enumerate(regions):
        a = np.asarray(region["box"]) * np.tile(image_size, 2)
        height = a[3] - a[1]
        neighbours = []
        for j, other in enumerate(regions):
            if i == j:
                continue
            b = np.asarray(other["box"]) * np.tile(image_size, 2)
            scale = max(height, b[3] - b[1])
            overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            gap = max(0, max(a[0], b[0]) - min(a[2], b[2]))
            if (
                min(height, b[3] - b[1]) >= 0.5 * scale
                and overlap >= 0.5 * min(height, b[3] - b[1])
                and gap <= 2 * scale
            ):
                neighbours.append((gap, j))
        if neighbours:
            group = tuple(sorted([i, *[j for _, j in sorted(neighbours)[:2]]]))
            local.add(group)
    groups.extend(sorted(local)[:max_candidates])
    if len(regions) > 1:
        groups.append(tuple(range(len(regions))))
    groups = list(dict.fromkeys(groups))
    members = torch.zeros(2 * max_candidates + 2, max_candidates, dtype=torch.bool)
    # Index zero is reserved for the live parent mask.
    for k, group in enumerate(groups, 1):
        members[k, list(group)] = True
    return members


def draw_hypothesis(polygons, members, size, *, origin=(0, 0), scale=(1, 1)):
    canvas = Image.new("L", size, 0)
    draw = ImageDraw.Draw(canvas)
    for index in members:
        points = [
            ((x - origin[0]) * scale[0], (y - origin[1]) * scale[1])
            for x, y in polygons[index]
        ]
        draw.polygon(points, fill=255)
    return canvas


def selection_inputs(image, record, max_candidates=24):
    result = routing_inputs(image, record, max_candidates)
    regions = candidates(record, max_candidates)
    polygons = [r["polygon"] for r in regions]
    members = hypothesis_members(regions, max_candidates, image.size)
    masks = torch.zeros(len(members), SCORE_SIZE, SCORE_SIZE)
    width, height = image.size
    for k in range(1, len(members)):
        ids = members[k].nonzero().flatten().tolist()
        if ids:
            canvas = draw_hypothesis(
                polygons,
                ids,
                (SCORE_SIZE * 2, SCORE_SIZE * 2),
                scale=(SCORE_SIZE * 2 / width, SCORE_SIZE * 2 / height),
            ).resize((SCORE_SIZE, SCORE_SIZE), Image.Resampling.BOX)
            masks[k] = torch.from_numpy(np.array(canvas, dtype=np.float32) / 255)
    result.update(
        hypothesis_members=members,
        hypothesis_masks=masks,
        evidence_polygons=polygons,
    )
    return result


def native_hypothesis(size, polygons, members):
    """Exact native polygon union used by inference/export, without a 768px detour."""
    return np.asarray(draw_hypothesis(polygons, members, size)) > 0


def selection_targets(mask, inputs, *, include_word_purity=False):
    truth = np.asarray(mask) > 127
    height, width = truth.shape
    total = int(truth.sum())
    polygons = inputs["evidence_polygons"]
    members = inputs["hypothesis_members"]
    quality = torch.zeros(len(members))
    purity = torch.zeros(members.shape[1]) if include_word_purity else None
    for k in range(1, len(members)):
        ids = members[k].nonzero().flatten().tolist()
        if not ids:
            continue
        # PIL edge rounding is not perfectly translation invariant. Draw in the
        # original coordinates, then crop only for counting against native GT.
        points = np.asarray([p for i in ids for p in polygons[i]])
        left, top = np.maximum(np.floor(points.min(0)), 0).astype(int)
        right, bottom = np.minimum(np.floor(points.max(0)) + 1, [width, height]).astype(
            int
        )
        if right <= left or bottom <= top:
            continue
        local = (
            np.asarray(
                draw_hypothesis(polygons, ids, (width, height)).crop(
                    (left, top, right, bottom)
                )
            )
            > 0
        )
        intersection = int((local & truth[top:bottom, left:right]).sum())
        area = int(local.sum())
        quality[k] = intersection / max(1, area + total - intersection)
        if include_word_purity and len(ids) == 1:
            purity[ids[0]] = intersection / max(1, area)
    return {
        "hypothesis_quality": quality,
        "original_mask": torch.from_numpy(truth.copy())[None],
        **({"ocr_word_purity": purity} if include_word_purity else {}),
    }


class OCRSelectionDataset(OCRGroundingDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supervision_digest = None
        if self.split == "train":
            sources = {"annotations": file_hash(self.root / "train_grounding.json")}
            sources.update(
                {
                    name: file_hash(
                        self.root
                        / "binary_masks_png"
                        / "train"
                        / (name.rsplit(".", 1)[0] + ".png")
                    )
                    for name in self.names
                }
            )
            self.supervision_digest = hashlib.sha256(
                json.dumps(sources, sort_keys=True).encode()
            ).hexdigest()

    def _native_inputs(self, image, record):
        return selection_inputs(image, record, self.max_candidates)

    def _training_targets(self, mask, inputs):
        # Answer text never enters the returned model inputs.
        return selection_targets(mask, inputs)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self.split == "train":
            # Reuse targets from the first pass; only the small answer labels differ.
            answer = normalize_text(
                self.records[self.names[index]].get("most_common_answer") or ""
            )
            matches = torch.tensor(
                [
                    bool(answer and f" {answer} " in f" {normalize_text(t)} ")
                    for t in sample["ocr_texts"]
                ]
            )
            # Single hypotheses occupy indices 1..number of OCR candidates.
            q = sample["hypothesis_quality"][1 : 1 + self.max_candidates]
            # IoU >= .5 implies both precision and recall >= .5.
            supported = matches & (q >= 0.5) & sample["ocr_valid"]
            sample["answer_candidate_target"] = supported.float()
        return sample
