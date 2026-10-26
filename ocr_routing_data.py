"""Cached OCR polygons and GT evidence supervision; compatible with the existing cache."""

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor

from models.region_ops import PATCH_SIZE
from ocr_cache import candidates
from ocr_data import OCRGroundingDataset, evidence_targets, native_candidates

ROUTE_COVERAGE = 0.8
TOKEN_PURITY = 0.5
TARGET_VERSION = "native-evidence-coverage-v1"


def routing_inputs(image, record, max_candidates=24):
    inputs = native_candidates(image, record, max_candidates)
    polygon_patches = torch.zeros(max_candidates, 1, *PATCH_SIZE)
    width, height = image.size
    for i, region in enumerate(candidates(record, max_candidates)):
        x1, y1, x2, y2 = region["box"]
        # Supersampled soft polygon prior in the same native crop coordinates.
        shape = (PATCH_SIZE[1] * 2, PATCH_SIZE[0] * 2)
        canvas = Image.new("L", shape, 0)
        points = [
            (
                (x / width - x1) / (x2 - x1) * shape[0],
                (y / height - y1) / (y2 - y1) * shape[1],
            )
            for x, y in region["polygon"]
        ]
        ImageDraw.Draw(canvas).polygon(points, fill=255)
        polygon_patches[i] = to_tensor(
            canvas.resize(PATCH_SIZE[::-1], Image.Resampling.BOX)
        )
    inputs["ocr_polygon_patches"] = polygon_patches
    inputs["ocr_valid"] &= polygon_patches.sum((1, 2, 3)) > 0
    return inputs


def routing_targets(mask, inputs):
    result = evidence_targets(mask, inputs["ocr_boxes"], inputs["ocr_valid"])
    polygons = inputs["ocr_polygon_patches"].float()
    target = result["ocr_patch_target"]
    purity = (polygons * target).sum((1, 2, 3)) / polygons.sum((1, 2, 3)).clamp_min(
        1e-6
    )
    purity = purity.clamp(0, 1) * inputs["ocr_valid"]
    truth = np.asarray(mask) > 127
    support = np.zeros_like(truth)
    height, width = truth.shape
    xs, ys = (np.arange(width) + 0.5) / width, (np.arange(height) + 0.5) / height
    for i in (
        ((purity >= TOKEN_PURITY) & inputs["ocr_valid"]).nonzero().flatten().tolist()
    ):
        x1, y1, x2, y2 = inputs["ocr_boxes"][i].tolist()
        columns, rows = (
            np.flatnonzero((xs >= x1) & (xs < x2)),
            np.flatnonzero((ys >= y1) & (ys < y2)),
        )
        if len(columns) and len(rows):
            support[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1] = True
    coverage = float((support & truth).sum() / max(1, truth.sum()))
    result.update(
        ocr_inclusion_target=purity,
        ocr_route_target=torch.tensor(
            float(coverage >= ROUTE_COVERAGE and truth.any())
        ),
        ocr_route_coverage=torch.tensor(coverage),
    )
    return result


class OCRRoutingDataset(OCRGroundingDataset):
    def _native_inputs(self, image, record):
        return routing_inputs(image, record, self.max_candidates)

    def _training_targets(self, mask, inputs):
        return routing_targets(mask, inputs)
