"""Identical native final/global/ROI supervision for the matched variants."""

import math

import torch
from torch.nn import functional as F

from native_refinement_ops import local_window, native_loss, pixel_window

OBJECTIVE_VERSION = "matched-joint-native-v1"
DEFAULTS = {
    "pyramid_weight": 0.25,
    "roi_weight": 0.25,
    "aux96_weight": 0.1,
    "aux192_weight": 0.2,
    "aux384_weight": 0.3,
}


class JointObjective:
    def __init__(self, **settings):
        if settings.keys() - DEFAULTS.keys():
            raise ValueError("Unknown joint objective setting")
        self.settings = {**DEFAULTS, **settings}
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0
            for v in self.settings.values()
        ):
            raise ValueError("Objective weights must be finite and nonnegative")

    def __call__(self, output, batch):
        final = []
        pyramid = []
        roi = []
        aux = []
        for i, truth in enumerate(batch["original_mask"]):
            size = tuple(truth.shape[-2:])
            raw = output["pyramid_native"][i]
            # Share one native target across global, auxiliary and crop losses.
            # Repeated CPU->GPU copies would also retain duplicate full-size masks.
            truth = truth.to(raw.device, dtype=torch.float32, non_blocking=True)
            final.append(native_loss(output["native_logits"][i], truth))
            pyramid.append(native_loss(raw, truth))
            parts = []
            available = output["candidate_valid"][i].detach().cpu().tolist()
            for j, box in enumerate(output["refine_boxes"][i].detach().cpu().tolist()):
                if not available[j]:
                    continue
                l, t, r, b = pixel_window(box, size)
                if r <= l or b <= t:
                    continue
                if "local_logits" in output:
                    _, predicted = local_window(output["local_logits"][i, j], box, size)
                else:
                    predicted = raw[:, t:b, l:r]
                parts.append(native_loss(predicted, truth[:, t:b, l:r]))
            roi.append(torch.stack(parts).mean() if parts else raw.float().sum() * 0)
            auxiliary = raw.float().sum() * 0
            for key, pred in output["auxiliary_logits"].items():
                z = F.interpolate(
                    pred[i : i + 1].float(), size, mode="bilinear", align_corners=False
                )[0]
                auxiliary = auxiliary + self.settings[f"aux{key}_weight"] * native_loss(
                    z, truth
                )
            aux.append(auxiliary)
        f, p, r, a = [torch.stack(x).mean() for x in (final, pyramid, roi, aux)]
        loss = (
            f
            + self.settings["pyramid_weight"] * p
            + self.settings["roi_weight"] * r
            + a
        )
        if "local_logits" in output:
            loss = loss + output["local_logits"].float().sum() * 0
        return loss, {
            "final": f.detach(),
            "pyramid": p.detach(),
            "roi": r.detach(),
            "auxiliary": a.detach(),
        }
