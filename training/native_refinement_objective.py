"""Exact native crop objectives and importance-corrected point supervision."""

import torch
from torch.nn import functional as F

from native_refinement_ops import global_native, local_window, native_loss


class NativeRefinementObjective:
    def __init__(self, architecture):
        self.architecture = architecture

    def __call__(self, output, batch):
        main = []
        auxiliary = []
        pixel = []
        balanced = []
        for i, truth in enumerate(batch["original_mask"]):
            base, delta = (
                output["base_logits"][i : i + 1],
                output["global_delta"][i : i + 1],
            )
            global_logits = global_native(base, delta, tuple(truth.shape[-2:]))
            if self.architecture == "native_crop":
                main.append(native_loss(output["native_logits"][i], truth))
                local = []
                available = output["candidate_valid"][i].detach().cpu().tolist()
                for j, box in enumerate(
                    output["refine_boxes"][i].detach().cpu().tolist()
                ):
                    if available[j]:
                        sample = local_window(
                            output["local_logits"][i, j], box, tuple(truth.shape[-2:])
                        )
                        if sample is not None:
                            (l, t, r, b), logits = sample
                            local.append(native_loss(logits, truth[:, t:b, l:r]))
                zero = output["local_logits"][i].float().sum() * 0
                auxiliary.append(
                    (torch.stack(local).mean() if local else zero)
                    + native_loss(global_logits, truth)
                )
            else:
                samples = output["point_samples"][i]
                target = (
                    truth.to(base.device, non_blocking=True)
                    .flatten()[samples["indices"]]
                    .float()
                )
                with torch.autocast(base.device.type, enabled=False):
                    bce = F.binary_cross_entropy_with_logits(
                        samples["logits"].float(), target, reduction="none"
                    )
                    p = (bce.double() * samples["pixel_weights"]).sum()
                    c = (bce.double() * samples["balanced_weights"]).sum()
                pixel.append(p)
                balanced.append(c)
                main.append(p + 0.25 * c)
                auxiliary.append(2 * native_loss(global_logits, truth))
        primary = torch.stack(main).mean()
        aux = torch.stack(auxiliary).mean()
        loss = primary + 0.25 * aux + output["local_logits"].float().sum() * 0
        return loss, {
            "native_main": primary.detach(),
            "native_aux": aux.detach(),
            "point_pixel_bce": torch.stack(pixel).mean().detach()
            if pixel
            else primary.detach() * 0,
            "point_balanced_bce": torch.stack(balanced).mean().detach()
            if balanced
            else primary.detach() * 0,
        }
