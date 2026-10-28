"""Matched pyramid control and end-to-end native crop grounding."""

from collections import OrderedDict

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.model import GroundingModel
from models.native_refiner import NativeRefiner
from models.text_encoder import masked_mean
from native_refinement_ops import global_native, sample_native_regions

EXPERIMENT = "joint-native-crop-v1"


class CheckpointedCrop(NativeRefiner):
    def __init__(self, *args, activation_checkpointing=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation_checkpointing = activation_checkpointing

    def _packed_crops(self, crops, semantic, boxes, tokens, valid):
        ids = valid.flatten().nonzero().flatten()
        b, n = valid.shape
        # Sample each image's valid semantic regions without replicating its full map.
        semantics = []
        for i in range(b):
            selected = boxes[i, valid[i]]
            if len(selected):
                semantics.append(
                    sample_native_regions(
                        semantic[i : i + 1],
                        selected[None],
                        ((crops.shape[-2] + 3) // 4, (crops.shape[-1] + 3) // 4),
                    )[0]
                )
        condition = self.condition(tokens[:, :n].flatten(0, 1)[ids])
        decoded = self._decode_crops(
            crops.flatten(0, 1)[ids], torch.cat(semantics), condition
        )
        full = decoded.new_zeros(b * n, 1, *decoded.shape[-2:]).index_copy(
            0, ids, decoded
        )
        return full.reshape(b, n, 1, *decoded.shape[-2:])

    def _crops(self, crops, semantic, boxes, tokens, valid=None):
        if valid is None:
            return super()._crops(crops, semantic, boxes, tokens)
        if not valid.any():
            # Keep skipped CNN parameters in the DDP graph, without executing it.
            anchor = sum(
                p.flatten()[0].float() * 0
                for module in (
                    self.encoder1,
                    self.encoder2,
                    self.encoder3,
                    self.middle,
                    self.decode2,
                    self.decode1,
                )
                for p in module.parameters()
            )
            return crops.new_zeros(*crops.shape[:2], 1, *crops.shape[-2:]) + anchor
        if self.training and self.activation_checkpointing:
            return checkpoint(
                self._packed_crops,
                crops,
                semantic,
                boxes,
                tokens,
                valid,
                use_reentrant=False,
            )
        return self._packed_crops(crops, semantic, boxes, tokens, valid)


class JointGroundingModel(nn.Module):
    def __init__(
        self,
        variant="pyramid",
        *,
        tiny=False,
        gradient_checkpointing=True,
        ocr_dropout=0.2,
        text_dropout=0.2,
    ):
        super().__init__()
        if variant not in {"pyramid", "joint-crop"}:
            raise ValueError("variant must be pyramid or joint-crop")
        if not 0 <= ocr_dropout < 1 or not 0 <= text_dropout < 1:
            raise ValueError("Invalid dropout")
        # Construct this first in BOTH variants, before any crop-specific RNG draws.
        self.pyramid = GroundingModel(
            tiny=tiny, gradient_checkpointing=gradient_checkpointing
        )
        self.variant, self.ocr_dropout, self.text_dropout = (
            variant,
            ocr_dropout,
            text_dropout,
        )
        self.image_size, self.detail_size = (
            self.pyramid.image_size,
            self.pyramid.detail_size,
        )
        if variant == "joint-crop":
            self.crop = CheckpointedCrop(
                self.pyramid.text_encoder.output_dim,
                self.pyramid.experiment_config["detail_width"],
                self.pyramid.decoder.output_channels,
                32 if tiny else 64,
                "native_crop",
                tiny,
                activation_checkpointing=gradient_checkpointing,
            )
        self._ocr_embeddings = OrderedDict()
        self._text_cache_version = None
        self.inference_view = "full"
        self.experiment_config = {
            "experiment": EXPERIMENT,
            "variant": variant,
            "tiny": tiny,
            "gradient_checkpointing": gradient_checkpointing,
            "ocr_dropout": ocr_dropout,
            "text_dropout": text_dropout,
            "pyramid": self.pyramid.experiment_config,
            "initialization": "pretrained-encoders-fresh-grounding-layers",
            "output_protocol": "native-logits-v1",
            "geometry": "float64-centres-and-native-logit-interpolation-v1",
            "crop_size": [192, 192],
            "context_scale": 2.0,
            "minimum_context_pixels": 32,
            "checkpointed_crop": gradient_checkpointing,
        }

    def forward(self, batch):
        # No no_grad/detach around pyramid: the crop losses must reach its features.
        features, question = self.pyramid.encode_inputs(batch)
        base = self.pyramid.decode_features(
            batch, features, question, return_detail=True
        )
        valid = batch["ocr_valid"].clone()
        b, n = valid.shape
        text_keep = valid.clone()
        if self.training:
            valid &= torch.rand(b, 1, device=valid.device) >= self.ocr_dropout
            text_keep = valid & (
                torch.rand(b, n, device=valid.device) >= self.text_dropout
            )
        else:
            if self.inference_view == "missing":
                valid.zero_()
            text_keep = valid.clone()
            if self.inference_view == "geometry":
                text_keep.zero_()
        sizes = [tuple(rgb.shape[-2:]) for (rgb,) in batch["native_rgb"]]
        raw_native = [
            global_native(
                base["logits"][i : i + 1], base["logits"][i : i + 1] * 0, size
            )
            for i, size in enumerate(sizes)
        ]
        output = {
            "pyramid_logits": base["logits"],
            "auxiliary_logits": base["auxiliary_logits"],
            "native_logits": raw_native,
            "pyramid_native": raw_native,
            "candidate_valid": valid,
            "refine_boxes": batch["refine_boxes"],
        }
        if self.variant == "pyramid" or (
            not self.training and self.inference_view == "pyramid"
        ):
            return output
        recognized = question.tokens.new_zeros(b * n, question.tokens.shape[-1])
        ids = text_keep.flatten().nonzero().flatten()
        if len(ids):
            texts = [text for row in batch["ocr_texts"] for text in row]
            active = [texts[i] for i in ids.tolist()]
            # Frozen text parameters naturally avoid an autograd graph; an optional
            # unfrozen encoder follows the caller's normal gradient context.
            recognized[ids] = self._recognized_embeddings(active).to(recognized.dtype)
        refined = self.crop(
            detail=base["detail_features"],
            semantic=base["semantic_features"],
            base_logits=base["logits"],
            question_tokens=question.tokens,
            question_mask=question.attention_mask,
            recognized_tokens=recognized.reshape(b, n, -1),
            patches=batch["ocr_patches"],
            boxes=batch["ocr_boxes"] * valid[..., None],
            confidence=batch["ocr_confidence"],
            valid=valid,
            refine_boxes=batch["refine_boxes"],
            native_rgb=batch["native_rgb"],
            native_crops=batch["native_crops"],
            global_only=not self.training and self.inference_view == "global",
            reuse_native_windows=self.training,
        )
        output.update(refined)
        if not self.training and self.inference_view == "pyramid":
            output["native_logits"] = raw_native
        return output

    def _recognized_embeddings(self, texts):
        encoder = self.pyramid.text_encoder
        if any(p.requires_grad for p in encoder.parameters()):
            self._ocr_embeddings.clear()
            encoded = encoder(texts)
            return masked_mean(encoded.tokens, encoded.valid_mask)
        device = next(encoder.parameters()).device
        # Invalidate on load_state_dict, device/precision change or encoder mode change.
        version = (
            tuple(p._version for p in encoder.parameters()),
            str(device),
            encoder.training,
            torch.is_autocast_enabled(device.type),
            torch.get_autocast_dtype(device.type),
        )
        if version != self._text_cache_version:
            self._ocr_embeddings.clear()
            self._text_cache_version = version
        missing = list(dict.fromkeys(t for t in texts if t not in self._ocr_embeddings))
        if missing:
            encoded = encoder(missing)
            pooled = masked_mean(encoded.tokens, encoded.valid_mask).detach().cpu()
            for text, value in zip(missing, pooled):
                self._ocr_embeddings[text] = value
        result = torch.stack([self._ocr_embeddings[t] for t in texts]).to(device)
        for text in texts:
            self._ocr_embeddings.move_to_end(text)
        while len(self._ocr_embeddings) > 16384:
            self._ocr_embeddings.popitem(last=False)
        return result
