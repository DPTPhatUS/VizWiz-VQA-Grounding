import os

import torch
from torch import nn
from PIL import Image, ImageOps
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor

from .image_encoder import ImageEncoder, VISION_MODEL
from .text_encoder import TextEncoder
from .mask_decoder import UNetDecoder
from metrics import compute_iou_per_sample
from utils import align_state_dict, to_device


class GroundingModel(nn.Module):
    def __init__(self, n_heads=8, max_num_patches=512, text_backbone="clip", pretrained=True,
                 vision_config=None, text_config=None, image_processor=None,
                 feature_layers=(4, 7, 9), decoder_channels=(1024, 512, 256, 128), text_encoder=None):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained, vision_config, feature_layers,
                                         max_num_patches, image_processor)
        self.text_encoder = text_encoder or TextEncoder(text_backbone, pretrained, text_config)
        self.hidden_dim = self.image_encoder.out_channels
        self.text_proj = nn.Linear(self.text_encoder.output_dim, self.hidden_dim)
        self.cross_attn = nn.MultiheadAttention(self.hidden_dim, n_heads, batch_first=True)
        self.decoder = UNetDecoder(in_channels=self.hidden_dim, mid_channels=tuple(decoder_channels))
        self.model_config = {
            "architecture": "siglip2-naflex-grounding-v1", "vision_model": VISION_MODEL,
            "max_num_patches": max_num_patches, "n_heads": n_heads, "text_backbone": text_backbone,
            "vision_config": self.image_encoder.vision_encoder.config.to_dict(),
            "text_config": self.text_encoder.model.config.to_dict(),
            "image_processor": self.image_encoder.processor_config,
            "feature_layers": list(feature_layers), "decoder_channels": list(decoder_channels),
        }

    def enable_gradient_checkpointing(self):
        options = {"use_reentrant": False}
        self.image_encoder.vision_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=options)
        self.text_encoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=options)

    def forward(self, image, text):
        f1, f2, f3, bottleneck = self.image_encoder(image)
        text_tokens, padding_mask = self.text_encoder(text, return_mask=True)
        text_tokens = self.text_proj(text_tokens)
        fused, _ = self.cross_attn(bottleneck.to(text_tokens.dtype), text_tokens, text_tokens,
                                  key_padding_mask=padding_mask, need_weights=False)
        predictions = [None] * len(text)
        for indices, features in self.image_encoder.spatial_groups((fused, f3, f2, f1), image):
            decoded = self.decoder(*features)
            for j, i in enumerate(indices):
                predictions[i] = decoded[j:j + 1]
        return predictions

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None):
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_grounding_model(checkpoint.get("model_config"), pretrained=False)
        model.load_state_dict(align_state_dict(checkpoint["model_state_dict"], model))
        return model.to(device).eval()

    @torch.no_grad()
    def predict(self, image, text=None, question=None, answer=None, gt_mask=None, threshold=0.5):
        self.eval()
        if isinstance(image, (str, os.PathLike)):
            with Image.open(image) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
        else:
            image = ImageOps.exif_transpose(image).convert("RGB")
        if text is None:
            q, a = str(question or "").strip(), str(answer or "").strip()
            text = f"Q: {q} A: {a}" if a else f"Q: {q}"
        inputs = self.image_encoder.make_processor()(images=image, return_tensors="pt")
        inputs = to_device(dict(inputs), next(self.parameters()).device)
        logits = self(inputs, [text])[0]
        logits = F.interpolate(logits.float(), (image.height, image.width), mode="bilinear", align_corners=False)
        prob = logits.sigmoid()[0, 0].cpu()
        binary = (prob > threshold).float()
        truth, iou = None, None
        if gt_mask is not None:
            if isinstance(gt_mask, (str, os.PathLike)):
                with Image.open(gt_mask) as m:
                    mask = m.convert("L")
            else:
                mask = gt_mask.convert("L")
            if mask.size != image.size:
                raise ValueError("Native image/mask dimensions differ")
            truth = (to_tensor(mask)[0] > 0.5).float()
            iou = compute_iou_per_sample(binary[None, None], truth[None, None]).item()
        return {"text": text, "image_pil": image, "prob_map": prob, "pred_mask": binary,
                "pred_pil": to_pil_image(binary), "gt_mask": truth, "iou": iou}


def build_grounding_model(config, pretrained=True):
    if not config:
        raise ValueError("Use a NaFlex checkpoint with model_config; CLIP checkpoints belong to baseline-fixes")
    config = dict(config)
    if config.pop("architecture", None) != "siglip2-naflex-grounding-v1":
        raise ValueError("Unsupported checkpoint architecture")
    if config.pop("vision_model", None) != VISION_MODEL:
        raise ValueError(f"This branch is fixed to {VISION_MODEL}; use the matching branch")
    return GroundingModel(pretrained=pretrained, **config)
