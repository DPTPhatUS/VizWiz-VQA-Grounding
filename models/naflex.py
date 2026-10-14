"""SigLIP 2 vision tower with native-aspect-ratio dense grounding.

Text encoder, cross-attention and U-Net recipe follow baseline-fixes. Different
patch grids are decoded separately, so padding never becomes spatial content.
"""
from collections import defaultdict

import torch
from torch import nn
from transformers import Siglip2ImageProcessor, Siglip2VisionConfig, Siglip2VisionModel

from .mask_decoder import UNetDecoder
from .text_encoder import TextEncoder

BASE_NAFLEX = "google/siglip2-base-patch16-naflex"
SO400M_NAFLEX = "google/siglip2-so400m-patch16-naflex"
CLIP_BASELINE = "openai/clip-vit-large-patch14-336"
VISION_MODELS = (BASE_NAFLEX, SO400M_NAFLEX, CLIP_BASELINE)


def make_naflex_processor(config):
    """Use checkpointed image preprocessing, with no second global resize."""
    return Siglip2ImageProcessor(**config["image_processor"])


class NaFlexImageEncoder(nn.Module):
    def __init__(self, model_name, config=None, pretrained=True, feature_layers=(4, 7, 9)):
        super().__init__()
        config = (Siglip2VisionConfig.from_pretrained(model_name) if config is None
                  else Siglip2VisionConfig(**config))
        config.vision_use_head = False  # pooled embeddings are unused by segmentation
        self.vision_encoder = (Siglip2VisionModel.from_pretrained(model_name, config=config)
                               if pretrained else Siglip2VisionModel(config))
        self.out_channels = config.hidden_size
        self.patch_size = config.patch_size
        self.feature_layers = tuple(feature_layers)
        if len(self.feature_layers) != 3 or not all(0 < i <= config.num_hidden_layers for i in self.feature_layers):
            raise ValueError("Exactly three valid hidden-state layer indices are required")

    def forward(self, inputs):
        outputs = self.vision_encoder(**inputs, output_hidden_states=True)
        return tuple(outputs.hidden_states[i] for i in self.feature_layers) + (outputs.last_hidden_state,)


class NaFlexGroundingModel(nn.Module):
    def __init__(self, vision_model=BASE_NAFLEX, max_num_patches=512, n_heads=8,
                 pretrained=True, vision_config=None, text_config=None,
                 image_processor=None, feature_layers=(4, 7, 9),
                 decoder_channels=(1024, 512, 256, 128), text_encoder=None):
        super().__init__()
        if vision_model not in (BASE_NAFLEX, SO400M_NAFLEX):
            raise ValueError("Use an official Base/So400m NaFlex checkpoint; Large is FixRes")
        if max_num_patches <= 0:
            raise ValueError("max_num_patches must be positive")
        self.image_encoder = NaFlexImageEncoder(vision_model, vision_config, pretrained, feature_layers)
        self.text_encoder = text_encoder or TextEncoder(pretrained=pretrained, config=text_config)
        self.hidden_dim = self.image_encoder.out_channels
        if self.hidden_dim % n_heads:
            raise ValueError("Vision hidden size must be divisible by attention heads")
        self.text_proj = nn.Linear(self.text_encoder.output_dim, self.hidden_dim)
        self.cross_attn = nn.MultiheadAttention(self.hidden_dim, n_heads, batch_first=True)
        self.decoder = UNetDecoder(in_channels=self.hidden_dim, mid_channels=tuple(decoder_channels))
        if image_processor is None:
            processor = Siglip2ImageProcessor.from_pretrained(vision_model)
            image_processor = processor.to_dict()
        image_processor = dict(image_processor)
        image_processor.update(max_num_patches=max_num_patches, patch_size=self.image_encoder.patch_size)
        self.model_config = {
            "architecture": "siglip2-naflex-grounding-v1", "vision_model": vision_model,
            "max_num_patches": max_num_patches, "n_heads": n_heads,
            "vision_config": self.image_encoder.vision_encoder.config.to_dict(),
            "text_config": self.text_encoder.model.config.to_dict() if hasattr(self.text_encoder, "model") else text_config,
            "image_processor": image_processor, "feature_layers": list(feature_layers),
            "decoder_channels": list(decoder_channels),
        }

    def forward(self, images, texts):
        grids = images["spatial_shapes"].tolist()
        padding = images["pixel_attention_mask"]
        if len(grids) != len(texts):
            raise ValueError("Image/text batch sizes differ")
        for i, (h, w) in enumerate(grids):
            n = h * w
            if h <= 0 or w <= 0 or n > padding.shape[1]:
                raise ValueError("Invalid NaFlex spatial grid")
            if not bool(padding[i, :n].all()) or bool(padding[i, n:].any()):
                raise ValueError("NaFlex valid patches must be a contiguous prefix matching spatial_shapes")
        f1, f2, f3, bottleneck = self.image_encoder(images)
        text_tokens, text_padding = self.text_encoder(texts, return_mask=True)
        text_tokens = self.text_proj(text_tokens)
        fused, _ = self.cross_attn(bottleneck.to(text_tokens.dtype), text_tokens, text_tokens,
                                  key_padding_mask=text_padding, need_weights=False)
        groups = defaultdict(list)
        for i, grid in enumerate(grids):
            groups[tuple(grid)].append(i)
        predictions = [None] * len(grids)
        for (h, w), indices in groups.items():
            n = h * w
            def spatial(tokens):
                return tokens[indices, :n].reshape(len(indices), h, w, self.hidden_dim).permute(0, 3, 1, 2).contiguous()
            decoded = self.decoder(spatial(fused), spatial(f3), spatial(f2), spatial(f1))
            for j, i in enumerate(indices):
                predictions[i] = decoded[j:j + 1]
        return predictions

    @torch.no_grad()
    def predict(self, image, text=None, question=None, answer=None, gt_mask=None, threshold=0.5):
        from PIL import Image, ImageOps
        from torchvision.transforms.functional import to_pil_image, to_tensor
        import os
        import torch.nn.functional as F
        from utils import to_device
        from metrics import compute_iou_per_sample
        self.eval()
        if isinstance(image, (str, os.PathLike)):
            with Image.open(image) as im:
                pil = ImageOps.exif_transpose(im).convert("RGB")
        else:
            pil = ImageOps.exif_transpose(image).convert("RGB")
        if text is None:
            q, a = str(question or "").strip(), str(answer or "").strip()
            text = f"Q: {q} A: {a}" if a else f"Q: {q}"
        inputs = make_naflex_processor(self.model_config)(images=pil, return_tensors="pt")
        inputs = to_device(dict(inputs), next(self.parameters()).device)
        logits = self(inputs, [text])[0]
        logits = F.interpolate(logits.float(), (pil.height, pil.width), mode="bilinear", align_corners=False)
        prob = logits.sigmoid()[0, 0].cpu()
        binary = (prob > threshold).float()
        truth, iou = None, None
        if gt_mask is not None:
            if isinstance(gt_mask, (str, os.PathLike)):
                with Image.open(gt_mask) as m:
                    mask = m.convert("L")
            else:
                mask = gt_mask.convert("L")
            if mask.size != pil.size:
                raise ValueError("Native image/mask dimensions differ")
            truth = (to_tensor(mask)[0] > 0.5).float()
            iou = compute_iou_per_sample(binary[None, None], truth[None, None]).item()
        return {"text": text, "image_pil": pil, "prob_map": prob, "pred_mask": binary,
                "pred_pil": to_pil_image(binary), "gt_mask": truth, "iou": iou}


def build_grounding_model(config=None, pretrained=True):
    if not config:
        from .model import GroundingModel
        return GroundingModel()
    config = dict(config)
    if config.pop("architecture", None) != "siglip2-naflex-grounding-v1":
        raise ValueError("Unsupported checkpoint model architecture")
    return NaFlexGroundingModel(pretrained=pretrained, **config)
