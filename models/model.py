import os
import torch
from torch import nn
from PIL import Image, ImageOps
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image

from dataset import CLIP_MEAN, CLIP_STD
from metrics import compute_iou, compute_iou_per_sample
from .image_encoder import ImageEncoder, BASE_NAFLEX, SO400M_NAFLEX, CLIP_BASELINE
from .text_encoder import TextEncoder
from .mask_decoder import UNetDecoder
from utils import align_state_dict, to_device


class GroundingModel(nn.Module):
    def __init__(self, n_heads=8, vision_model=CLIP_BASELINE, max_num_patches=512,
                 pretrained=True, vision_config=None, text_config=None, image_processor=None,
                 feature_layers=(4, 7, 9), decoder_channels=(1024, 512, 256, 128), text_encoder=None):
        super().__init__()
        if vision_model not in (CLIP_BASELINE, BASE_NAFLEX, SO400M_NAFLEX):
            raise ValueError("Use CLIP, Base NaFlex or So400m NaFlex; Large is FixRes")
        self.image_encoder = ImageEncoder(vision_model, pretrained, vision_config,
                                         feature_layers, max_num_patches, image_processor)
        self.text_encoder = text_encoder or TextEncoder(pretrained=pretrained, config=text_config)

        self.hidden_dim = self.image_encoder.out_channels
        if self.hidden_dim % n_heads:
            raise ValueError("Vision hidden size must be divisible by attention heads")

        # project text dim to match hidden dim
        self.text_proj = nn.Linear(
            self.text_encoder.output_dim, self.hidden_dim
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=n_heads, batch_first=True
        )
        self.decoder = UNetDecoder(in_channels=self.hidden_dim, mid_channels=tuple(decoder_channels))
        self.model_config = None
        if self.image_encoder.is_naflex:
            self.model_config = {
                "architecture": "siglip2-naflex-grounding-v1", "vision_model": vision_model,
                "max_num_patches": max_num_patches, "n_heads": n_heads,
                "vision_config": self.image_encoder.vision_encoder.config.to_dict(),
                "text_config": self.text_encoder.model.config.to_dict() if hasattr(self.text_encoder, "model") else text_config,
                "image_processor": self.image_encoder.processor_config, "feature_layers": list(feature_layers),
                "decoder_channels": list(decoder_channels),
            }

    def forward(self, image, text):
        enc_feat1, enc_feat2, enc_feat3, bottleneck = self.image_encoder(image)  # each (B, D_v, H_p, W_p)
        if self.image_encoder.is_naflex:
            if bottleneck.shape[0] != len(text):
                raise ValueError("Image/text batch sizes differ")
            img_tokens = bottleneck
        else:
            B, D, H, W = bottleneck.shape
            img_tokens = bottleneck.flatten(2).permute(0, 2, 1)

        text_tokens, key_padding_mask = self.text_encoder(
            text,
            return_mask=True,
        )  # (B, L, D_t), (B, L)
        text_tokens = self.text_proj(text_tokens)  # (B, L, D_v)
        img_tokens = img_tokens.to(text_tokens.dtype)

        attn_output, _ = self.cross_attn(
            query=img_tokens,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )  # (B, N, D_v)
        if self.image_encoder.is_naflex:
            predictions = [None] * len(text)
            for indices, maps in self.image_encoder.spatial_groups(
                    (attn_output, enc_feat3, enc_feat2, enc_feat1), image):
                decoded = self.decoder(*maps)
                for j, i in enumerate(indices):
                    predictions[i] = decoded[j:j + 1]
            return predictions
        fused = attn_output.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # (B, D_v, H_p, W_p)

        output = self.decoder(fused, enc_feat3, enc_feat2, enc_feat1)  # (B, 1, 16*H_p, 16*W_p)
        return output

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None, **kwargs):
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = ckpt.get("model_config") if isinstance(ckpt, dict) else None
        if config:
            if kwargs:
                raise ValueError("NaFlex architecture is restored from checkpoint metadata; do not override it")
            model = build_grounding_model(config, pretrained=False)
        else:
            model = cls(**kwargs)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state_dict = align_state_dict(state_dict, model)
        model.load_state_dict(state_dict)
        return model.to(device).eval()

    @torch.no_grad()
    def predict(
        self,
        image,
        text=None,
        question=None,
        answer=None,
        gt_mask=None,
        threshold=0.5,
        image_size=(336, 336),
    ):
        self.eval()
        device = next(self.parameters()).device
        size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)

        if text is None:
            q, a = str(question or "").strip(), str(answer or "").strip()
            text = f"Q: {q} A: {a}" if a else f"Q: {q}"

        if isinstance(image, (str, os.PathLike)):
            with Image.open(image) as img:
                pil_img = ImageOps.exif_transpose(img).convert("RGB")
        else:
            pil_img = ImageOps.exif_transpose(image).convert("RGB")

        if self.image_encoder.is_naflex:
            size = (pil_img.height, pil_img.width)
            inputs = self.image_encoder.make_processor()(images=pil_img, return_tensors="pt")
            prediction = self.forward(to_device(dict(inputs), device), [text])[0]
        else:
            pil_img = TF.resize(pil_img, size, interpolation=T.InterpolationMode.BICUBIC, antialias=True)
            img_tensor = TF.normalize(TF.to_tensor(pil_img), mean=CLIP_MEAN, std=CLIP_STD).unsqueeze(0).to(device)
            prediction = self.forward(img_tensor, [text])

        logits = torch.nn.functional.interpolate(
            prediction.float(), size=size, mode="bilinear", align_corners=False
        )  # (1, 1, H, W)
        prob = torch.sigmoid(logits)  # (1, 1, H, W)
        pred_bin = (prob > threshold).float()  # (1, 1, H, W)

        iou, gt_out = None, None
        if gt_mask is not None:
            if isinstance(gt_mask, (str, os.PathLike)):
                with Image.open(gt_mask) as m:
                    gt_pil = m.convert("L")
            else:
                gt_pil = gt_mask.convert("L")
            if self.image_encoder.is_naflex and gt_pil.size != pil_img.size:
                raise ValueError("Native image/mask dimensions differ")
            gt_pil = TF.resize(gt_pil, size, interpolation=T.InterpolationMode.NEAREST)
            gt_4d = (TF.to_tensor(gt_pil) > 0.5).float().unsqueeze(0).to(device)  # (1, 1, H, W)
            iou = (compute_iou_per_sample(pred_bin, gt_4d).item() if self.image_encoder.is_naflex
                   else round(compute_iou(pred_bin, gt_4d, threshold=0.5), 6))
            gt_out = gt_4d[0, 0].cpu()  # (H, W)

        pred_2d = pred_bin[0, 0].cpu()  # (H, W)
        return {
            "text": text,
            "image_pil": pil_img,
            "prob_map": prob[0, 0].cpu(),
            "pred_mask": pred_2d,
            "pred_pil": to_pil_image(pred_2d),
            "gt_mask": gt_out,
            "iou": iou,
        }


def build_grounding_model(config=None, pretrained=True):
    if not config:
        return GroundingModel()
    config = dict(config)
    if config.pop("architecture", None) != "siglip2-naflex-grounding-v1":
        raise ValueError("Unsupported checkpoint model architecture")
    return GroundingModel(pretrained=pretrained, **config)
