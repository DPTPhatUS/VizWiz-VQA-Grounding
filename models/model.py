import os
import torch
from torch import nn
from PIL import Image, ImageOps
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image

from dataset import CLIP_MEAN, CLIP_STD
from metrics import compute_iou
from models import ImageEncoder, TextEncoder, UNetDecoder


class GroundingModel(nn.Module):
    def __init__(self, n_heads=8):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

        # fixed based on image hidden dim (1024 for ViT-L/14-336)
        self.hidden_dim = self.image_encoder.out_channels

        # project text dim to match hidden dim
        self.text_proj = nn.Linear(
            self.text_encoder.output_dim, self.hidden_dim
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=n_heads, batch_first=True
        )
        self.decoder = UNetDecoder(in_channels=self.hidden_dim)

    def forward(self, image, text):
        enc_feat1, enc_feat2, enc_feat3, bottleneck = self.image_encoder(image)
        B, D, H, W = bottleneck.shape
        img_tokens = bottleneck.flatten(2).permute(0, 2, 1)  # (B, N, D)

        text_tokens, key_padding_mask = self.text_encoder(
            text,
            return_mask=True,
        )  # (B, L, D_text), (B, L)
        text_tokens = self.text_proj(text_tokens)  # align to (B, L, D)
        img_tokens = img_tokens.to(text_tokens.dtype)

        attn_output, _ = self.cross_attn(
            query=img_tokens,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        fused = attn_output.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        output = self.decoder(fused, enc_feat3, enc_feat2, enc_feat1)
        return output

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None, **kwargs):
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = cls(**kwargs).to(device)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        return model.eval()

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

        if text is None:
            q, a = str(question or "").strip(), str(answer or "").strip()
            text = f"Q: {q} A: {a}" if a else f"Q: {q}"

        if isinstance(image, (str, os.PathLike)):
            with Image.open(image) as img:
                pil_img = ImageOps.exif_transpose(img).convert("RGB")
        else:
            pil_img = ImageOps.exif_transpose(image).convert("RGB")

        pil_img = TF.resize(pil_img, image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True)
        img_tensor = TF.normalize(TF.to_tensor(pil_img), mean=CLIP_MEAN, std=CLIP_STD).unsqueeze(0).to(device)

        logits = self.forward(img_tensor, [text])
        prob = torch.sigmoid(logits)
        pred_bin = (prob > threshold).float()

        iou, gt_out = None, None
        if gt_mask is not None:
            if isinstance(gt_mask, (str, os.PathLike)):
                with Image.open(gt_mask) as m:
                    gt_pil = m.convert("L")
            else:
                gt_pil = gt_mask.convert("L")
            gt_pil = TF.resize(gt_pil, image_size, interpolation=T.InterpolationMode.NEAREST)
            gt_4d = (TF.to_tensor(gt_pil) > 0.5).float().unsqueeze(0).to(device)
            iou = round(compute_iou(pred_bin, gt_4d, threshold=0.5), 6)
            gt_out = gt_4d[0, 0].cpu()

        pred_2d = pred_bin[0, 0].cpu()
        return {
            "text": text,
            "image_pil": pil_img,
            "prob_map": prob[0, 0].cpu(),
            "pred_mask": pred_2d,
            "pred_pil": to_pil_image(pred_2d),
            "gt_mask": gt_out,
            "iou": iou,
        }
