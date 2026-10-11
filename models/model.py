"""Question-conditioned grounding models."""
import os
import torch
from torch import nn
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_pil_image, to_tensor
from losses import per_image_iou
from models.image_encoder import ImageEncoder, TinyVision
from models.text_encoder import TextEncoder
from models.mask_decoder import CompactDecoder
from utils import initialize_weights, is_tiny_state, read_checkpoint

class CompactGroundingModel(nn.Module):
    architecture = "compact"

    def __init__(self, tiny=False):
        super().__init__()
        self.image_encoder = TinyVision() if tiny else ImageEncoder()
        self.text_encoder = TextEncoder(tiny)
        self.visual_dim = self.image_encoder.out_channels
        self.text_dim = self.text_encoder.output_dim
        self.text_proj = nn.Linear(self.text_dim, self.visual_dim)
        self.cross_attn = nn.MultiheadAttention(self.visual_dim, 2 if tiny else 8, batch_first=True)
        self.residual_scale = nn.Parameter(torch.tensor(.01))
        self._init_decoder(tiny)
        self.register_buffer("image_mean", torch.tensor([.48145466, .4578275, .40821073]).view(1,3,1,1))  # (1, 3, 1, 1)
        self.register_buffer("image_std", torch.tensor([.26862954, .26130258, .27577711]).view(1,3,1,1))  # (1, 3, 1, 1)
        self.experiment_config = {"experiment": "controls", "architecture": self.architecture, "tiny": tiny,
                                  "protocol": "normalized-masked-question-v1"}

    def _init_decoder(self, tiny):
        self.decoder = CompactDecoder(self.visual_dim, self.text_dim, width=16 if tiny else 128)

    def _decode_mask(self, fused, s3, s2, s1, text):
        return self.decoder(fused, s3, s2, s1, text)

    def encode_image(self, image):
        return self.image_encoder((image - self.image_mean) / self.image_std)  # 4 x (B, D_v, H_p, W_p)

    def decode(self, features, texts):
        s1, s2, s3, vision = features  # each (B, D_v, H_p, W_p)
        text = self.text_encoder(texts)  # tokens: (B, L, D_t), masks: (B, L)
        query = vision.flatten(2).transpose(1, 2)  # (B, N, D_v), N = H_p * W_p
        projected = self.text_proj(text.tokens)  # (B, L, D_v)
        query = query.to(projected.dtype)
        attended, _ = self.cross_attn(query, projected, projected,
            key_padding_mask=~text.attention_mask, need_weights=False)  # (B, N, D_v)
        fused = query + self.residual_scale.to(query.dtype) * attended  # (B, N, D_v)
        fused = fused.transpose(1, 2).reshape_as(vision).contiguous()  # (B, D_v, H_p, W_p)
        logits = self._decode_mask(fused, s3, s2, s1, text)  # (B, 1, 16*H_p, 16*W_p)
        return {"logits": logits}

    def forward(self, batch):
        return self.decode(self.encode_image(batch["image"]), batch["text"])

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None, **kwargs):
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        saved = read_checkpoint(checkpoint_path)
        config, run = saved.get("experiment_config") or {}, saved["run_config"]
        default_tiny = config.get("tiny", run.get("tiny", is_tiny_state(saved["model_state_dict"])))
        tiny = kwargs.pop("tiny", default_tiny)
        model = cls(tiny=tiny, **kwargs)
        initialize_weights(saved, model)
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
        image_size=336,
    ):
        self.eval()
        device = next(self.parameters()).device
        size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)

        q = str(question or "").strip()
        a = str(answer or "").strip()
        q_text = f"Q: {q}" if q else str(text or "").strip().split(" A: ", 1)[0]
        if text is None:
            text = f"{q_text} A: {a}" if a else q_text
        else:
            text = str(text).strip()

        if isinstance(image, (str, os.PathLike)):
            with Image.open(image) as img:
                pil_img = ImageOps.exif_transpose(img).convert("RGB")
        else:
            pil_img = ImageOps.exif_transpose(image).convert("RGB")

        pil_img = pil_img.resize(size, Image.Resampling.BICUBIC)
        img_tensor = to_tensor(pil_img).unsqueeze(0).to(device)  # (1, 3, H, W)
        batch = {"image": img_tensor, "text": [text], "question_text": [q_text]}

        logits = torch.nn.functional.interpolate(
            self.forward(batch)["logits"].float(), size=size, mode="bilinear", align_corners=False
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
            gt_pil = gt_pil.resize(size, Image.Resampling.NEAREST)
            gt_4d = (to_tensor(gt_pil) > 0.5).float().unsqueeze(0).to(device)  # (1, 1, H, W)
            iou = round(per_image_iou(logits, gt_4d).item(), 6)
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


class TeacherModel(CompactGroundingModel):
    """Question/answer teacher using the compact decoder architecture."""


EXPERIMENT = 'answer-value-distillation'


class GroundingModel(nn.Module):
    def __init__(self, tiny=False):
        super().__init__()
        self.coarse = CompactGroundingModel(tiny=tiny)
        self.experiment_config = dict(self.coarse.experiment_config, experiment="answer-value-distillation")

    def forward(self, batch):
        return self.coarse(dict(batch, text=batch.get("question_text", batch["text"])))

    from_checkpoint = classmethod(CompactGroundingModel.from_checkpoint.__func__)
    predict = CompactGroundingModel.predict
