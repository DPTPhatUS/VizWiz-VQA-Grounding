"""Question-conditioned grounding models."""
import os
import torch
from torch import nn
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_pil_image, to_tensor
from losses import per_image_iou
from models.image_encoder import ImageEncoder, TinyVision
from models.text_encoder import TextEncoder, masked_mean
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
        self.visual_to_text = nn.Linear(self.visual_dim, self.text_dim)
        self.answer_query = nn.Parameter(torch.zeros(1, 1, self.text_dim))  # (1, 1, D_t)
        self.answer_attn = nn.MultiheadAttention(self.text_dim, 2 if tiny else 8, batch_first=True)
        self.answer_norm = nn.LayerNorm(self.text_dim)
        self.answer_mlp = nn.Sequential(
            nn.Linear(self.text_dim, self.text_dim),
            nn.GELU(),
            nn.Linear(self.text_dim, self.text_dim),
        )
        nn.init.zeros_(self.answer_mlp[-1].weight)
        nn.init.zeros_(self.answer_mlp[-1].bias)
        self._init_decoder(tiny)
        self.register_buffer("image_mean", torch.tensor([.48145466, .4578275, .40821073]).view(1,3,1,1))  # (1, 3, 1, 1)
        self.register_buffer("image_std", torch.tensor([.26862954, .26130258, .27577711]).view(1,3,1,1))  # (1, 3, 1, 1)
        self.experiment_config = {"experiment": "controls", "architecture": self.architecture, "tiny": tiny,
                                  "protocol": "normalized-masked-question-v1"}

    def _init_decoder(self, tiny):
        self.decoder = CompactDecoder(self.visual_dim, self.text_dim, width=16 if tiny else 128)

    def _decode_mask(self, fused, s3, s2, s1, text, answer_vec=None, *, projected_skips=None):
        return self.decoder(fused, s3, s2, s1, text, answer_vec=answer_vec, projected_skips=projected_skips)

    def encode_image(self, image):
        return self.image_encoder((image - self.image_mean) / self.image_std)  # 4 x (B, D_v, H_p, W_p)

    def infer_latent_answer(self, vision, question_vec):
        B, _, H_p, W_p = vision.shape
        patches = vision.flatten(2).transpose(1, 2)  # (B, N, D_v), N = H_p * W_p
        visual_kv = self.visual_to_text(patches)  # (B, N, D_t)
        query = (question_vec.unsqueeze(1) + self.answer_query.to(question_vec.dtype)).to(visual_kv.dtype)  # (B, 1, D_t)
        evidence, attn_weights = self.answer_attn(
            query, visual_kv, visual_kv, need_weights=True, average_attn_weights=True
        )  # (B, 1, D_t), (B, 1, N)
        latent = self.answer_norm(question_vec.to(evidence.dtype) + evidence.squeeze(1))  # (B, D_t)
        latent = latent + self.answer_mlp(latent)  # (B, D_t)
        attn_map = attn_weights.view(B, 1, H_p, W_p)  # (B, 1, H_p, W_p)
        return latent, attn_map

    def decode(self, features, texts, answer_vec=None, *, projected_skips=None, question_vec=None):
        s1, s2, s3, vision = features  # each (B, D_v, H_p, W_p)
        text = texts if hasattr(texts, "tokens") else self.text_encoder(texts)
        q = question_vec if question_vec is not None else masked_mean(text.tokens, text.valid_mask)  # (B, D_t)
        if answer_vec is None:
            latent, attn_map = self.infer_latent_answer(vision, q)  # (B, D_t), (B, 1, H_p, W_p)
        else:
            latent, attn_map = answer_vec.to(text.tokens.dtype), None  # (B, D_t), None
        ans_token = latent.to(text.tokens.dtype).unsqueeze(1)  # (B, 1, D_t)
        kv_tokens = torch.cat([text.tokens, ans_token], dim=1)  # (B, L + 1, D_t)
        ans_mask = torch.ones(ans_token.shape[:2], dtype=torch.bool, device=ans_token.device)  # (B, 1)
        kv_mask = torch.cat([text.attention_mask, ans_mask], dim=1)  # (B, L + 1)
        query = vision.flatten(2).transpose(1, 2)  # (B, N, D_v)
        projected = self.text_proj(kv_tokens)  # (B, L + 1, D_v)
        query = query.to(projected.dtype)
        attended, _ = self.cross_attn(query, projected, projected,
            key_padding_mask=~kv_mask, need_weights=False)  # (B, N, D_v)
        fused = query + self.residual_scale.to(query.dtype) * attended  # (B, N, D_v)
        fused = fused.transpose(1, 2).reshape_as(vision).contiguous()  # (B, D_v, H_p, W_p)
        logits = self._decode_mask(fused, s3, s2, s1, q, latent, projected_skips=projected_skips)  # (B, 1, 16*H_p, 16*W_p)
        return {"logits": logits, "latent_answer": latent, "answer_attn": attn_map, "question_vec": q.detach()}

    def forward(self, batch):
        features = self.encode_image(batch["image"])  # 4 x (B, D_v, H_p, W_p)
        questions = batch.get("question_text", batch["text"])
        text = self.text_encoder(questions)
        q = masked_mean(text.tokens, text.valid_mask)  # (B, D_t)
        has_answer = self.training and ("answer_only_text" in batch or "answer_text" in batch)
        self_distill = has_answer and batch.get("self_distill", True)
        s1, s2, s3, _ = features
        proj_skips = self.decoder.project_skips(s3, s2, s1) if self_distill and hasattr(self.decoder, "project_skips") else None
        student = self.decode(features, text, projected_skips=proj_skips, question_vec=q)
        if has_answer:
            ans_texts = batch.get("answer_only_text") or [
                f"A: {t.split(' A: ', 1)[1]}" if " A: " in t else t for t in batch["answer_text"]
            ]
            with torch.no_grad():
                ans_feat = self.text_encoder(ans_texts)
                target_answer = masked_mean(ans_feat.tokens, ans_feat.valid_mask)  # (B, D_t)
            student["target_answer"] = target_answer
            if self_distill:
                detached_skips = [s.detach() for s in proj_skips] if proj_skips is not None else None
                with torch.no_grad():
                    question_logits = self.decode(
                        features, text, student["question_vec"], projected_skips=detached_skips, question_vec=student["question_vec"]
                    )["logits"]  # (B, 1, 16*H_p, 16*W_p)
                teacher_logits = self.decode(
                    features, text, target_answer, projected_skips=proj_skips, question_vec=q
                )["logits"]  # (B, 1, 16*H_p, 16*W_p)
                student.update(
                    teacher_logits=teacher_logits,
                    question_logits=question_logits,
                )
        return student

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None, **kwargs):
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        saved = read_checkpoint(checkpoint_path)
        default_tiny = saved.get("run_config", {}).get("tiny", is_tiny_state(saved["model_state_dict"]))
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

        out = self.forward(batch)
        logits = torch.nn.functional.interpolate(
            out["logits"].float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, H, W)
        prob = torch.sigmoid(logits)  # (1, 1, H, W)
        pred_bin = (prob > threshold).float()  # (1, 1, H, W)
        attn_out = None
        if out.get("answer_attn") is not None:
            attn_out = torch.nn.functional.interpolate(
                out["answer_attn"].float(), size=size, mode="bilinear", align_corners=False
            )[0, 0].cpu()  # (H, W)

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
            "answer_attn": attn_out,
            "gt_mask": gt_out,
            "iou": iou,
        }


class TeacherModel(CompactGroundingModel):
    """Question/answer teacher using the compact decoder architecture."""

    def forward(self, batch):
        return self.decode(self.encode_image(batch["image"]), batch["text"])


EXPERIMENT = 'latent-answer-distillation'


class GroundingModel(nn.Module):
    def __init__(self, tiny=False):
        super().__init__()
        self.coarse = CompactGroundingModel(tiny=tiny)
        self.experiment_config = dict(self.coarse.experiment_config, experiment=EXPERIMENT)

    def forward(self, batch):
        return self.coarse(batch)

    from_checkpoint = classmethod(CompactGroundingModel.from_checkpoint.__func__)
    predict = CompactGroundingModel.predict
