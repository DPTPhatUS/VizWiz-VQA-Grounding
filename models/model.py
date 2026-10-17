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
from utils import load_model_weights, is_tiny_state, read_checkpoint

class CompactGroundingModel(nn.Module):
    """Shared encoders, latent answer predictor and conditioned mask decoder."""
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

    def set_text_encoder_frozen(self, frozen=True):
        """Freeze the shared Q/A encoder; keep projections and answer predictor trainable."""
        self._text_encoder_frozen = frozen
        self.text_encoder.requires_grad_(not frozen)
        self.text_encoder.train(self.training and not frozen)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "_text_encoder_frozen", False):
            self.text_encoder.eval()
        return self

    def _decode_mask(self, fused, s3, s2, s1, text, answer_vec=None, *, projected_skips=None):
        return self.decoder(fused, s3, s2, s1, text, answer_vec=answer_vec, projected_skips=projected_skips)

    def encode_image(self, image):
        return self.image_encoder((image - self.image_mean) / self.image_std)  # 4 x (B, D_v, H_p, W_p)

    def encode_question(self, texts, question_vec=None):
        """Accept raw text or cached token features and return their pooled question."""
        text = texts if hasattr(texts, "tokens") else self.text_encoder(texts)
        q = question_vec if question_vec is not None else masked_mean(text.tokens, text.valid_mask)  # (B, D_t)
        return text, q

    @torch.no_grad()
    def encode_answer_target(self, batch):
        """Encode real answers for privileged supervision, without target gradients."""
        answers = batch.get("answer_only_text") or [
            f"A: {t.split(' A: ', 1)[1]}" if " A: " in t else t for t in batch["answer_text"]
        ]
        text = self.text_encoder(answers)
        return masked_mean(text.tokens, text.valid_mask)  # (B, D_t)

    def infer_latent_answer(self, vision, question_vec):
        """Predict A′ by attending to image patches with the pooled question."""
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

    def _answer_conditioning(self, text, question_vec, answer_vec, available):
        """Append an answer token; missing answers use Q-only attention and skips."""
        token = answer_vec.to(text.tokens.dtype).unsqueeze(1)  # (B, 1, D_t)
        mask = torch.ones(token.shape[:2], dtype=torch.bool, device=token.device)
        if available is not None:
            mask = torch.as_tensor(available, device=token.device, dtype=torch.bool).view(-1, 1)
            answer_vec = torch.where(mask, answer_vec, question_vec.to(answer_vec.dtype))
            token = torch.where(mask.unsqueeze(-1), token, torch.zeros_like(token))
        return (torch.cat([text.tokens, token], dim=1),
                torch.cat([text.attention_mask, mask], dim=1), answer_vec)

    def _grounding_logits(self, features, tokens, attention_mask, question_vec, answer_vec,
                         projected_skips=None):
        """The same visual/text fusion and mask decoder for every conditioning view."""
        s1, s2, s3, vision = features  # each (B, D_v, H_p, W_p)
        query = vision.flatten(2).transpose(1, 2)  # (B, N, D_v)
        projected = self.text_proj(tokens)  # (B, L or L + 1, D_v)
        query = query.to(projected.dtype)
        attended, _ = self.cross_attn(query, projected, projected,
            key_padding_mask=~attention_mask, need_weights=False)  # (B, N, D_v)
        fused = query + self.residual_scale.to(query.dtype) * attended  # (B, N, D_v)
        fused = fused.transpose(1, 2).reshape_as(vision).contiguous()  # (B, D_v, H_p, W_p)
        return self._decode_mask(fused, s3, s2, s1, question_vec, answer_vec,
                                 projected_skips=projected_skips)  # (B, 1, 16*H_p, 16*W_p)

    def decode_question(self, features, text, question_vec, *, projected_skips=None):
        """Q-only reference: no answer token and Q-only skip conditioning."""
        logits = self._grounding_logits(features, text.tokens, text.attention_mask,
                                       question_vec, None, projected_skips)
        return {"logits": logits, "latent_answer": None, "answer_attn": None,
                "question_vec": question_vec.detach()}

    def decode_answer(self, features, text, question_vec, answer_vec, *,
                      answer_available=None, projected_skips=None):
        """Ground with a supplied answer vector (real A or a diagnostic intervention)."""
        tokens, mask, answer = self._answer_conditioning(
            text, question_vec, answer_vec.to(text.tokens.dtype), answer_available)
        logits = self._grounding_logits(features, tokens, mask, question_vec, answer, projected_skips)
        return {"logits": logits, "latent_answer": answer, "answer_attn": None,
                "question_vec": question_vec.detach()}

    def decode_predicted(self, features, text, question_vec, *, detach_predicted_answer=False,
                         answer_available=None, projected_skips=None):
        """Ground with A′, optionally cutting only its direct mask-loss gradient."""
        latent, attn_map = self.infer_latent_answer(features[-1], question_vec)
        conditioning = latent.detach() if detach_predicted_answer else latent
        tokens, mask, answer = self._answer_conditioning(text, question_vec, conditioning, answer_available)
        logits = self._grounding_logits(features, tokens, mask, question_vec, answer, projected_skips)
        # Alignment still needs the original A′ graph when grounding uses detached A′.
        return {"logits": logits, "latent_answer": latent if detach_predicted_answer else answer,
                "answer_attn": attn_map, "question_vec": question_vec.detach()}

    def decode(self, features, texts, answer_vec=None, *, projected_skips=None, question_vec=None,
               question_only=False, answer_available=None, detach_predicted_answer=False):
        """Compatibility entry point for existing prediction and diagnostic callers."""
        text, q = self.encode_question(texts, question_vec)
        if question_only:
            return self.decode_question(features, text, q, projected_skips=projected_skips)
        if answer_vec is not None:
            return self.decode_answer(features, text, q, answer_vec,
                                      answer_available=answer_available, projected_skips=projected_skips)
        return self.decode_predicted(features, text, q, detach_predicted_answer=detach_predicted_answer,
                                     answer_available=answer_available, projected_skips=projected_skips)

    def forward(self, batch):
        """Encode once, predict the mask, then add only the requested training views.

        Existing objectives select views through the batch controls below. All
        views share this model's weights and stay inside one DDP forward call.
        """
        features = self.encode_image(batch["image"])  # 4 x (B, D_v, H_p, W_p)
        text, q = self.encode_question(batch.get("question_text", batch["text"]))

        if getattr(self, "question_only", False):
            return self.decode_question(features, text, q)

        # Resolve the recipe's view requests at the forward boundary.
        has_target = (self.training and batch.get("encode_answer_target", True)
                      and ("answer_only_text" in batch or "answer_text" in batch))
        needs_question = has_target and batch.get("self_distill", True)
        needs_oracle = needs_question or (has_target and batch.get("oracle_supervision", False))
        s1, s2, s3, _ = features
        projected_skips = None
        if needs_oracle and hasattr(self.decoder, "project_skips"):
            projected_skips = self.decoder.project_skips(s3, s2, s1)

        output = self.decode_predicted(features, text, q, projected_skips=projected_skips,
                                      detach_predicted_answer=self.training and batch.get("detach_predicted_answer", False))
        if not has_target:
            return output

        target_answer = self.encode_answer_target(batch)
        output["target_answer"] = target_answer
        if needs_question:
            output["question_logits"] = self.decode_question(
                features, text, q, projected_skips=projected_skips)["logits"]
        if needs_oracle:
            # Historical output name: this is the shared model's oracle view.
            output["teacher_logits"] = self.decode_answer(
                features, text, q, target_answer, answer_available=batch.get("answer_available"),
                projected_skips=projected_skips)["logits"]
        return output

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None, **kwargs):
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        saved = read_checkpoint(checkpoint_path)
        default_tiny = saved.get("run_config", {}).get("tiny", is_tiny_state(saved["model_state_dict"]))
        tiny = kwargs.pop("tiny", default_tiny)
        model = cls(tiny=tiny, **kwargs)
        load_model_weights(model, saved)
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
