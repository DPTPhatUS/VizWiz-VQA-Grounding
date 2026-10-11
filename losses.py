import math
import torch
from torch.nn import functional as F


def resize_logits(logits, target):
    if logits.shape[-2:] == target.shape[-2:]:
        return logits
    return F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)


def segmentation_loss(logits, target, dice_weight=0.0):
    logits = resize_logits(logits, target).float()  # (B, 1, H, W)
    target = target.float()  # (B, 1, H, W)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    if not dice_weight:
        return bce
    probability = logits.sigmoid()  # (B, 1, H, W)
    axes = tuple(range(1, target.ndim))
    dice = 1 - (2*(probability*target).sum(axes)+1) / (
        probability.sum(axes)+target.sum(axes)+1)  # (B,)
    return bce + dice_weight*dice.mean()


def per_image_iou(logits, target):
    prediction = resize_logits(logits, target) > 0  # (B, 1, H, W)
    truth = target > .5  # (B, 1, H, W)
    axes = tuple(range(1,target.ndim))
    intersection = (prediction & truth).sum(axes).float()  # (B,)
    union = (prediction | truth).sum(axes).float()  # (B,)
    return torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union))  # (B,)


def balanced_weights(values, target):
    """Balance foreground/background focus while scaling each region by its effective gain."""
    values = values.detach().float().clamp(0, 1)  # (B, 1, H, W)
    result = torch.zeros_like(values)  # (B, 1, H, W)
    regions = (target > .5, target <= .5)  # 2 x (B, 1, H, W)
    nonempty = sum(r.flatten(1).any(1).float() for r in regions).view(-1, 1, 1, 1).clamp_min(1)  # (B, 1, 1, 1)
    for region in regions:
        masked = values * region  # (B, 1, H, W)
        mass = masked.flatten(1).sum(1).view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        dist = masked / mass.clamp_min(1e-8)  # (B, 1, H, W)
        scale = (dist * masked).flatten(1).sum(1).view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        result = result + scale * dist  # (B, 1, H, W)
    return result / nonempty  # (B, 1, H, W)


def distillation_weights(answer_logits, question_logits, target, probability=None, entropy=None):
    """Positive answer-induced BCE reduction on teacher-correct pixels, weighted by confidence."""
    with torch.no_grad():
        answer = resize_logits(answer_logits, target).float()  # (B, 1, H, W)
        question = resize_logits(question_logits, target).float()  # (B, 1, H, W)
        target_f = target.float()  # (B, 1, H, W)
        truth = target > .5  # (B, 1, H, W)
        correct = (answer > 0) == truth  # (B, 1, H, W)
        if probability is None:
            probability = answer.sigmoid()  # (B, 1, H, W)
        error = F.binary_cross_entropy_with_logits(answer, target_f, reduction="none")  # (B, 1, H, W)
        question_error = F.binary_cross_entropy_with_logits(question, target_f, reduction="none")  # (B, 1, H, W)
        if entropy is None:
            entropy = F.binary_cross_entropy_with_logits(answer, probability, reduction="none")  # (B, 1, H, W)
        confidence = (1 - entropy / math.log(2)).clamp(0, 1)  # (B, 1, H, W)
        gain = 1 - torch.exp(-(question_error - error).clamp_min(0))  # (B, 1, H, W)
        values = gain * confidence * correct  # (B, 1, H, W)
        return balanced_weights(values, target)  # (B, 1, H, W)


def latent_answer_loss(latent, target, available=None):
    """Align inferred latent answer vector with true CLIP answer vector in direction and scale."""
    latent, target = latent.float(), target.detach().float()  # (B, D_t), (B, D_t)
    cosine = 1 - F.cosine_similarity(latent, target, dim=-1, eps=1e-6)  # (B,)
    l1 = F.smooth_l1_loss(latent, target, reduction="none").mean(dim=-1)  # (B,)
    loss = cosine + l1  # (B,)
    if available is not None:
        mask = torch.as_tensor(available, dtype=latent.dtype, device=latent.device).view(-1)  # (B,)
        loss = loss * mask  # (B,)
    return loss.mean()  # ()


def attention_guidance_loss(attn_map, weights, target, available=None):
    """Guide latent-answer spatial cross-attention toward counterfactual answer-gain foreground patches."""
    attn = attn_map.float().flatten(1).clamp_min(1e-6)  # (B, N), N = H_p * W_p
    with torch.no_grad():
        fg = (target > 0.5).float()  # (B, 1, H, W)
        salience = weights.detach().float() * fg + 0.25 * fg  # (B, 1, H, W)
        pooled = F.adaptive_avg_pool2d(salience, attn_map.shape[-2:]).flatten(1)  # (B, N)
        mass = pooled.sum(dim=-1, keepdim=True)  # (B, 1)
        valid = (mass.squeeze(-1) > 1e-6).float()  # (B,)
        if available is not None:
            valid = valid * torch.as_tensor(available, dtype=valid.dtype, device=valid.device).view(-1)  # (B,)
        target_dist = pooled / mass.clamp_min(1e-6)  # (B, N)
    kl = (target_dist * (target_dist.clamp_min(1e-6).log() - attn.log())).sum(dim=-1)  # (B,)
    return (kl * valid).mean()  # ()


class DistillationObjective:
    def __init__(self, teacher=None, *, dice_weight=0.5, oracle_weight=0.5, latent_weight=0.5, attn_weight=0.2):
        self.teacher = teacher.requires_grad_(False).eval() if teacher is not None else None
        self.dice_weight = dice_weight
        self.oracle_weight = oracle_weight
        self.latent_weight = latent_weight
        self.attn_weight = attn_weight

    def __call__(self, model, batch):
        student_batch = dict(batch, self_distill=False) if self.teacher is not None else batch
        output = model(student_batch)
        target = batch["mask"]  # (B, 1, H, W)
        student = resize_logits(output["logits"], target).float()  # (B, 1, H, W)
        supervised = segmentation_loss(student, target, self.dice_weight)  # ()
        if self.teacher is not None:
            self.teacher.eval()
            with torch.no_grad():
                features = self.teacher.encode_image(batch["image"])
                answer_logits = self.teacher.decode(features, batch["answer_text"])["logits"]  # (B, 1, 16*H_p, 16*W_p)
                question_logits = self.teacher.decode(features, batch["question_text"])["logits"]  # (B, 1, 16*H_p, 16*W_p)
                del features
            oracle = torch.zeros_like(supervised)  # ()
        else:
            answer_logits = output["teacher_logits"]  # (B, 1, 16*H_p, 16*W_p)
            question_logits = output["question_logits"]  # (B, 1, 16*H_p, 16*W_p)
            oracle = segmentation_loss(answer_logits, target, self.dice_weight)  # ()
        with torch.no_grad():
            answer = resize_logits(answer_logits.detach(), target).float()  # (B, 1, H, W)
            question = resize_logits(question_logits.detach(), target).float()  # (B, 1, H, W)
            probability = answer.sigmoid()  # (B, 1, H, W)
            entropy = F.binary_cross_entropy_with_logits(answer, probability, reduction="none")  # (B, 1, H, W)
            weights = distillation_weights(answer, question, target, probability=probability, entropy=entropy)  # (B, 1, H, W)
            if "answer_available" in batch:
                avail = torch.as_tensor(batch["answer_available"], dtype=weights.dtype, device=weights.device)
                weights = weights * avail.view(-1, 1, 1, 1)  # (B, 1, 1, 1) -> (B, 1, H, W)
            active = weights.flatten(1).sum(1).mean()  # ()
        divergence = F.binary_cross_entropy_with_logits(student, probability, reduction="none") - entropy  # (B, 1, H, W)
        # Batch mean: no-answer/no-gain examples contribute zero KD.
        kd = (weights * divergence).flatten(1).sum(1).mean()  # ()
        latent = torch.zeros_like(supervised)  # ()
        if "latent_answer" in output and "target_answer" in output:
            latent = latent_answer_loss(output["latent_answer"], output["target_answer"], batch.get("answer_available"))
        attn_align = torch.zeros_like(supervised)  # ()
        if output.get("answer_attn") is not None:
            attn_align = attention_guidance_loss(output["answer_attn"], weights, target, batch.get("answer_available"))
        total = (
            supervised
            + self.oracle_weight * oracle
            + kd
            + self.latent_weight * latent
            + self.attn_weight * attn_align
        )
        return total, {
            "segmentation": supervised.detach(),
            "oracle_segmentation": oracle.detach(),
            "distillation": kd.detach(),
            "latent_alignment": latent.detach(),
            "attn_guidance": attn_align.detach(),
            "active_fraction": active,
        }


class SupervisedObjective:
    def __init__(self, dice_weight=0.):
        self.dice_weight = dice_weight
    def __call__(self, model, batch):
        output = model(batch)
        loss = segmentation_loss(output["logits"], batch["mask"], self.dice_weight)  # ()
        return loss, {"segmentation": loss.detach()}
