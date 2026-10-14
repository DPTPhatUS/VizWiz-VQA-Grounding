import math
import torch
import torch.distributed as dist
from torch.nn import functional as F


def resize_logits(logits, target):
    if logits.shape[-2:] == target.shape[-2:]:
        return logits
    return F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)


def segmentation_loss(logits, target, dice_weight=0.0, *, reduction="mean"):
    """Mean batch loss, or one scalar per image for scheduled supervision."""
    logits = resize_logits(logits, target).float()  # (B, 1, H, W)
    target = target.float()  # (B, 1, H, W)
    axes = tuple(range(1, target.ndim))
    if reduction == "mean":
        bce = F.binary_cross_entropy_with_logits(logits, target)
    elif reduction == "none":
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(axes)
    else:
        raise ValueError("Segmentation reduction must be mean or none")
    if not dice_weight:
        return bce
    probability = logits.sigmoid()  # (B, 1, H, W)
    dice = 1 - (2*(probability*target).sum(axes)+1) / (
        probability.sum(axes)+target.sum(axes)+1)  # (B,)
    return bce + dice_weight * (dice.mean() if reduction == "mean" else dice)


def masked_loss_mean(values, available=None, *, valid_mean=False):
    """Legacy batch mean or mean over examples with a valid privileged target."""
    if available is None:
        return values.mean()
    mask = torch.as_tensor(available, dtype=values.dtype, device=values.device).view(-1)
    total = (values * mask).sum()
    if not valid_mean:
        return (values * mask).mean()
    count = mask.sum().detach().clone()
    world = 1
    if dist.is_initialized():
        dist.all_reduce(count)
        world = dist.get_world_size()
    # DDP averages gradients; each rank contributes to the global valid-example mean.
    return total * world / count.clamp_min(1)


def bernoulli_kl(student_logits, teacher_logits):
    """Pointwise KL(teacher || student), evaluated stably in log space."""
    student = student_logits.float()
    teacher = teacher_logits.detach().float()
    p = teacher.sigmoid()
    return p * (F.logsigmoid(teacher) - F.logsigmoid(student)) + (1 - p) * (
        F.logsigmoid(-teacher) - F.logsigmoid(-student))


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


def latent_answer_loss(latent, target, available=None, *, loss_version="legacy"):
    """Align inferred latent answer vector with true CLIP answer vector in direction and scale."""
    latent, target = latent.float(), target.detach().float()  # (B, D_t), (B, D_t)
    cosine = 1 - F.cosine_similarity(latent, target, dim=-1, eps=1e-6)  # (B,)
    l1 = F.smooth_l1_loss(latent, target, reduction="none").mean(dim=-1)  # (B,)
    loss = cosine + l1  # (B,)
    return masked_loss_mean(loss, available, valid_mean=loss_version == "stable")


def attention_target_distribution(weights, target, size, available=None, gain_mix=0.75):
    """Mix separately normalized patch gain and foreground distributions; no gain falls back to foreground."""
    if not 0 <= gain_mix <= 1:
        raise ValueError("gain_mix must be in [0,1]")
    with torch.no_grad():
        fg = (target > 0.5).float()
        fg_pooled = F.adaptive_avg_pool2d(fg, size).flatten(1)
        gain_pooled = F.adaptive_avg_pool2d(weights.detach().float() * fg, size).flatten(1)
        fg_mass = fg_pooled.sum(-1, keepdim=True)
        gain_mass = gain_pooled.sum(-1, keepdim=True)
        prior = fg_pooled / fg_mass.clamp_min(1e-12)
        gain = gain_pooled / gain_mass.clamp_min(1e-12)
        mixed = gain_mix * gain + (1 - gain_mix) * prior
        distribution = torch.where(gain_mass > 1e-12, mixed, prior)
        valid = (fg_mass.squeeze(-1) > 0).float()
        if available is not None:
            valid *= torch.as_tensor(available, device=valid.device, dtype=valid.dtype).view(-1)
        return distribution, valid


def attention_guidance_loss(attn_map, weights, target, available=None, gain_mix=0.75, *, loss_version="legacy"):
    target_dist, valid = attention_target_distribution(weights, target, attn_map.shape[-2:], available, gain_mix)
    if loss_version == "stable":
        attn = attn_map.float().flatten(1) + 1e-8
        attn = attn / attn.sum(-1, keepdim=True)
        kl = (torch.xlogy(target_dist, target_dist) - target_dist * attn.log()).sum(-1)
        return masked_loss_mean(kl, valid, valid_mean=True)
    attn = attn_map.float().flatten(1).clamp_min(1e-6)
    kl = (target_dist * (target_dist.clamp_min(1e-6).log() - attn.log())).sum(-1)
    return (kl * valid).mean()


@torch.no_grad()
def distillation_diagnostics(answer, question, target, weights):
    qa_iou = per_image_iou(answer, target)
    q_iou = per_image_iou(question, target)
    return {
        "mean_kd_weight_mass": weights.flatten(1).sum(1).mean(),
        "kd_pixel_fraction": (weights > 0).float().flatten(1).mean(1).mean(),
        "teacher_qa_iou": qa_iou.mean(),
        "teacher_q_iou": q_iou.mean(),
        "teacher_iou_gain": (qa_iou - q_iou).mean(),
        "attention_gain_image_fraction": ((weights * (target > .5)).flatten(1).sum(1) > 1e-12).float().mean(),
    }


class DistillationObjective:
    def __init__(self, teacher=None, *, dice_weight=0.5, oracle_weight=0.5, question_weight=0.5,
                 latent_weight=0.5, attn_weight=0.2, kd_weight=1., gain_mix=0.75,
                 loss_version="legacy", detach_predicted_answer=False):
        self.teacher = teacher.requires_grad_(False).eval() if teacher is not None else None
        self.dice_weight = dice_weight
        self.oracle_weight = oracle_weight
        self.question_weight = question_weight
        self.gain_mix = gain_mix
        self.latent_weight = latent_weight
        self.attn_weight = attn_weight
        self.kd_weight = kd_weight
        self.loss_version = loss_version
        self.detach_predicted_answer = detach_predicted_answer

    def __call__(self, model, batch, epoch=0):
        student_batch = dict(batch, self_distill=False) if self.teacher is not None else batch
        if self.detach_predicted_answer:
            student_batch = dict(student_batch, detach_predicted_answer=True)
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
            question_supervised = torch.zeros_like(supervised)
        else:
            answer_logits = output["teacher_logits"]  # (B, 1, 16*H_p, 16*W_p)
            question_logits = output["question_logits"]  # (B, 1, 16*H_p, 16*W_p)
            if self.loss_version == "stable":
                oracle = masked_loss_mean(segmentation_loss(answer_logits, target, self.dice_weight, reduction="none"),
                                          batch.get("answer_available"), valid_mean=True)
            else:
                oracle = segmentation_loss(answer_logits, target, self.dice_weight)  # ()
            question_supervised = segmentation_loss(question_logits, target, self.dice_weight)
        with torch.no_grad():
            answer = resize_logits(answer_logits.detach(), target).float()  # (B, 1, H, W)
            question = resize_logits(question_logits.detach(), target).float()  # (B, 1, H, W)
            probability = answer.sigmoid()  # (B, 1, H, W)
            entropy = F.binary_cross_entropy_with_logits(answer, probability, reduction="none")  # (B, 1, H, W)
            weights = distillation_weights(answer, question, target, probability=probability, entropy=entropy)  # (B, 1, H, W)
            if "answer_available" in batch:
                avail = torch.as_tensor(batch["answer_available"], dtype=weights.dtype, device=weights.device)
                weights = weights * avail.view(-1, 1, 1, 1)  # (B, 1, 1, 1) -> (B, 1, H, W)
            diagnostics = distillation_diagnostics(answer, question, target, weights)
        divergence = (bernoulli_kl(student, answer) if self.loss_version == "stable" else
                      F.binary_cross_entropy_with_logits(student, probability, reduction="none") - entropy)
        # Batch mean: no-answer/no-gain examples contribute zero KD.
        kd = (weights * divergence).flatten(1).sum(1).mean()  # ()
        latent = torch.zeros_like(supervised)  # ()
        if "latent_answer" in output and "target_answer" in output:
            latent = latent_answer_loss(output["latent_answer"], output["target_answer"], batch.get("answer_available"),
                                        loss_version=self.loss_version)
        attn_align = torch.zeros_like(supervised)  # ()
        if output.get("answer_attn") is not None:
            attn_align = attention_guidance_loss(output["answer_attn"], weights, target, batch.get("answer_available"),
                                                 self.gain_mix, loss_version=self.loss_version)
        total = (
            supervised
            + self.oracle_weight * oracle
            + self.question_weight * question_supervised
            + self.kd_weight * kd
            + self.latent_weight * latent
            + self.attn_weight * attn_align
        )
        return total, {
            "segmentation": supervised.detach(),
            "oracle_segmentation": oracle.detach(),
            "question_segmentation": question_supervised.detach(),
            "weighted_oracle_segmentation": (self.oracle_weight * oracle).detach(),
            "weighted_question_segmentation": (self.question_weight * question_supervised).detach(),
            "distillation": kd.detach(),
            "latent_alignment": latent.detach(),
            "attn_guidance": attn_align.detach(),
            "weighted_distillation": (self.kd_weight * kd).detach(),
            "weighted_latent_alignment": (self.latent_weight * latent).detach(),
            "weighted_attn_guidance": (self.attn_weight * attn_align).detach(),
            **diagnostics,
        }


class SupervisedObjective:
    def __init__(self, dice_weight=0.):
        self.dice_weight = dice_weight
    def __call__(self, model, batch, epoch=0):
        output = model(dict(batch, self_distill=False, encode_answer_target=False))
        loss = segmentation_loss(output["logits"], batch["mask"], self.dice_weight)  # ()
        return loss, {"segmentation": loss.detach()}


class AnswerAlignmentObjective:
    """Predicted-answer supervision, optionally warmed up with the real answer.

    Schedule mixes per-image losses, not answer embeddings. Missing real answers
    always use predicted-answer grounding, including during warm-up.
    """
    def __init__(self, *, dice_weight=.5, latent_weight=.5, loss_version="legacy",
                 detach_predicted_answer=False, warmup_epochs=0, transition_epochs=0):
        self.dice_weight = dice_weight
        self.latent_weight = latent_weight
        self.loss_version = loss_version
        self.detach_predicted_answer = detach_predicted_answer
        self.warmup_epochs = warmup_epochs
        self.transition_epochs = transition_epochs

    def predicted_fraction(self, epoch):
        if epoch < self.warmup_epochs:
            return 0.
        if not self.transition_epochs:
            return 1.
        return min(1., (epoch - self.warmup_epochs + 1) / self.transition_epochs)

    def __call__(self, model, batch, epoch=0):
        fraction = self.predicted_fraction(epoch)
        output = model(dict(batch, self_distill=False, encode_answer_target=True,
                            oracle_supervision=fraction < 1,
                            detach_predicted_answer=self.detach_predicted_answer))
        target = batch["mask"]
        predicted = segmentation_loss(output["logits"], target, self.dice_weight, reduction="none")
        available = batch["answer_available"].to(device=predicted.device, dtype=predicted.dtype)
        oracle = torch.zeros_like(predicted)
        if fraction < 1:
            oracle = segmentation_loss(output["teacher_logits"], target, self.dice_weight, reduction="none")
        real_fraction = (1 - fraction) * available
        supervised = ((1 - real_fraction) * predicted + real_fraction * oracle).mean()
        latent = latent_answer_loss(output["latent_answer"], output["target_answer"], available,
                                    loss_version=self.loss_version)
        total = supervised + self.latent_weight * latent
        return total, {"segmentation": supervised.detach(), "predicted_segmentation": predicted.mean().detach(),
                       "oracle_segmentation": masked_loss_mean(oracle, available, valid_mean=True).detach(),
                       "latent_alignment": latent.detach(),
                       "weighted_latent_alignment": (self.latent_weight * latent).detach(),
                       "predicted_fraction": predicted.new_tensor(fraction),
                       "oracle_image_fraction": available.mean().detach() * (1 - fraction)}
