"""Question-conditioned grounding models."""
import torch
from torch import nn
from torch.nn import functional as F

from losses import segmentation_loss, per_image_iou
from models.image_encoder import ImageEncoder, TinyVision
from models.text_encoder import TextEncoder, masked_mean
from models.mask_decoder import CompactDecoder


class CompactGroundingModel(nn.Module):
    def __init__(self, tiny=False):
        super().__init__()
        self.image_encoder = TinyVision() if tiny else ImageEncoder()
        self.text_encoder = TextEncoder(tiny)
        self.visual_dim = self.image_encoder.out_channels
        self.text_dim = self.text_encoder.output_dim
        self.text_proj = nn.Linear(self.text_dim, self.visual_dim)
        self.cross_attn = nn.MultiheadAttention(self.visual_dim, 2 if tiny else 8, batch_first=True)
        self.residual_scale = nn.Parameter(torch.tensor(.01))
        self.decoder = CompactDecoder(self.visual_dim, self.text_dim, width=16 if tiny else 128)
        self.register_buffer("image_mean", torch.tensor([.48145466, .4578275, .40821073]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([.26862954, .26130258, .27577711]).view(1, 3, 1, 1))
        self.experiment_config = {"experiment": "controls", "architecture": "compact", "tiny": tiny,
                                  "protocol": "normalized-masked-question-v1"}

    def encode_image(self, image):
        return self.image_encoder((image - self.image_mean) / self.image_std)

    def decode(self, features, texts):
        s1, s2, s3, vision = features
        text = self.text_encoder(texts)
        query = vision.flatten(2).transpose(1, 2)
        projected = self.text_proj(text.tokens)
        attended, _ = self.cross_attn(
            query, projected, projected, key_padding_mask=~text.attention_mask, need_weights=False)
        fused = query + self.residual_scale * attended
        fused = fused.transpose(1, 2).reshape_as(vision)
        logits = self.decoder(fused, s3, s2, s1, text)
        return {"logits": logits, "visual": fused,
                "pooled_text": masked_mean(text.tokens, text.valid_mask)}

    def forward(self, batch):
        return self.decode(self.encode_image(batch["image"]), batch["text"])


EXPERIMENT = 'gain-guided-refinement'


def candidate_windows(height, width):
    """Integer half-open boxes; each regular scale tiles the entire canvas."""
    if min(height, width) < 3:
        raise ValueError('Detail canvas must be at least 3 pixels in each dimension')
    return [(y * height // grid, x * width // grid,
             (y + 1) * height // grid, (x + 1) * width // grid)
            for grid in (2, 3) for y in range(grid) for x in range(grid)]


def crop(tensor, box):
    y0, x0, y1, x1 = box
    return tensor[..., y0:y1, x0:x1]


def blend_residuals(coarse, residuals, boxes):
    """Mean overlapping corrections; untouched pixels retain coarse logits."""
    total, count = torch.zeros_like(coarse), torch.zeros_like(coarse)
    height, width = coarse.shape[-2:]
    for residual, (y0, x0, y1, x1) in zip(residuals, boxes):
        resized = F.interpolate(residual, size=(y1 - y0, x1 - x0),
                                mode='bilinear', align_corners=False)
        padding = (x0, width - x1, y0, height - y1)
        total = total + F.pad(resized, padding)
        count = count + F.pad(torch.ones_like(resized), padding)
    return coarse + total / count.clamp_min(1)


def box_iou(a, b):
    ay, ax, by, bx = a
    cy, cx, dy, dx = b
    overlap = max(0, min(by, dy) - max(ay, cy)) * max(0, min(bx, dx) - max(ax, cx))
    return overlap / ((by - ay) * (bx - ax) + (dy - cy) * (dx - cx) - overlap)


def select_windows(scores, boxes, budget, skip_nonpositive=False, diversity_iou=.3):
    selected = []
    for row in scores.detach().float().cpu():
        chosen = []
        for index in torch.argsort(row, descending=True, stable=True).tolist():
            if skip_nonpositive and row[index] <= 0:
                continue
            if all(box_iou(boxes[index], boxes[j]) <= diversity_iou for j in chosen):
                chosen.append(index)
            if len(chosen) == budget:
                break
        selected.append(chosen)
    return selected


class CropRefiner(nn.Module):
    def __init__(self, context_dim, width):
        super().__init__()
        self.context = nn.Linear(context_dim, width)
        self.layers = nn.Sequential(
            nn.Conv2d(width + 4, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, 1, 1))

    def forward(self, rgb, coarse, context):
        context = self.context(context)[..., None, None].expand(-1, -1, *rgb.shape[-2:])
        return self.layers(torch.cat([rgb, coarse, context], dim=1))


class GroundingModel(nn.Module):
    def __init__(self, coarse=None, stage='refiner', crop_size=336, budget=2, tiny=False):
        super().__init__()
        if stage not in ('refiner', 'router'):
            raise ValueError('Unknown refinement stage')
        if crop_size <= 0 or budget not in (1, 2, 4):
            raise ValueError('Invalid crop size or budget')
        coarse = CompactGroundingModel(tiny=tiny) if coarse is None else coarse
        self.coarse = coarse.requires_grad_(False)
        self.stage, self.crop_size, self.budget = stage, crop_size, budget
        self.policy = 'fixed' if stage == 'refiner' else 'gain'
        self.diversity_iou, self.skip_nonpositive = .3, True
        width = 16 if coarse.experiment_config['tiny'] else 64
        context_dim = coarse.visual_dim + coarse.text_dim
        self.refiner = CropRefiner(context_dim, width)
        self.router = nn.Sequential(nn.Linear(context_dim + 7, width), nn.GELU(), nn.Linear(width, 1))
        self.refiner.requires_grad_(stage == 'refiner')
        self.router.requires_grad_(stage == 'router')
        self.experiment_config = {
            'experiment': 'gain-guided-refinement', 'coarse': coarse.experiment_config.copy(),
            'crop_size': crop_size, 'grids': [2, 3], 'width': width, 'protocol': 'residual-mean-gain-v1'}
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.coarse.eval()
        self.refiner.train(mode and self.stage == 'refiner')
        self.router.train(mode and self.stage == 'router')
        return self

    def features(self, batch):
        with torch.no_grad():
            # Never consume answer-bearing text, including during training.
            output = self.coarse({'image': batch['image'], 'text': batch['question_text']})
            height, width = batch['detail_image'].shape[-2:]
            feature_height, feature_width = output['visual'].shape[-2:]
            boxes = candidate_windows(height, width)
            logits = F.interpolate(output['logits'], size=(height, width),
                                   mode='bilinear', align_corners=False)
            contexts, descriptors = [], []
            probability = logits.sigmoid() if self.stage == 'router' else None
            for box in boxes:
                y0, x0, y1, x1 = box
                # Floor the start and ceil the end, retaining at least one feature cell.
                # Pool on the native feature map to avoid upsampling the CLIP features.
                feature_y0 = y0 * feature_height // height
                feature_x0 = x0 * feature_width // width
                feature_y1 = max(feature_y0 + 1, (y1 * feature_height + height - 1) // height)
                feature_x1 = max(feature_x0 + 1, (x1 * feature_width + width - 1) // width)
                feature_box = (feature_y0, feature_x0, feature_y1, feature_x1)
                visual = crop(output['visual'], feature_box).mean((-2, -1))
                context = torch.cat([visual, output['pooled_text']], dim=1)
                contexts.append(context)
                if self.stage == 'router':
                    crop_probability = crop(probability, box)
                    mean_probability = crop_probability.mean((1, 2, 3))
                    uncertainty = (4 * crop_probability * (1 - crop_probability)).mean((1, 2, 3))
                    foreground_fraction = (crop_probability > .5).float().mean((1, 2, 3))
                    statistics = torch.stack([mean_probability, uncertainty, foreground_fraction], dim=1)
                    coordinates = logits.new_tensor([y0 / height, x0 / width, y1 / height, x1 / width])
                    coordinates = coordinates.expand(len(logits), -1)
                    descriptors.append(torch.cat([context, statistics, coordinates], dim=1))
        descriptors = torch.stack(descriptors, 1) if descriptors else None
        return logits, boxes, torch.stack(contexts, 1), descriptors

    def residual(self, batch, coarse, box, context):
        rgb = crop(batch['detail_image'], box)
        rgb = F.interpolate(rgb, size=(self.crop_size, self.crop_size),
                            mode='bilinear', align_corners=False)
        rgb = (rgb - self.coarse.image_mean) / self.coarse.image_std
        logits = F.interpolate(crop(coarse, box), size=rgb.shape[-2:],
                               mode='bilinear', align_corners=False)
        return self.refiner(rgb, logits, context)

    def _refiner_loss(self, batch, coarse, boxes, contexts):
        losses = []
        for image_index in range(len(coarse)):
            window_index = torch.randint(len(boxes), ()).item()
            box = boxes[window_index]
            local = {'detail_image': batch['detail_image'][image_index:image_index + 1]}
            local_coarse = coarse[image_index:image_index + 1]
            context = contexts[image_index:image_index + 1, window_index]
            residual = self.residual(local, local_coarse, box, context)
            # Supervise the sampled crop directly.
            coarse_crop = crop(local_coarse, box)
            prediction = coarse_crop + F.interpolate(residual, size=coarse_crop.shape[-2:],
                                                     mode='bilinear', align_corners=False)
            target = crop(batch['detail_mask'][image_index:image_index + 1], box)
            losses.append(segmentation_loss(prediction, target))
        return {'loss': torch.stack(losses).mean()}

    def _router_loss(self, batch, coarse, boxes, contexts, scores):
        with torch.no_grad():
            # Stream candidates to avoid retaining 13 detail-resolution predictions.
            baseline = per_image_iou(coarse, batch['detail_mask'])
            gains = []
            for window_index, box in enumerate(boxes):
                residual = self.residual(batch, coarse, box, contexts[:, window_index])
                prediction = blend_residuals(coarse, [residual], [box])
                gains.append(per_image_iou(prediction, batch['detail_mask']) - baseline)
            targets = torch.stack(gains, dim=1)
        return {'logits': coarse, 'loss': F.mse_loss(scores.float(), targets),
                'gain_targets': targets, 'gain_scores': scores}

    def _predict(self, batch, coarse, boxes, contexts, scores):
        chosen = select_windows(scores, boxes, self.budget,
                                self.skip_nonpositive and self.policy == 'gain', self.diversity_iou)
        predictions = []
        for image_index, indices in enumerate(chosen):
            local = {'detail_image': batch['detail_image'][image_index:image_index + 1]}
            local_coarse = coarse[image_index:image_index + 1]
            residuals = [self.residual(local, local_coarse, boxes[index],
                                      contexts[image_index:image_index + 1, index]) for index in indices]
            predictions.append(blend_residuals(local_coarse, residuals, [boxes[index] for index in indices]))
        return {'logits': torch.cat(predictions), 'gain_scores': scores,
                'crop_counts': torch.tensor([len(indices) for indices in chosen], device=coarse.device)}

    def forward(self, batch, compute_loss=False):
        coarse, boxes, contexts, descriptors = self.features(batch)
        if compute_loss and self.stage == 'refiner':
            return self._refiner_loss(batch, coarse, boxes, contexts)
        if self.stage == 'refiner':
            scores = -torch.arange(len(boxes), device=coarse.device, dtype=coarse.dtype)
            scores = scores.expand(len(coarse), -1)
        else:
            scores = self.router(descriptors).squeeze(-1)
        if compute_loss:
            return self._router_loss(batch, coarse, boxes, contexts, scores)
        return self._predict(batch, coarse, boxes, contexts, scores)
