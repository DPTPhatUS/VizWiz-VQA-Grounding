"""Training-only point classification of answer evidence versus annotated context."""

import torch
from torch import nn
from torch.nn import functional as F

from answer_data import INVALID_ANSWERS, normalize_answer, vocabulary_digest


def sample_points(mask, prediction, *, random=True):
    """64 GT points,32 high-logit background points,32 uniform background points.

    Coordinates come from the training mask grid, not thresholded low-resolution
    masks. Discrete mining is detached; auxiliary gradients update visual features.
    At evaluation the point selection is deterministic and shared by all probes.
    """
    _, _, height, width = mask.shape
    score = F.interpolate(
        prediction.detach().float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    coordinates, valid_sets = [], []
    for truth, confidence in zip(mask.detach()[:, 0] > 0.5, score[:, 0]):
        foreground = truth.flatten().nonzero().flatten()
        background = (~truth).flatten().nonzero().flatten()

        def pick(candidates, count):
            if not len(candidates):
                return torch.zeros(count, dtype=torch.long, device=mask.device)
            if random:
                positions = torch.randint(len(candidates), (count,), device=mask.device)
            else:
                positions = (
                    (torch.arange(count, device=mask.device).float() + 0.5)
                    * len(candidates)
                    / count
                ).long()
            return candidates[positions]

        positive = pick(foreground, 64)
        if len(background):
            candidates = confidence.flatten()[background]
            hard = background[candidates.topk(min(32, len(background))).indices]
            hard = hard[torch.arange(32, device=mask.device) % len(hard)]
        else:
            hard = torch.zeros(32, dtype=torch.long, device=mask.device)
        selected = torch.cat((positive, hard, pick(background, 32)))
        y, x = selected // width, selected % width
        coordinates.append(
            torch.stack(
                ((x.float() + 0.5) * 2 / width - 1, (y.float() + 0.5) * 2 / height - 1),
                -1,
            )
        )
        valid_sets.append((bool(len(foreground)), bool(len(background))))
    return torch.stack(coordinates)[:, :, None], valid_sets


class AnswerHead(nn.Module):
    def __init__(self, channels, vocabulary, *, tiny=False):
        super().__init__()
        if not vocabulary or list(vocabulary) != sorted(set(vocabulary)):
            raise ValueError("Provide the sorted, unique train-answer vocabulary")
        if any(normalize_answer(s) != s or s in INVALID_ANSWERS for s in vocabulary):
            raise ValueError("Vocabulary contains invalid or non-normalized answers")
        self.lookup = {answer: i + 1 for i, answer in enumerate(vocabulary)}
        self.register_buffer(
            "vocabulary_hash",
            torch.tensor(list(vocabulary_digest(vocabulary)), dtype=torch.uint8),
        )
        width = 16 if tiny else 128
        hidden = 8 if tiny else 64
        self.features = nn.Sequential(
            nn.Linear(channels, width), nn.GELU(), nn.Linear(width, hidden), nn.GELU()
        )
        self.classifier = nn.Linear(hidden, len(vocabulary) + 1)

    def forward(self, feature, mask, answers, prediction=None):
        if prediction is None:
            prediction = torch.zeros_like(mask)
        coordinates, valid = sample_points(mask, prediction, random=self.training)
        sampled = (
            F.grid_sample(feature.float(), coordinates, align_corners=False)
            .squeeze(-1)
            .transpose(1, 2)
        )
        logits = self.classifier(self.features(sampled))
        targets = torch.full(
            logits.shape[:2], -100, dtype=torch.long, device=feature.device
        )
        for index, (answer, (foreground, background)) in enumerate(zip(answers, valid)):
            label = self.lookup.get(normalize_answer(answer))
            # Missing/OOV/empty-region answers do not silently become background labels.
            if label is not None and foreground:
                targets[index, :64] = label
                if background:
                    targets[index, 64:] = 0
        return {"answer_logits": logits, "answer_targets": targets}
