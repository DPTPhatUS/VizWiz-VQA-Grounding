"""Branch architecture and checkpoint-driven model construction."""
from torch import nn
from models.backbone import BaseGroundingModel

EXPERIMENT = 'answer-value-distillation'


class GroundingModel(nn.Module):
    def __init__(self, tiny=False):
        super().__init__()
        self.coarse = BaseGroundingModel("compact", tiny)
        self.experiment_config = dict(self.coarse.experiment_config, experiment="answer-value-distillation")

    def forward(self, batch):
        return self.coarse(dict(batch, text=batch.get("question_text", batch["text"])))


def build_model(args):
    # Evaluation never opens the training-time teacher checkpoint.
    if getattr(args, "teacher_training", False):
        return BaseGroundingModel("joint", getattr(args, "tiny", False))
    return GroundingModel(getattr(args, "tiny", False))
