"""Shared control / answer-dropout teacher branch."""
from research.model import ResearchGrounder
from research.losses import segmentation_loss


def add_arguments(parser):
    pass


def validate_args(args):
    pass


def build_model(args):
    return ResearchGrounder(args.architecture, args.tiny)


class SupervisedObjective:
    def __init__(self, dice_weight):
        self.dice_weight = dice_weight
    def __call__(self, model, batch):
        output = model(batch)
        loss = segmentation_loss(output["logits"], batch["mask"], self.dice_weight)
        return loss, {"segmentation": loss.detach()}


def build_objective(args, model, device):
    return SupervisedObjective(args.dice_weight)
