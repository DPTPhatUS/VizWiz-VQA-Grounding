"""Branch architecture and checkpoint-driven model construction."""
from models.backbone import BaseGroundingModel

EXPERIMENT = 'controls'


class GroundingModel(BaseGroundingModel):
    """Selectable baseline, residual, joint-skip, or compact control."""
    pass


def build_model(args):
    return GroundingModel(args.architecture, args.tiny)
