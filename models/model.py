"""Branch architecture and checkpoint-driven model construction."""
from models.backbone import BaseGroundingModel

EXPERIMENT = 'controls'


class GroundingModel(BaseGroundingModel):
    """Compact question-only control."""
    pass


def build_model(args):
    return GroundingModel(tiny=getattr(args, "tiny", False))
