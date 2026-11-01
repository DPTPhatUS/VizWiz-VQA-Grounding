"""Construct the CM, FTC, and capacity-control variants."""
from models.cm import GroundingModel


def build_model(model='cm', tiny=False, connection_mode='shared', context=True, membership=True, pretrained=True):
    if model == 'cm':
        return GroundingModel(tiny, connection_mode, context, membership, pretrained)
    if model not in ('ftc', 'ftc-capacity'):
        raise ValueError(f'Unknown model: {model}')
    from models.ftc import FTCModel
    return FTCModel(tiny=tiny, capacity=model == 'ftc-capacity', pretrained=pretrained)
