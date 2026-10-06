"""Experiment C: question-conditioned evidence location and extent."""
import math
from research.extent import EvidenceExtentGrounder, ExtentObjective

EXPERIMENT = 'evidence-extent'


def add_arguments(parser):
    parser.add_argument('--evidence-width', type=int, default=None, help='Token width (default 64; tiny 16)')
    parser.add_argument('--support-weight', type=float, default=.2)
    parser.add_argument('--area-weight', type=float, default=.1)
    parser.add_argument('--pairs', help='Verified training question pairs with each question own mask')
    parser.add_argument('--pair-delta-weight', type=float, default=0.)
    parser.add_argument('--pair-consistency-weight', type=float, default=0.)


def validate_args(args):
    if args.evidence_width is not None and (args.evidence_width <= 0 or args.evidence_width % (2 if args.tiny else 4)):
        raise ValueError('Evidence width must be positive and divisible by the attention head count')
    if args.architecture != 'compact':
        raise ValueError('Evidence extent requires the compact architecture')
    if args.text_mode != 'question':
        raise ValueError('Evidence extent requires question-only training and inference')
    for name in ('support_weight', 'area_weight', 'pair_delta_weight', 'pair_consistency_weight'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if (args.pair_delta_weight or args.pair_consistency_weight) and not args.pairs:
        raise ValueError('Paired losses require an explicit verified --pairs manifest')


def build_model(args):
    if args.architecture != 'compact':
        raise ValueError('Evidence extent requires the compact architecture')
    return EvidenceExtentGrounder(args.tiny, args.evidence_width)


def build_objective(args, model, device):
    return ExtentObjective(args.dice_weight, args.support_weight, args.area_weight,
                           args.pair_delta_weight, args.pair_consistency_weight)
