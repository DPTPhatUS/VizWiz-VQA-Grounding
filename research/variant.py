"""Gain-guided high-resolution refinement experiment CLI contract."""
import argparse
from research.model import ResearchGrounder
from research.refinement import GainGuidedGrounder
from research.checkpoint import read_checkpoint

EXPERIMENT='gain-guided-refinement'
POLICIES=('gain','uncertainty','random','fixed','relevance')


def add_arguments(parser):
    parser.add_argument('--stage',choices=('refiner','router'),default='refiner')
    parser.add_argument('--crop-size',type=int,default=336)
    parser.add_argument('--policy',choices=POLICIES,default=None)
    parser.add_argument('--budget',type=int,choices=(1,2,4),default=2)
    parser.add_argument('--diversity-iou',type=float,default=.3)
    parser.add_argument('--skip-nonpositive',action=argparse.BooleanOptionalAction,default=True)


def validate_args(args):
    args.needs_detail=True
    args.policy=args.policy or ('fixed' if args.stage=='refiner' else 'gain')
    if args.architecture!='compact' or args.text_mode!='question':
        raise ValueError('Refinement requires compact architecture and question-only text')
    if args.crop_size <= 0 or args.detail_size < args.image_size or not 0 <= args.diversity_iou <= 1:
        raise ValueError('Require positive crop size, detail size >= image size, and diversity IoU in [0,1]')
    if args.policy=='gain' and args.stage=='refiner':
        raise ValueError('Gain policy requires a trained router; use fixed during refiner training')
    if not (args.init_checkpoint or args.resume_checkpoint):
        raise ValueError('Frozen coarse reference requires --init-checkpoint or --resume-checkpoint')
    if args.init_checkpoint:
        checkpoint=read_checkpoint(args.init_checkpoint)
        config=checkpoint['experiment_config']
        if args.stage=='router' and (config.get('experiment')!=EXPERIMENT or
                                    checkpoint['run_config'].get('stage') not in ('refiner','router')):
            raise ValueError('Router initialization requires a trained refinement checkpoint')
        if config.get('experiment')=='controls' and checkpoint['run_config'].get('text_mode')!='question':
            raise ValueError('Frozen coarse initialization requires a question-only controls checkpoint')


def build_model(args):
    return GainGuidedGrounder(ResearchGrounder(args.architecture,args.tiny),stage=args.stage,
                             crop_size=args.crop_size,policy=args.policy,budget=args.budget,
                             dice_weight=args.dice_weight,diversity_iou=args.diversity_iou,
                             skip_nonpositive=args.skip_nonpositive,routing_seed=args.seed)


class RefinementObjective:
    def __call__(self,model,batch):
        output=model(batch,compute_loss=True)
        return output['loss'],{'stage_loss':output['loss'].detach()}


def build_objective(args,model,device):
    return RefinementObjective()


def add_eval_arguments(parser):
    parser.add_argument('--policy',choices=POLICIES)
    parser.add_argument('--budget',type=int,choices=(1,2,4))
    parser.add_argument('--diversity-iou',type=float)
    parser.add_argument('--skip-nonpositive',action=argparse.BooleanOptionalAction,default=None)


def configure_evaluation(model,args):
    for name in ('policy','budget','diversity_iou','skip_nonpositive'):
        value=getattr(args,name)
        if value is not None:
            setattr(model,name,value)
    if not 0 <= model.diversity_iou <= 1:
        raise ValueError('Diversity IoU must be in [0,1]')
    if model.policy=='gain' and model.stage!='router':
        raise ValueError('Gain evaluation requires a trained router-stage checkpoint')
    return {name:getattr(model,name) for name in ('policy','budget','diversity_iou','skip_nonpositive','routing_seed')}
