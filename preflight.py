"""Check dataset alignment, run options, and finite gradients before a training session."""
import json
from pathlib import Path
import sys
from PIL import Image
import torch
from dataset import VizWizGroundingDataset, collate_samples
from models.factory import build_model
from losses import SupervisedObjective
from train import training_parser, configure_run, seed_everything
from runtime import parameter_counts, to_device


def validate_data(root,splits=('train','val')):
    result={}
    for split in splits:
        root=Path(root);records=json.loads((root/f'{split}_grounding.json').read_text())
        if not records:raise ValueError(f'Empty {split} split')
        for name,record in records.items():
            if 'question' not in record or not isinstance(record['question'],str):
                raise ValueError(f'Missing question for {name}')
            with Image.open(root/split/name) as image:
                shape=image.size
                if image.getexif().get(274,1) in (5,6,7,8):
                    shape=shape[::-1]
            with Image.open(root/'binary_masks_png'/split/Path(name).with_suffix('.png')) as mask:
                if shape!=mask.size:raise ValueError(f'Image/mask dimensions differ: {name}')
        result[split]=len(records)
    return result


def main(argv=None):
    parser=training_parser();parser.description=__doc__
    parser.add_argument('--steps',type=int,default=1)
    parser.add_argument('--dataset-check-only',action='store_true')
    argv=sys.argv[1:] if argv is None else argv
    args=parser.parse_args(argv)
    args._explicit={a[2:].split('=')[0].replace('-','_') for a in argv if a.startswith('--')}
    args.device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    configure_run(args)
    if args.steps<1:raise ValueError('Preflight steps must be positive')
    summary={'datasets':validate_data(args.data_root)}
    if args.dataset_check_only:
        print(json.dumps(summary));return summary
    seed_everything(args.seed)
    model=build_model(args.model,tiny=args.tiny,connection_mode=args.connection_mode,
                      context=not args.disable_context,membership=not args.disable_membership,
                      pretrained=True).to(args.device)
    dataset=VizWizGroundingDataset(args.data_root,'train',32 if args.tiny else 336)
    objective=SupervisedObjective(args.edge_loss_weight)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    losses=[]
    # CLIP computes a pooled output that this dense model intentionally does not consume.
    unused_pooler = {'image_encoder.vision_encoder.vision_model.post_layernorm.'+k for k in ('weight','bias')}
    inactive=[]
    model.train()
    for step in range(args.steps):
        batch=to_device(collate_samples([dataset[step%len(dataset)]]),args.device)
        optimizer.zero_grad(set_to_none=True)
        loss,_=objective(model,batch);loss.backward()
        inactive=[name for name,p in model.named_parameters() if p.requires_grad and p.grad is None]
        if not torch.isfinite(loss) or any(name not in unused_pooler for name in inactive) or any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
            raise ValueError('Nonfinite loss/gradient or disconnected trainable parameter')
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise ValueError('Disabled parameter received a gradient')
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step();losses.append(float(loss.detach()))
    summary.update(parameters=parameter_counts(model),losses=losses,unused_pooler_parameters=inactive,
                   experiment_config=model.experiment_config)
    print(json.dumps(summary));return summary


if __name__=='__main__':main()
