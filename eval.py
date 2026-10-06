"""Evaluate this branch's checkpoint with question-only text and export masks."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from dataset import VizWizGroundingDataset, make_loader
from models.model import build_model
from models.checkpoint import read_checkpoint, load_checkpoint
from metrics import evaluate


POLICIES = ('gain', 'uncertainty', 'random', 'fixed', 'relevance')


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


def eval_main(argv=None):
    parser = argparse.ArgumentParser(description="Question-only evaluation using checkpoint architecture")
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--data-root",default="data/vizwiz")
    parser.add_argument("--dataset",choices=["val","test"],default="val")
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--num-workers",type=int,default=4)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--metric-resolution",choices=["original","local"],default="original")
    add_eval_arguments(parser)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count")
    saved = read_checkpoint(args.checkpoint)
    config = SimpleNamespace(**saved["run_config"])
    model = build_model(config)
    load_checkpoint(args.checkpoint,model)
    inference = configure_evaluation(model,args)
    device = torch.device(args.device); model.to(device)
    dataset = VizWizGroundingDataset(args.data_root,args.dataset,config.image_size,
        config.detail_size if getattr(config,"needs_detail",False) else None)
    loader = make_loader(dataset,args,args.batch_size)
    out = Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    scores = evaluate(model,loader,device,args.metric_resolution,out)
    summary = {"num_samples":len(scores),"mean_iou":sum(scores.values())/len(scores),
               "dataset":args.dataset,"metric_resolution":args.metric_resolution,
               "text_mode":"question","checkpoint":str(args.checkpoint),
               "experiment_config":model.experiment_config,"inference":inference}
    (out/"metrics.json").write_text(json.dumps({"summary":summary,"per_sample":scores},indent=2))
    print(json.dumps(summary),flush=True)


main = eval_main

if __name__ == "__main__":
    main()
