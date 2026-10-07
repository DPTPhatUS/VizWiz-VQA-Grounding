"""Evaluate this branch's checkpoint with question-only text and export masks."""
import argparse
import json
from pathlib import Path
import torch
from dataset import VizWizGroundingDataset, make_loader
from models.model import GroundingModel, TeacherModel
from utils import read_checkpoint, load_model_weights
from metrics import evaluate


def eval_main(argv=None):
    parser = argparse.ArgumentParser(description="Question-only evaluation using saved model architecture")
    parser.add_argument("--checkpoint",required=True,help="Raw model weights or a training checkpoint")
    parser.add_argument("--data-root",default="data/vizwiz")
    parser.add_argument("--dataset",choices=["val","test"],default="val")
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--num-workers",type=int,default=4)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--teacher", action="store_true", help="Evaluate the joint-skip teacher instead of the student")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count")
    saved = read_checkpoint(args.checkpoint)
    config = saved["run_config"]
    model_type = TeacherModel if args.teacher or config.get("teacher_training", False) else GroundingModel
    model = model_type(tiny=config.get("tiny", False))
    load_model_weights(model, saved)
    device = torch.device(args.device); model.to(device)
    dataset = VizWizGroundingDataset(args.data_root,args.dataset,config.get("image_size", 336))
    loader = make_loader(dataset,args,args.batch_size)
    out = Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    scores = evaluate(model,loader,device,out)
    summary = {"num_samples":len(scores),"mean_iou":sum(scores.values())/len(scores),
               "dataset":args.dataset,"metric_resolution":"original",
               "text_mode":"question","checkpoint":str(args.checkpoint),
               "experiment_config":model.experiment_config}
    (out/"metrics.json").write_text(json.dumps({"summary":summary,"per_sample":scores},indent=2))
    print(json.dumps(summary),flush=True)


main = eval_main

if __name__ == "__main__":
    main()
