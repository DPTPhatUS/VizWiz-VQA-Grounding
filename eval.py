"""Evaluate this branch's checkpoint with question-only text and export masks."""
import argparse
import json
from pathlib import Path
import torch
from dataset import VizWizGroundingDataset, make_loader
from checkpoints import read_checkpoint, model_from_checkpoint
from artifacts import sha256_file
from runtime import environment
from metrics import evaluate_counts, summarize_counts, sample_metrics


def eval_main(argv=None):
    parser = argparse.ArgumentParser(description="Question-only evaluation using saved model architecture")
    parser.add_argument("--checkpoint",required=True,help="Checkpoint from this experiment")
    parser.add_argument("--data-root",default="data/vizwiz")
    parser.add_argument("--dataset",choices=["val","test"],default="val")
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--num-workers",type=int,default=4)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir",required=True)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count")
    saved = read_checkpoint(args.checkpoint)
    config = saved["run_config"]
    model = model_from_checkpoint(saved)
    del saved
    device = torch.device(args.device); model.to(device)
    dataset = VizWizGroundingDataset(args.data_root,args.dataset,config['image_size'])
    loader = make_loader(dataset,args,args.batch_size)
    out = Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    counts = evaluate_counts(model,loader,device,out)
    scores = {name: sample_metrics(row)["iou"] for name,row in counts.items()}
    summary = {**summarize_counts(counts),
               "dataset":args.dataset,"metric_resolution":"original",
               "text_mode":"question","checkpoint":str(args.checkpoint),
               "checkpoint_sha256":sha256_file(args.checkpoint), "environment":environment(),
               "experiment_config":model.experiment_config}
    (out/"metrics.json").write_text(json.dumps({"summary":summary,"per_sample":scores,"counts":counts},indent=2))
    print(json.dumps(summary),flush=True)


main = eval_main

if __name__ == "__main__":
    main()
