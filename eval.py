"""Evaluate this branch's checkpoint with question-only text and export masks."""
import argparse
import json
import os
from pathlib import Path
import torch
from dataset import VizWizGroundingDataset, make_loader
from checkpoints import read_checkpoint, model_from_checkpoint, checkpoint_info
from artifacts import write_json
from metrics import evaluate_counts, summarize_counts, sample_metrics
from runtime import environment
from predictions import LogitWriter


def eval_main(argv=None):
    parser = argparse.ArgumentParser(description="Question-only evaluation using saved model architecture")
    parser.add_argument("--checkpoint",required=True,help="Checkpoint from this experiment")
    parser.add_argument("--data-root",default="data/vizwiz")
    parser.add_argument("--dataset",choices=["val","test"],default="val")
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--num-workers",type=int,default=4)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir",required=True)
    parser.add_argument('--save-masks', action='store_true', help='Also export native binary PNG masks')
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count")
    saved = read_checkpoint(args.checkpoint)
    config = saved["run_config"]
    info = checkpoint_info(args.checkpoint,saved)
    record_path = Path(args.checkpoint).parent/'run.json'
    if not record_path.exists():
        write_json(record_path,{'run_config':config,'experiment':saved['experiment_config'],
                               'provenance':saved.get('provenance') or environment()})
    model = model_from_checkpoint(saved)
    del saved
    device = torch.device(args.device); model.to(device)
    dataset = VizWizGroundingDataset(args.data_root,args.dataset,config['image_size'])
    loader = make_loader(dataset,args,args.batch_size)
    out = Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    with LogitWriter(out/'logits.npz', args.dataset, info, config['image_size']) as logits:
        counts = evaluate_counts(model,loader,device,out if args.save_masks else None,
                                 logit_writer=logits,boundary=True)
    summary = {**summarize_counts(counts),
               "dataset":args.dataset,"metric_resolution":"original",
               "text_mode":"question","checkpoint":info,
               "threshold":.5,
               "boundary_definition":"mean image-wise inner-band IoU; 3x3 zero-padded erosion, "
                                     "max(1, round(0.02 * native diagonal)) iterations"}
    write_json(out/'metrics.json', {'summary':summary,
        'run':os.path.relpath(record_path,out),
        'per_sample':{name:{**row, **sample_metrics(row)} for name,row in counts.items()},
        'logits':'logits.npz', 'masks':args.save_masks})
    print(json.dumps(summary),flush=True)


main = eval_main

if __name__ == "__main__":
    main()
