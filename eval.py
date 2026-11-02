"""Evaluate a checkpoint on val or test with the official native-resolution mIoU.

Usage:
    uv run eval.py --checkpoint OUT/checkpoint_epoch20.pt --split val --data-root DATA --output-dir OUT/eval_val_ep20
    torchrun --nproc_per_node 2 eval.py ...        # images are split across GPUs, scored once each

Writes OUTPUT_DIR/metrics.json ({"summary": ..., "per_sample": {filename: IoU}}) and, with
--save-masks, the predicted masks as PNGs at native resolution. Refuses to overwrite results.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from dataset import TEXT_MODES, VizWizGroundingDataset, collate
from evaluation import ShardSampler, evaluate, gather_scores, mean_iou
from models import GroundingModel
from utils import cleanup_distributed, setup_distributed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--data-root", default="data/vizwiz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8, help="Images per GPU")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-masks", action="store_true", help="Also save predicted masks as PNGs")
    parser.add_argument("--text-mode", choices=TEXT_MODES, default=None,
                        help="Default: the checkpoint's training text mode. Override for diagnostics, "
                             "e.g. a qa-trained model evaluated with the question only")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers nonnegative")
    if not Path(args.checkpoint).is_file():
        parser.error(f"Checkpoint is not a file: {args.checkpoint}")
    return args


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    if (out / "metrics.json").exists():
        raise FileExistsError(f"Refusing to overwrite {out / 'metrics.json'}")
    rank, world_size, _, device = setup_distributed()
    try:
        model, metadata = GroundingModel.from_checkpoint(args.checkpoint, device)
        trained_mode = ((metadata.get("run_config") or {}).get("recipe") or {}).get("text_mode", "question")
        text_mode = args.text_mode or trained_mode
        dataset = VizWizGroundingDataset(args.data_root, args.split, native_mask=True, text_mode=text_mode)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=ShardSampler(len(dataset), rank, world_size),
                            num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate)
        save = None
        if args.save_masks:
            (out / "masks").mkdir(parents=True, exist_ok=True)

            def save(name, mask):
                to_pil_image(mask.to("cpu", torch.uint8) * 255).save(out / "masks" / f"{Path(name).stem}.png")

        scores = gather_scores(evaluate(model, loader, device, save), world_size)
        if rank == 0:
            summary = {
                "checkpoint": str(args.checkpoint),
                "epoch": metadata.get("epoch"),
                "split": args.split,
                "num_samples": len(scores),
                "mean_iou": mean_iou(scores, len(dataset)),
                "protocol": "native resolution, bilinear logits, threshold 0, empty union = 1, FP32",
                "text_mode": text_mode,
                "trained_text_mode": trained_mode,
                "answer_in_input": text_mode == "qa",
                "model_config": metadata["model_config"],
                "recipe": (metadata.get("run_config") or {}).get("recipe"),
            }
            out.mkdir(parents=True, exist_ok=True)
            (out / "metrics.json").write_text(json.dumps({"summary": summary, "per_sample": scores}, indent=1))
            print(f"{args.split}: mean IoU {100 * summary['mean_iou']:.2f} over {len(scores)} images -> {out / 'metrics.json'}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
