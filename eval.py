"""Evaluate a trained model on val or test set.

Usage:
    uv run eval.py --checkpoint outputs/checkpoint_epoch20.pt --dataset val
    uv run eval.py --checkpoint outputs/checkpoint_epoch20.pt --dataset test
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
from datetime import datetime
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from dataset import VizWizGroundingDataset, collate_eval_samples
from models import GroundingModel, build_grounding_model, make_naflex_processor
from metrics import compute_iou_per_sample
from utils import align_state_dict, to_device


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--data-root", type=str, default="data/vizwiz")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--text-mode", choices=["qa", "question"], default=None,
                        help="Default: checkpoint text mode, or qa for legacy checkpoints")
    parser.add_argument("--metric-resolution", choices=["original", "resized"], default="original",
                        help="Native GT scoring (default); resized reproduces the legacy 336-pixel protocol")
    args = parser.parse_args(argv)

    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("Batch size must be positive and worker count nonnegative")

    if not os.path.isfile(args.checkpoint):
        parser.error(f"Checkpoint is not a file: {args.checkpoint}")

    device = torch.device(args.device)
    image_size = (336, 336)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
    args.text_mode = args.text_mode or (checkpoint.get("text_mode", "qa") if isinstance(checkpoint, dict) else "qa")
    model = build_grounding_model(model_config, pretrained=False) if model_config else GroundingModel()
    processor = make_naflex_processor(model_config) if model_config else None
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = align_state_dict(checkpoint["model_state_dict"], model)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')})")
    else:
        state_dict = align_state_dict(checkpoint, model)
        model.load_state_dict(state_dict)
        print("Loaded raw state dict")
    del state_dict, checkpoint
    model = model.to(device).eval()

    # --- Dataset paths ---
    if args.dataset == "val":
        json_path = os.path.join(args.data_root, "val_grounding.json")
        image_root = os.path.join(args.data_root, "val")
        mask_root = os.path.join(args.data_root, "binary_masks_png", "val")
    else:  # test
        json_path = os.path.join(args.data_root, "test_grounding.json")
        image_root = os.path.join(args.data_root, "test")
        mask_root = os.path.join(args.data_root, "binary_masks_png", "test")

    # --- Dataset & Loader ---
    dataset = VizWizGroundingDataset(
        json_path=json_path,
        image_root=image_root,
        mask_root=mask_root,
        image_size=image_size,
        is_test=True,  # disable augmentation
        text_mode=args.text_mode,
        return_original_mask=args.metric_resolution == "original" or processor is not None,
        naflex_processor=processor,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate_eval_samples,
    )

    # --- Inference ---
    per_sample_iou = {}

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    loop = tqdm(loader, desc=f"Eval ({args.dataset})", dynamic_ncols=True)
    with torch.no_grad():
        for batch in loop:
            images = to_device(batch["image"], device)
            texts = batch["text"]
            masks = batch["mask"]
            filenames = batch["filename"]

            pred = model(images, texts)  # (B, 1, 16*H_p, 16*W_p)
            for i, fname in enumerate(filenames):
                if args.metric_resolution == "original" or processor is not None:
                    truth = batch["original_mask"][i].unsqueeze(0).to(device)
                    if args.metric_resolution == "resized":
                        truth = F.interpolate(truth, size=image_size, mode="nearest")
                else:
                    truth = masks[i:i+1].to(device)
                sample_pred = pred[i] if isinstance(pred, list) else pred[i:i+1]
                logits = F.interpolate(sample_pred.float(), size=truth.shape[-2:],
                                       mode="bilinear", align_corners=False)
                prediction = (logits > 0).float()
                score = compute_iou_per_sample(prediction, truth,
                                               legacy=args.metric_resolution == "resized").item()
                per_sample_iou[fname] = round(score, 6) if args.metric_resolution == "resized" else score
                if args.output_dir:
                    mask_path = os.path.join(args.output_dir, os.path.splitext(fname)[0] + ".png")
                    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                    to_pil_image(prediction[0, 0].cpu()).save(mask_path)

    if not per_sample_iou:
        raise ValueError("Evaluation dataset is empty")

    # --- Results JSON ---
    if args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(args.output_dir, f"results_{args.dataset}_{timestamp}.json")
        iou_values = list(per_sample_iou.values())
        summary = {
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "text_mode": args.text_mode,
            "metric_resolution": args.metric_resolution,
            "vision_model": model_config["vision_model"] if model_config else "openai/clip-vit-large-patch14-336",
            "max_num_patches": model_config["max_num_patches"] if model_config else 576,
            "empty_union_score": 1.0 if args.metric_resolution == "original" else 0.0,
            "num_samples": len(per_sample_iou),
            "mean_iou": round(sum(iou_values) / len(iou_values), 6),
            "min_iou": round(min(iou_values), 6),
            "max_iou": round(max(iou_values), 6),
        }
        results = {
            "summary": summary,
            "per_sample": per_sample_iou,
        }
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved → {results_file}")

    # --- Report ---
    iou_values = list(per_sample_iou.values())
    print(f"\nEvaluated {len(per_sample_iou)} samples on {args.dataset} set")
    print(f"Mean IoU: {sum(iou_values) / len(iou_values):.4f}")
    if args.output_dir:
        print(f"Predicted masks saved to {args.output_dir}")


if __name__ == "__main__":
    main()
