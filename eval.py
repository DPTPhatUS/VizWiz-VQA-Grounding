"""Evaluate a trained model on val or test set.

Usage:
    uv run eval.py --checkpoint outputs/checkpoint_epoch20.pt --dataset val
    uv run eval.py --checkpoint outputs/checkpoint_epoch20.pt --dataset test
    uv run eval.py --dataset val                  # uses base model (random weights, no checkpoint)
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
from datetime import datetime
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from dataset import VizWizGroundingDataset
from models import GroundingModel
from metrics import compute_iou_per_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--data-root", type=str, default="data/vizwiz")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    image_size = (args.image_size, args.image_size)

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
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- Model ---
    model = GroundingModel().to(device)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')})")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded raw state dict")
    else:
        print("No checkpoint provided — using base model (random weights)")
    model.eval()

    # --- Inference ---
    per_sample_iou = {}

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    loop = tqdm(loader, desc=f"Eval ({args.dataset})")
    with torch.no_grad():
        for batch in loop:
            images = batch["image"].to(device)
            texts = batch["text"]
            masks = batch["mask"]
            filenames = batch["filename"]

            pred = model(images, texts)
            pred = F.interpolate(pred, size=masks.shape[-2:], mode="bilinear")

            iou_per_sample = compute_iou_per_sample(pred, masks.to(device))
            for fname, iou_val in zip(filenames, iou_per_sample.tolist()):
                per_sample_iou[fname] = round(iou_val, 6)

            if args.output_dir:
                pred_bin = (torch.sigmoid(pred) > 0.5).float()
                for i, fname in enumerate(filenames):
                    mask_path = os.path.join(args.output_dir, fname.replace(".jpg", ".png"))
                    from torchvision.transforms.functional import to_pil_image
                    to_pil_image(pred_bin[i, 0].cpu()).save(mask_path)

    # --- Results JSON ---
    if args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(args.output_dir, f"results_{args.dataset}_{timestamp}.json")
        iou_values = list(per_sample_iou.values())
        summary = {
            "checkpoint": args.checkpoint or "none (base model)",
            "dataset": args.dataset,
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
