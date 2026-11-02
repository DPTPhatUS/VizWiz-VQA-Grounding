"""External baseline: CLIPSeg (Lüddecke and Ecker, CVPR 2022) trained on VizWiz-VQA-Grounding.

Protocol, fixed on 2026-11-02 before any run:
- Model: CIDAS/clipseg-rd64-refined (CLIP ViT-B/16). The CLIP encoders stay frozen, as in the
  original method; only the decoder (1.13M parameters, initialised from the released weights)
  is trained.
- Input: the full frame resized to 352 x 352 with the dataset's bicubic resize (aspect ratio not
  kept, as for QGDP-Net), normalised with the checkpoint's image statistics. Prompt: the
  question as asked; the answer is never used.
- Loss: logits upsampled to the 768 x 768 training mask (bilinear), BCE + 0.5 soft Dice
  (losses.segmentation_loss, the main term of QGDP-Net's objective).
- Optimiser: Adam, no weight decay, constant learning rate, gradient-norm clipping at 1.0,
  batch 4, 20 epochs, one GPU, FP32.
- Decoder weights saved at epochs 10 and 20; both evaluated on val and test with
  metrics.evaluate (native resolution, per-image IoU, empty union = 1). The reported
  checkpoint is the one with the higher validation mIoU.
- Learning rate: chosen once on seed 42 from {1e-4, 1e-5} by validation mIoU at each run's
  selected checkpoint; seeds 43 and 44 use the chosen value.

Usage:
  python clipseg_baseline.py train --data-root DATA --output-dir OUT --seed 42 --lr 1e-4
  python clipseg_baseline.py eval --data-root DATA --run-dir OUT   # epochs 10, 20; val, test
  python clipseg_baseline.py choose-lr RUN_DIR [RUN_DIR ...]       # prints the chosen run dir
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dataset import VizWizGroundingDataset, make_loader
from losses import segmentation_loss
from metrics import evaluate
from utils import to_device

MODEL_NAME = "CIDAS/clipseg-rd64-refined"
INPUT_SIZE = 352
MASK_SIZE = 768
PROTOCOL = "clipseg-frozen-clip-decoder-v1"


class CLIPSegGrounding(nn.Module):
    """CLIPSeg with the interface of GroundingModel: forward(batch) -> {"logits": (B, 1, H, W)}."""

    def __init__(self, name: str = MODEL_NAME):
        super().__init__()
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self.model = CLIPSegForImageSegmentation.from_pretrained(name)
        processor = CLIPSegProcessor.from_pretrained(name)
        self.tokenizer = processor.tokenizer
        image = processor.image_processor
        if (image.size["height"], image.size["width"]) != (INPUT_SIZE, INPUT_SIZE):
            raise ValueError("Unexpected CLIPSeg input size")
        self.register_buffer("mean", torch.tensor(image.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(image.image_std).view(1, 3, 1, 1), persistent=False)
        self.model.clip.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.clip.eval()  # frozen encoders (no dropout in CLIP; kept explicit)
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, batch: dict) -> dict:
        image = batch["image"]  # (B, 3, 352, 352) in [0, 1]
        if image.shape[-2:] != (INPUT_SIZE, INPUT_SIZE):
            raise ValueError(f"CLIPSeg expects {INPUT_SIZE} px images")
        questions = [q[3:] if q.startswith("Q: ") else q for q in batch["question_text"]]
        text = self.tokenizer(questions, padding=True, truncation=True, max_length=77, return_tensors="pt")
        out = self.model(
            pixel_values=(image - self.mean) / self.std,
            input_ids=text.input_ids.to(image.device),
            attention_mask=text.attention_mask.to(image.device),
        )
        return {"logits": out.logits.reshape(image.shape[0], 1, INPUT_SIZE, INPUT_SIZE)}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataset(root, split):
    return VizWizGroundingDataset(root, split, image_size=INPUT_SIZE, detail_size=MASK_SIZE)


def train(args):
    out = Path(args.output_dir)
    if (out / "history.jsonl").exists():
        raise ValueError(f"Refusing to reuse {out}")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    model = CLIPSegGrounding().to(device)
    params = model.trainable_parameters()
    n_trainable = sum(p.numel() for p in params)
    generator = torch.Generator().manual_seed(args.seed)
    loader = make_loader(dataset(args.data_root, "train"), args, args.batch_size, None, True, generator)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    config = {"protocol": PROTOCOL, "model_name": MODEL_NAME, "seed": args.seed, "lr": args.lr,
              "batch_size": args.batch_size, "num_epochs": args.num_epochs, "optimizer": "adam",
              "weight_decay": 0.0, "gradient_clip_norm": 1.0, "loss": "bce + 0.5 dice at 768 px",
              "input_size": INPUT_SIZE, "trainable_parameters": n_trainable, "clip_frozen": True,
              "torch_version": str(torch.__version__)}
    (out / "run_config.json").write_text(json.dumps(config, indent=1))
    print(json.dumps(config), flush=True)
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        start, total, count = time.time(), 0.0, 0
        for step, batch in enumerate(loader):
            if args.max_steps and step >= args.max_steps:
                break
            batch = to_device(batch, device)
            logits = F.interpolate(model(batch)["logits"], size=batch["mask"].shape[-2:],
                                   mode="bilinear", align_corners=False)
            loss = segmentation_loss(logits, batch["mask"], overlap_loss="dice", overlap_weight=0.5)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total += loss.item() * batch["mask"].shape[0]
            count += batch["mask"].shape[0]
        entry = {"epoch": epoch, "train_loss": total / count, "samples": count,
                 "seconds": round(time.time() - start, 1), "training_config": config}
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps({k: v for k, v in entry.items() if k != "training_config"}), flush=True)
        if epoch in args.checkpoint_epochs:
            torch.save({"epoch": epoch, "decoder_state_dict": model.model.decoder.state_dict(),
                        "run_config": config}, out / f"decoder_epoch{epoch}.pt")


def load(run_dir, epoch, device):
    model = CLIPSegGrounding().to(device)
    saved = torch.load(Path(run_dir) / f"decoder_epoch{epoch}.pt", map_location=device, weights_only=True)
    model.model.decoder.load_state_dict(saved["decoder_state_dict"])
    return model, saved["run_config"]


def evaluate_run(args):
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    for epoch in args.checkpoint_epochs:
        model, config = load(run_dir, epoch, device)
        for split in ("val", "test"):
            target = run_dir / f"eval_{split}_ep{epoch}"
            if (target / "metrics.json").exists():
                raise ValueError(f"Refusing to overwrite {target}")
            loader = make_loader(dataset(args.data_root, split), args, args.batch_size)
            scores = evaluate(model, loader, device, target)
            summary = {"num_samples": len(scores), "mean_iou": sum(scores.values()) / len(scores),
                       "dataset": split, "metric_resolution": "original", "text_mode": "question",
                       "checkpoint": str(run_dir / f"decoder_epoch{epoch}.pt"), **config}
            (target / "metrics.json").write_text(json.dumps({"summary": summary, "per_sample": scores}))
            print(split, epoch, f"{100 * summary['mean_iou']:.2f}", flush=True)


def selected_val(run_dir):
    paths = sorted(Path(run_dir).glob("eval_val_ep*/metrics.json"))
    if not paths:
        raise ValueError(f"No validation results in {run_dir}")
    return max(json.loads(p.read_text())["summary"]["mean_iou"] for p in paths)


def choose_lr(run_dirs):
    """Validation only: the run whose selected checkpoint has the highest val mIoU."""
    scored = {d: selected_val(d) for d in run_dirs}
    for d, v in scored.items():
        print(f"{d}: selected val mIoU {100 * v:.2f}", file=sys.stderr)
    best = max(scored, key=scored.get)
    print(json.loads((Path(best) / "run_config.json").read_text())["lr"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--data-root", required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--num-workers", type=int, default=2)
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--checkpoint-epochs", default="10,20",
                       type=lambda s: tuple(int(x) for x in s.split(",")),
                       help="Epochs whose decoder weights are saved (train) and evaluated (eval)")
    t = sub.choices["train"]
    t.add_argument("--output-dir", required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--lr", type=float, required=True)
    t.add_argument("--num-epochs", type=int, default=20)
    t.add_argument("--max-steps", type=int, default=0, help="Smoke tests only: stop each epoch early")
    sub.choices["eval"].add_argument("--run-dir", required=True)
    c = sub.add_parser("choose-lr")
    c.add_argument("run_dirs", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "train":
        if not math.isfinite(args.lr) or args.lr <= 0:
            raise ValueError("lr must be positive")
        train(args)
    elif args.command == "eval":
        evaluate_run(args)
    else:
        choose_lr(args.run_dirs)


if __name__ == "__main__":
    main()
