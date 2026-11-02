"""Train the question-only grounding model. Defaults build B0 (ocrg decisions D016-D017).

B0 on two GPUs (global batch 4 = 2 per GPU):
    torchrun --nproc_per_node 2 train.py --data-root DATA --output-dir OUT --num-epochs 20 --batch-size 4 \
        --dice-weight 0.5 --freeze-text-encoder --no-freeze-vision-encoder --validate-every 2 --save-every 10

Outputs in OUTPUT_DIR: run_config_from_epoch<E>.json, history.jsonl (one line per epoch),
val_curve.jsonl (native-resolution validation mIoU), checkpoint_epoch<E>.pt (always at the last epoch).
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from dataset import VizWizGroundingDataset, collate
from evaluation import ShardSampler, evaluate, gather_scores, mean_iou
from losses import segmentation_loss
from models import GroundingModel, ModelConfig
from run_record import RECIPE_KEYS, append_jsonl, check_expectations, check_resume_recipe, git_state, parse_expectations
from training_options import TrainingArgumentParser, TrainingControls, public_arguments, restore_control_arguments
from utils import align_state_dict, cleanup_distributed, setup_distributed


def parse_args():
    parser = TrainingArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/vizwiz")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4, help="Global batch size, split across GPUs")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4, help="Data-loading workers per GPU process")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--validate-every", type=int, default=0, help="Validation mIoU every N epochs (0 = never)")
    parser.add_argument("--save-every", type=int, default=10, help="Checkpoint every N epochs; the last epoch is always saved")
    parser.add_argument("--dice-weight", type=float, default=0.0,
                        help="Weight of the per-image soft Dice term added to BCE (0 = BCE only)")
    parser.add_argument("--grad-clip-norm", type=float, default=None, help="Clip the gradient norm (off by default)")
    parser.add_argument("--decoder", choices=("fusion", "unet"), default=ModelConfig.decoder)
    parser.add_argument("--attention-residual", action=argparse.BooleanOptionalAction, default=ModelConfig.attention_residual,
                        help="F = V + alpha * MHA(V, T) (default) instead of F = MHA(V, T)")
    parser.add_argument("--attention-residual-init", type=float, default=ModelConfig.attention_residual_init)
    parser.add_argument("--expect", action="append", default=[], metavar="KEY=VALUE",
                        help="Stop before training unless the resolved recipe has this value (repeatable)")
    args = parser.parse_args()
    if min(args.num_epochs, args.batch_size) <= 0 or min(args.num_workers, args.validate_every, args.save_every) < 0:
        parser.error("Epochs and batch size must be positive; workers and intervals nonnegative")
    if args.dice_weight < 0 or (args.grad_clip_norm is not None and args.grad_clip_norm <= 0):
        parser.error("--dice-weight must be nonnegative and --grad-clip-norm positive")
    if args.resume_checkpoint is not None and not os.path.isfile(args.resume_checkpoint):
        parser.error(f"Resume checkpoint is not a file: {args.resume_checkpoint}")
    try:
        args.expected = parse_expectations(args.expect)
    except ValueError as error:
        parser.error(str(error))
    return args


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_atomically(state: dict, path: Path):
    """Write to a temporary file, then rename, so an interrupted save never leaves a broken checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def train_one_epoch(model, loader, controls, scaler, device, args, show_progress):
    """One pass over loader. Returns (sum of batch losses, number of batches) as one device tensor,
    and the seconds spent waiting for data. No per-step GPU-to-CPU synchronisation."""
    optimizer = controls.optimizer
    totals = torch.zeros(2, device=device, dtype=torch.float64)  # (loss sum, batches)
    data_wait, waited_from = 0.0, time.perf_counter()
    for batch in tqdm(loader, disable=not show_progress, dynamic_ncols=True, leave=False):
        data_wait += time.perf_counter() - waited_from
        images = batch["image"].to(device, non_blocking=True)  # (B, 3, S, S)
        masks = batch["mask"].to(device, non_blocking=True)  # (B, 1, S, S)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device.type, enabled=scaler.is_enabled()):
            logits = model(images, batch["text"])  # (B, 1, h, w)
            logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = segmentation_loss(logits, masks, args.dice_weight)  # ()
        scaler.scale(loss).backward()
        if args.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(controls.parameters, args.grad_clip_norm)
        controls.step(scaler)
        totals[0] += loss.detach()
        totals[1] += 1
        waited_from = time.perf_counter()
    return totals, data_wait


def main():
    args = parse_args()
    checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False) if args.resume_checkpoint else None
    restore_control_arguments(args, checkpoint)
    rank, world_size, local_rank, device = setup_distributed()
    try:
        if args.batch_size % world_size:
            raise ValueError(f"--batch-size {args.batch_size} must be a multiple of the {world_size} processes")

        # Resolved recipe: checked against --expect and the resume checkpoint before anything is built.
        model_config = ModelConfig(decoder=args.decoder, attention_residual=args.attention_residual,
                                   attention_residual_init=args.attention_residual_init)
        resolved = {**public_arguments(args), "world_size": world_size}
        resolved.pop("expected", None)
        recipe = {key: resolved[key] for key in RECIPE_KEYS}
        check_expectations(resolved, args.expected)
        if checkpoint is not None:
            check_resume_recipe(checkpoint, recipe)
            if checkpoint.get("model_config") != model_config.to_dict():
                raise ValueError("Resume model_config differs from the checkpoint")

        seed_everything(args.seed)
        torch.backends.cudnn.benchmark = True  # fixed input sizes
        out = Path(args.output_dir)
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)

        # --- Data ---
        pin = device.type == "cuda"
        train_set = VizWizGroundingDataset(args.data_root, "train")
        generator = torch.Generator()
        sampler = (DistributedSampler(train_set, world_size, rank, shuffle=True, seed=args.seed) if world_size > 1
                   else RandomSampler(train_set, generator=generator))
        loader_options = dict(num_workers=args.num_workers, pin_memory=pin, collate_fn=collate,
                              persistent_workers=args.num_workers > 0,
                              prefetch_factor=4 if args.num_workers > 0 else None)
        train_loader = DataLoader(train_set, batch_size=args.batch_size // world_size, sampler=sampler, **loader_options)
        val_set = val_loader = None
        if args.validate_every:
            val_set = VizWizGroundingDataset(args.data_root, "val", native_mask=True)
            val_loader = DataLoader(val_set, batch_size=args.batch_size // world_size,
                                    sampler=ShardSampler(len(val_set), rank, world_size), **loader_options)

        # --- Model and optimisation ---
        model = GroundingModel(model_config).to(device)
        controls = TrainingControls(model, args, checkpoint, include_frozen=True)
        parameters = {"total": sum(p.numel() for p in model.parameters()),
                      "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        scaler = GradScaler("cuda", enabled=device.type == "cuda")
        start_epoch = 0
        if checkpoint is not None:
            model.load_state_dict(align_state_dict(checkpoint["model_state_dict"], model))
            controls.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint["epoch"]
            del checkpoint
        if world_size > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # no-op for the GroupNorm decoder
            # Every trainable parameter is used in each step, so DDP needs no unused-parameter search
            # (it raises if that ever stops being true).
            model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        core = model.module if world_size > 1 else model
        controls.configure_schedule(args.num_epochs, len(train_loader), start_epoch)

        run_config = {"recipe": recipe, "arguments": resolved, "model_config": model_config.to_dict(),
                      "training_controls": dict(controls.config), "parameters": parameters,
                      "git": git_state(Path(__file__).resolve().parent), "start_epoch": start_epoch,
                      "versions": {"torch": torch.__version__}}
        if rank == 0:
            import transformers
            run_config["versions"]["transformers"] = transformers.__version__
            print("Run config:", json.dumps(run_config), flush=True)
            (out / f"run_config_from_epoch{start_epoch}.json").write_text(json.dumps(run_config, indent=1))

        for epoch in range(start_epoch + 1, args.num_epochs + 1):
            if world_size > 1:
                sampler.set_epoch(epoch)
            else:
                generator.manual_seed(args.seed + epoch)
            model.train()
            controls.set_encoder_modes()
            controls.log_learning_rates(epoch)
            started = time.perf_counter()
            totals, data_wait = train_one_epoch(model, train_loader, controls, scaler, device, args, rank == 0)
            if world_size > 1:
                dist.all_reduce(totals)
            train_loss = (totals[0] / totals[1]).item()  # one synchronisation per epoch
            if rank == 0:
                record = {"epoch": epoch, "train_loss": train_loss, "lr": [g["lr"] for g in controls.optimizer.param_groups],
                          "seconds": time.perf_counter() - started, "data_wait_seconds": data_wait,
                          "training_config": recipe}
                append_jsonl(out / "history.jsonl", record)
                print(f"[epoch {epoch}] train loss {train_loss:.4f}, {record['seconds']:.0f} s "
                      f"({data_wait:.0f} s waiting for data)", flush=True)

            if args.validate_every and epoch % args.validate_every == 0:
                scores = gather_scores(evaluate(core, val_loader, device), world_size)
                if rank == 0:
                    val_miou = mean_iou(scores, len(val_set))
                    append_jsonl(out / "val_curve.jsonl", {"epoch": epoch, "val_miou": val_miou,
                                                           "num_images": len(scores), "training_config": recipe})
                    print(f"[epoch {epoch}] validation mIoU {100 * val_miou:.2f}", flush=True)

            if rank == 0 and ((args.save_every and epoch % args.save_every == 0) or epoch == args.num_epochs):
                save_atomically({"epoch": epoch, "model_state_dict": core.state_dict(),
                                 "model_config": model_config.to_dict(), "run_config": run_config,
                                 "training_controls": controls.state_dict(),
                                 "optimizer_state_dict": controls.optimizer.state_dict(),
                                 "scaler_state_dict": scaler.state_dict()}, out / f"checkpoint_epoch{epoch}.pt")
                print(f"[epoch {epoch}] checkpoint saved", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
