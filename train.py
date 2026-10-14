import argparse
from contextlib import nullcontext
import json
import os
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

from dataset import VizWizGroundingDataset, collate_eval_samples
from models import GroundingModel, build_grounding_model, make_naflex_processor
from models.image_encoder import VISION_MODEL
from utils import align_state_dict, to_device, grounding_loss


def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0, False
    rank, world_size, local_rank = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    else:
        dist.init_process_group("gloo")  # enables CPU distributed contract tests
    return rank, world_size, local_rank, True


def set_deterministic(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    random.seed(torch.initial_seed() % (2**32))


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device,
                    accumulation_steps=1, use_amp=False, rank=0, is_dist=False, description="Train"):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    local_samples = len(loader.sampler)
    for step, batch in enumerate(tqdm(loader, desc=description, disable=rank != 0, dynamic_ncols=True)):
        batch = to_device(batch, device)
        n = len(batch["text"])
        last_step = step + 1 == len(loader)
        update = (step + 1) % accumulation_steps == 0 or last_step
        # Weight by actual images, including an incomplete final accumulation window.
        window_start = (step // accumulation_steps) * accumulation_steps
        window_samples = min(accumulation_steps * loader.batch_size,
                             local_samples - window_start * loader.batch_size)
        sync = model.no_sync() if is_dist and not update else nullcontext()
        with sync:
            with autocast(device.type, enabled=use_amp):
                pred = model(batch["image"], batch["text"])
                loss = grounding_loss(pred, batch["mask"], loss_fn)
            scaler.scale(loss * n / window_samples).backward()
        if update:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        totals[0] += loss.detach().double() * n
        totals[1] += n
    if is_dist:
        dist.all_reduce(totals)
    return (totals[0] / totals[1]).item()


@torch.no_grad()
def validate(model, loader, loss_fn, device, use_amp=False, rank=0, is_dist=False):
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    for batch in tqdm(loader, desc="Validation", disable=rank != 0, dynamic_ncols=True):
        batch = to_device(batch, device)
        n = len(batch["text"])
        with autocast(device.type, enabled=use_amp):
            loss = grounding_loss(model(batch["image"], batch["text"]), batch["mask"], loss_fn)
        totals[0] += loss.double() * n
        totals[1] += n
    if is_dist:
        dist.all_reduce(totals)
    return (totals[0] / totals[1]).item()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/vizwiz")
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Global micro-batch across all GPUs (fresh default 4)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-num-patches", type=int, default=None)
    parser.add_argument("--text-encoder", choices=["clip", "siglip2"], default=None)
    parser.add_argument("--text-mode", choices=["qa", "question"], default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--resume-checkpoint", default=None)
    args = parser.parse_args(argv)

    checkpoint = (torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
                  if args.resume_checkpoint else None)
    config = checkpoint.get("model_config") if checkpoint else None
    previous = checkpoint.get("training_args", {}) if checkpoint else {}
    defaults = {"batch_size": 4, "gradient_accumulation_steps": 1, "amp": True,
                "gradient_checkpointing": False, "text_encoder": "clip", "text_mode": "qa", "max_num_patches": 512}
    if checkpoint:
        if not config or config.get("vision_model") != VISION_MODEL:
            parser.error(f"This branch requires a {VISION_MODEL} checkpoint")
        saved = {"text_encoder": config.get("text_backbone", "clip"), "max_num_patches": config["max_num_patches"],
                 "text_mode": checkpoint.get("text_mode", "qa")}
        for key, value in saved.items():
            if getattr(args, key) is not None and getattr(args, key) != value:
                parser.error(f"Resume {key} differs from checkpoint; start a separate run")
        defaults.update(previous)
        defaults.update(saved)
    for key in ("batch_size", "gradient_accumulation_steps", "amp", "gradient_checkpointing",
                "text_encoder", "text_mode", "max_num_patches"):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])
    if min(args.batch_size, args.gradient_accumulation_steps, args.max_num_patches, args.num_epochs) <= 0:
        parser.error("Batch, accumulation, patch budget and epochs must be positive")
    if args.num_workers < 0 or args.save_every < 0 or args.validate_every < 0:
        parser.error("Worker count and save/validation intervals must be nonnegative")

    rank, world_size, local_rank, is_dist = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if args.batch_size % world_size:
        parser.error("Global micro-batch must be divisible by GPU/process count")
    set_deterministic(args.seed)
    model = (build_grounding_model(config, pretrained=False) if checkpoint else
             GroundingModel(max_num_patches=args.max_num_patches, text_backbone=args.text_encoder))
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    processor = make_naflex_processor(model.model_config)
    model_config = model.model_config
    datasets = {split: VizWizGroundingDataset(
        os.path.join(args.data_root, f"{split}_grounding.json"), os.path.join(args.data_root, split),
        os.path.join(args.data_root, "binary_masks_png", split), processor,
        is_test=split == "val", text_mode=args.text_mode) for split in ("train", "val")}
    samplers = {split: DistributedSampler(data, num_replicas=world_size, rank=rank,
                                         shuffle=split == "train", seed=args.seed) if is_dist else None
                for split, data in datasets.items()}
    generator = torch.Generator().manual_seed(args.seed + rank)
    loaders = {split: DataLoader(data, batch_size=args.batch_size // world_size,
        shuffle=split == "train" and samplers[split] is None, sampler=samplers[split],
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        prefetch_factor=2 if args.num_workers else None, worker_init_fn=seed_worker,
        generator=generator if split == "train" else None, collate_fn=collate_eval_samples)
        for split, data in datasets.items()}

    model = model.to(device)
    # Local BN: ranks may decode different numbers of aspect-ratio groups.
    if is_dist:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                    find_unused_parameters=False)
    underlying = model.module if is_dist else model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    start_epoch = 0
    if checkpoint:
        underlying.load_state_dict(align_state_dict(checkpoint["model_state_dict"], underlying))
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        if start_epoch >= args.num_epochs:
            parser.error("--num-epochs must exceed the checkpoint epoch when resuming")
        del checkpoint
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Vision: {VISION_MODEL}; text: {args.text_encoder}; patches: {args.max_num_patches}; "
              f"per GPU: {args.batch_size // world_size}; effective batch: {args.batch_size * args.gradient_accumulation_steps}")
        with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
            json.dump({"model_config": model_config, "training_args": vars(args), "world_size": world_size}, f, indent=2)
    log = open(os.path.join(args.output_dir, f"train_loss_{start_epoch + 1}_{args.num_epochs}.txt"), "w") if rank == 0 else None
    try:
        for epoch in range(start_epoch, args.num_epochs):
            generator.manual_seed(args.seed + epoch * world_size + rank)
            if samplers["train"] is not None:
                samplers["train"].set_epoch(epoch)
            train_loss = train_one_epoch(model, loaders["train"], optimizer, loss_fn, scaler, device,
                args.gradient_accumulation_steps, use_amp, rank, is_dist, f"Epoch {epoch + 1}/{args.num_epochs}")
            if rank == 0:
                print(f"[Epoch {epoch + 1}] train_loss={train_loss:.6f}")
                log.write(f"train epoch={epoch + 1} loss={train_loss:.6f}\n")
            if args.validate_every and (epoch + 1) % args.validate_every == 0:
                val_loss = validate(model, loaders["val"], loss_fn, device, use_amp, rank, is_dist)
                if rank == 0:
                    print(f"[Epoch {epoch + 1}] val_loss={val_loss:.6f}")
                    log.write(f"val epoch={epoch + 1} loss={val_loss:.6f}\n")
            state = {"epoch": epoch + 1, "model_state_dict": underlying.state_dict(),
                     "model_config": model_config, "text_mode": args.text_mode, "training_args": vars(args)}
            if rank == 0:
                log.flush()
                if args.save_every and (epoch + 1) % args.save_every == 0:
                    torch.save(dict(state, optimizer_state_dict=optimizer.state_dict(),
                                    scaler_state_dict=scaler.state_dict(), loss=train_loss),
                               os.path.join(args.output_dir, f"checkpoint_epoch{epoch + 1}.pt"))
        if rank == 0:
            torch.save(state, os.path.join(args.output_dir, f"model_final_epoch{args.num_epochs}.pt"))
    finally:
        if log is not None:
            log.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
