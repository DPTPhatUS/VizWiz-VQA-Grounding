import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from dataset import VizWizGroundingDataset
from utils import to_device
from models import GroundingModel


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def setup_distributed():
    """Initialize NCCL process group from torchrun env vars.
    Returns (rank, world_size, local_rank, is_distributed).
    Gracefully falls back to single-GPU when env vars are absent.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


def set_deterministic(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- CLI args (parsed before DDP so every rank sees identical args) ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/vizwiz")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Distributed init ---
    rank, world_size, local_rank, is_dist = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Deterministic mode  (before any model creation)
    set_deterministic(args.seed)

    # Only rank 0 writes to disk
    if rank == 0:
        os.makedirs("outputs", exist_ok=True)

    # --- Config ---
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    config["dataset"]["train_json"] = os.path.join(args.data_root, "train_grounding.json")
    config["dataset"]["val_json"]   = os.path.join(args.data_root, "val_grounding.json")
    config["dataset"]["train_image_root"] = os.path.join(args.data_root, "train")
    config["dataset"]["train_mask_root"]  = os.path.join(args.data_root, "binary_masks_png", "train")
    config["dataset"]["val_image_root"]   = os.path.join(args.data_root, "val")
    config["dataset"]["val_mask_root"]    = os.path.join(args.data_root, "binary_masks_png", "val")

    if args.num_epochs is not None:
        config["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["lr"] = args.lr
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.resume_checkpoint is not None:
        config["resume_checkpoint"] = args.resume_checkpoint

    # Per-GPU batch size  (keeps effective batch size == config["batch_size"])
    per_gpu_bs = config["batch_size"] // world_size

    # --- Datasets ---
    train_set = VizWizGroundingDataset(
        json_path=config["dataset"]["train_json"],
        image_root=config["dataset"]["train_image_root"],
        mask_root=config["dataset"]["train_mask_root"],
        image_size=tuple(config["image_size"]),
    )
    val_set = VizWizGroundingDataset(
        json_path=config["dataset"]["val_json"],
        image_root=config["dataset"]["val_image_root"],
        mask_root=config["dataset"]["val_mask_root"],
        image_size=tuple(config["image_size"]),
    )

    # --- Samplers & Loaders ---
    train_sampler = (
        DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        if is_dist else None
    )
    train_loader = DataLoader(
        train_set,
        batch_size=per_gpu_bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config["num_workers"],
        pin_memory=True,
        prefetch_factor=2,
    )

    # Validation: run on EVERY rank with DistributedSampler so the union
    # covers the full val set.  Loss is reduced across ranks below.
    val_sampler = (
        DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)
        if is_dist else None
    )
    val_loader = DataLoader(
        val_set,
        batch_size=per_gpu_bs,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config["num_workers"],
        pin_memory=True,
        prefetch_factor=2,
    )

    # --- Model ---
    model = GroundingModel().to(device)

    # SyncBatchNorm: makes per-GPU BatchNorm stats identical to single-GPU
    if is_dist:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wrap with DDP  (after SyncBN conversion)
    if is_dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True)

    # --- Optimizer / Loss / Scaler ---
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # --- Resume checkpoint ---
    resume_path = config.get("resume_checkpoint", None)
    start_epoch = 0

    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location="cpu")
        # Always load into the *unwrapped* model so state-dict keys match
        # regardless of DDP / SyncBN wrapping.
        underlying_model = model.module if is_dist else model

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            underlying_model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            if rank == 0:
                print(f"✅ Resumed full training state from {resume_path} (epoch {start_epoch})")
        else:
            underlying_model.load_state_dict(checkpoint)
            if rank == 0:
                print(f"✅ Resumed model from {resume_path}")
            try:
                start_epoch = int(resume_path.split("epoch")[1].split(".")[0])
            except Exception:
                start_epoch = 0

        del checkpoint
        torch.cuda.empty_cache()

    # Sync all ranks before starting training
    if is_dist:
        dist.barrier()

    # ===================================================================
    #  Training loop
    # ===================================================================
    for epoch in range(start_epoch, config["num_epochs"]):
        # Shuffle partitions differently each epoch  (mandatory for DDP)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ---- Train ----
        model.train()
        train_loss_sum = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} (Train)", disable=(rank != 0))
        for batch in loop:
            batch = to_device(batch, device)
            images, masks, texts = batch["image"], batch["mask"], batch["text"]

            with autocast("cuda"):
                pred = model(images, texts)
                pred = nn.functional.interpolate(pred, size=masks.shape[-2:], mode="bilinear")
                loss = loss_fn(pred, masks)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            if rank == 0:
                loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss_sum / len(train_loader)

        # Reduce train loss across all ranks  (same value everywhere after this)
        if is_dist:
            t = torch.tensor([avg_train_loss], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg_train_loss = t.item() / world_size

        if rank == 0:
            print(f"[Epoch {epoch+1}] Average Training Loss: {avg_train_loss:.4f}")

        # ---- Validation ----
        model.eval()
        val_loss_sum = 0.0

        loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} (Val)", disable=(rank != 0))
        with torch.no_grad():
            for batch in loop:
                batch = to_device(batch, device)
                images, masks, texts = batch["image"], batch["mask"], batch["text"]

                pred = model(images, texts)
                pred = nn.functional.interpolate(pred, size=masks.shape[-2:], mode="bilinear")
                loss = loss_fn(pred, masks)
                val_loss_sum += loss.item()
                if rank == 0:
                    loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss_sum / len(val_loader)

        if is_dist:
            t = torch.tensor([avg_val_loss], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg_val_loss = t.item() / world_size

        if rank == 0:
            print(f"[Epoch {epoch+1}] Average Validation Loss: {avg_val_loss:.4f}")

        # ---- Checkpoint (rank 0 only) ----
        if rank == 0 and (epoch + 1) % 10 == 0:
            ckpt_path = f"outputs/checkpoint_epoch{epoch+1}.pt"
            underlying_model = model.module if is_dist else model
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": underlying_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "loss": avg_train_loss,
                },
                ckpt_path,
                _use_new_zipfile_serialization=False,
            )
            print(f"✅ Checkpoint saved at {ckpt_path}")

    # ---- Final model (rank 0 only) ----
    if rank == 0:
        underlying_model = model.module if is_dist else model
        final_path = f"outputs/model_final_epoch{config['num_epochs']}.pt"
        torch.save(underlying_model.state_dict(), final_path)
        print(f"🔚 Final model saved → {final_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
