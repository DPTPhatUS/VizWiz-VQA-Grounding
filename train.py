from training_options import (
    TrainingArgumentParser,
    TrainingControls,
    restore_control_arguments,
    public_arguments,
)
import json
import os
import random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import VizWizGroundingDataset, collate_eval_samples
from evaluation import score_sample
from losses import segmentation_loss
from run_record import (
    RECIPE_KEYS,
    append_jsonl,
    check_expectations,
    check_resume_recipe,
    git_state,
    parse_expectations,
)
from utils import align_state_dict, to_device
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
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


def set_deterministic(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = TrainingArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/vizwiz")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--validate-every", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-mode", choices=["qa", "question"], default="qa",
                        help="QA baseline (default) or a separately trained question-only control")
    parser.add_argument("--dice-weight", type=float, default=0.0,
                        help="Weight of the per-image soft Dice term added to BCE; 0 keeps the upstream BCE-only loss")
    parser.add_argument("--expect", action="append", default=[], metavar="KEY=VALUE",
                        help="Stop before training unless the resolved recipe has this value (repeatable)")
    args = parser.parse_args()
    if args.dice_weight < 0:
        parser.error("--dice-weight must be nonnegative")
    try:
        expected = parse_expectations(args.expect)
    except ValueError as error:
        parser.error(str(error))

    checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False) if args.resume_checkpoint else None
    restore_control_arguments(args, checkpoint)

    if args.resume_checkpoint is not None and not os.path.isfile(args.resume_checkpoint):
        parser.error(f"Resume checkpoint is not a file: {args.resume_checkpoint}")

    # --- Distributed init ---
    rank, world_size, local_rank, is_dist = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if args.batch_size < world_size or args.batch_size % world_size != 0:
        parser.error(
            f"--batch-size ({args.batch_size}) must be a positive multiple of world_size ({world_size})"
        )

    # Resolved recipe: checked against --expect and the resume checkpoint before anything is trained.
    recipe = {key: getattr(args, key) for key in RECIPE_KEYS if key != "world_size"}
    recipe["world_size"] = world_size
    resolved = {**public_arguments(args), "world_size": world_size}
    check_expectations(resolved, expected)
    if checkpoint is not None:
        check_resume_recipe(checkpoint, recipe)

    # Deterministic mode  (before any model creation)
    set_deterministic(args.seed)

    # Only rank 0 writes to disk
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    image_size = (336, 336)

    # Per-GPU batch size  (keeps effective batch size == args.batch_size)
    per_gpu_bs = args.batch_size // world_size

    # --- Datasets ---
    train_set = VizWizGroundingDataset(
        json_path=os.path.join(args.data_root, "train_grounding.json"),
        image_root=os.path.join(args.data_root, "train"),
        mask_root=os.path.join(args.data_root, "binary_masks_png", "train"),
        image_size=image_size,
        text_mode=args.text_mode,
    )
    val_set = VizWizGroundingDataset(
        json_path=os.path.join(args.data_root, "val_grounding.json"),
        image_root=os.path.join(args.data_root, "val"),
        mask_root=os.path.join(args.data_root, "binary_masks_png", "val"),
        image_size=image_size,
        is_test=True,  # validation must not use random training augmentation
        text_mode=args.text_mode,
        return_original_mask=True,  # validation mIoU uses the official native-resolution protocol
    )

    # --- Samplers & Loaders ---
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed + rank)

    train_sampler = (
        DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        if is_dist else None
    )
    train_loader = DataLoader(
        train_set,
        batch_size=per_gpu_bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=loader_generator,
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
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
        worker_init_fn=seed_worker,
        collate_fn=collate_eval_samples,
    )

    # --- Model ---
    model = GroundingModel().to(device)
    controls = TrainingControls(model, args, checkpoint, include_frozen=True)

    # SyncBatchNorm: makes per-GPU BatchNorm stats identical to single-GPU
    if is_dist:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wrap with DDP  (after SyncBN conversion)
    if is_dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True)

    # --- Optimizer / Loss / Scaler ---
    use_amp = (device.type == "cuda")
    optimizer = controls.optimizer
    scaler = GradScaler("cuda", enabled=use_amp)

    # --- Resume checkpoint ---
    start_epoch = 0

    if args.resume_checkpoint is not None:
        # Always load into the *unwrapped* model so state-dict keys match
        # regardless of DDP / SyncBN wrapping.
        underlying_model = model.module if is_dist else model

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = align_state_dict(checkpoint["model_state_dict"], underlying_model)
            underlying_model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for group in optimizer.param_groups:
                group["lr"] = args.lr
            if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            if rank == 0:
                print(f"✅ Resumed full training state from {args.resume_checkpoint} (epoch {start_epoch})")
        else:
            state_dict = align_state_dict(checkpoint, underlying_model)
            underlying_model.load_state_dict(state_dict)
            if rank == 0:
                print(f"✅ Resumed model from {args.resume_checkpoint}")
            try:
                start_epoch = int(args.resume_checkpoint.split("epoch")[1].split(".")[0])
            except Exception:
                start_epoch = 0

        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    controls.configure_schedule(args.num_epochs, len(train_loader), start_epoch)

    run_config = {
        "recipe": recipe,
        "arguments": resolved,
        "training_controls": dict(controls.config),
        "git": git_state(os.path.dirname(os.path.abspath(__file__))),
        "versions": {"torch": str(torch.__version__)},
        "start_epoch": start_epoch,
    }
    if rank == 0:
        try:
            import transformers
            run_config["versions"]["transformers"] = str(transformers.__version__)
        except ImportError:
            run_config["versions"]["transformers"] = None
        print("Run config:", json.dumps(run_config))
        with open(os.path.join(args.output_dir, f"run_config_from_epoch{start_epoch}.json"), "w") as f:
            json.dump(run_config, f, indent=1)

    # --- Log file (rank 0 only) ---
    log_file = None
    log_path = None
    if rank == 0:
        log_name = f"train_loss_{start_epoch+1}_{args.num_epochs}.txt"
        log_path = os.path.join(args.output_dir, log_name)
        log_file = open(log_path, "w")

    # Sync all ranks before starting training
    if is_dist:
        dist.barrier()

    # ===================================================================
    #  Training loop
    # ===================================================================
    try:
        for epoch in range(start_epoch, args.num_epochs):
            # Shuffle partitions differently each epoch  (mandatory for DDP)
            loader_generator.manual_seed(args.seed + epoch * world_size + rank)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # ---- Train ----
            model.train()
            controls.set_encoder_modes()
            controls.log_learning_rates(epoch + 1)
            train_loss_sum = torch.zeros(1, device=device, dtype=torch.float64)  # (1,)

            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} (Train)", disable=(rank != 0), dynamic_ncols=True)
            for batch in loop:
                batch = to_device(batch, device)
                images, masks, texts = batch["image"], batch["mask"], batch["text"]  # (B, 3, H, W), (B, 1, H, W), list[str]

                optimizer.zero_grad(set_to_none=True)
                with autocast(device.type, enabled=use_amp):
                    pred = model(images, texts)  # (B, 1, 16*H_p, 16*W_p)
                    pred = nn.functional.interpolate(
                        pred, size=masks.shape[-2:], mode="bilinear", align_corners=False
                    )  # (B, 1, H, W)
                    loss = segmentation_loss(pred, masks, args.dice_weight)  # ()

                scaler.scale(loss).backward()
                controls.step(scaler)

                train_loss_sum += loss.detach().to(torch.float64)

            # Reduce train loss across all ranks once per epoch (avoids per-batch CPU-GPU syncs)
            if is_dist:
                dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)
            avg_train_loss = (train_loss_sum / (len(train_loader) * world_size)).item()

            if rank == 0:
                print(f"[Epoch {epoch+1}] Average Training Loss: {avg_train_loss:.4f}")
                log_file.write(f"train epoch={epoch+1} loss={avg_train_loss:.6f}\n")
                log_file.flush()
                append_jsonl(os.path.join(args.output_dir, "history.jsonl"), {
                    "epoch": epoch + 1, "train_loss": avg_train_loss,
                    "lr": [group["lr"] for group in optimizer.param_groups],
                    "training_config": recipe})

            # ---- Validation ----
            # FP32 forward, scored with the same function as eval.py, so the logged mIoU equals a
            # later eval.py run on the saved checkpoint. DistributedSampler pads with repeated
            # images; scores are keyed by filename, so the repeats are counted once.
            if args.validate_every > 0 and (epoch + 1) % args.validate_every == 0:
                model.eval()
                val_loss_sum = torch.zeros(1, device=device, dtype=torch.float64)  # (1,)
                scores = {}

                loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} (Val)", disable=(rank != 0), dynamic_ncols=True)
                with torch.no_grad():
                    for batch in loop:
                        images = batch["image"].to(device, non_blocking=True)  # (B, 3, H, W)
                        masks = batch["mask"].to(device, non_blocking=True)  # (B, 1, H, W)
                        pred = model(images, batch["text"])  # (B, 1, 16*H_p, 16*W_p)
                        resized = nn.functional.interpolate(
                            pred, size=masks.shape[-2:], mode="bilinear", align_corners=False
                        )  # (B, 1, H, W)
                        val_loss_sum += segmentation_loss(resized, masks, args.dice_weight).detach().to(torch.float64)
                        for i, fname in enumerate(batch["filename"]):
                            truth = batch["original_mask"][i].unsqueeze(0).to(device)  # (1, 1, H_i, W_i)
                            scores[fname] = score_sample(pred[i:i+1], truth)[1]

                if is_dist:
                    dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
                    gathered = [None] * world_size
                    dist.all_gather_object(gathered, scores)
                    scores = {k: v for part in gathered for k, v in part.items()}
                avg_val_loss = (val_loss_sum / (len(val_loader) * world_size)).item()

                if rank == 0:
                    if len(scores) != len(val_set):
                        raise RuntimeError(f"Validation scored {len(scores)} of {len(val_set)} images")
                    val_miou = sum(scores.values()) / len(scores)
                    print(f"[Epoch {epoch+1}] Validation loss {avg_val_loss:.4f}, mIoU {100 * val_miou:.2f}")
                    log_file.write(f"val epoch={epoch+1} loss={avg_val_loss:.6f} miou={val_miou:.6f}\n")
                    log_file.flush()
                    append_jsonl(os.path.join(args.output_dir, "val_curve.jsonl"), {
                        "epoch": epoch + 1, "val_loss": avg_val_loss, "val_miou": val_miou,
                        "num_images": len(scores), "training_config": recipe})

            # ---- Checkpoint (rank 0 only) ----
            if rank == 0 and args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt")
                underlying_model = model.module if is_dist else model
                ckpt = {
                    "epoch": epoch + 1,
                    "model_state_dict": underlying_model.state_dict(),
                    "training_controls": controls.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "run_config": run_config,
                    "scaler_state_dict": scaler.state_dict(),
                    "loss": avg_train_loss,
                }
                torch.save(ckpt, ckpt_path, _use_new_zipfile_serialization=False)
                print(f"✅ Checkpoint saved at {ckpt_path}")

        # ---- Final model (rank 0 only) ----
        if rank == 0:
            underlying_model = model.module if is_dist else model
            final_path = os.path.join(args.output_dir, f"model_final_epoch{args.num_epochs}.pt")
            torch.save(underlying_model.state_dict(), final_path, _use_new_zipfile_serialization=False)
            print(f"🔚 Final model saved → {final_path}")
    finally:
        if log_file is not None:
            log_file.close()
            if rank == 0:
                print(f"📝 Training log saved → {log_path}")
        cleanup_distributed()


if __name__ == "__main__":
    main()
