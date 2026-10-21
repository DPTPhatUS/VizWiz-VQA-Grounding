"""Train this branch's grounding experiment on CPU, CUDA, or torchrun DDP."""
import math
from training_options import (
    TrainingArgumentParser,
    TrainingControls,
    public_arguments,
    restore_control_arguments,
)
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DistributedSampler
from tqdm.auto import tqdm
from dataset import VizWizGroundingDataset, make_loader
from utils import to_device, read_checkpoint, load_model_weights, initialize_weights
from models.model import GroundingModel, EXPERIMENT


def validate_initialization(args, saved):
    config = saved.get("experiment_config") or {}
    run = saved["run_config"]
    if not config:
        if args.stage == "router" and not any(key.startswith("refiner.") for key in saved["model_state_dict"]):
            raise ValueError("Router initialization requires trained refiner weights")
        return
    if config.get("experiment") == EXPERIMENT:
        allowed = ("refiner",) if args.stage == "refiner" else ("refiner", "router")
        if run.get("stage") not in allowed:
            raise ValueError("Refiner initialization requires refiner weights; router initialization requires refiner or router weights")
    elif args.stage == "refiner" and config.get("experiment") == "controls" and config.get("architecture") == "compact":
        if run.get("text_mode", "question") != "question":
            raise ValueError("Refiner initialization requires question-only controls")
    else:
        raise ValueError("Refiner initialization requires compact controls or refiner weights; router initialization requires refiner or router weights")


IMAGE_SIZE = 336
DETAIL_SIZE = 672
CROP_SIZE = 336


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def training_parser():
    parser = TrainingArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/vizwiz")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4, help="Global batch size across all ranks")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save a resumable checkpoint every N epochs; 0 disables all model saves")
    parser.add_argument("--resume-checkpoint", help="Resume a full checkpoint; raw weights use --init-checkpoint")
    parser.add_argument("--stage", choices=("refiner", "router"), default="refiner")
    parser.add_argument("--init-checkpoint", help="Compact controls checkpoint for refiner; refiner checkpoint for router")
    return parser


def check_arguments(args):
    args.detail_size = DETAIL_SIZE
    args.crop_size = CROP_SIZE
    if not (args.init_checkpoint or args.resume_checkpoint):
        raise ValueError("A frozen reference requires --init-checkpoint or --resume-checkpoint")
    if min(args.num_epochs, args.batch_size) <= 0:
        raise ValueError("Epochs and batch size must be positive")
    if args.save_every < 0:
        raise ValueError("save-every cannot be negative")
    if args.num_workers < 0 or args.lr <= 0:
        raise ValueError("Workers cannot be negative; learning rate must be positive")
    if args.resume_checkpoint and args.init_checkpoint:
        raise ValueError("Use either resume or weight initialization, not both")
    for name in ('resume_checkpoint', 'init_checkpoint'):
        path = getattr(args, name)
        if path and not Path(path).is_file():
            raise ValueError(f"{name} does not exist: {path}")


def train_main(argv=None):
    args = training_parser().parse_args(argv)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    check_arguments(args)
    rank, world, local = (int(os.environ.get(k,default)) for k,default in (
        ("RANK",0),("WORLD_SIZE",1),("LOCAL_RANK",0)))
    if args.batch_size % world:
        raise ValueError("Global batch size must be divisible by world size")
    distributed = world > 1
    device = torch.device(args.device)
    if distributed:
        if device.type == "cuda":
            device = torch.device("cuda", local)
            torch.cuda.set_device(device)
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        _train(args, rank, world, distributed, device)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def _train(args, rank, world, distributed, device):
    checkpoint = read_checkpoint(args.resume_checkpoint) if args.resume_checkpoint else None
    restore_control_arguments(args, checkpoint)
    initialization = read_checkpoint(args.init_checkpoint) if args.init_checkpoint else None
    if initialization is not None:
        validate_initialization(args, initialization)
    seed_everything(args.seed)
    run_config = public_arguments(args).copy()
    run_config.update(world_size=world, image_size=IMAGE_SIZE, text_mode="question")
    out = Path(args.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    train_set = VizWizGroundingDataset(args.data_root, "train", IMAGE_SIZE, DETAIL_SIZE)
    model = GroundingModel(stage=args.stage, crop_size=CROP_SIZE).to(device)
    if initialization is not None:
        initialize_weights(initialization, model)
    del initialization
    controls = TrainingControls(model, args, checkpoint)
    parameters = controls.parameters
    optimizer = controls.optimizer
    scaler = torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    start = 0
    if args.resume_checkpoint:
        load_model_weights(model, checkpoint)
        if checkpoint["run_config"].get("stage") != args.stage:
            raise ValueError("Resume requires the same training stage; use --init-checkpoint to change stage")
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError("Raw model weights have no optimizer state; use --init-checkpoint for a new run")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        # CPU checkpoints contain an empty disabled-scaler state. A CUDA continuation
        # starts a fresh scaler; a disabled CPU scaler safely ignores CUDA state.
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start = checkpoint["epoch"]
        if start >= args.num_epochs:
            raise ValueError("num-epochs must exceed the resumed epoch")
        # Release the loaded CPU copies after restoring all training state.
        del checkpoint
    if distributed:
        model = nn.parallel.DistributedDataParallel(model,
            device_ids=[device.index] if device.type=="cuda" else None)
    sampler = DistributedSampler(train_set,num_replicas=world,rank=rank,seed=args.seed) if distributed else None
    generator = torch.Generator()
    train_loader = make_loader(train_set,args,args.batch_size//world,sampler,sampler is None,generator)
    controls.configure_schedule(
        args.num_epochs,
        math.ceil(len(train_loader) / getattr(args, "gradient_accumulation_steps", 1)),
        start,
    )
    run_config["optimization_config"] = controls.config
    run_config["experiment_config"] = unwrap(model).experiment_config
    run_config["tiny"] = unwrap(model).coarse.experiment_config["tiny"]
    if rank == 0:
        print(json.dumps({"parameters":sum(p.numel() for p in unwrap(model).parameters()),
                          "trainable":sum(p.numel() for p in parameters),"device":str(device),
                          "experiment":unwrap(model).experiment_config}),flush=True)
    for epoch in range(start,args.num_epochs):
        # Epoch-keyed RNG permits reproducible continuation, including randomized training crops.
        seed_everything(args.seed+epoch*world+rank)
        generator.manual_seed(args.seed+epoch*world+rank)
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        controls.set_encoder_modes()
        controls.log_learning_rates(epoch + 1)
        totals = torch.zeros(2,device=device,dtype=torch.float64)
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}",
                          unit="batch", disable=rank != 0, dynamic_ncols=True):
            batch = to_device(batch,device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,enabled=device.type=="cuda"):
                loss = model(batch, compute_loss=True)["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters,1.)
            controls.step(scaler)
            n = len(batch["filename"])
            totals[0].add_(loss.detach(),alpha=n)
            totals[1].add_(n)
        if distributed:
            dist.all_reduce(totals)
        if rank == 0:
            epoch_loss = (totals[0]/totals[1]).item()
            record = {"epoch":epoch+1,"train_loss":epoch_loss,"stage_loss":epoch_loss}
            print(json.dumps(record),flush=True)
            with (out/"history.jsonl").open("a") as handle:
                handle.write(json.dumps(record)+"\n")
            if args.save_every and (epoch + 1) % args.save_every == 0:
                checkpoint_path = out / f"checkpoint_epoch{epoch + 1}.pt"
                torch.save({"epoch": epoch + 1,
                            "model_state_dict": unwrap(model).state_dict(),
                            "training_controls": controls.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            "experiment_config": unwrap(model).experiment_config,
                            "run_config": run_config}, checkpoint_path, _use_new_zipfile_serialization=False)
                print(f"Checkpoint saved: {checkpoint_path}", flush=True)
    if distributed:
        dist.barrier()

main = train_main

if __name__ == "__main__":
    main()
