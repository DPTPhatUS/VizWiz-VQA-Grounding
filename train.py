"""Train the question-guided detail pyramid with native-resolution evaluation kept separate."""

import argparse
import hashlib
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DistributedSampler
from tqdm.auto import tqdm

from dataset import VizWizGroundingDataset, make_loader
from losses import OBJECTIVE_VERSION, TRAINING_DEFAULTS, SegmentationObjective
from models.model import GroundingModel
from training_image_protocol import image_names, parent_provenance
from training_options import (
    TrainingArgumentParser,
    TrainingControls,
    public_arguments,
    recipe_config,
    restore_control_arguments,
)
from utils import (
    checkpoint_model_kwargs,
    load_model_weights,
    read_checkpoint,
    save_checkpoint,
    to_device,
)


def training_parser():
    parser = TrainingArgumentParser(recipe_keys=TRAINING_DEFAULTS, description=__doc__)
    parser.add_argument("--data-root", default="data/vizwiz")
    parser.add_argument(
        "--train-image-list", help="Optional JSON list of train image names"
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Global microbatch size across ranks before accumulation",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Recompute vision/CNN/decoder activations to reduce memory",
    )
    parser.add_argument(
        "--resume-checkpoint",
        help="Full checkpoint from this detail-pyramid experiment",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON for overlap loss and auxiliary supervision weights",
    )
    return parser


def resolve_training_arguments(args, saved=None):
    requested = recipe_config(args, TRAINING_DEFAULTS)
    if not isinstance(requested, dict) or set(requested) - TRAINING_DEFAULTS.keys():
        raise ValueError("Unknown detail-pyramid training config keys")
    settings = dict(TRAINING_DEFAULTS)
    if saved is not None:
        run = saved["run_config"]
        if run.get("objective_version") != OBJECTIVE_VERSION:
            raise ValueError("Resume requires the same detail-pyramid objective")
        settings.update(run["training_config"])
        if any(settings[key] != value for key, value in requested.items()):
            raise ValueError("Resume loss configuration differs from checkpoint")
        for key in (
            "batch_size",
            "lr",
            "seed",
            "world_size",
            "gradient_accumulation_steps",
        ):
            if getattr(args, key) != run[key]:
                raise ValueError(f"Resume {key} differs from checkpoint")
    else:
        settings.update(requested)
    SegmentationObjective(**settings)
    if args.gradient_checkpointing is None:
        args.gradient_checkpointing = (
            saved["run_config"].get("gradient_checkpointing", False) if saved else False
        )
    return settings


def check_arguments(args):
    if (
        min(args.num_epochs, args.batch_size, args.gradient_accumulation_steps) <= 0
        or args.save_every < 0
        or args.num_workers < 0
    ):
        raise ValueError("Invalid epoch, batch, save interval or worker count")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    for key in ("config", "resume_checkpoint"):
        if getattr(args, key) and not Path(getattr(args, key)).is_file():
            raise ValueError(f"{key} does not exist")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def train_main(argv=None):
    args = training_parser().parse_args(argv)
    check_arguments(args)
    rank, world, local = (
        int(os.environ.get(key, default))
        for key, default in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0))
    )
    if args.batch_size < world or args.batch_size % world:
        raise ValueError("Global batch size must be a multiple of world size")
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.world_size = world
    device = (
        torch.device(args.device, local)
        if args.device == "cuda"
        else torch.device("cpu")
    )
    if args.device == "cuda":
        torch.cuda.set_device(device)
    distributed = world > 1
    if distributed:
        if args.device == "cuda":
            dist.init_process_group("nccl", device_id=device)
        else:
            dist.init_process_group("gloo")
    try:
        _train(args, rank, world, distributed, device)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def _train(args, rank, world, distributed, device):
    checkpoint = (
        read_checkpoint(args.resume_checkpoint) if args.resume_checkpoint else None
    )
    restore_control_arguments(args, checkpoint)
    options = checkpoint_model_kwargs(checkpoint) if checkpoint else {}
    settings = resolve_training_arguments(args, checkpoint)
    seed_everything(args.seed)
    out = Path(args.output_dir)
    history_path = out / "history.jsonl"
    if not checkpoint and (history_path.exists() or any(out.glob("*.pt"))):
        raise ValueError("Output already contains a run; choose a new directory")
    if checkpoint and history_path.exists():
        records = [
            json.loads(line)
            for line in history_path.read_text().splitlines()
            if line.strip()
        ]
        if records and records[-1]["epoch"] != checkpoint["epoch"]:
            raise ValueError(
                "Output history does not end at the resumed epoch; use a new output directory"
            )
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    model = GroundingModel(
        **options, gradient_checkpointing=args.gradient_checkpointing
    ).to(device)
    train_set = VizWizGroundingDataset(
        args.data_root, "train", model.image_size, detail_size=model.detail_size
    )
    train_set.names = image_names(args.train_image_list, train_set.names)
    provenance = parent_provenance(
        train_set.names, checkpoint.get("run_config", {}) if checkpoint else None
    )
    objective = SegmentationObjective(**settings)
    controls = TrainingControls(model, args, checkpoint)
    parameters = controls.parameters
    optimizer = controls.optimizer
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start = 0
    if checkpoint:
        load_model_weights(model, checkpoint)
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError("Resume needs a full checkpoint with optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start = checkpoint["epoch"]
        del checkpoint
        if start >= args.num_epochs:
            raise ValueError("num-epochs must exceed the resumed epoch")
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
        )
    sampler = (
        DistributedSampler(train_set, num_replicas=world, rank=rank, seed=args.seed)
        if distributed
        else None
    )
    generator = torch.Generator()
    train_loader = make_loader(
        train_set, args, args.batch_size // world, sampler, sampler is None, generator
    )
    controls.configure_schedule(
        args.num_epochs,
        math.ceil(len(train_loader) / getattr(args, "gradient_accumulation_steps", 1)),
        start,
    )
    version = OBJECTIVE_VERSION
    run_config = dict(
        public_arguments(args),
        training_config=settings,
        objective_version=version,
        image_size=unwrap(model).image_size,
        detail_size=unwrap(model).detail_size,
        experiment_config=unwrap(model).experiment_config,
        freeze_text_encoder=controls.is_encoder_frozen("text"),
        text_mode="question",
        torch_version=str(torch.__version__),
        tiny=unwrap(model).experiment_config["tiny"],
    )
    run_config.update(provenance)
    run_config["optimization_config"] = controls.config
    run_config.update(
        gradient_clip_norm=1.0,
        gradient_clip_scope="all-trainable",
        effective_batch_size=args.batch_size * args.gradient_accumulation_steps,
    )
    source_root = Path(__file__).resolve().parent
    run_config["source_sha256"] = {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in (
            "train.py",
            "training_image_protocol.py",
            "training_options.py",
            "models/model.py",
            "models/image_encoder.py",
            "models/text_encoder.py",
            "models/detail_pyramid.py",
            "models/mask_decoder.py",
            "losses.py",
            "dataset.py",
            "metrics.py",
            "utils.py",
        )
    }
    if rank == 0:
        print(
            json.dumps(
                {
                    "parameters": sum(p.numel() for p in unwrap(model).parameters()),
                    "trainable": sum(p.numel() for p in parameters),
                    "device": str(device),
                    "experiment": unwrap(model).experiment_config,
                    "training_config": settings,
                    "seed": args.seed,
                    "lr": args.lr,
                    "global_microbatch": args.batch_size,
                    "effective_batch_size": run_config["effective_batch_size"],
                    "gradient_checkpointing": args.gradient_checkpointing,
                }
            ),
            flush=True,
        )
    for epoch in range(start, args.num_epochs):
        # Epoch-keyed RNG permits reproducible continuation at an epoch boundary.
        seed_everything(args.seed + epoch * world + rank)
        generator.manual_seed(args.seed + epoch * world + rank)
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        controls.set_encoder_modes()
        controls.log_learning_rates(epoch + 1)
        totals = torch.zeros(
            2, device=device, dtype=torch.float64
        )  # (2,): [weighted_loss_sum, sample_count]
        component_sums = {}
        optimizer.zero_grad(set_to_none=True)
        local_samples = len(sampler) if sampler is not None else len(train_set)
        local_batch = args.batch_size // world
        for index, batch in enumerate(
            tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{args.num_epochs}",
                unit="batch",
                disable=rank != 0,
                dynamic_ncols=True,
            )
        ):
            batch = to_device(batch, device)
            n = len(batch["filename"])
            accumulation = args.gradient_accumulation_steps
            window_start = (index // accumulation) * accumulation
            window_end = min(window_start + accumulation, len(train_loader))
            window_samples = (
                min(window_end * local_batch, local_samples)
                - window_start * local_batch
            )
            update = index + 1 == window_end
            context = model.no_sync() if distributed and not update else nullcontext()
            with context:
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    loss, components = objective(model, batch, epoch=epoch)
                if device.type != "cuda" and not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(loss * n / window_samples).backward()
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    parameters, 1.0, error_if_nonfinite=device.type != "cuda"
                )
                controls.step(scaler)
                optimizer.zero_grad(set_to_none=True)
            totals[0] += loss.detach().to(torch.float64) * n
            totals[1] += n
            for name, value in components.items():
                val64 = value.detach().to(device=device, dtype=torch.float64) * n
                component_sums[name] = (
                    component_sums[name] + val64 if name in component_sums else val64
                )
        names = sorted(component_sums)
        stats = torch.stack(
            [totals[0], totals[1], *(component_sums[k] for k in names)]
        )  # (2 + C,)
        if distributed:
            dist.all_reduce(stats)
        if not torch.isfinite(stats[0]):
            raise FloatingPointError("Nonfinite training loss")
        values = (stats / stats[1]).tolist()
        record = {
            "epoch": epoch + 1,
            "train_loss": values[0],
            "objective_version": version,
        }
        record["training_config"] = run_config["training_config"]
        for name, val in zip(names, values[2:]):
            record[name] = val
        if rank == 0:
            print(json.dumps(record), flush=True)
            with (out / "history.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if args.save_every and (epoch + 1) % args.save_every == 0:
                checkpoint_path = out / f"checkpoint_epoch{epoch + 1}.pt"
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": unwrap(model).state_dict(),
                        "training_controls": controls.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "experiment_config": unwrap(model).experiment_config,
                        "run_config": run_config,
                    },
                    checkpoint_path,
                )
                print(f"Checkpoint saved: {checkpoint_path}", flush=True)
    if rank == 0:
        final_path = out / f"model_final_epoch{args.num_epochs}.pt"
        save_checkpoint(unwrap(model).state_dict(), final_path)
        print(f"Final model saved: {final_path}", flush=True)
    if distributed:
        dist.barrier()


main = train_main

if __name__ == "__main__":
    main()
