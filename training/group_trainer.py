"""Controlled crop-dose, foreground-retention and large-image recall experiments."""

import argparse
import hashlib
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from checkpointing import io_process_group, run_primary_io
from dataset import VizWizGroundingDataset, collate_samples, seed_worker
from losses import TRAINING_DEFAULTS, SegmentationObjective
from models.model import GroundingModel
from train import seed_everything
from training.context_crop import ContextCrop
from training.crop_support import RECIPES, CropSupportObjective
from training.projection import constrained_step, gradient_vector, parameter_vector
from training.size_groups import GROUP_NAMES, BalancedGroupBatches, mask_groups
from training.size_objective import resolve_group_weights
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

VERSION = "crop-support-updates-v1"
BASE_VERSION = "size-weighted-updates-v1"
LOSS_DEFAULTS = {**TRAINING_DEFAULTS, "overlap_loss": "dice"}


def parser(preflight=False):
    result = TrainingArgumentParser(description=__doc__, recipe_keys=LOSS_DEFAULTS)
    result.add_argument(
        "--resume-checkpoint",
        required=True,
        help="Full B epoch5 checkpoint, or this experiment's own full checkpoint",
    )
    result.add_argument("--variant", choices=tuple(RECIPES), default="retain")
    result.add_argument("--data-root", required=True)
    result.add_argument("--output-dir", default="outputs")
    result.add_argument("--num-epochs", type=int, default=10)
    result.add_argument(
        "--focus-alpha",
        type=float,
        default=0.25,
        help="Blend natural group frequencies with small/medium focus; 0 is natural",
    )
    result.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="Global total over all three groups, before accumulation",
    )
    result.add_argument("--gradient-accumulation-steps", type=int, default=1)
    result.add_argument("--num-workers", type=int, default=2)
    result.add_argument("--lr", type=float, default=1e-5)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--save-every", type=int, default=10)
    result.add_argument(
        "--save-final-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional extra final weights file; disabled to save disk",
    )
    result.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    result.add_argument(
        "--project-updates", action=argparse.BooleanOptionalAction, default=False
    )
    if preflight:
        result.add_argument("--steps", type=int, default=3)
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_gradient(model, batches, objective, controls, scaler, device, group):
    """Stream microbatches and reduce one mean group gradient across DDP ranks."""
    controls.optimizer.zero_grad(set_to_none=True)
    statistics = [0.0] * 7
    for index, batch in enumerate(batches):
        batch = to_device(batch, device)
        area = batch.get("source_area_fraction")
        if area is None:
            area = batch["mask"].float().mean((1, 2, 3))
        actual = torch.where(area < 0.01, 0, torch.where(area < 0.1, 1, 2))
        if bool((actual != group).any()):
            raise ValueError("Sampler and loaded target area disagree")
        last = index + 1 == len(batches)
        context = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and not last
            else nullcontext()
        )
        with context:
            with torch.autocast(
                device.type, dtype=torch.float16, enabled=scaler.is_enabled()
            ):
                loss, components = objective(model, batch)
            bad = (~torch.isfinite(loss)).to(torch.int32)
            if dist.is_initialized():
                dist.all_reduce(bad, op=dist.ReduceOp.MAX)
            if bool(bad):
                raise FloatingPointError(
                    "Nonfinite group forward loss on at least one rank"
                )
            scaler.scale(loss / len(batches)).backward()
        values = [loss.detach()] + [
            torch.as_tensor(
                components.get(key, loss.detach() if key == "base_objective" else 0.0),
                device=device,
            ).detach()
            for key in (
                "base_objective",
                "retention_term",
                "recall_delta",
                "support_fraction",
                "protected_coverage",
                "retention_shortfall",
            )
        ]
        statistics = [
            a + b / len(batches)
            for a, b in zip(statistics, torch.stack(values).double().tolist())
        ]
    vector = gradient_vector(controls.parameters, scaler.get_scale())
    controls.optimizer.zero_grad(set_to_none=True)
    return vector, statistics


def one_update(
    model,
    group_batches,
    objective,
    controls,
    scaler,
    device,
    project,
    group_weights=(0.5, 0.5, 0.0),
):
    gradients, records = [], []
    for group, batches in enumerate(group_batches):
        gradient, statistics = collect_gradient(
            model, batches, objective, controls, scaler, device, group
        )
        gradients.append(gradient)
        records.append(statistics)
    metrics = constrained_step(
        controls,
        scaler,
        torch.stack(gradients),
        project=project,
        group_weights=group_weights,
    )
    statistics = torch.tensor(records, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(statistics)
        statistics /= dist.get_world_size()
    losses, base, retain, recall, eligible, coverage, shortfall = statistics.T.tolist()
    metrics.update({"loss_" + name: loss for name, loss in zip(GROUP_NAMES, base)})
    metrics["macro_group_loss"] = sum(base) / 3
    for name, values in (
        ("base_segmentation_objective", base),
        ("retention_objective_contribution", retain),
        ("recall_objective_delta", recall),
        ("train_loss", losses),
    ):
        metrics[name] = sum(
            weight * value for weight, value in zip(group_weights, values)
        )
    metrics["proposal_loss"] = metrics["train_loss"]
    metrics["support_fraction_large"] = eligible[2]
    metrics["protected_coverage_large"] = coverage[2]
    metrics["retention_shortfall_large"] = shortfall[2]
    if "crop_applied" in group_batches[0][0]:
        statistics = []
        for batches in group_batches:
            statistics.append(
                [
                    torch.cat([batch[key].reshape(-1) for batch in batches])
                    .mean()
                    .item()
                    for key in (
                        "crop_requested",
                        "crop_applied",
                        "crop_input_area_fraction",
                    )
                ]
            )
        statistics = torch.tensor(statistics, device=device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(statistics)
            statistics /= dist.get_world_size()
        for name, values in zip(GROUP_NAMES, statistics.tolist()):
            for key, value in zip(
                ("crop_requested", "crop_applied", "crop_input_area_fraction"), values
            ):
                metrics[f"{key}_{name}"] = value
        metrics["crop_applied_fraction"] = float(statistics[:, 1].mean())
    return metrics


def execute(args, rank, world, device, preflight):
    saved = read_checkpoint(args.resume_checkpoint)
    source_version = saved.get("run_config", {}).get("objective_version")
    entering = source_version == BASE_VERSION
    if entering and (
        saved.get("epoch") != 5 or saved["run_config"].get("focus_alpha") != 0.25
    ):
        raise ValueError(
            "Start this experiment from the original B/mild_focus epoch5 checkpoint"
        )
    if "optimizer_state_dict" not in saved or source_version not in {
        VERSION,
        BASE_VERSION,
    }:
        raise ValueError("Use a full size-weighted B epoch5 or context-crop checkpoint")
    restore_control_arguments(args, saved)
    for key in (
        "batch_size",
        "gradient_accumulation_steps",
        "seed",
        "world_size",
        "lr",
        "project_updates",
        "focus_alpha",
        "gradient_checkpointing",
    ):
        if getattr(args, key) != saved["run_config"][key]:
            raise ValueError(f"Resume {key} differs from checkpoint")
    recipe = RECIPES[args.variant]
    crop = ContextCrop(recipe.crop_probability)
    if not entering and (
        saved["run_config"].get("context_crop") != crop.config
        or saved["run_config"].get("support_recipe") != recipe.config
    ):
        raise ValueError("Resume crop/support policy differs from checkpoint")
    options = checkpoint_model_kwargs(saved)
    requested = recipe_config(args, LOSS_DEFAULTS)
    settings = {
        **LOSS_DEFAULTS,
        **saved["run_config"]["training_config"],
    }
    if any(settings[key] != value for key, value in requested.items()):
        raise ValueError("Resume loss configuration differs")
    settings.update(requested)
    objective = SegmentationObjective(**settings)
    seed_everything(args.seed)
    model = GroundingModel(
        **options, gradient_checkpointing=args.gradient_checkpointing
    )
    load_model_weights(model, saved)
    model.to(device)
    initial = saved["run_config"]["group_protocol"]["initial_checkpoint"]
    start = saved["epoch"]
    controls = TrainingControls(model, args, saved)
    if (
        not controls.config["freeze_vision_encoder"]
        or not controls.config["freeze_text_encoder"]
    ):
        raise ValueError("This controlled continuation keeps both encoders frozen")
    if not entering and not isinstance(saved.get("support_state"), dict):
        raise ValueError("Resume is missing the original support-reference state")
    objective = CropSupportObjective(
        objective, model, recipe, saved.get("support_state") if not entering else None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    controls.optimizer.load_state_dict(saved["optimizer_state_dict"])
    if saved.get("scaler_state_dict"):
        scaler.load_state_dict(saved["scaler_state_dict"])
    dataset = VizWizGroundingDataset(
        args.data_root,
        "train",
        model.image_size,
        detail_size=model.detail_size,
        train_transform=crop,
    )
    groups, digest = mask_groups(dataset)
    natural_weights, proposal_weights = resolve_group_weights(
        list(map(len, groups)), args.focus_alpha
    )
    if (
        tuple(saved["run_config"]["group_protocol"]["proposal_weights"])
        != proposal_weights
    ):
        raise ValueError("Resume group objective weights changed")
    if saved["run_config"]["group_protocol"]["mask_group_sha256"] != digest:
        raise ValueError("Training masks/group assignments changed on resume")
    annotation = file_hash(Path(args.data_root) / "train_grounding.json")
    if saved["run_config"]["train_annotation_sha256"] != annotation:
        raise ValueError("Training annotations changed on resume")
    augmentation_origin = (
        {
            "path": str(args.resume_checkpoint),
            "sha256": file_hash(args.resume_checkpoint),
            "epoch": start,
            "objective_version": source_version,
        }
        if entering
        else saved["run_config"]["augmentation_origin"]
    )
    del saved
    if not preflight and start >= args.num_epochs:
        raise ValueError("num-epochs must exceed the resumed epoch")
    sampler = BalancedGroupBatches(
        groups,
        args.batch_size,
        args.gradient_accumulation_steps,
        rank=rank,
        world=world,
        seed=args.seed,
        steps=args.steps if preflight else None,
    )
    generator = torch.Generator()
    kwargs = {"prefetch_factor": 2} if args.num_workers else {}
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=device.type == "cuda",
        **kwargs,
    )
    controls.configure_schedule(
        1 if preflight else args.num_epochs, sampler.steps, 0 if preflight else start
    )
    if world > 1:
        model = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    protocol = {
        "group_names": GROUP_NAMES,
        "group_counts": list(map(len, groups)),
        "mask_group_sha256": digest,
        "sampling": "balanced-cyclic-shuffled",
        "projection": args.project_updates,
        "constraint_relative_tolerance": 1e-6,
        "focus_alpha": args.focus_alpha,
        "context_crop": crop.config,
        "support_recipe": recipe.config,
        "reference_parameters_training_only": objective.reference_parameters,
        "natural_weights": list(natural_weights),
        "proposal_weights": list(proposal_weights),
        "initial_checkpoint": initial,
        "updates_per_epoch": sampler.steps,
    }
    config = {
        **public_arguments(args),
        "objective_version": VERSION,
        "context_crop": crop.config,
        "support_recipe": recipe.config,
        "augmentation_origin": augmentation_origin,
        "training_config": settings,
        "experiment_config": raw.experiment_config,
        "tiny": options["tiny"],
        "group_protocol": protocol,
        "optimization_config": controls.config,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "train_annotation_sha256": annotation,
        "image_size": raw.image_size,
        "detail_size": raw.detail_size,
        "text_mode": "question",
        "torch_version": str(torch.__version__),
    }
    source = Path(__file__).resolve().parents[1]
    config["source_sha256"] = {
        name: file_hash(source / name)
        for name in (
            "train_group.py",
            "preflight_group.py",
            "training/group_trainer.py",
            "training/size_groups.py",
            "training/size_objective.py",
            "training/context_crop.py",
            "training/crop_support.py",
            "training/projection.py",
            "train.py",
            "training_options.py",
            "losses.py",
            "models/model.py",
            "models/image_encoder.py",
            "models/text_encoder.py",
            "models/detail_pyramid.py",
            "models/mask_decoder.py",
            "dataset.py",
            "utils.py",
            "checkpointing.py",
            "metrics.py",
        )
    }
    out = Path(args.output_dir)
    history = out / "history.jsonl"
    if not preflight:
        if not history.exists() and any(out.glob("*.pt")):
            raise ValueError(
                "Output contains checkpoints without matching history; use a fresh folder"
            )
        if history.exists():
            records = [
                json.loads(line)
                for line in history.read_text().splitlines()
                if line.strip()
            ]
            if records and records[-1]["epoch"] != start:
                raise ValueError(
                    "History does not end at resumed epoch; use a fresh output directory"
                )
        run_primary_io(lambda: out.mkdir(parents=True, exist_ok=True), args._io_group)
    if rank == 0:
        print(
            json.dumps(
                {
                    "group_protocol": protocol,
                    "training_config": settings,
                    "trainable_parameters": sum(p.numel() for p in controls.parameters),
                }
            ),
            flush=True,
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start, (start + 1) if preflight else args.num_epochs):
        seed_everything(args.seed + epoch * world + rank)
        generator.manual_seed(args.seed + epoch * world + rank)
        sampler.set_epoch(epoch)
        model.train()
        controls.set_encoder_modes()
        controls.log_learning_rates(epoch + 1)
        iterator = iter(loader)
        totals = {}
        for _ in tqdm(
            range(sampler.steps), desc=f"Group epoch {epoch + 1}", disable=rank != 0
        ):
            group_batches = [[] for _ in range(3)]
            for _micro in range(args.gradient_accumulation_steps):
                for group in range(3):
                    group_batches[group].append(next(iterator))
            retries = 0
            while True:
                metrics = one_update(
                    model,
                    group_batches,
                    objective,
                    controls,
                    scaler,
                    device,
                    args.project_updates,
                    proposal_weights,
                )
                if not preflight or metrics["optimizer_update"]:
                    break
                retries += 1
                if retries > 8:
                    raise FloatingPointError(
                        "Group preflight overflow persists after8 scale reductions"
                    )
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
        record = {
            "epoch": epoch + 1,
            **{key: value / sampler.steps for key, value in totals.items()},
            "objective_version": VERSION,
            "training_config": settings,
            "group_protocol": protocol,
        }
        if rank == 0:
            print(json.dumps(record), flush=True)
        if preflight:
            if world > 1:
                reference = parameter_vector(controls.parameters)
                dist.broadcast(reference, src=0)
                differs = (
                    ~torch.eq(reference, parameter_vector(controls.parameters)).all()
                ).to(torch.int32)
                dist.all_reduce(differs, op=dist.ReduceOp.MAX)
                if bool(differs):
                    raise RuntimeError(
                        "Ranks diverged after constrained optimizer updates"
                    )
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "status": "passed",
                        "steps": sampler.steps,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                        / 2**30
                        if device.type == "cuda"
                        else None,
                        "interpretation": "Execution/gradient check only; no checkpoints written",
                    }
                ),
                flush=True,
            )
        else:

            def export_epoch(record=record, epoch=epoch):
                with history.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                if args.save_every and (epoch + 1) % args.save_every == 0:
                    save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": raw.state_dict(),
                            "optimizer_state_dict": controls.optimizer.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            "support_state": objective.state_dict(),
                            "training_controls": controls.state_dict(),
                            "experiment_config": raw.experiment_config,
                            "run_config": config,
                        },
                        out / f"checkpoint_epoch{epoch + 1}.pt",
                    )

            run_primary_io(export_epoch, args._io_group)
    if not preflight and args.save_final_weights:

        def export_final():
            save_checkpoint(
                {
                    "model_state_dict": raw.state_dict(),
                    "experiment_config": raw.experiment_config,
                    "run_config": config,
                },
                out / f"model_final_epoch{args.num_epochs}.pt",
            )

        run_primary_io(export_final, args._io_group)
    if world > 1:
        dist.barrier()


def run_main(argv=None, *, preflight=False):
    args = parser(preflight).parse_args(argv)
    rank, world, local = [
        int(os.environ.get(k, d))
        for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0))
    ]
    args.world_size = world
    if (
        min(args.num_epochs, args.batch_size, args.gradient_accumulation_steps) <= 0
        or args.batch_size % (3 * world)
        or args.num_workers < 0
        or args.save_every < 0
    ):
        raise ValueError(
            "Use positive epochs/accumulation, nonnegative workers/save interval and global batch divisible by3*world"
        )
    ContextCrop(RECIPES[args.variant].crop_probability)
    if args.project_updates:
        raise ValueError("Context-crop experiment requires projection disabled")
    if not math.isfinite(args.focus_alpha) or not 0 <= args.focus_alpha <= 1:
        raise ValueError("focus-alpha must be finite and in [0, 1]")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("LR must be finite and positive")
    if preflight and args.steps < 2:
        raise ValueError("Preflight needs at least2 steps")
    if not Path(args.resume_checkpoint).is_file():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {args.resume_checkpoint!r}. "
            "Attach the full B/mild_focus epoch5 checkpoint and set B_CKPT to its path."
        )
    for kind in ("vision", "text"):
        key = f"freeze_{kind}_encoder"
        if getattr(args, key) is None:
            setattr(args, key, True)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = (
        torch.device("cuda", local) if args.device == "cuda" else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    args._io_group = None
    try:
        if world > 1 and not preflight:
            args._io_group = io_process_group()
        execute(args, rank, world, device, preflight)
    finally:
        if dist.is_initialized():
            if args._io_group is not None:
                dist.destroy_process_group(args._io_group)
            dist.destroy_process_group()
