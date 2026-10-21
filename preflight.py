"""Short real-backbone forward/backward/DDP check before a full Kaggle run."""

import argparse
from train import TRAINING_DEFAULTS
from training_options import TrainingArgumentParser, TrainingControls
import json
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from dataset import VizWizGroundingDataset, make_loader
from losses import SegmentationObjective
from models.model import GroundingModel
from train import (
    model_options,
    objective_options,
    resolve_training_arguments,
    seed_everything,
)
from utils import to_device


def checked_step(
    model, wrapped, objective, batch, optimizer, scaler, device, *, max_scale_retries=8, training_controls=None
):
    """Require one successful update; let AMP calibrate without hiding broken gradients."""
    world = dist.get_world_size() if dist.is_initialized() else 1

    def rank_counts(flags):
        values = torch.tensor(flags, dtype=torch.int32, device=device)
        if dist.is_initialized():
            dist.all_reduce(values)
        return values.tolist()

    for attempt in range(max_scale_retries + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device.type, dtype=torch.float16, enabled=scaler.is_enabled()
        ):
            loss, _ = objective(wrapped, batch)
        if rank_counts([not bool(torch.isfinite(loss))])[0]:
            raise FloatingPointError(
                "Preflight forward loss is nonfinite on at least one rank"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        missing, nonfinite, frozen = [], [], []
        parameters = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameters.append(parameter)
                if parameter.grad is None:
                    missing.append(name)
                elif not torch.isfinite(parameter.grad).all():
                    nonfinite.append(name)
            elif parameter.grad is not None:
                frozen.append(name)
        missing_ranks, frozen_ranks, overflow_ranks = rank_counts(
            [bool(missing), bool(frozen), bool(nonfinite)]
        )
        if missing_ranks:
            raise RuntimeError(
                f"Missing gradient (not AMP overflow); local parameters: {missing}; affected ranks: {missing_ranks}"
            )
        if frozen_ranks:
            raise RuntimeError(
                f"Frozen parameters received gradients: {frozen}; affected ranks: {frozen_ranks}"
            )
        if overflow_ranks:
            if not scaler.is_enabled():
                raise FloatingPointError(
                    f"Nonfinite unscaled gradients in FP32: {nonfinite[:5]}"
                )
            if overflow_ranks != world:
                raise RuntimeError(
                    "Ranks disagree on gradient overflow after DDP reduction"
                )
            if attempt == max_scale_retries:
                raise FloatingPointError(
                    f"AMP gradients remain nonfinite after {max_scale_retries} scale reductions; "
                    f"scale={scaler.get_scale()}, parameters={nonfinite[:5]}. "
                    "Preflight has not passed; investigate numerical stability."
                )
            previous = scaler.get_scale()
            # unscale_ recorded Inf/NaN: GradScaler skips this update on every rank.
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous:
                raise RuntimeError("AMP overflow did not reduce the loss scale")
            print(
                json.dumps(
                    dict(
                        event="amp_loss_scale_backoff",
                        rank=dist.get_rank() if dist.is_initialized() else 0,
                        previous_scale=previous,
                        next_scale=scaler.get_scale(),
                        retry=attempt + 1,
                        nonfinite_parameters=nonfinite[:3],
                        optimizer_update=False,
                    )
                ),
                flush=True,
            )
            continue
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        if training_controls is None:
            scaler.step(optimizer)
            scaler.update()
        else:
            training_controls.step(scaler)
        return attempt
    raise AssertionError("Unreachable preflight retry state")


def main(argv=None):
    parser = TrainingArgumentParser(description=__doc__, recipe_keys=TRAINING_DEFAULTS)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Global microbatch size across ranks"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--config", help="Same optional experiment JSON used by train.py"
    )
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)
    world, rank, local = [
        int(os.environ.get(k, v))
        for k, v in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0))
    ]
    if (
        args.batch_size <= 0
        or args.batch_size % world
        or args.steps < 2
        or args.num_workers < 0
    ):
        raise ValueError(
            "Use a positive global batch divisible by world size, >=2 steps and nonnegative workers"
        )
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = (
        torch.device("cuda", local) if args.device == "cuda" else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        seed_everything(42)
        settings = resolve_training_arguments(args)
        model = (
            GroundingModel(
                gradient_checkpointing=args.gradient_checkpointing,
                **model_options(settings),
            )
            .to(device)
            .train()
        )
        dataset = VizWizGroundingDataset(
            args.data_root, "train", model.image_size, detail_size=model.detail_size
        )
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world,
                rank=rank,
                shuffle=False,
            )
            if world > 1
            else None
        )
        loader = make_loader(dataset, args, args.batch_size // world, sampler)
        if args.steps > len(loader):
            raise ValueError("Not enough batches for requested preflight steps")
        controls = TrainingControls(model, args)
        controls.configure_schedule(1, args.steps)
        optimizer = controls.optimizer
        criterion = SegmentationObjective(**objective_options(settings))

        def objective(module, batch):
            # Exercise active ranking even when the real run starts with a warm-up.
            return criterion(module, batch, epoch=settings["ranking_warmup_epochs"])

        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        wrapped = (
            DistributedDataParallel(
                model, device_ids=[local] if device.type == "cuda" else None
            )
            if world > 1
            else model
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        scale_retries = 0
        for index, batch in enumerate(loader):
            scale_retries += checked_step(
                model,
                wrapped,
                objective,
                to_device(batch, device),
                optimizer,
                scaler,
                device,
                training_controls=controls,
            )
            if index + 1 == args.steps:
                break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print(
            json.dumps(
                dict(
                    rank=rank,
                    status="passed",
                    steps=args.steps,
                    amp_scale_retries=scale_retries,
                    final_loss_scale=scaler.get_scale(),
                    device=str(device),
                    seconds=time.perf_counter() - start,
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else None,
                    interpretation="Execution/gradient check, not an accuracy result; no checkpoints written.",
                )
            ),
            flush=True,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
