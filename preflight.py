"""Short real-backbone forward/backward/DDP check before a full Kaggle run."""

import argparse
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
from train import seed_everything
from utils import to_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Global microbatch size across ranks"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
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
        model = (
            GroundingModel(gradient_checkpointing=args.gradient_checkpointing)
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
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=1e-5)
        objective = SegmentationObjective()
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
        for index, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss, _ = objective(wrapped, to_device(batch, device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Preflight loss is nonfinite")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    if (
                        parameter.grad is None
                        or not torch.isfinite(parameter.grad).all()
                    ):
                        raise RuntimeError(f"Missing/nonfinite gradient: {name}")
                elif parameter.grad is not None:
                    raise RuntimeError(f"Frozen parameter has a gradient: {name}")
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
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
