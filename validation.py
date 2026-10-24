"""Checkpoint-free, unpadded distributed validation using the standalone eval rules."""

import json
import random
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Subset

from dataset import VizWizGroundingDataset, make_loader
from eval import write_evaluation_metrics
from metrics import evaluate


def gather_results(value):
    if not dist.is_initialized():
        return [value]
    results = [None] * dist.get_world_size()
    dist.all_gather_object(results, value)
    return results


@contextmanager
def evaluation_state(model, device):
    """Restore module modes and RNG so validation cannot change subsequent training."""
    modes = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def validate_epoch(model, args, device, config, epoch):
    """Evaluate each validation image exactly once, then combine native pixel counts."""
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    output = Path(args.output_dir) / f"eval_val_ep{epoch}"
    error = None
    if rank == 0:
        try:
            if output.exists() and any(output.iterdir()):
                raise ValueError(f"Validation output must be new or empty: {output}")
            output.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001 - share failures before any rank continues
            error = str(exc)
    errors = gather_results(error)
    if any(errors):
        raise RuntimeError(f"Validation setup failed: {errors}")

    with evaluation_state(model, device):
        error, scores, totals = None, {}, {}
        try:
            dataset = VizWizGroundingDataset(
                args.data_root, "val", model.image_size, detail_size=model.detail_size
            )
            # DistributedSampler pads uneven splits, duplicating images and pixel counts.
            subset = Subset(dataset, range(rank, len(dataset), world))
            loader = make_loader(
                subset,
                SimpleNamespace(num_workers=args.num_workers, device=str(device)),
                args.val_batch_size,
                generator=torch.Generator().manual_seed(0),
            )
            # Bypass the DDP wrapper: shards can have different numbers of forward calls.
            scores = evaluate(model, loader, device, output, pixel_totals=totals)
        except Exception as exc:  # noqa: BLE001 - share failures before any rank continues
            error = f"Rank {rank}: {type(exc).__name__}: {exc}"
        shards = gather_results((error, scores, totals))
        errors = [item[0] for item in shards if item[0]]
        if errors:
            raise RuntimeError(f"Validation failed: {errors}")
        combined, pixels = {}, {"intersection": 0, "union": 0}
        for _, scores, totals in shards:
            if combined.keys() & scores.keys():
                raise ValueError("Duplicate samples across validation shards")
            combined.update(scores)
            for name in pixels:
                pixels[name] += totals.get(name, 0)
        if len(combined) != len(dataset):
            raise ValueError("Validation shard coverage differs from the full dataset")
        combined = {name: combined[name] for name in dataset.names}
        error, summary = None, None
        if rank == 0:
            try:
                checkpoint = Path(args.output_dir) / f"checkpoint_epoch{epoch}.pt"
                summary = write_evaluation_metrics(
                    model,
                    combined,
                    pixels,
                    output,
                    config,
                    dataset="val",
                    epoch=epoch,
                    checkpoint=checkpoint
                    if args.save_every and epoch % args.save_every == 0
                    else None,
                )
                print(json.dumps({"validation": summary}), flush=True)
            except Exception as exc:  # noqa: BLE001 - share failures before any rank continues
                error = str(exc)
        errors = gather_results(error)
        if any(errors):
            raise RuntimeError(f"Validation export failed: {errors}")
    return summary
