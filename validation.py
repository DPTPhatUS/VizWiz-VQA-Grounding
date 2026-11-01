"""Deterministic, non-padding validation shards without changing training state."""
import contextlib
import random
import numpy as np
import torch
import torch.distributed as dist
from dataset import make_loader
from metrics import evaluate_counts, sample_metrics


@contextlib.contextmanager
def preserve_training_state(model):
    modes = [(module, module.training) for module in model.modules()]
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python); np.random.set_state(numpy); torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        for module, mode in modes:
            module.training = mode


def validation_iou(model, dataset, args, device, rank=0, world=1):
    # Each rank gets a strided shard; unlike DistributedSampler there is no padding.
    sampler = list(range(rank, len(dataset), world))
    core = model.module if hasattr(model, 'module') else model
    with preserve_training_state(core):
        loader = make_loader(dataset, args, max(1, args.batch_size // world), sampler=sampler,
                             generator=torch.Generator().manual_seed(0))
        records = evaluate_counts(core, loader, device)
        sums = torch.tensor([sum(sample_metrics(r)['iou'] for r in records.values()), len(records)],
                            dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(sums)
    if int(sums[1]) != len(dataset):
        raise ValueError('Validation shards did not cover the split exactly once')
    return float(sums[0] / sums[1])
