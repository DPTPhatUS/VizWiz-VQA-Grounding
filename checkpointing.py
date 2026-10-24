"""Coordinate rank-zero disk I/O without parking peers in a GPU collective."""

from datetime import timedelta

import torch.distributed as dist


def io_process_group():
    """All ranks create this CPU group once, after initializing training DDP."""
    return dist.new_group(backend="gloo", timeout=timedelta(hours=1))


def run_primary_io(action, group=None):
    """Report disk failures to every rank before any enters its next GPU collective."""
    distributed = dist.is_initialized()
    if distributed and group is None:
        raise ValueError("Distributed checkpoint I/O requires the CPU process group")
    error = None
    if not distributed or dist.get_rank() == 0:
        try:
            action()
        except Exception as exc:  # noqa: BLE001 - propagate the primary failure to all ranks
            error = f"{type(exc).__name__}: {exc}"
    if distributed:
        status = [error]
        # Slow disk writes wait on Gloo, not the NCCL watchdog's GPU queue.
        dist.broadcast_object_list(status, src=0, group=group)
        error = status[0]
    if error is not None:
        raise RuntimeError(f"Rank-zero output failed: {error}")
