"""Deterministic balanced sampling of mask-area groups, independent of backbone."""

import hashlib
import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Sampler

GROUP_NAMES = ("small_lt_1pct", "medium_1_to_10pct", "large_ge_10pct")


def mask_groups(dataset):
    """Index training masks only, on the same nearest-resized grid as the loader."""
    if dataset.split != "train":
        raise ValueError("Group sampling is training-only")
    groups = [[], [], []]
    digest = hashlib.sha256()
    for index, name in enumerate(dataset.names):
        path = (
            dataset.root / "binary_masks_png" / "train" / Path(name).with_suffix(".png")
        )
        with Image.open(path) as source:
            mask = source.convert("L").resize(
                (dataset.detail_size, dataset.detail_size), Image.Resampling.NEAREST
            )
        histogram = mask.histogram()
        area = sum(histogram[128:]) / (dataset.detail_size**2)
        group = 0 if area < 0.01 else 1 if area < 0.1 else 2
        groups[group].append(index)
        binary = mask.point(lambda value: 255 if value > 127 else 0)
        digest.update(name.encode("utf8"))
        digest.update(binary.tobytes())
    if any(not indices for indices in groups):
        raise ValueError("Constrained training needs nonempty <1%, 1-10%, >=10% groups")
    return groups, digest.hexdigest()


class BalancedGroupBatches(Sampler):
    """Yield group0,1,2 microbatches repeatedly, identical global draws across ranks.

    batch_size counts all three groups globally before accumulation. A balanced
    epoch draws approximately N samples, rounded up to whole optimizer windows.
    Sampling cycles shuffled group members, rather than dropping a small group.
    """

    def __init__(
        self,
        groups,
        batch_size,
        accumulation=1,
        *,
        rank=0,
        world=1,
        seed=42,
        steps=None,
    ):
        if (
            batch_size <= 0
            or batch_size % (3 * world)
            or accumulation <= 0
            or not 0 <= rank < world
        ):
            raise ValueError(
                "Global batch must be divisible by 3*world; accumulation must be positive"
            )
        if len(groups) != 3 or any(not group for group in groups):
            raise ValueError("Three nonempty groups are required")
        self.groups, self.batch_size, self.accumulation = (
            groups,
            batch_size,
            accumulation,
        )
        self.rank, self.world, self.seed, self.epoch = rank, world, seed, 0
        self.steps = (
            steps
            if steps is not None
            else math.ceil(sum(map(len, groups)) / (batch_size * accumulation))
        )
        if self.steps <= 0:
            raise ValueError("At least one optimizer step is required")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.steps * self.accumulation * 3

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 10007)
        queues = [[] for _ in self.groups]
        cursor = [0, 0, 0]
        global_count = self.batch_size // 3
        for _ in range(self.steps * self.accumulation):
            for group, members in enumerate(self.groups):
                selected = []
                while len(selected) < global_count:
                    if cursor[group] == len(queues[group]):
                        order = torch.randperm(
                            len(members), generator=generator
                        ).tolist()
                        queues[group] = [members[i] for i in order]
                        cursor[group] = 0
                    count = min(
                        global_count - len(selected), len(queues[group]) - cursor[group]
                    )
                    selected.extend(
                        queues[group][cursor[group] : cursor[group] + count]
                    )
                    cursor[group] += count
                yield selected[self.rank :: self.world]


class UniformBatches(BalancedGroupBatches):
    """Control sampler: same step count and microbatch layout, but size-agnostic.

    Each optimizer window yields three microbatch slots, like the balanced
    sampler, but every slot draws batch_size // 3 global images from one
    shuffled queue over the whole training set, so the batch composition
    follows the natural size distribution.
    """

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 10007)
        members = [index for group in self.groups for index in group]
        queue, cursor = [], 0
        global_count = self.batch_size // 3
        for _ in range(self.steps * self.accumulation):
            for _slot in range(3):
                selected = []
                while len(selected) < global_count:
                    if cursor == len(queue):
                        order = torch.randperm(len(members), generator=generator).tolist()
                        queue, cursor = [members[i] for i in order], 0
                    count = min(global_count - len(selected), len(queue) - cursor)
                    selected.extend(queue[cursor : cursor + count])
                    cursor += count
                yield selected[self.rank :: self.world]
