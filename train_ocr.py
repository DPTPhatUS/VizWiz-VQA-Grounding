"""Train only the OCR adapter; retain the complete frozen parent as a separate artifact."""

import argparse
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from tqdm.auto import tqdm

from checkpointing import io_process_group, run_primary_io
from dataset import make_loader
from ocr_cache import atomic_json, file_hash
from ocr_data import OCRGroundingDataset
from ocr_pointer_data import TARGET_VERSION as POINTER_TARGET_VERSION
from ocr_pointer_data import OCRPointerDataset
from ocr_routing_data import (
    ROUTE_COVERAGE,
    TARGET_VERSION,
    TOKEN_PURITY,
    OCRRoutingDataset,
)
from ocr_selection_data import HYPOTHESIS_VERSION, OCRSelectionDataset
from train import seed_everything
from training.ocr_checkpoint import checkpoint_version, load_ocr_model
from training.ocr_objective import OCREvidenceObjective
from training.ocr_pointer_objective import PointerObjective
from training.ocr_routing_objective import RoutingObjective
from training.ocr_scope_objective import SCOPE_TARGET_VERSION, ScopeObjective
from training.ocr_selection_objective import SelectionObjective
from utils import save_checkpoint, to_device


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--parent-checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume-checkpoint")
    p.add_argument(
        "--config",
        help="Flat JSON overrides CLI; architecture and training options only",
    )
    p.add_argument(
        "--architecture",
        choices=(
            "proposal",
            "pyramid",
            "residual",
            "hypotheses",
            "independent",
            "pointer",
        ),
        default="proposal",
    )
    p.add_argument(
        "--selector-recipe",
        choices=("ranking", "scope"),
        default="ranking",
        help="Independent scorer: pairwise control or scope-focused training",
    )
    p.add_argument(
        "--pointer-recipe",
        choices=("categorical", "structured"),
        default="categorical",
        help="Native OCR sufficiency pointer: plain or token-set supervised",
    )
    p.add_argument(
        "--answer-weight",
        type=float,
        default=0.0,
        help="Train-only supported answer retrieval; hypotheses only",
    )
    p.add_argument(
        "--min-gain",
        type=float,
        default=0.02,
        help="Predicted IoU gain required to replace base; mask selectors only",
    )
    p.add_argument("--route-weight", type=float, default=0.2)
    p.add_argument("--evidence-weight", type=float, default=0.5)
    p.add_argument("--evidence-mode", choices=("full", "geometry"), default="full")
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument(
        "--batch-size", type=int, default=4, help="Global batch across ranks"
    )
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw")
    p.add_argument("--max-candidates", type=int, default=24)
    p.add_argument("--ocr-dropout", type=float, default=0.2)
    p.add_argument("--text-dropout", type=float, default=0.2)
    p.add_argument("--selection-weight", type=float, default=0.1)
    p.add_argument("--local-weight", type=float, default=0.1)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return p


def arguments(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.config:
        requested = json.loads(Path(args.config).read_text())
        actions = {a.dest: a for a in p._actions}
        allowed = set(actions) - {
            "config",
            "help",
            "parent_checkpoint",
            "data_root",
            "cache_root",
            "output_dir",
            "resume_checkpoint",
        }
        if not isinstance(requested, dict) or set(requested) - allowed:
            p.error("Unknown config key or path override; paths stay on CLI")
        for key, value in requested.items():
            action = actions[key]
            typ = bool if key == "amp" else action.type or str
            if (
                type(value) not in (int, float)
                if typ is float
                else type(value) is not typ
            ):
                p.error(f"Invalid config type for {key}")
            if action.choices and value not in action.choices:
                p.error(f"Invalid config choice for {key}")
            setattr(args, key, value)
    for key in (
        "num_epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "max_candidates",
        "save_every",
    ):
        if getattr(args, key) <= 0:
            p.error(f"{key} must be positive")
    if args.num_workers < 0 or args.max_candidates > 128:
        p.error("Invalid worker/candidate count")
    for key in (
        "lr",
        "weight_decay",
        "ocr_dropout",
        "text_dropout",
        "selection_weight",
        "local_weight",
        "route_weight",
        "evidence_weight",
        "answer_weight",
        "min_gain",
    ):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0 or (key == "lr" and value == 0):
            p.error(f"Invalid {key}")
    if args.min_gain >= 1 or args.ocr_dropout >= 1 or args.text_dropout >= 1:
        p.error("Gain/dropout probabilities must be below one")
    if args.answer_weight and args.architecture != "hypotheses":
        p.error("answer-weight is only supported by hypotheses")
    if args.selector_recipe != "ranking" and args.architecture != "independent":
        p.error("selector-recipe applies only to independent architecture")
    if args.pointer_recipe != "categorical" and args.architecture != "pointer":
        p.error("pointer-recipe applies only to pointer architecture")
    return args


def main(argv=None):
    args = arguments(argv)
    rank, world, local = [
        int(os.environ.get(k, d))
        for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0))
    ]
    if args.batch_size % world or args.batch_size < world:
        raise ValueError("Global batch must be a positive multiple of world size")
    device = (
        torch.device("cuda", local)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    args.device = str(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        if device.type == "cuda":
            dist.init_process_group("nccl", device_id=device)
        else:
            dist.init_process_group("gloo")
    group = io_process_group() if world > 1 else None
    try:
        train(args, rank, world, device, group)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def train(args, rank, world, device, group):
    seed_everything(args.seed)
    model, saved, digest = load_ocr_model(
        args.parent_checkpoint,
        args.resume_checkpoint,
        architecture=args.architecture,
        evidence_mode=args.evidence_mode,
        ocr_dropout=args.ocr_dropout,
        text_dropout=args.text_dropout,
        **(
            {"min_gain": args.min_gain}
            if args.architecture in {"hypotheses", "independent"}
            else {}
        ),
        **(
            {"pointer_recipe": args.pointer_recipe}
            if args.architecture == "pointer"
            else {}
        ),
        **(
            {"selector_recipe": args.selector_recipe}
            if args.architecture == "independent"
            else {}
        ),
    )
    if saved and args.architecture != model.experiment_config.get(
        "architecture", "residual"
    ):
        raise ValueError(
            "Resume architecture differs; explicitly select the saved architecture"
        )
    dataset_class = {
        "residual": OCRGroundingDataset,
        "proposal": OCRRoutingDataset,
        "pyramid": OCRRoutingDataset,
        "hypotheses": OCRSelectionDataset,
        "independent": OCRSelectionDataset,
        "pointer": OCRPointerDataset,
    }[args.architecture]
    data = dataset_class(
        args.data_root,
        "train",
        args.cache_root,
        model.image_size,
        detail_size=model.detail_size,
        max_candidates=args.max_candidates,
        **(
            {"partition": "fit", "split_seed": args.seed}
            if args.architecture == "pointer"
            else {}
        ),
    )
    target_protocol = None
    if args.architecture == "pointer":
        target_protocol = {
            "version": POINTER_TARGET_VERSION,
            "source_supervision_digest": data.source_supervision_digest,
            "sufficient_iou": 0.5,
            "tolerance": 0.05,
            "retrieval_weight": 0.25,
            "token_weight": 0.25,
            "retrieval_min_iou": 0.1,
            "parent_quality_target": False,
            "pointer_recipe": args.pointer_recipe,
            "split_protocol": data.split_protocol,
        }
    elif args.architecture == "independent":
        target_protocol = {
            "version": SCOPE_TARGET_VERSION,
            "hypotheses": HYPOTHESIS_VERSION,
            "quality": "native-hard-IoU",
            "selector_recipe": args.selector_recipe,
            "ranking_gap": 0.05,
            "minimum_ocr_quality": 0.1,
            "ocr_choice_weight": 2.0,
            "choice_score_margin": 0.1,
            "quality_augmentation": model.experiment_config["quality_augmentation"],
        }
    elif args.architecture == "hypotheses":
        target_protocol = {
            "version": HYPOTHESIS_VERSION,
            "answer_match": "normalized-phrase-and-single-IoU-ge-0.5",
            "quality": "native-hard-IoU",
            "ranking_gap": 0.05,
            "quality_augmentation": "half-base-shift-or-morphology-v1",
        }
    elif args.architecture != "residual":
        target_protocol = {
            "version": TARGET_VERSION,
            "token_purity": TOKEN_PURITY,
            "route_coverage": ROUTE_COVERAGE,
        }
    config = dict(
        vars(args),
        cache_digest=data.cache_digest,
        cache_protocol=data.cache_protocol,
        world_size=world,
        experiment_config=model.experiment_config,
        parent_sha256=digest,
        schedule="constant",
        parent_frozen=True,
        supervision_digest=getattr(data, "supervision_digest", None),
        target_protocol=target_protocol,
    )
    config["source_sha256"] = {
        name: file_hash(Path(__file__).parent / name)
        for name in (
            "train_ocr.py",
            "train.py",
            "utils.py",
            "models/text_encoder.py",
            "models/detail_pyramid.py",
            "models/mask_decoder.py",
            "ocr_data.py",
            "ocr_cache.py",
            "models/ocr_evidence.py",
            "models/region_ops.py",
            "dataset.py",
            "losses.py",
            "checkpointing.py",
            "training/ocr_checkpoint.py",
            "models/ocr_model.py",
            "models/model.py",
            "training/ocr_objective.py",
            "models/ocr_routing.py",
            "models/evidence_ops.py",
            "models/ocr_routing_model.py",
            "ocr_routing_data.py",
            "training/ocr_routing_objective.py",
            "models/ocr_selection.py",
            "models/ocr_selection_model.py",
            "ocr_selection_data.py",
            "training/ocr_selection_objective.py",
            "evidence_selection.py",
            "models/ocr_scope.py",
            "models/ocr_scope_model.py",
            "training/ocr_scope_objective.py",
            "models/ocr_pointer.py",
            "models/ocr_pointer_model.py",
            "ocr_pointer_data.py",
            "training/ocr_pointer_objective.py",
            "eval_ocr.py",
            "predict_ocr.py",
            "diagnose_ocr_pointer.py",
        )
    }
    out = Path(args.output_dir)
    if saved:
        required = (
            "architecture",
            "route_weight",
            "evidence_weight",
            "target_protocol",
            "evidence_mode",
            "ocr_dropout",
            "text_dropout",
            "max_candidates",
            "batch_size",
            "gradient_accumulation_steps",
            "seed",
            "world_size",
            "cache_digest",
            "cache_protocol",
            "lr",
            "optimizer",
            "weight_decay",
            "selection_weight",
            "local_weight",
            "amp",
        )
        if args.architecture in {"hypotheses", "independent"}:
            required += ("answer_weight", "min_gain", "supervision_digest")
        if args.architecture == "independent":
            required += ("selector_recipe",)
        if args.architecture == "pointer":
            required += ("pointer_recipe", "supervision_digest")
        if (
            args.architecture == "residual"
            and "architecture" not in saved["run_config"]
        ):
            required = tuple(
                key
                for key in required
                if key
                not in {
                    "architecture",
                    "route_weight",
                    "evidence_weight",
                    "target_protocol",
                }
            )
        if any(saved["run_config"][key] != config[key] for key in required):
            raise ValueError(
                "Resume protocol differs; keep training settings and cache unchanged"
            )

    def prepare():
        if out.exists() and any(out.iterdir()):
            raise ValueError(
                "Use an empty output directory, including for a resumed segment"
            )
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(config, out / "run_config.json")

    run_primary_io(prepare, group)
    model.to(device)
    parameters = list(model.adapter.parameters())
    optimizer_type = (
        torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    )
    optimizer = optimizer_type(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start = 0
    if saved:
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scaler.load_state_dict(saved["scaler_state_dict"])
        start = saved["epoch"]
    if start >= args.num_epochs:
        raise ValueError("num-epochs must exceed resumed epoch")
    del saved
    if rank == 0:
        print(
            json.dumps(
                {
                    "adapter_parameters": sum(p.numel() for p in parameters),
                    "frozen_parent_parameters": sum(
                        p.numel() for p in model.parent.parameters()
                    ),
                    "parent_sha256": digest,
                }
            ),
            flush=True,
        )
    raw_model = model
    if world > 1:
        model = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    sampler = (
        DistributedSampler(data, world, rank, seed=args.seed) if world > 1 else None
    )
    generator = torch.Generator()
    loader = make_loader(
        data, args, args.batch_size // world, sampler, sampler is None, generator
    )
    if args.architecture == "pointer":
        objective = PointerObjective(args.pointer_recipe)
    elif args.architecture == "independent":
        objective = ScopeObjective(args.selector_recipe)
    elif args.architecture == "hypotheses":
        objective = SelectionObjective(args.answer_weight)
    elif args.architecture == "residual":
        objective = OCREvidenceObjective(args.selection_weight, args.local_weight)
    else:
        objective = RoutingObjective(
            args.selection_weight,
            args.local_weight,
            args.route_weight,
            args.evidence_weight,
        )
    for epoch in range(start, args.num_epochs):
        seed_everything(args.seed + epoch * world + rank)
        generator.manual_seed(args.seed + epoch * world + rank)
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = None
        skipped = 0
        local_batch = args.batch_size // world
        local_samples = len(sampler) if sampler else len(data)
        for index, batch in enumerate(
            tqdm(loader, desc=f"OCR epoch {epoch + 1}", disable=rank != 0)
        ):
            batch = to_device(batch, device)
            n = len(batch["filename"])
            accumulation = args.gradient_accumulation_steps
            begin = index // accumulation * accumulation
            end = min(begin + accumulation, len(loader))
            window_samples = min(end * local_batch, local_samples) - begin * local_batch
            update = index + 1 == end
            context = model.no_sync() if world > 1 and not update else nullcontext()
            with context:
                with torch.autocast(
                    device.type, enabled=args.amp and device.type == "cuda"
                ):
                    loss, parts = objective(model(batch), batch)
                if not torch.isfinite(loss):
                    # All ranks must fail together before a gradient collective.
                    finite = torch.tensor(0, device=device)
                else:
                    finite = torch.tensor(1, device=device)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError("Nonfinite adapter objective")
                scaler.scale(loss * n / window_samples).backward()
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped += int(scaler.get_scale() < scale)
                optimizer.zero_grad(set_to_none=True)
            values = [loss.detach(), *parts.values()]
            if totals is None:
                totals = torch.zeros(2 + len(parts), device=device, dtype=torch.float64)
            totals[0] += n
            totals[1:] += torch.stack(values).to(torch.float64) * n
        if world > 1:
            dist.all_reduce(totals)
        stats = (totals[1:] / totals[0]).tolist()
        record = dict(
            epoch=epoch + 1,
            train_loss=stats[0],
            **dict(zip(parts, stats[1:])),
            skipped_updates_local=skipped,
        )

        def write_epoch(record=record, epoch=epoch):
            with (out / "history.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.num_epochs:
                # Complete adapter resume state, excluding the fixed 1.73GB parent.
                save_checkpoint(
                    {
                        "version": checkpoint_version(raw_model),
                        "epoch": epoch + 1,
                        "parent_sha256": digest,
                        "adapter_state_dict": raw_model.adapter.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "experiment_config": raw_model.experiment_config,
                        "run_config": config,
                    },
                    out / f"checkpoint_epoch{epoch + 1}.pt",
                )

        run_primary_io(write_epoch, group)


if __name__ == "__main__":
    main()
