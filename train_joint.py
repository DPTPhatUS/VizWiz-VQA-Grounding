"""Fresh matched pyramid/control training with compact epoch-boundary recovery."""

import argparse
import json
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from tqdm.auto import tqdm

from checkpointing import io_process_group, run_primary_io
from dataset import make_loader
from models.joint_grounding import JointGroundingModel
from native_refinement_data import TARGET_VERSION, NativeRefinementDataset
from ocr_cache import atomic_json, file_hash
from train import seed_everything
from training.joint_checkpoint import (
    VERSION,
    compact_state,
    frozen_layout,
    read_joint,
    storage_estimate,
)
from training.joint_objective import DEFAULTS, OBJECTIVE_VERSION, JointObjective
from training_options import (
    TrainingArgumentParser,
    TrainingControls,
    public_arguments,
    recipe_config,
)
from utils import save_checkpoint, to_device


def arguments(argv=None):
    p = TrainingArgumentParser(description=__doc__, recipe_keys=DEFAULTS)
    p.add_argument("--data-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--variant", choices=("pyramid", "joint-crop"), required=True)
    p.add_argument(
        "--resume-checkpoint", help="latest_resume.pt, never a weights-only snapshot"
    )
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument(
        "--batch-size", type=int, default=2, help="Global microbatch across all GPUs"
    )
    p.add_argument("--gradient-accumulation-steps", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-candidates", type=int, default=24)
    p.add_argument("--ocr-dropout", type=float, default=0.2)
    p.add_argument("--text-dropout", type=float, default=0.2)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--tiny", action="store_true", help="Offline contracts only, not a research run"
    )
    args = p.parse_args(argv)
    if args.config:
        requested = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        if set(requested) & {
            "data_root",
            "cache_root",
            "output_dir",
            "resume_checkpoint",
        }:
            p.error(
                "Paths stay on CLI; config overrides training/model/objective settings"
            )
    for name in (
        "num_epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "max_candidates",
        "save_every",
    ):
        if getattr(args, name) <= 0:
            p.error(f"{name} must be positive")
    if args.num_workers < 0 or args.max_candidates > 128:
        p.error("Invalid worker/candidate count")
    for name in ("ocr_dropout", "text_dropout"):
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) < 1:
            p.error(f"{name} must be in [0,1)")
    return args


def main(argv=None):
    args = arguments(argv)
    rank, world, local = [
        int(os.environ.get(k, d))
        for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0))
    ]
    if args.batch_size < world or args.batch_size % world:
        raise ValueError("Global microbatch must be a positive multiple of world size")
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
    saved = None
    if args.resume_checkpoint:
        model, saved = read_joint(args.resume_checkpoint)
        if saved.get("kind") != "resume":
            raise ValueError(
                "Exact resume requires latest_resume.pt with optimizer/scaler/controls"
            )
        if (
            model.variant != args.variant
            or model.experiment_config["tiny"] != args.tiny
            or model.experiment_config["gradient_checkpointing"]
            != args.gradient_checkpointing
            or model.ocr_dropout != args.ocr_dropout
            or model.text_dropout != args.text_dropout
        ):
            raise ValueError("Resume model settings differ; use the original settings")
    else:
        model = JointGroundingModel(
            args.variant,
            tiny=args.tiny,
            gradient_checkpointing=args.gradient_checkpointing,
            ocr_dropout=args.ocr_dropout,
            text_dropout=args.text_dropout,
        )
    # Explicit defaults for this experiment; TrainingControls restores saved policies on resume.
    if args.freeze_vision_encoder is None:
        args.freeze_vision_encoder = False
    if args.freeze_text_encoder is None:
        args.freeze_text_encoder = True
    model.to(device)
    controls = TrainingControls(model, args, saved)
    optimizer, parameters = controls.optimizer, controls.parameters
    objective = JointObjective(**recipe_config(args, DEFAULTS))
    data = NativeRefinementDataset(
        args.data_root,
        "train",
        args.cache_root,
        model.image_size,
        detail_size=model.detail_size,
        max_candidates=args.max_candidates,
        mode="native_crop" if model.variant == "joint-crop" else "native_point",
    )
    sampler = (
        DistributedSampler(data, world, rank, seed=args.seed) if world > 1 else None
    )
    generator = torch.Generator()
    loader = make_loader(
        data, args, args.batch_size // world, sampler, sampler is None, generator
    )
    start = saved["epoch"] if saved else 0
    if start >= args.num_epochs:
        raise ValueError("num-epochs must exceed resumed epoch")
    controls.configure_schedule(
        args.num_epochs,
        math.ceil(len(loader) / args.gradient_accumulation_steps),
        start,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    if saved:
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scaler.load_state_dict(saved["scaler_state_dict"])
    omitted = frozen_layout(model)
    protocol = {
        "data_version": TARGET_VERSION,
        "cache_digest": data.cache_digest,
        "cache_protocol": data.cache_protocol,
        "supervision_digest": data.supervision_digest,
        "training_images": sorted(data.names),
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "world_size": world,
        "seed": args.seed,
        "amp": args.amp,
        "max_candidates": args.max_candidates,
        "objective_version": OBJECTIVE_VERSION,
        "objective": objective.settings,
        "random_policy": "reseed-each-epoch-with-seed+epoch*world+rank",
        "sampler_policy": "DistributedSampler-padding-when-not-divisible",
    }
    if saved and (
        saved["run_config"]["protocol"] != protocol
        or saved["omitted_frozen_encoders"] != omitted
    ):
        raise ValueError("Resume data/objective/batch/frozen-encoder protocol differs")
    snapshots = math.ceil(args.num_epochs / args.save_every)
    config = dict(
        public_arguments(args),
        experiment_config=model.experiment_config,
        protocol=protocol,
        optimization=controls.config,
        storage_estimate=storage_estimate(model, parameters, omitted, snapshots),
        omitted_frozen_encoders=omitted,
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in parameters),
    )
    sources = (
        "train_joint.py",
        "eval_joint.py",
        "predict_joint.py",
        "training_options.py",
        "models/joint_grounding.py",
        "models/model.py",
        "models/image_encoder.py",
        "models/text_encoder.py",
        "models/detail_pyramid.py",
        "models/mask_decoder.py",
        "models/native_refiner.py",
        "models/region_ops.py",
        "models/ocr_evidence.py",
        "native_refinement_data.py",
        "native_refinement_ops.py",
        "ocr_data.py",
        "ocr_routing_data.py",
        "ocr_cache.py",
        "dataset.py",
        "checkpointing.py",
        "utils.py",
        "training/joint_checkpoint.py",
        "training/joint_objective.py",
    )
    config["source_sha256"] = {
        name: file_hash(Path(__file__).parent / name) for name in sources
    }
    history = list(saved.get("history", [])) if saved else []
    out = Path(args.output_dir)

    def prepare():
        if out.exists() and any(out.iterdir()):
            raise ValueError(
                "Use an empty output directory, including for a resumed segment"
            )
        out.mkdir(parents=True, exist_ok=True)
        # Include old/new resume coexistence, serializer margin and room for results.
        required = (
            math.ceil(config["storage_estimate"]["peak_checkpoint_bytes"] * 1.05)
            + 1024**3
        )
        available = shutil.disk_usage(out).free
        if available < required:
            raise OSError(
                f"Projected checkpoint peak plus 1 GiB result reserve needs {required / 1024**3:.2f} GiB; "
                f"only {available / 1024**3:.2f} GiB is free. Use --save-every 10 in both runs "
                "or free disk space before starting. No checkpoints were deleted."
            )
        config["storage_estimate"].update(
            start_free_bytes=available, required_with_reserve_bytes=required
        )
        atomic_json(config, out / "run_config.json")
        (out / "history.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in history)
        )
        print(
            json.dumps(
                {
                    k: config[k]
                    for k in (
                        "total_parameters",
                        "trainable_parameters",
                        "storage_estimate",
                        "optimization",
                    )
                }
            ),
            flush=True,
        )

    run_primary_io(prepare, group)
    del saved
    raw = model
    if world > 1:
        model = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    for epoch in range(start, args.num_epochs):
        seed_everything(args.seed + epoch * world + rank)
        generator.manual_seed(args.seed + epoch * world + rank)
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        controls.set_encoder_modes()
        controls.log_learning_rates(epoch + 1)
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(6, device=device, dtype=torch.float64)
        skipped = 0
        local_batch = args.batch_size // world
        local_samples = len(sampler) if sampler is not None else len(data)
        for index, batch in enumerate(
            tqdm(loader, desc=f"{args.variant} epoch {epoch + 1}", disable=rank != 0)
        ):
            batch = to_device(batch, device)
            n = len(batch["filename"])
            begin = (
                index
                // args.gradient_accumulation_steps
                * args.gradient_accumulation_steps
            )
            end = min(begin + args.gradient_accumulation_steps, len(loader))
            window_samples = min(end * local_batch, local_samples) - begin * local_batch
            update = index + 1 == end
            context = model.no_sync() if world > 1 and not update else nullcontext()
            with context:
                with torch.autocast(
                    device.type, enabled=args.amp and device.type == "cuda"
                ):
                    loss, parts = objective(model(batch), batch)
                finite = torch.isfinite(loss).to(torch.int32)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError("Nonfinite joint native objective")
                scaler.scale(loss * n / window_samples).backward()
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    parameters, 1.0, error_if_nonfinite=not scaler.is_enabled()
                )
                scale = scaler.get_scale()
                controls.step(scaler)
                skipped += int(scaler.get_scale() < scale)
                optimizer.zero_grad(set_to_none=True)
            totals[0] += n
            totals[1:] += torch.stack([loss.detach(), *parts.values()]).double() * n
        if world > 1:
            dist.all_reduce(totals)
        stats = (totals[1:] / totals[0]).tolist()
        record = dict(
            epoch=epoch + 1,
            train_loss=stats[0],
            **dict(zip(parts, stats[1:])),
            optimizer_steps=controls.completed_steps,
            skipped_updates_local=skipped,
        )
        history.append(record)

        def write_epoch(record=record, epoch=epoch):
            with (out / "history.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.num_epochs:
                common = {
                    "version": VERSION,
                    "epoch": epoch + 1,
                    "model_state_dict": compact_state(raw, omitted),
                    "omitted_frozen_encoders": omitted,
                    "experiment_config": raw.experiment_config,
                    "run_config": config,
                }
                save_checkpoint(
                    dict(common, kind="weights"),
                    out / f"checkpoint_epoch{epoch + 1}.pt",
                )
                save_checkpoint(
                    dict(
                        common,
                        kind="resume",
                        optimizer_state_dict=optimizer.state_dict(),
                        scaler_state_dict=scaler.state_dict(),
                        training_controls=controls.state_dict(),
                        history=history,
                    ),
                    out / "latest_resume.pt",
                )

        run_primary_io(write_epoch, group)


if __name__ == "__main__":
    main()
