"""Train this branch's grounding experiment on CPU, CUDA, or torchrun DDP."""
import argparse
import json
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DistributedSampler
from tqdm.auto import tqdm
from dataset import VizWizGroundingDataset, make_loader
from utils import (to_device, read_checkpoint, load_model_weights, initialize_weights,
                   checkpoint_model_kwargs, save_checkpoint)
from models.model import GroundingModel
from losses import SupervisedObjective, DistillationObjective, AnswerAlignmentObjective


IMAGE_SIZE = 336

TRAINING_DEFAULTS = {
    "training_recipe": "full", "loss_version": "legacy", "detach_predicted_answer": False,
    "dice_weight": .5, "latent_weight": .5, "oracle_weight": .5, "question_weight": .5,
    "kd_weight": 1., "attn_weight": .2, "gain_mix": .75,
    "warmup_epochs": 0, "answer_transition_epochs": 0,
    "freeze_text_encoder": False, "question_only": False,
}


def objective_version(args):
    if args.question_only:
        return "question-supervised-v1"
    versions = {"full": "latent-gain-v2" if args.loss_version == "legacy" else "latent-gain-v3",
                "grounding-only": "latent-supervised-v1",
                "alignment": "latent-alignment-v1" if args.loss_version == "legacy" else "latent-alignment-v2",
                "warmup": "latent-warmup-v1" if args.loss_version == "legacy" else "latent-warmup-v2"}
    return versions[args.training_recipe] + ("-detached" if args.detach_predicted_answer else "")


def resolve_training_arguments(args, saved=None):
    """Restore recipe settings on resume; never silently switch objective or schedule."""
    requested = {}
    if args.config:
        requested = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        if not isinstance(requested, dict) or set(requested) - TRAINING_DEFAULTS.keys():
            raise ValueError("Config must be a JSON object containing only keys from TRAINING_DEFAULTS")
        for key, value in requested.items():
            default = TRAINING_DEFAULTS[key]
            numeric = type(default) is float and type(value) in (int, float)
            if not numeric and type(value) is not type(default):
                raise ValueError(f"Config {key} has the wrong value type")
    for key in ("training_recipe", "loss_version"):
        value = getattr(args, key)
        if value is not None:
            if key in requested and requested[key] != value:
                raise ValueError(f"Explicit {key} conflicts with config")
            requested[key] = value
    if saved is not None:
        run = saved["run_config"]
        if run.get("teacher_training") or run.get("teacher_checkpoint"):
            raise ValueError("Historical teacher runs must resume with personal/legacy/latent_training.py or latent_teacher.py; the main trainer is single-stage")
        settings = dict(TRAINING_DEFAULTS)
        settings.update(run.get("loss_config", {}))
        settings.update({k: run[k] for k in TRAINING_DEFAULTS if k in run})
        settings.update(run.get("training_config", {}))
        if "training_recipe" not in run and "training_config" not in run:
            settings["training_recipe"] = "grounding-only" if run.get("supervised_only") else "full"
        for key in TRAINING_DEFAULTS:
            if key in requested and requested[key] != settings[key]:
                raise ValueError(f"Resume {key} differs from checkpoint; use --init-checkpoint for a new experiment")
    else:
        settings = dict(TRAINING_DEFAULTS)
        settings.update(requested)
        if settings["training_recipe"] == "warmup":
            for key in ("warmup_epochs", "answer_transition_epochs"):
                if key not in requested:
                    settings[key] = 5
    for key in TRAINING_DEFAULTS:
        setattr(args, key, settings[key])
    if args.training_recipe not in ("full", "grounding-only", "alignment", "warmup") or args.loss_version not in ("legacy", "stable"):
        raise ValueError("Checkpoint contains an unsupported training recipe or loss version")
    args.supervised_only = args.training_recipe == "grounding-only"
    if args.question_only and args.training_recipe != "grounding-only":
        raise ValueError("question_only requires the grounding-only recipe")
    if saved is not None:
        recorded_version = saved["run_config"].get("objective_version")
        if recorded_version and recorded_version != objective_version(args):
            raise ValueError("Saved objective version differs; use --init-checkpoint for mixed-objective continuation")
    if args.training_recipe != "full" and any(k in requested for k in
            ("oracle_weight", "question_weight", "kd_weight", "attn_weight", "gain_mix")):
        raise ValueError("Oracle/Q-reference/KD/attention config weights apply only to the full recipe")
    for key in ("dice_weight", "latent_weight", "oracle_weight", "question_weight", "kd_weight", "attn_weight"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not math.isfinite(args.gain_mix) or not 0 <= args.gain_mix <= 1:
        raise ValueError("gain-mix must be in [0,1]")
    if min(args.warmup_epochs, args.answer_transition_epochs) < 0:
        raise ValueError("Warm-up/transition epochs must be nonnegative")
    if args.training_recipe == "warmup":
        if args.warmup_epochs < 1 or args.num_epochs < args.warmup_epochs + max(args.answer_transition_epochs, 1):
            raise ValueError("Warmup needs positive warm-up epochs and enough epochs to reach predicted-only training")
    elif args.warmup_epochs or args.answer_transition_epochs:
        raise ValueError("Warm-up/transition settings require --training-recipe warmup")
    if args.detach_predicted_answer and (args.training_recipe == "grounding-only" or
            (args.latent_weight == 0 and (args.training_recipe != "full" or args.attn_weight == 0))):
        raise ValueError("Detaching A′ requires a predictor supervision loss; grounding-only must not detach")


def build_objective(args):
    if args.training_recipe == "grounding-only":
        return SupervisedObjective(args.dice_weight)
    common = {k: getattr(args, k) for k in ("dice_weight", "latent_weight", "loss_version", "detach_predicted_answer")}
    if args.training_recipe == "full":
        return DistillationObjective(**common, **{k: getattr(args, k) for k in
            ("oracle_weight", "question_weight", "kd_weight", "attn_weight", "gain_mix")})
    return AnswerAlignmentObjective(**common, warmup_epochs=args.warmup_epochs,
                                    transition_epochs=args.answer_transition_epochs)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def training_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/vizwiz")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4, help="Global batch size across all ranks")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save a resumable checkpoint every N epochs; 0 disables periodic saves")
    parser.add_argument("--resume-checkpoint", help="Resume a full checkpoint; raw weights use --init-checkpoint")
    parser.add_argument("--init-checkpoint", help="Optional compact control weights; starts a new run")
    parser.add_argument("--training-recipe", choices=["full", "grounding-only", "alignment", "warmup"], default=None,
                        help="Fresh default full (no warm-up); resume restores saved settings")
    parser.add_argument("--loss-version", choices=["legacy", "stable"], default=None,
                        help="Fresh default legacy; stable selects audited loss calculations")
    parser.add_argument("--config", help="Optional JSON for loss weights, detachment, warm-up or freezing text; no file needed for normal runs")
    parser.add_argument("--spatial-upsample-stages", type=int, nargs="*", default=None,
                        help="Upsampling stages to apply spatial evidence interaction (default: 0 1)")
    parser.add_argument("--spatial-detail", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable semantic-guided region-pixel detail fusion (default: True)")
    parser.add_argument("--detail-upsample-stages", type=int, nargs="*", default=None,
                        help="Upsampling stages to apply region-pixel detail fusion (default: 2 3)")
    parser.add_argument("--detail-dim", type=int, default=None,
                        help="Channel dimension of detail features from pixel detail stem (default: 16)")
    return parser


def check_arguments(args):
    from models.mask_decoder import (resolve_spatial_upsample_stages, resolve_detail_upsample_stages,
                                     validate_detail_dim)
    if args.spatial_upsample_stages is not None:
        resolve_spatial_upsample_stages(args.spatial_upsample_stages)
    if getattr(args, "detail_upsample_stages", None) is not None:
        resolve_detail_upsample_stages(args.detail_upsample_stages)
    if args.detail_dim is not None:
        validate_detail_dim(args.detail_dim)
    if min(args.num_epochs, args.batch_size) <= 0:
        raise ValueError("Epochs and batch size must be positive")
    if args.save_every < 0:
        raise ValueError("save-every cannot be negative")
    if args.num_workers < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Workers cannot be negative; learning rate must be positive")
    if args.resume_checkpoint and getattr(args, "init_checkpoint", None):
        raise ValueError("Use either resume or weight initialization, not both")
    for name in ('resume_checkpoint', 'init_checkpoint', 'config'):
        path = getattr(args, name, None)
        if path and not Path(path).is_file():
            raise ValueError(f"{name} does not exist: {path}")


def train_main(argv=None):
    args = training_parser().parse_args(argv)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    check_arguments(args)
    rank, world, local = (int(os.environ.get(k,default)) for k,default in (
        ("RANK",0),("WORLD_SIZE",1),("LOCAL_RANK",0)))
    if args.batch_size < world or args.batch_size % world != 0:
        raise ValueError("Global batch size must be a positive multiple of world size")
    distributed = world > 1
    device = torch.device(args.device)
    if distributed:
        if device.type == "cuda":
            device = torch.device("cuda", local)
            torch.cuda.set_device(device)
            dist.init_process_group("nccl", device_id=device)
        else:
            dist.init_process_group("gloo")
    try:
        _train(args, rank, world, distributed, device)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def resolve_model_arguments(args, checkpoint=None):
    """Restore architecture on resume; reject any conflicting explicit override."""
    from models.mask_decoder import resolve_spatial_upsample_stages, resolve_detail_upsample_stages
    requested = {key: getattr(args, key) for key in (
        "spatial_upsample_stages", "spatial_detail", "detail_upsample_stages", "detail_dim")
        if getattr(args, key) is not None}
    for key, resolve in (("spatial_upsample_stages", resolve_spatial_upsample_stages),
                         ("detail_upsample_stages", resolve_detail_upsample_stages)):
        if key in requested:
            requested[key] = resolve(requested[key])
    if checkpoint is None:
        return requested
    options = checkpoint_model_kwargs(checkpoint)
    for key, value in requested.items():
        if value != options.get(key, ()):
            raise ValueError(f"Resume {key} differs from the checkpoint; start a new experiment instead")
    return options


def _train(args, rank, world, distributed, device):
    checkpoint = read_checkpoint(args.resume_checkpoint) if args.resume_checkpoint else None
    model_options = resolve_model_arguments(args, checkpoint)
    resolve_training_arguments(args, checkpoint)
    seed_everything(args.seed)
    run_config = vars(args).copy()
    run_config.update(world_size=world, image_size=IMAGE_SIZE,
                      teacher_training=False, text_mode="question", answer_dropout=.5)
    version = objective_version(args)
    run_config["objective_version"] = version
    run_config["training_config"] = {k: getattr(args, k) for k in TRAINING_DEFAULTS}
    out = Path(args.output_dir)
    history_path = out / "history.jsonl"
    if not checkpoint and (history_path.exists() or any(out.glob("*.pt"))):
        raise ValueError("Output already contains a run; choose a new directory or resume explicitly")
    if checkpoint and history_path.exists():
        records = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
        if records and records[-1]["epoch"] != checkpoint["epoch"]:
            raise ValueError("Existing history does not end at the resumed epoch; use a new output directory")
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    train_set = VizWizGroundingDataset(args.data_root, "train", IMAGE_SIZE, text_mode="question")
    model = GroundingModel(**model_options)
    if getattr(args, "init_checkpoint", None):
        initialize_weights(args.init_checkpoint, model)
    model = model.to(device)
    model.coarse.question_only = args.question_only
    if args.freeze_text_encoder:
        model.coarse.set_text_encoder_frozen()
    objective = build_objective(args)
    if isinstance(objective, DistillationObjective):
        run_config["loss_config"] = {name: getattr(objective, name) for name in
                                   ("dice_weight", "oracle_weight", "question_weight", "latent_weight", "attn_weight", "gain_mix")}
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    start = 0
    if args.resume_checkpoint:
        load_model_weights(model, checkpoint)
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError("Raw model weights have no optimizer state; use --init-checkpoint for a new run")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        # CPU checkpoints contain an empty disabled-scaler state. A CUDA continuation
        # starts a fresh scaler; a disabled CPU scaler safely ignores CUDA state.
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start = checkpoint["epoch"]
        del checkpoint
        if start >= args.num_epochs:
            raise ValueError("num-epochs must exceed the resumed epoch")
    if distributed:
        if device.type == "cuda":
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    sampler = DistributedSampler(train_set,num_replicas=world,rank=rank,seed=args.seed) if distributed else None
    generator = torch.Generator()
    train_loader = make_loader(train_set,args,args.batch_size//world,sampler,sampler is None,generator)
    run_config["experiment_config"] = unwrap(model).experiment_config
    run_config["spatial_evidence"] = unwrap(model).coarse.decoder.spatial_evidence_enabled
    run_config["spatial_upsample_stages"] = list(unwrap(model).coarse.decoder.spatial_upsample_stages)
    run_config["spatial_detail"] = unwrap(model).coarse.decoder.spatial_detail_enabled
    run_config["detail_upsample_stages"] = list(unwrap(model).coarse.decoder.detail_upsample_stages)
    run_config["detail_dim"] = unwrap(model).coarse.decoder.detail_dim
    run_config["tiny"] = getattr(unwrap(model), "coarse", unwrap(model)).experiment_config.get("tiny", False)
    if rank == 0:
        print(json.dumps({"parameters":sum(p.numel() for p in unwrap(model).parameters()),
                          "trainable":sum(p.numel() for p in parameters),"device":str(device),
                          "experiment":unwrap(model).experiment_config}),flush=True)
    for epoch in range(start,args.num_epochs):
        # Epoch-keyed RNG permits reproducible continuation, including loader/answer dropout.
        seed_everything(args.seed+epoch*world+rank)
        generator.manual_seed(args.seed+epoch*world+rank)
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        totals = torch.zeros(2,device=device,dtype=torch.float64)  # (2,): [weighted_loss_sum, sample_count]
        component_sums = {}
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}",
                          unit="batch", disable=rank != 0, dynamic_ncols=True):
            batch = to_device(batch,device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,enabled=device.type=="cuda"):
                loss, components = objective(model, batch, epoch=epoch)
            components["answer_available_fraction"] = batch["answer_available"].float().mean()
            if device.type != "cuda" and not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters,1.)
            scaler.step(optimizer); scaler.update()
            n = len(batch["filename"])
            totals[0] += loss.detach().to(torch.float64) * n
            totals[1] += n
            for name,value in components.items():
                val64 = value.detach().to(device=device, dtype=torch.float64) * n
                component_sums[name] = component_sums[name] + val64 if name in component_sums else val64
        names = sorted(component_sums)
        stats = torch.stack([totals[0], totals[1], *(component_sums[k] for k in names)])  # (2 + C,)
        if distributed:
            dist.all_reduce(stats)
        if not torch.isfinite(stats[0]):
            raise FloatingPointError("Nonfinite training loss")
        values = (stats / stats[1]).tolist()
        record = {"epoch":epoch+1,"train_loss":values[0],"objective_version":version}
        record["training_config"] = run_config["training_config"]
        for name, val in zip(names, values[2:]):
            record[name] = val
        if rank == 0:
            print(json.dumps(record),flush=True)
            with (out/"history.jsonl").open("a") as handle:
                handle.write(json.dumps(record)+"\n")
            if args.save_every and (epoch + 1) % args.save_every == 0:
                checkpoint_path = out / f"checkpoint_epoch{epoch + 1}.pt"
                save_checkpoint({"epoch": epoch + 1,
                            "model_state_dict": unwrap(model).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            "experiment_config": unwrap(model).experiment_config,
                            "run_config": run_config}, checkpoint_path)
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
