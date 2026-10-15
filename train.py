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
from utils import to_device, read_checkpoint, load_model_weights, initialize_weights, is_tiny_state, LATENT_ANSWER_KEYS
from models.model import GroundingModel
from losses import SupervisedObjective, DistillationObjective, AnswerAlignmentObjective
from models.model import CompactGroundingModel, TeacherModel


def load_teacher(path, device):
    saved = read_checkpoint(path)
    config, run = saved.get("experiment_config") or {}, saved["run_config"]
    state = saved["model_state_dict"]
    if (
        (config and config.get("experiment") != "controls")
        or (run and (run.get("text_mode") != "dropout" or not 0 < run.get("answer_dropout", 0) < 1))
        or any(k.startswith("coarse.") for k in state)
    ):
        raise ValueError("Teacher must be a corrected model trained with answer dropout")
    if run.get("image_size", IMAGE_SIZE) != IMAGE_SIZE:
        raise ValueError("Teacher/student image sizes must match")
    if not LATENT_ANSWER_KEYS <= set(state):
        raise ValueError("Frozen teacher lacks trained latent-answer tensors; compact checkpoints may only initialize a student via --init-checkpoint")
    tiny = config.get("tiny", run.get("tiny", is_tiny_state(state)))
    teacher = TeacherModel(tiny=tiny)
    load_model_weights(teacher, saved)
    return teacher.to(device)


IMAGE_SIZE = 336

TRAINING_DEFAULTS = {
    "training_recipe": "full", "loss_version": "legacy", "detach_predicted_answer": False,
    "dice_weight": .5, "latent_weight": .5, "oracle_weight": .5, "question_weight": .5,
    "kd_weight": 1., "attn_weight": .2, "gain_mix": .75,
    "warmup_epochs": 0, "answer_transition_epochs": 0,
}


def objective_version(args):
    if args.teacher_training:
        return "supervised-teacher-v1"
    versions = {"full": "latent-gain-v2" if args.loss_version == "legacy" else "latent-gain-v3",
                "grounding-only": "latent-supervised-v1",
                "alignment": "latent-alignment-v1" if args.loss_version == "legacy" else "latent-alignment-v2",
                "warmup": "latent-warmup-v1" if args.loss_version == "legacy" else "latent-warmup-v2"}
    return versions[args.training_recipe] + ("-detached" if args.detach_predicted_answer else "")


def resolve_training_arguments(args, saved=None):
    """Restore recipe settings on resume; never silently switch objective or schedule."""
    if args.teacher_training:
        return
    requested = {key: getattr(args, key) for key in TRAINING_DEFAULTS}
    if args.supervised_only:
        if requested["training_recipe"] not in (None, "grounding-only"):
            raise ValueError("--supervised-only is an alias for --training-recipe grounding-only")
        requested["training_recipe"] = "grounding-only"
    if saved is not None:
        run = saved["run_config"]
        if run.get("teacher_training"):
            raise ValueError("A teacher checkpoint cannot resume student training")
        settings = dict(TRAINING_DEFAULTS)
        settings.update(run.get("loss_config", {}))
        settings.update({k: run[k] for k in TRAINING_DEFAULTS if k in run})
        settings.update(run.get("training_config", {}))
        if "training_recipe" not in run and "training_config" not in run:
            settings["training_recipe"] = "grounding-only" if run.get("supervised_only") else "full"
        for key in TRAINING_DEFAULTS:
            if requested[key] is not None and requested[key] != settings[key]:
                raise ValueError(f"Resume {key} differs from checkpoint; use --init-checkpoint for a new experiment")
        if args.teacher_checkpoint is None:
            args.teacher_checkpoint = run.get("teacher_checkpoint")
        elif not run.get("teacher_checkpoint"):
            raise ValueError("Resume cannot add an external teacher; use --init-checkpoint")
    else:
        settings = dict(TRAINING_DEFAULTS)
        settings.update({k: v for k, v in requested.items() if v is not None})
        if settings["training_recipe"] == "warmup":
            for key in ("warmup_epochs", "answer_transition_epochs"):
                if requested[key] is None:
                    settings[key] = 5
    for key in TRAINING_DEFAULTS:
        setattr(args, key, settings[key])
    if args.training_recipe not in ("full", "grounding-only", "alignment", "warmup") or args.loss_version not in ("legacy", "stable"):
        raise ValueError("Checkpoint contains an unsupported training recipe or loss version")
    args.supervised_only = args.training_recipe == "grounding-only"
    if saved is not None:
        recorded_version = saved["run_config"].get("objective_version")
        if recorded_version and recorded_version != objective_version(args):
            raise ValueError("Saved objective version differs; use --init-checkpoint for mixed-objective continuation")
    if args.training_recipe != "full" and args.teacher_checkpoint:
        raise ValueError("External teachers are supported only by the full recipe")
    if args.training_recipe != "full" and any(requested[k] is not None for k in
            ("oracle_weight", "question_weight", "kd_weight", "attn_weight", "gain_mix")):
        raise ValueError("Oracle/Q-reference/KD/attention weight flags apply only to the full recipe")
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
        raise ValueError("Warm-up/transition flags require --training-recipe warmup")
    if args.detach_predicted_answer and (args.training_recipe == "grounding-only" or
            (args.latent_weight == 0 and (args.training_recipe != "full" or args.attn_weight == 0))):
        raise ValueError("Detaching A′ requires a predictor supervision loss; grounding-only must not detach")


def build_objective(args, teacher=None):
    if args.teacher_training:
        return SupervisedObjective()
    if args.training_recipe == "grounding-only":
        return SupervisedObjective(args.dice_weight)
    common = {k: getattr(args, k) for k in ("dice_weight", "latent_weight", "loss_version", "detach_predicted_answer")}
    if args.training_recipe == "full":
        return DistillationObjective(teacher, **common, **{k: getattr(args, k) for k in
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


def training_parser(teacher_training=False, *, advanced=False):
    parser = argparse.ArgumentParser(description="Train answer-dropout teacher" if teacher_training else __doc__)
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
    if not teacher_training:
        parser.add_argument("--help-advanced", action="store_true", help="Show loss weights, warm-up schedule and compatibility options")
        recipes = parser.add_argument_group("training recipe", "Choose one recipe; defaults restore automatically on resume.")
        recipes.add_argument("--training-recipe", choices=["full", "grounding-only", "alignment", "warmup"], default=None,
                            help="full: all losses, no warm-up (legacy default); grounding-only: mask loss; alignment: mask + answer alignment; warmup: oracle-to-predicted schedule")
        recipes.add_argument("--loss-version", choices=["legacy", "stable"], default=None,
                            help="Use stable for new comparisons; legacy preserves old training (fresh default)")
        tuning = parser.add_argument_group("advanced training options" if advanced else argparse.SUPPRESS)
        tuning.add_argument("--teacher-checkpoint", default=None, help="Optional fully trained latent teacher, full recipe only" if advanced else argparse.SUPPRESS)
        tuning.add_argument("--supervised-only", action="store_true", default=None,
                            help="Compatibility alias for --training-recipe grounding-only" if advanced else argparse.SUPPRESS)
        tuning.add_argument("--detach-predicted-answer", action=argparse.BooleanOptionalAction, default=None,
                            help="Cut the direct grounding gradient through A'; default off" if advanced else argparse.SUPPRESS)
        tuning.add_argument("--warmup-epochs", type=int, default=None,
                            help="Oracle-only epochs for warmup recipe; fresh default 5" if advanced else argparse.SUPPRESS)
        tuning.add_argument("--answer-transition-epochs", type=int, default=None,
                            help="Oracle-to-predicted transition for warmup recipe; fresh default 5" if advanced else argparse.SUPPRESS)
        for flag in ("dice-weight", "latent-weight", "oracle-weight", "question-weight", "kd-weight", "attn-weight", "gain-mix"):
            default = TRAINING_DEFAULTS[flag.replace("-", "_")]
            scope = "Full recipe only; " if flag in ("oracle-weight", "question-weight", "kd-weight", "attn-weight", "gain-mix") else ""
            tuning.add_argument(f"--{flag}", type=float, default=None,
                                help=f"{scope}fresh default {default:g}" if advanced else argparse.SUPPRESS)
    return parser


def check_arguments(args):
    if min(args.num_epochs, args.batch_size) <= 0:
        raise ValueError("Epochs and batch size must be positive")
    if args.save_every < 0:
        raise ValueError("save-every cannot be negative")
    if args.num_workers < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Workers cannot be negative; learning rate must be positive")
    if args.resume_checkpoint and getattr(args, "init_checkpoint", None):
        raise ValueError("Use either resume or weight initialization, not both")
    if getattr(args, "supervised_only", False) and getattr(args, "teacher_checkpoint", None):
        raise ValueError("A supervised-only control must not use an external teacher")
    for name in ('resume_checkpoint', 'init_checkpoint', 'teacher_checkpoint'):
        path = getattr(args, name, None)
        if path and not Path(path).is_file():
            raise ValueError(f"{name} does not exist: {path}")


def train_main(argv=None, teacher_training=False):
    args = training_parser(teacher_training).parse_args(argv)
    if not teacher_training:
        if args.help_advanced:
            training_parser(advanced=True).print_help()
            return
        del args.help_advanced
    args.teacher_training = teacher_training
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


def _train(args, rank, world, distributed, device):
    checkpoint = read_checkpoint(args.resume_checkpoint) if args.resume_checkpoint else None
    resolve_training_arguments(args, checkpoint)
    check_arguments(args)  # validate a teacher path restored from checkpoint metadata
    seed_everything(args.seed)
    run_config = vars(args).copy()
    run_config.update(world_size=world, image_size=IMAGE_SIZE,
                      text_mode="dropout" if args.teacher_training else "question", answer_dropout=.5)
    version = objective_version(args)
    run_config["objective_version"] = version
    if not args.teacher_training:
        run_config["training_config"] = {k: getattr(args, k) for k in TRAINING_DEFAULTS}
    out = Path(args.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    train_set = VizWizGroundingDataset(args.data_root, "train", IMAGE_SIZE,
                text_mode="dropout" if args.teacher_training else "question")
    tiny = checkpoint["run_config"].get("tiny", is_tiny_state(checkpoint["model_state_dict"])) if checkpoint else False
    model = TeacherModel(tiny=tiny) if args.teacher_training else GroundingModel(tiny=tiny)
    if getattr(args, "init_checkpoint", None):
        initialize_weights(args.init_checkpoint, model)
    model = model.to(device)
    teacher = load_teacher(args.teacher_checkpoint, device) if getattr(args, "teacher_checkpoint", None) else None
    objective = build_objective(args, teacher)
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
        if not args.teacher_training:
            record["training_config"] = run_config["training_config"]
        for name, val in zip(names, values[2:]):
            record[name] = val
        if rank == 0:
            print(json.dumps(record),flush=True)
            with (out/"history.jsonl").open("a") as handle:
                handle.write(json.dumps(record)+"\n")
            if args.save_every and (epoch + 1) % args.save_every == 0:
                checkpoint_path = out / f"checkpoint_epoch{epoch + 1}.pt"
                torch.save({"epoch": epoch + 1,
                            "model_state_dict": unwrap(model).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            "experiment_config": unwrap(model).experiment_config,
                            "run_config": run_config}, checkpoint_path, _use_new_zipfile_serialization=False)
                print(f"Checkpoint saved: {checkpoint_path}", flush=True)
    if rank == 0:
        final_path = out / f"model_final_epoch{args.num_epochs}.pt"
        torch.save(unwrap(model).state_dict(), final_path, _use_new_zipfile_serialization=False)
        print(f"Final model saved: {final_path}", flush=True)
    if distributed:
        dist.barrier()

main = train_main

if __name__ == "__main__":
    main()
