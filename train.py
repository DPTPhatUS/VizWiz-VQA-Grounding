"""Train this branch's grounding experiment on CPU, CUDA, or torchrun DDP."""
import math
from training_options import (
    TrainingArgumentParser,
    TrainingControls,
    public_arguments,
    restore_control_arguments,
)
import json
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
from utils import to_device, read_checkpoint, load_model_weights, initialize_weights, is_tiny_state
from models.model import GroundingModel
from losses import SupervisedObjective, DistillationObjective
from models.model import CompactGroundingModel, TeacherModel


def load_teacher(path, device):
    saved = read_checkpoint(path)
    config, run = saved.get("experiment_config") or {}, saved["run_config"]
    if (config and config.get("experiment") != "controls") or (run and (run.get("text_mode") != "dropout" or not 0 < run.get("answer_dropout", 0) < 1)):
        raise ValueError("Teacher must be a corrected model trained with answer dropout")
    if run.get("image_size", IMAGE_SIZE) != IMAGE_SIZE:
        raise ValueError("Teacher/student image sizes must match")
    teacher_type = CompactGroundingModel if config.get("architecture") == "compact" else TeacherModel
    tiny = config.get("tiny", run.get("tiny", is_tiny_state(saved["model_state_dict"])))
    teacher = teacher_type(tiny=tiny)
    load_model_weights(teacher, saved)
    return teacher.to(device)


IMAGE_SIZE = 336


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def training_parser(teacher_training=False):
    parser = TrainingArgumentParser(description="Train answer-dropout teacher" if teacher_training else __doc__)
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
        parser.add_argument("--teacher-checkpoint", help="Answer-dropout compact teacher from train_teacher.py")
        parser.add_argument("--supervised-only", action="store_true", help="Train the same Q-only compact student with BCE only")
    return parser


def check_arguments(args):
    if min(args.num_epochs, args.batch_size) <= 0:
        raise ValueError("Epochs and batch size must be positive")
    if args.save_every < 0:
        raise ValueError("save-every cannot be negative")
    if args.num_workers < 0 or args.lr <= 0:
        raise ValueError("Workers cannot be negative; learning rate must be positive")
    if args.resume_checkpoint and getattr(args, "init_checkpoint", None):
        raise ValueError("Use either resume or weight initialization, not both")
    if not getattr(args, "teacher_training", False):
        if not args.supervised_only and not args.teacher_checkpoint:
            raise ValueError("Distillation requires --teacher-checkpoint; use --supervised-only for the compact control")
        if args.supervised_only and args.teacher_checkpoint:
            raise ValueError("A supervised-only control must not use a teacher checkpoint")
    for name in ('resume_checkpoint', 'init_checkpoint', 'teacher_checkpoint'):
        path = getattr(args, name, None)
        if path and not Path(path).is_file():
            raise ValueError(f"{name} does not exist: {path}")


def train_main(argv=None, teacher_training=False):
    args = training_parser(teacher_training).parse_args(argv)
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
    restore_control_arguments(args, checkpoint)
    seed_everything(args.seed)
    run_config = public_arguments(args).copy()
    run_config.update(world_size=world, image_size=IMAGE_SIZE,
                      text_mode="dropout" if args.teacher_training else "question", answer_dropout=.5)
    out = Path(args.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    train_set = VizWizGroundingDataset(args.data_root, "train", IMAGE_SIZE,
                text_mode="dropout" if args.teacher_training else "question")
    model = (TeacherModel() if args.teacher_training else GroundingModel()).to(device)
    if getattr(args, "init_checkpoint", None):
        initialize_weights(args.init_checkpoint,model)
    objective = (SupervisedObjective() if args.teacher_training or getattr(args, "supervised_only", False) else
                 DistillationObjective(load_teacher(args.teacher_checkpoint, device)))
    controls = TrainingControls(model, args, checkpoint)
    parameters = controls.parameters
    optimizer = controls.optimizer
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
    controls.configure_schedule(
        args.num_epochs,
        math.ceil(len(train_loader) / getattr(args, "gradient_accumulation_steps", 1)),
        start,
    )
    run_config["optimization_config"] = controls.config
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
        controls.set_encoder_modes()
        controls.log_learning_rates(epoch + 1)
        totals = torch.zeros(2,device=device,dtype=torch.float64)  # (2,) [weighted_loss_sum, sample_count]
        component_sums = {}
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}",
                          unit="batch", disable=rank != 0, dynamic_ncols=True):
            batch = to_device(batch,device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,enabled=device.type=="cuda"):
                loss, components = objective(model,batch)  # loss: ()
            if device.type != "cuda" and not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters,1.)
            controls.step(scaler)
            n = len(batch["filename"])
            totals[0] += loss.detach().to(torch.float64) * n
            totals[1] += n
            for name,value in components.items():
                val = value.detach().to(device=device, dtype=torch.float64) * n  # ()
                component_sums[name] = component_sums[name] + val if name in component_sums else val
        names = sorted(component_sums)
        stats = torch.cat([totals, torch.stack([component_sums[k] for k in names])]) if names else totals  # (2 + C,)
        if distributed:
            dist.all_reduce(stats)
        if not torch.isfinite(stats[0]):
            raise FloatingPointError("Nonfinite training loss")
        stats_cpu = stats.tolist()
        count = stats_cpu[1]
        record = {"epoch":epoch+1,"train_loss":stats_cpu[0]/count}
        for i, name in enumerate(names):
            record[name] = stats_cpu[2 + i] / count
        if rank == 0:
            print(json.dumps(record),flush=True)
            with (out/"history.jsonl").open("a") as handle:
                handle.write(json.dumps(record)+"\n")
            if args.save_every and (epoch + 1) % args.save_every == 0:
                checkpoint_path = out / f"checkpoint_epoch{epoch + 1}.pt"
                torch.save({"epoch": epoch + 1,
                            "model_state_dict": unwrap(model).state_dict(),
                            "training_controls": controls.state_dict(),
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
