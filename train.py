"""Train this branch's grounding experiment on CPU, CUDA, or torchrun DDP."""
import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DistributedSampler, Subset
from dataset import VizWizGroundingDataset, make_loader
from utils import to_device
from models.model import build_model, EXPERIMENT
from models.checkpoint import (load_checkpoint, read_checkpoint, save_checkpoint,
                               initialize_weights)
from metrics import evaluate
from losses import SupervisedObjective, DistillationObjective
from models.backbone import BaseGroundingModel


def add_experiment_arguments(parser):
    parser.add_argument("--init-checkpoint", help="Optional compact control weights; starts a new run")
    if not parser.teacher_training:
        parser.add_argument("--teacher-checkpoint", required=True, help="Answer-dropout teacher from train_teacher.py")


def validate_experiment_args(args):
    pass


def build_objective(args, model, device):
    if args.teacher_training:
        return SupervisedObjective()
    saved = read_checkpoint(args.teacher_checkpoint)
    config, run = saved["experiment_config"], saved["run_config"]
    if config.get("experiment") != "controls" or run.get("text_mode") != "dropout" or not 0 < run.get("answer_dropout", 0) < 1:
        raise ValueError("Teacher must be a corrected model trained with answer dropout")
    if run.get("image_size") != IMAGE_SIZE:
        raise ValueError("Teacher/student image sizes must match")
    teacher = BaseGroundingModel(config["architecture"], config.get("tiny", False)).to(device)
    load_checkpoint(args.teacher_checkpoint, teacher)
    return DistillationObjective(teacher, "incremental", 1., 0.)


IMAGE_SIZE = 336


def build_datasets(args):
    return (VizWizGroundingDataset(args.data_root, "train", IMAGE_SIZE,
                text_mode="dropout" if args.teacher_training else "question"),
            VizWizGroundingDataset(args.data_root, "val", IMAGE_SIZE))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def training_parser(teacher_training=False):
    parser = argparse.ArgumentParser(description="Train answer-dropout teacher" if teacher_training else __doc__)
    parser.teacher_training = teacher_training
    parser.add_argument("--data-root", default="data/vizwiz")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4, help="Global batch size across all ranks")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10, help="Numbered checkpoint interval; 0 keeps only last/best")
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--resume-checkpoint")
    add_experiment_arguments(parser)
    return parser


def check_arguments(args):
    if min(args.num_epochs, args.batch_size, args.validate_every) <= 0:
        raise ValueError("Epochs, batch size and validation interval must be positive")
    if args.num_workers < 0 or args.lr <= 0 or args.save_every < 0:
        raise ValueError("Workers/save interval cannot be negative; learning rate must be positive")
    if args.resume_checkpoint and getattr(args, "init_checkpoint", None):
        raise ValueError("Use either resume or weight initialization, not both")
    for name in ("resume_checkpoint", "init_checkpoint", "teacher_checkpoint", "pairs"):
        path = getattr(args, name, None)
        if path and not Path(path).is_file():
            raise ValueError(f"{name} does not exist: {path}")
    validate_experiment_args(args)


def train_main(argv=None, teacher_training=False):
    args = training_parser(teacher_training).parse_args(argv)
    args.teacher_training = teacher_training
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    check_arguments(args)
    rank, world, local = (int(os.environ.get(k,default)) for k,default in (
        ("RANK",0),("WORLD_SIZE",1),("LOCAL_RANK",0)))
    if args.batch_size % world:
        raise ValueError("Global batch size must be divisible by world size")
    distributed = world > 1
    device = torch.device(args.device)
    if distributed:
        if device.type == "cuda":
            device = torch.device("cuda", local)
            torch.cuda.set_device(device)
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        _train(args, rank, world, distributed, device)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def _train(args, rank, world, distributed, device):
    seed_everything(args.seed)
    run_config = vars(args).copy()
    run_config.update(world_size=world, image_size=IMAGE_SIZE,
                      text_mode="dropout" if args.teacher_training else "question", answer_dropout=.5)
    out = Path(args.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if not args.resume_checkpoint and (out/"last.pt").exists():
        raise ValueError("Output already contains a run; use --resume-checkpoint or a fresh directory")
    if args.resume_checkpoint and (out / "last.pt").exists() and Path(args.resume_checkpoint).resolve() != (out / "last.pt").resolve():
        raise ValueError("Output already contains a run; resume its last.pt or choose a fresh output directory")
    if distributed:
        dist.barrier()
    train_set, val_set = build_datasets(args)
    model = build_model(args).to(device)
    if getattr(args, "init_checkpoint", None):
        initialize_weights(args.init_checkpoint,model)
    objective = build_objective(args,model,device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    start, best = 0, -1.
    if args.resume_checkpoint:
        checkpoint = load_checkpoint(args.resume_checkpoint,model)
        if checkpoint["run_config"].get("stage") != getattr(args, "stage", None):
            raise ValueError("Resume requires the same training stage; use --init-checkpoint to change stage")
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError("Resume requires optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        # CPU checkpoints contain an empty disabled-scaler state. A CUDA continuation
        # starts a fresh scaler; a disabled CPU scaler safely ignores CUDA state.
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start,best = checkpoint["epoch"],checkpoint["best_iou"]
        # A new output directory selects its best checkpoint from the continued epochs.
        if not (out / "best.pt").exists():
            best = -1.
        if start >= args.num_epochs:
            raise ValueError("num-epochs must exceed the resumed epoch")
    if distributed:
        model = nn.parallel.DistributedDataParallel(model,
            device_ids=[device.index] if device.type=="cuda" else None,
            find_unused_parameters=True)
    sampler = DistributedSampler(train_set,num_replicas=world,rank=rank,seed=args.seed) if distributed else None
    generator = torch.Generator()
    train_loader = make_loader(train_set,args,args.batch_size//world,sampler,sampler is None,generator)
    # No padded validation examples and no DDP forward collectives for unequal shard lengths.
    val_subset = Subset(val_set,list(range(rank,len(val_set),world)))
    val_loader = make_loader(val_subset,args,args.batch_size//world)
    if rank == 0:
        (out/"config.json").write_text(json.dumps(run_config,indent=2))
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
        totals = torch.zeros(2,device=device,dtype=torch.float64)
        component_sums = {}
        for batch in train_loader:
            batch = to_device(batch,device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,enabled=device.type=="cuda"):
                loss, components = objective(model,batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters,1.)
            scaler.step(optimizer); scaler.update()
            n = len(batch["filename"])
            totals += torch.tensor([loss.detach().item()*n,n],device=device)
            for name,value in components.items():
                component_sums[name] = component_sums.get(name,0.)+float(value.detach())*n
        if distributed:
            dist.all_reduce(totals)
        record = {"epoch":epoch+1,"train_loss":(totals[0]/totals[1]).item()}
        for name in sorted(component_sums):
            value = torch.tensor(component_sums[name],device=device,dtype=torch.float64)
            if distributed:
                dist.all_reduce(value)
            record[name] = (value/totals[1]).item()
        improved = False
        if (epoch+1)%args.validate_every == 0 or epoch+1 == args.num_epochs:
            if distributed:
                # Includes e.g. BatchNorm running statistics in wide teacher controls.
                for buffer in unwrap(model).buffers():
                    dist.broadcast(buffer,src=0)
            scores = evaluate(unwrap(model),val_loader,device,"original")
            statistics = torch.tensor([sum(scores.values()),len(scores)],device=device,dtype=torch.float64)
            if distributed:
                dist.all_reduce(statistics)
            score = (statistics[0]/statistics[1]).item()
            record["val_mean_iou"] = score
            improved = score > best
            best = max(best,score)
        if rank == 0:
            print(json.dumps(record),flush=True)
            with (out/"history.jsonl").open("a") as handle:
                handle.write(json.dumps(record)+"\n")
            options = (unwrap(model),optimizer,scaler,epoch+1,best,run_config)
            save_checkpoint(out/"last.pt",*options)
            if improved:
                save_checkpoint(out/"best.pt",*options)
            if args.save_every and (epoch+1)%args.save_every == 0:
                save_checkpoint(out/f"checkpoint_epoch{epoch+1}.pt",*options)
        if distributed:
            dist.barrier()


main = train_main

if __name__ == "__main__":
    main()
