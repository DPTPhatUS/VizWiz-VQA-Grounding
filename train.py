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
from torch.utils.data import DistributedSampler
from dataset import VizWizGroundingDataset, make_loader
from utils import to_device, read_checkpoint, load_model_weights, initialize_weights
from models.model import GroundingModel, EXPERIMENT
from losses import ExtentObjective


IMAGE_SIZE = 336


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
    parser.add_argument('--init-checkpoint', help='Initialize model weights for a new run')
    parser.add_argument('--pairs', help='Verified same-image question pairs with their own masks')
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
    for name in ('resume_checkpoint', 'init_checkpoint', 'pairs'):
        path = getattr(args, name, None)
        if path and not Path(path).is_file():
            raise ValueError(f"{name} does not exist: {path}")


def train_main(argv=None):
    args = training_parser().parse_args(argv)
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
    run_config.update(world_size=world, image_size=IMAGE_SIZE, text_mode="question")
    out = Path(args.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.pt")):
        raise ValueError("Output already contains model files; choose a fresh output directory")
    if distributed:
        dist.barrier()
    train_set = VizWizGroundingDataset(args.data_root, "train", IMAGE_SIZE, pairs=args.pairs)
    model = GroundingModel().to(device)
    if getattr(args, "init_checkpoint", None):
        initialize_weights(args.init_checkpoint,model)
    pair_weight = .1 if args.pairs else 0.
    objective = ExtentObjective(support_weight=.2, area_weight=.1,
                                pair_delta_weight=pair_weight, pair_consistency_weight=pair_weight)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    start = 0
    if args.resume_checkpoint:
        checkpoint = read_checkpoint(args.resume_checkpoint)
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
        if start >= args.num_epochs:
            raise ValueError("num-epochs must exceed the resumed epoch")
    if distributed:
        model = nn.parallel.DistributedDataParallel(model,
            device_ids=[device.index] if device.type=="cuda" else None,
            find_unused_parameters=True)
    sampler = DistributedSampler(train_set,num_replicas=world,rank=rank,seed=args.seed) if distributed else None
    generator = torch.Generator()
    train_loader = make_loader(train_set,args,args.batch_size//world,sampler,sampler is None,generator)
    run_config["experiment_config"] = unwrap(model).experiment_config
    run_config["tiny"] = getattr(unwrap(model), "coarse", unwrap(model)).experiment_config.get("tiny", False)
    if rank == 0:
        (out/"config.json").write_text(json.dumps(run_config,indent=2))
        print(json.dumps({"parameters":sum(p.numel() for p in unwrap(model).parameters()),
                          "trainable":sum(p.numel() for p in parameters),"device":str(device),
                          "experiment":unwrap(model).experiment_config}),flush=True)
    for epoch in range(start,args.num_epochs):
        # Epoch-keyed RNG permits reproducible continuation, including loader and paired-question sampling.
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
                            "run_config": run_config}, checkpoint_path)
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
