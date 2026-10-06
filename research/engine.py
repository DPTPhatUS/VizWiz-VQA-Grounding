"""Shared CPU/CUDA/DDP training and original-resolution evaluation."""
import argparse
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler, Subset
from PIL import Image

from research import variant
from research.data import GroundingDataset, collate_samples, to_device
from research.checkpoint import (load_checkpoint, read_checkpoint, save_checkpoint,
                                 initialize_weights, rng_state, restore_rng)
from research.losses import per_image_iou


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(args):
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", True
    annotations = {}
    for split in ("train", "val"):
        path = Path(args.data_root) / f"{split}_grounding.json"
        if path.exists():
            annotations[split] = file_hash(path)
            masks = hashlib.sha256()
            for filename in sorted(json.loads(path.read_text())):
                mask_path = Path(args.data_root) / "binary_masks_png" / split / Path(filename).with_suffix(".png")
                masks.update(json.dumps([filename, file_hash(mask_path)]).encode())
            annotations[f"{split}_masks"] = masks.hexdigest()
    if getattr(args, "pairs", None):
        annotations["pairs"] = file_hash(args.pairs)
        manifest = Path(args.pairs)
        masks = hashlib.sha256()
        for pair in json.loads(manifest.read_text()):
            for key in ("mask1", "mask2"):
                masks.update(json.dumps([pair[key], file_hash(manifest.parent / pair[key])]).encode())
        annotations["pair_masks"] = masks.hexdigest()
    return {"revision": revision, "dirty": dirty, "torch": torch.__version__,
            "annotation_sha256": annotations}


def training_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/vizwiz")
    parser.add_argument("--output-dir", default="outputs-research")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4, help="Global batch size across all ranks")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--encoder-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--dice-weight", type=float, default=0.)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--detail-size", type=int, default=672)
    parser.add_argument("--architecture", choices=["compact","joint","residual","baseline"], default="compact")
    parser.add_argument("--text-mode", choices=["question","answer","dropout"], default="question")
    parser.add_argument("--answer-dropout", type=float, default=.5)
    parser.add_argument("--freeze-encoders", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--metric-resolution", choices=["original","local"], default="original")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--init-checkpoint", help="Weights only; start a new optimizer and epoch counter")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tiny", action="store_true", help="Offline synthetic test model, not a research result")
    variant.add_arguments(parser)
    return parser


def check_arguments(args):
    if min(args.num_epochs,args.batch_size,args.save_every,args.validate_every,args.image_size) <= 0:
        raise ValueError("Epochs, batch size, save/validation intervals and image size must be positive")
    if args.num_workers < 0 or args.lr <= 0 or (args.encoder_lr is not None and args.encoder_lr <= 0):
        raise ValueError("Invalid workers or learning rate")
    if args.dice_weight < 0 or args.weight_decay < 0:
        raise ValueError("Loss weights and weight decay cannot be negative")
    if not args.tiny and args.image_size != 336:
        raise ValueError("CLIP-L/14-336 requires --image-size 336")
    if args.resume_checkpoint and args.init_checkpoint:
        raise ValueError("Use either resume or weight initialization, not both")
    for name in ("resume_checkpoint","init_checkpoint","teacher_checkpoint","pairs"):
        path = getattr(args,name,None)
        if path and not Path(path).is_file():
            raise ValueError(f"{name} does not exist: {path}")
    if args.resume_checkpoint and Path(args.resume_checkpoint).resolve().parent != Path(args.output_dir).resolve():
        raise ValueError("Resume requires the same output directory as its checkpoint; use --init-checkpoint for a new run")
    if args.resume_checkpoint and Path(args.resume_checkpoint).name != "last.pt":
        raise ValueError("Strict continuation requires last.pt; use --init-checkpoint for older or best weights")
    variant.validate_args(args)


def make_loader(dataset, args, batch_size, sampler=None, shuffle=False, generator=None):
    kwargs = {"batch_size":batch_size,"sampler":sampler,"shuffle":shuffle,
              "num_workers":args.num_workers,"collate_fn":collate_samples,
              "pin_memory":str(args.device).startswith("cuda"), "worker_init_fn":seed_worker,
              "generator":generator}
    if args.num_workers:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def metric_samples(logits, batch, resolution):
    for i, name in enumerate(batch["filename"]):
        truth = batch["original_mask"][i].unsqueeze(0).to(logits.device) if resolution == "original" else batch["mask"][i:i+1]
        resized = F.interpolate(logits[i:i+1].float(), size=truth.shape[-2:],
                                mode="bilinear", align_corners=False)
        score = per_image_iou(resized, truth).item()
        yield name, score, resized[0,0] > 0


@torch.no_grad()
def evaluate(model, loader, device, resolution, output=None):
    model.eval()
    scores = {}
    for batch in loader:
        batch = to_device(batch, device)
        # Dataset validation/test always supplies question-only text.
        logits = model(batch)["logits"]
        for name, score, prediction in metric_samples(logits, batch, resolution):
            scores[name] = score
            if output is not None:
                path = Path(output) / Path(name).with_suffix(".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((prediction.cpu().numpy().astype(np.uint8)*255)).save(path)
    return scores


def train_main(argv=None):
    args = training_parser().parse_args(argv)
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
    run_config["world_size"] = world
    out = Path(args.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        if not args.resume_checkpoint and (out/"last.pt").exists():
            raise ValueError("Output already contains a run; use --resume-checkpoint or a fresh directory")
    if distributed:
        dist.barrier()
    train_set = GroundingDataset(args.data_root,"train",args.image_size,
        args.detail_size if getattr(args,"needs_detail",False) else None,
        args.text_mode,args.answer_dropout,pairs=getattr(args,"pairs",None))
    val_set = GroundingDataset(args.data_root,"val",args.image_size,
        args.detail_size if getattr(args,"needs_detail",False) else None)
    model = variant.build_model(args).to(device)
    if args.init_checkpoint:
        initialize_weights(args.init_checkpoint,model)
    coarse = getattr(model,"coarse",model)
    if args.freeze_encoders:
        for module in (coarse.image_encoder,coarse.text_encoder):
            module.requires_grad_(False)
    objective = variant.build_objective(args,model,device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    encoders = {id(p) for module in (coarse.image_encoder,coarse.text_encoder) for p in module.parameters()}
    groups = [{"params":[p for p in parameters if id(p) in encoders],"lr":args.encoder_lr or args.lr},
              {"params":[p for p in parameters if id(p) not in encoders],"lr":args.lr}]
    optimizer = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    info = provenance(args)
    start, best = 0, -1.
    if args.resume_checkpoint:
        checkpoint = load_checkpoint(args.resume_checkpoint,model)
        ignored = {"resume_checkpoint","init_checkpoint","num_epochs","output_dir","device","num_workers"}
        old = {k:v for k,v in checkpoint["run_config"].items() if k not in ignored}
        new = {k:v for k,v in run_config.items() if k not in ignored}
        if old != new:
            raise ValueError("Resume training configuration differs; use --init-checkpoint for a new run")
        if checkpoint.get("provenance",{}).get("annotation_sha256") != info["annotation_sha256"]:
            raise ValueError("Resume annotation hashes differ")
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError("Resume requires optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # CPU checkpoints contain an empty disabled-scaler state. A CUDA continuation
        # starts a fresh scaler; a disabled CPU scaler safely ignores CUDA state.
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start,best = checkpoint["epoch"],checkpoint["best_iou"]
        restore_rng(checkpoint["rng_states"][rank])
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
        if args.freeze_encoders:
            coarse.image_encoder.eval(); coarse.text_encoder.eval()
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
            scores = evaluate(unwrap(model),val_loader,device,args.metric_resolution)
            statistics = torch.tensor([sum(scores.values()),len(scores)],device=device,dtype=torch.float64)
            if distributed:
                dist.all_reduce(statistics)
            score = (statistics[0]/statistics[1]).item()
            record["val_mean_iou"] = score
            improved = score > best
            best = max(best,score)
        states = [None]*world if distributed else [rng_state()]
        if distributed:
            dist.all_gather_object(states,rng_state())
        if rank == 0:
            print(json.dumps(record),flush=True)
            with (out/"history.jsonl").open("a") as handle:
                handle.write(json.dumps(record)+"\n")
            options = (unwrap(model),optimizer,scaler,epoch+1,best,run_config,states,info)
            save_checkpoint(out/"last.pt",*options)
            if improved:
                save_checkpoint(out/"best.pt",*options)
            if (epoch+1)%args.save_every == 0:
                save_checkpoint(out/f"checkpoint_epoch{epoch+1}.pt",*options)
        if distributed:
            dist.barrier()


def eval_main(argv=None):
    parser = argparse.ArgumentParser(description="Question-only evaluation using checkpoint architecture")
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--data-root",default="data/vizwiz")
    parser.add_argument("--dataset",choices=["val","test"],default="val")
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--num-workers",type=int,default=4)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--metric-resolution",choices=["original","local"],default="original")
    if hasattr(variant,"add_eval_arguments"):
        variant.add_eval_arguments(parser)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count")
    saved = read_checkpoint(args.checkpoint)
    config = SimpleNamespace(**saved["run_config"])
    model = variant.build_model(config)
    load_checkpoint(args.checkpoint,model)
    inference = variant.configure_evaluation(model,args) if hasattr(variant,"configure_evaluation") else {}
    device = torch.device(args.device); model.to(device)
    dataset = GroundingDataset(args.data_root,args.dataset,config.image_size,
        config.detail_size if getattr(config,"needs_detail",False) else None)
    loader = make_loader(dataset,args,args.batch_size)
    out = Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    scores = evaluate(model,loader,device,args.metric_resolution,out)
    summary = {"num_samples":len(scores),"mean_iou":sum(scores.values())/len(scores),
               "dataset":args.dataset,"metric_resolution":args.metric_resolution,
               "text_mode":"question","checkpoint":str(args.checkpoint),
               "experiment_config":model.experiment_config,"inference":inference}
    (out/"metrics.json").write_text(json.dumps({"summary":summary,"per_sample":scores},indent=2))
    print(json.dumps(summary),flush=True)
