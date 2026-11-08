"""Train a fresh grounding model and retain its validation-selected checkpoint."""
import argparse
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DistributedSampler
from tqdm.auto import tqdm
from dataset import VizWizGroundingDataset, make_loader
from runtime import to_device, parameter_counts, environment
from checkpoints import save_checkpoint
from artifacts import write_json
from models.cm import GroundingModel
from models.factory import build_model
from models.connections import LOCAL_REFINEMENTS
from losses import SupervisedObjective
from validation import validation_iou


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
    parser.add_argument("--output-dir", help="Default: outputs/context-membership/<connection-mode>")
    parser.add_argument('--connection-mode', choices=('shared', 'separate'),
                        help='Default: shared')
    parser.add_argument('--edge-loss-weight', type=float,
                        help='Relation supervision weight; default .1 when enabled')
    parser.add_argument('--local-refinement', choices=LOCAL_REFINEMENTS,
                        help='CM local update rule; default membership (three roles)')
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4, help="Global batch size across all ranks")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--model', choices=('cm', 'ftc', 'ftc-capacity'))
    parser.add_argument('--disable-context', action='store_true', default=None)
    parser.add_argument('--disable-membership', action='store_true', default=None)
    parser.add_argument('--tiny', action='store_true', default=None, help='Offline smoke models only')
    parser.add_argument('--val-every', type=int, default=1, help='Validate every N epochs')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default=None)
    return parser


def check_arguments(args):
    if min(args.num_epochs, args.batch_size) <= 0:
        raise ValueError("Epochs and batch size must be positive")
    if not 1 <= args.val_every <= args.num_epochs:
        raise ValueError('Validation interval must be between 1 and num-epochs')
    if args.num_workers < 0 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Workers cannot be negative; learning rate must be positive")
    if args.edge_loss_weight is not None and (not np.isfinite(args.edge_loss_weight) or args.edge_loss_weight < 0):
        raise ValueError('edge-loss-weight must be finite and nonnegative')


def configure_run(args):
    explicit = getattr(args, '_explicit', set())
    args.model = args.model or 'cm'
    if args.model != 'cm' and any(k in explicit for k in ('connection_mode','disable_context','disable_membership','edge_loss_weight','local_refinement')):
        raise ValueError('CM-specific options are not supported for FTC models')
    args.connection_mode = args.connection_mode or 'shared'
    args.local_refinement = args.local_refinement or 'membership'
    args.tiny = bool(args.tiny)
    args.disable_context, args.disable_membership = bool(args.disable_context), bool(args.disable_membership)
    if args.model != 'cm' and args.local_refinement != 'membership':
        raise ValueError('Local refinement controls require a CM model')
    if args.disable_membership and args.local_refinement != 'membership':
        raise ValueError('A local refinement control cannot disable membership')
    if args.edge_loss_weight is None:
        args.edge_loss_weight = .1 if args.model == 'cm' and not args.disable_membership else 0.
    if args.disable_membership and args.edge_loss_weight > 0:
        raise ValueError('Disabled membership requires zero relation-loss weight')
    args.output_dir = args.output_dir or (f'outputs/context-membership/{args.connection_mode}' if args.model == 'cm' else f'outputs/{args.model}')
    check_arguments(args)


def train_main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = training_parser().parse_args(argv)
    args._explicit = {a[2:].split('=')[0].replace('-', '_') for a in argv if a.startswith('--')}
    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
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
    configure_run(args)
    run_config = {key:value for key,value in vars(args).items() if not key.startswith('_')}
    run_config.update(world_size=world, image_size=32 if args.tiny else IMAGE_SIZE,
                      text_mode='question', freeze_text_encoder=True,
                      run_id=f'{Path(args.output_dir).parent.name}/{Path(args.output_dir).name}')
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Refusing to reuse nonempty output directory: {out}')
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    image_size = 32 if args.tiny else IMAGE_SIZE
    train_set = VizWizGroundingDataset(args.data_root, 'train', image_size)
    val_set = VizWizGroundingDataset(args.data_root, 'val', image_size) if args.val_every else None
    if args.model == 'cm':
        options = {'connection_mode':args.connection_mode}
        if args.tiny: options['tiny'] = True
        if args.disable_context: options['context'] = False
        if args.disable_membership: options['membership'] = False
        if args.local_refinement != 'membership': options['local_refinement'] = args.local_refinement
        model = GroundingModel(**options)
    else:
        model = build_model(model=args.model, tiny=args.tiny)
    model = model.to(device)
    objective = SupervisedObjective(args.edge_loss_weight)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    best_score, best_epoch = -1., None
    provenance = environment()
    if distributed:
        model = nn.parallel.DistributedDataParallel(model,
            device_ids=[device.index] if device.type=="cuda" else None,
            find_unused_parameters=True)
    sampler = DistributedSampler(train_set,num_replicas=world,rank=rank,seed=args.seed) if distributed else None
    generator = torch.Generator()
    train_loader = make_loader(train_set,args,args.batch_size//world,sampler,sampler is None,generator)
    run_config["tiny"] = unwrap(model).experiment_config.get("tiny", False)
    if rank == 0:
        startup = {'parameters': parameter_counts(unwrap(model)), 'device': str(device),
                   'experiment': unwrap(model).experiment_config}
        print(json.dumps(startup), flush=True)
        run_record = {'run_config':run_config, 'provenance':provenance, **startup}
        write_json(out/'run.json', run_record)
    for epoch in range(args.num_epochs):
        # Keep data order and stochastic layers reproducible for each seed/epoch.
        seed_everything(args.seed+epoch*world+rank)
        generator.manual_seed(args.seed+epoch*world+rank)
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        totals = torch.zeros(2,device=device,dtype=torch.float64)
        component_sums = {}
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}",
                          unit="batch", disable=rank != 0, dynamic_ncols=True):
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
        if args.val_every and (epoch + 1) % args.val_every == 0:
            score = validation_iou(model, val_set, args, device, rank, world)
            record['validation_miou'] = score
            improved = score > best_score
            if improved:
                best_score, best_epoch = score, epoch + 1
        if rank == 0:
            print(json.dumps(record), flush=True)
            with (out/'history.jsonl').open('a') as handle:
                handle.write(json.dumps(record)+'\n')
            if improved:
                save_checkpoint(out/'best.pt', {'epoch':epoch+1,
                    'model_state_dict':unwrap(model).state_dict(),
                    'experiment_config':unwrap(model).experiment_config, 'run_config':run_config,
                    'validation_miou':best_score, 'provenance':provenance})
    if rank == 0:
        run_record['selection'] = {'epoch':best_epoch, 'validation_miou':best_score}
        write_json(out/'run.json',run_record)
    if distributed:
        dist.barrier()

main = train_main

if __name__ == "__main__":
    main()
