"""Fresh, matched continuation training and post-training snapshot evaluation.

Original train.py/eval.py remain unchanged. Weight-only snapshots are for
evaluation, not optimizer-state resume. No automatic cross-session extension.
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.functional import to_pil_image

from b1_data import OCRDataset, ocr_collate
from evaluation import ShardSampler, evaluate, gather_scores, mean_iou
from models.model import ModelConfig
from models.simple_ocr import ARMS, SimpleOCRModel
from run_record import append_jsonl, git_state
from scripts.export_b0_validation_masks import digest
from train import save_atomically, seed_everything, train_one_epoch
from training_options import TrainingControls
from utils import align_state_dict, cleanup_distributed, setup_distributed

PARENT_SHA = "49aeae8744fd2f52778ef783b13e5359965baed26935c249ee177cf3e4c0e15e"
EPOCHS = 10
SNAPSHOTS = (5, 10)
VIEWS = ("clean", "empty", "cross_image", "text_shuffle", "geometry")


def load_parent(path: Path, expected_sha: str = PARENT_SHA):
    if digest(path) != expected_sha:
        raise ValueError("Parent checkpoint SHA256 mismatch")
    c = torch.load(
        path, map_location="cpu", weights_only=False
    )  # own retained training checkpoint
    if (
        c["epoch"] != 20
        or c["run_config"]["recipe"]["seed"] != 42
        or c["run_config"]["recipe"]["text_mode"] != "question"
        or c["run_config"]["git"]
        != {"commit": "0677214ffbc3e05659719f66d30db519509eba4c", "dirty": False}
    ):
        raise ValueError("Parent identity/recipe mismatch")
    return c


def new_model(arm, parent):
    model = SimpleOCRModel(arm, ModelConfig(**parent["model_config"]))
    model.base.load_state_dict(
        align_state_dict(parent["model_state_dict"], model.base), strict=True
    )
    return model


def fresh_controls(model, epochs, batches):
    args = SimpleNamespace(
        lr=1e-5,
        backbone_lr=1e-6,
        optimizer="adamw",
        weight_decay=0.05,
        lr_scheduler="cosine",
        warmup_ratio=0.05,
        min_lr_ratio=0.1,
        freeze_text_encoder=True,
        freeze_vision_encoder=False,
    )
    controls = TrainingControls(model, args)
    new_ids = (
        {id(p) for p in model.ocr.parameters()} if model.ocr is not None else set()
    )
    groups = []
    for group in controls.optimizer.param_groups:
        for is_new in (False, True):
            params = [p for p in group["params"] if (id(p) in new_ids) == is_new]
            if params:
                groups.append(
                    {
                        **group,
                        "params": params,
                        "lr": 1e-4 if is_new else group["lr"],
                        "group_name": ("ocr_" if is_new else "existing_")
                        + group["group_name"],
                    }
                )
    controls.optimizer = torch.optim.AdamW(groups)
    controls.base_lrs = [g["lr"] for g in groups]
    ids = [id(p) for g in groups for p in g["params"]]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in controls.parameters}:
        raise ValueError("Optimizer group coverage mismatch")
    controls.configure_schedule(epochs, batches)
    return controls


def write_snapshot(model, path, epoch, config):
    # All parameters, including frozen encoders, are retained; no optimizer,
    # scaler or gradients. This makes each snapshot independently evaluable.
    clean_config = json.loads(json.dumps(config))
    save_atomically(
        {
            "format": "simple_ocr_weights_v1",
            "epoch": epoch,
            "arm": model.arm,
            "model_config": model.base.config.to_dict(),
            "run_config": clean_config,
            "model_state_dict": {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            },
        },
        path,
    )
    path.with_suffix(".sha256").write_text(digest(path) + "\n")


def read_snapshot(path, expected_sha=None):
    checksum = expected_sha or path.with_suffix(".sha256").read_text().strip()
    if digest(path) != checksum:
        raise ValueError("Snapshot integrity mismatch")
    c = torch.load(path, map_location="cpu", weights_only=True)
    if c["format"] != "simple_ocr_weights_v1" or c["epoch"] not in SNAPSHOTS:
        raise ValueError("Unexpected snapshot format/epoch")
    if any(k in c for k in ("optimizer_state_dict", "scaler_state_dict")):
        raise ValueError("Expected a weight-only snapshot")
    model = SimpleOCRModel(c["arm"], ModelConfig(**c["model_config"]))
    model.load_state_dict(c["model_state_dict"], strict=True)
    return model, {k: v for k, v in c.items() if k != "model_state_dict"}


def loader(
    data, split, bank, batch, workers, rank, world, *, training=False, view="clean"
):
    dataset = OCRDataset(data, split, bank[split], native_mask=not training, view=view)
    generator = torch.Generator()
    sampler = (
        (
            DistributedSampler(dataset, world, rank, shuffle=True, seed=42)
            if world > 1
            else RandomSampler(dataset, generator=generator)
        )
        if training
        else ShardSampler(len(dataset), rank, world)
    )
    result = DataLoader(
        dataset,
        batch_size=batch,
        sampler=sampler,
        collate_fn=ocr_collate,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    return dataset, result, sampler, generator


def train_arm(args):
    rank, world, local, device = setup_distributed()
    try:
        if args.batch_size % world:
            raise ValueError("Global batch must divide world size")
        if rank == 0 and args.output.exists():
            raise FileExistsError(args.output)
        seed_everything(42)
        torch.backends.cudnn.benchmark = True  # matches original B0 policy
        parent = load_parent(args.parent)
        model = new_model(args.arm, parent)
        del parent  # release original optimizer state before moving model to GPU
        model.to(device)
        bank = torch.load(args.features, map_location="cpu", weights_only=True)
        _train_set, train_loader, sampler, generator = loader(
            args.data,
            "train",
            bank,
            args.batch_size // world,
            args.workers,
            rank,
            world,
            training=True,
        )
        val_set, val_loader, _, _ = loader(
            args.data, "val", bank, args.eval_batch_size, args.workers, rank, world
        )
        controls = fresh_controls(model, EPOCHS, len(train_loader))
        config = {
            "arm": args.arm,
            "parent_sha256": PARENT_SHA,
            "seed": 42,
            "epochs": EPOCHS,
            "global_batch": args.batch_size,
            "world_size": world,
            "batches_per_epoch": len(train_loader),
            "expected_attempted_updates": EPOCHS * len(train_loader),
            "loss": "BCE+0.5 per-image Dice",
            "optimizer": "fresh AdamW",
            "scheduler": controls.config,
            "gradient_clip": 1.0,
            "parameter_groups": [
                {k: v for k, v in g.items() if k != "params"}
                | {"parameters": sum(p.numel() for p in g["params"])}
                for g in controls.optimizer.param_groups
            ],
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "ocr_parameters": sum(p.numel() for p in model.ocr.parameters())
            if model.ocr
            else 0,
            "features_sha256": digest(args.features),
            "git": git_state(Path(__file__).parent),
            "torch": str(torch.__version__),
            "text_mode": "question",
            "validation_images": len(val_set),
        }
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "run_config.json").write_text(json.dumps(config, indent=2))
        if world > 1:
            dist.barrier()
            wrapped = DDP(
                model,
                device_ids=[local] if device.type == "cuda" else None,
                gradient_as_bucket_view=True,
            )
        else:
            wrapped = model
        scaler = GradScaler("cuda", enabled=device.type == "cuda")
        train_args = SimpleNamespace(grad_clip_norm=1.0, dice_weight=0.5)
        for epoch in range(1, EPOCHS + 1):
            if world > 1:
                sampler.set_epoch(epoch)
            else:
                generator.manual_seed(42 + epoch)
            wrapped.train()
            controls.set_encoder_modes()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started, before = time.perf_counter(), controls.completed_steps
            first_lrs = [g["lr"] for g in controls.optimizer.param_groups]
            totals, wait = train_one_epoch(
                wrapped, train_loader, controls, scaler, device, train_args, rank == 0
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            peak = torch.tensor(
                [
                    torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else 0.0
                ],
                device=device,
            )
            if world > 1:
                dist.all_reduce(totals)
                dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            loss = float((totals[0] / totals[1]).item())
            if not torch.isfinite(torch.tensor(loss)):
                raise ValueError("Nonfinite epoch loss; experiment is incomplete")
            # Per-epoch validation only. No masks, interventions or comprehensive
            # diagnostics until the orchestrator has completed all three arms.
            scores = gather_scores(evaluate(model, val_loader, device), world)
            val_miou = mean_iou(scores, len(val_set))
            if rank == 0:
                successful = controls.completed_steps - before
                beta_grad = model.ocr.beta.grad if model.ocr is not None else None
                gradient_finite = (
                    bool(torch.isfinite(beta_grad).all())
                    if beta_grad is not None
                    else None
                )
                record = {
                    "epoch": epoch,
                    "train_loss": loss,
                    "val_miou": val_miou,
                    "attempted_updates": len(train_loader),
                    "successful_updates": successful,
                    "skipped_updates": len(train_loader) - successful,
                    "lr_start": first_lrs,
                    "lr_end": [g["lr"] for g in controls.optimizer.param_groups],
                    "seconds": seconds,
                    "data_wait_seconds": wait,
                    "peak_gpu_gib_max_rank": peak.item(),
                    "train_image_occurrences_per_rank": len(sampler),
                    "train_image_occurrences_global": len(sampler) * world,
                    "ocr_beta": float(model.ocr.beta.detach()) if model.ocr else None,
                    "ocr_beta_gradient": float(beta_grad) if gradient_finite else None,
                    "ocr_beta_gradient_finite": gradient_finite,
                }
                append_jsonl(args.output / "history.jsonl", record)
                append_jsonl(
                    args.output / "val_curve.jsonl",
                    {"epoch": epoch, "val_miou": val_miou, "num_images": len(scores)},
                )
                print(
                    f"{args.arm} epoch {epoch}/10: loss {loss:.5f}, val {100 * val_miou:.2f}, skipped {record['skipped_updates']}",
                    flush=True,
                )
                if epoch in SNAPSHOTS:
                    write_snapshot(
                        model,
                        args.output / f"checkpoint_epoch{epoch}.pt",
                        epoch,
                        config,
                    )
            if world > 1:
                dist.barrier()
        if rank == 0:
            (args.output / "TRAINING_COMPLETE.json").write_text(
                json.dumps({"epochs": EPOCHS, "snapshots": list(SNAPSHOTS)})
            )
    finally:
        cleanup_distributed()


def evaluate_snapshot(args):
    rank, world, _, device = setup_distributed()
    try:
        if rank == 0 and args.output.exists():
            raise FileExistsError(args.output)
        if args.parent_eval:
            parent = load_parent(args.parent)
            model = new_model("b0_continued", parent)
            metadata = {
                "epoch": 20,
                "arm": "original_b0",
                "run_config": {},
                "model_config": parent["model_config"],
            }
            del parent
        else:
            model, metadata = read_snapshot(args.checkpoint)
        model.to(device).eval()
        bank = torch.load(args.features, map_location="cpu", weights_only=True)
        dataset, val_loader, _, _ = loader(
            args.data,
            "val",
            bank,
            args.eval_batch_size,
            args.workers,
            rank,
            world,
            view=args.view,
        )
        if rank == 0:
            (args.output / "masks").mkdir(parents=True, exist_ok=False)
        if world > 1:
            dist.barrier()

        def save(name, mask):
            to_pil_image(mask.to("cpu", torch.uint8) * 255).save(
                args.output / "masks" / f"{Path(name).stem}.png"
            )

        scores = gather_scores(evaluate(model, val_loader, device, save), world)
        if rank == 0:
            summary = {
                "checkpoint": str(args.parent if args.parent_eval else args.checkpoint),
                "epoch": metadata["epoch"],
                "arm": metadata["arm"],
                "view": args.view,
                "split": "val",
                "num_samples": len(scores),
                "mean_iou": mean_iou(scores, len(dataset)),
                "text_mode": "question",
                "answer_in_input": False,
                "protocol": "native bilinear logits, threshold0, empty union1, FP32",
                "recipe": {"seed": 42, "text_mode": "question"},
                "model_config": metadata["model_config"],
                "continuation": metadata["run_config"],
            }
            (args.output / "metrics.json").write_text(
                json.dumps({"summary": summary, "per_sample": scores}, indent=1)
            )
            print(
                f"{metadata['arm']} ep{metadata['epoch']} {args.view}: {100 * summary['mean_iou']:.3f}",
                flush=True,
            )
    finally:
        cleanup_distributed()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("train", "evaluate"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--parent-eval", action="store_true")
    p.add_argument("--view", choices=VIEWS, default="clean")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    if args.batch_size < 1 or args.eval_batch_size < 1 or args.workers < 0:
        p.error("Invalid batch/worker count")
    if args.mode == "train" and args.arm is None:
        p.error("train requires --arm")
    if args.mode == "evaluate" and not args.parent_eval and args.checkpoint is None:
        p.error("evaluate requires --checkpoint or --parent-eval")
    (train_arm if args.mode == "train" else evaluate_snapshot)(args)


if __name__ == "__main__":
    main()
