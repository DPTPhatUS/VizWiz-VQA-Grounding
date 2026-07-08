import argparse
import torch
from torch.amp import autocast, GradScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

import os

from dataset import VizWizGroundingDataset
from utils import to_device, compute_iou
from models import TextEncoder, ImageEncoder, GroundingModel

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=str, default="data/vizwiz")
parser.add_argument("--num-epochs", type=int, default=None, help="Override num_epochs from config.yml")
parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size from config.yml")
parser.add_argument("--lr", type=float, default=None, help="Override lr from config.yml")
parser.add_argument("--num-workers", type=int, default=None, help="Override num_workers from config.yml")
parser.add_argument("--resume-checkpoint", type=str, default=None, help="Override resume_checkpoint from config.yml; pass empty string to disable resume")
args = parser.parse_args()

# create output directory if not exists
os.makedirs("outputs", exist_ok=True)

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

config["dataset"]["train_json"] = os.path.join(args.data_root, "train_grounding.json")
config["dataset"]["val_json"] = os.path.join(args.data_root, "val_grounding.json")
config["dataset"]["train_image_root"] = os.path.join(args.data_root, "train")
config["dataset"]["train_mask_root"] = os.path.join(args.data_root, "binary_masks_png", "train")
config["dataset"]["val_image_root"] = os.path.join(args.data_root, "val")
config["dataset"]["val_mask_root"] = os.path.join(args.data_root, "binary_masks_png", "val")

# CLI overrides — only apply if explicitly provided
if args.num_epochs is not None:
    config["num_epochs"] = args.num_epochs
if args.batch_size is not None:
    config["batch_size"] = args.batch_size
if args.lr is not None:
    config["lr"] = args.lr
if args.num_workers is not None:
    config["num_workers"] = args.num_workers
if args.resume_checkpoint is not None:
    config["resume_checkpoint"] = args.resume_checkpoint

# dataset
train_set = VizWizGroundingDataset(
    json_path=config["dataset"]["train_json"],
    image_root=config["dataset"]["train_image_root"],
    mask_root=config["dataset"]["train_mask_root"],
    image_size=tuple(config["image_size"])
)
val_set = VizWizGroundingDataset(
    json_path=config["dataset"]["val_json"],
    image_root=config["dataset"]["val_image_root"],
    mask_root=config["dataset"]["val_mask_root"],
    image_size=tuple(config["image_size"])
)
train_loader = DataLoader(
    train_set,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],  # reduce to 8~12 if OOM
    pin_memory=True,
    prefetch_factor=2  # reduced from 4 to 2 to lower cpu load
)
val_loader = DataLoader(
    val_set,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],  # reduce to 8~12 if OOM
    pin_memory=True,
    prefetch_factor=2  # reduced from 4 to 2 to lower cpu load
)

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GroundingModel().to(device)

# optimizer / loss
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
loss_fn = nn.BCEWithLogitsLoss()
scaler = GradScaler()

# resume checkpoint
resume_path = config.get("resume_checkpoint", None)
start_epoch = 0

if resume_path and os.path.exists(resume_path):
    checkpoint = torch.load(resume_path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"✅ Resumed full training state from {resume_path} (epoch {start_epoch})")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ Resumed model from {resume_path}")
        try:
            start_epoch = int(resume_path.split("epoch")[1].split(".")[0])
        except Exception:
            start_epoch = 0

# training loop
for epoch in range(start_epoch, config["num_epochs"]):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} (Training)")
    for batch in loop:
        batch = to_device(batch, device)
        images = batch["image"]
        masks = batch["mask"]
        texts = batch["text"]

        with autocast('cuda'):
            pred_masks = model(images, texts)
            pred_masks = nn.functional.interpolate(pred_masks, size=masks.shape[-2:], mode='bilinear')
            loss = loss_fn(pred_masks, masks)
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Average Training Loss: {avg_train_loss:.4f}")

    # validation
    model.eval()
    val_loss = 0
    val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} (Validation)")
    with torch.no_grad():
        for batch in val_loop:
            batch = to_device(batch, device)
            images = batch["image"]
            masks = batch["mask"]
            texts = batch["text"]

            pred_masks = model(images, texts)
            pred_masks = nn.functional.interpolate(pred_masks, size=masks.shape[-2:], mode='bilinear')

            loss = loss_fn(pred_masks, masks)
            val_loss += loss.item()
            val_loop.set_postfix(loss=loss.item())

    avg_val_loss = val_loss / len(val_loader)
    print(f"[Epoch {epoch+1}] Average Validation Loss: {avg_val_loss:.4f}")
    # save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint_path = f"outputs/checkpoint_epoch{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': avg_train_loss,
        }, checkpoint_path)
        print(f"✅ Checkpoint saved at {checkpoint_path}")

# save final model
torch.save(model.state_dict(), f"outputs/model_final_epoch{config['num_epochs']}.pt")
print(f" Final model saved")
