import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import os

from dataset import VizWizGroundingDataset
from utils import to_device, compute_iou
from models import GroundingModel

# === Load config ===
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

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
train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,
                          num_workers=config["num_workers"], pin_memory=True, prefetch_factor=2)
val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"], pin_memory=True, prefetch_factor=2)

# === Model + Optimizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GroundingModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
loss_fn = nn.BCEWithLogitsLoss()
scaler = GradScaler()

# === Continued training setup ===
start_epoch = 100
total_epochs = 300

# load checkpoint
checkpoint_path = f"outputs/checkpoint_epoch{start_epoch}.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint.get('epoch', start_epoch)
        print(f"✅ Resumed full training state from {checkpoint_path} (epoch {start_epoch})")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded model from {checkpoint_path}")
else:
    print(f"❌ Checkpoint {checkpoint_path} not found. Starting from epoch {start_epoch}.")

# Training loop
for epoch in range(start_epoch, total_epochs):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} (Training)")
    for batch in loop:
        batch = to_device(batch, device)
        images = batch["image"]
        masks = batch["mask"]
        texts = batch["text"]

        optimizer.zero_grad()

        with autocast():
            pred_masks = model(images, texts)
            pred_masks = nn.functional.interpolate(pred_masks, size=masks.shape[-2:], mode='bilinear')
            loss = loss_fn(pred_masks, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Avg Training Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} (Validation)")
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
    print(f"[Epoch {epoch+1}] Avg Validation Loss: {avg_val_loss:.4f}")

    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        ckpt_path = f"outputs/checkpoint_epoch{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': avg_train_loss,
        }, ckpt_path)
        print(f"✅ Checkpoint saved at {ckpt_path}")

# Save final model
torch.save(model.state_dict(), f"outputs/model_final_epoch{total_epochs}.pt")
print("✅ Final model saved.")