from dataset import CLIP_MEAN, CLIP_STD
from models.model import GroundingModel
from metrics import compute_iou
from PIL import Image, ImageOps
import torch, json, os
import torch.nn.functional as F

from torchvision import transforms as T

val_json = "data/vizwiz/val_grounding.json"
image_dir = "data/vizwiz/val"
mask_dir = "data/vizwiz/binary_masks_png/val"

# 1. load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GroundingModel.from_checkpoint("outputs/cross_model_final_epoch100.pt", device=device)

# 2. load val json
with open(val_json, "r", encoding="utf-8") as f:
    val_data = json.load(f)

# 3. compute IoU
ious = []

image_transform = T.Compose([
    T.Resize((336, 336), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])
mask_transform = T.Compose([
    T.Resize((336, 336), interpolation=T.InterpolationMode.NEAREST),
    T.ToTensor(),
])

for filename, meta in val_data.items():
    image_path = os.path.join(image_dir, filename)
    mask_path = os.path.join(mask_dir, filename.replace(".jpg", ".png"))
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        continue

    if getattr(model, "model_config", None):
        result = model.predict(image_path, question=meta.get("question", ""),
                               answer=meta.get("most_common_answer", ""), gt_mask=mask_path)
        ious.append(result["iou"])
        continue

    with Image.open(image_path) as img:
        image = image_transform(ImageOps.exif_transpose(img).convert("RGB")).unsqueeze(0).to(device)  # (1, 3, H, W)
    with Image.open(mask_path) as m:
        true_mask = (mask_transform(m.convert("L")) > 0.5).float().unsqueeze(0).to(device)  # (1, 1, H, W)
    q = str(meta.get("question", "")).strip()
    a = str(meta.get("most_common_answer", "")).strip()
    text = f"Q: {q} A: {a}" if a else f"Q: {q}"

    with torch.no_grad():
        pred_mask = model(image, text)  # (1, 1, 16*H_p, 16*W_p)
        pred_mask = F.interpolate(pred_mask.float(), size=true_mask.shape[-2:], mode="bilinear", align_corners=False)  # (1, 1, H, W)
        pred_mask = (torch.sigmoid(pred_mask) > 0.5).float()  # (1, 1, H, W)

    iou = compute_iou(pred_mask, true_mask)
    ious.append(iou)

# 4. output mean IoU
print(f"Mean IoU over {len(ious)} samples: {sum(ious)/len(ious):.4f}")
