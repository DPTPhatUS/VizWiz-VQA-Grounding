import torch
from PIL import Image, ImageOps
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import os
from dataset import CLIP_MEAN, CLIP_STD
from models import GroundingModel

# ===== Settings =====
image_path = "data/vizwiz/test/VizWiz_test_00000006.jpg"
question = "What does it say on here?"
checkpoint_path = "outputs/clip-vit-L-p14-336-wo-crop_epoch3.pt"
save_path = "result/clip-vit-L-p14-336-wo-crop_epoch3/VizWiz_test_00000006_pred_mask.png"
image_size = (336, 336)
device = torch.device("cpu")
# ===== Model load =====
model = GroundingModel.from_checkpoint(checkpoint_path, device=device)

# ===== Preprocess =====
transform = T.Compose([
    T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

with Image.open(image_path) as img:
    image = ImageOps.exif_transpose(img).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, H, W)
text_input = [f"Q: {question}"]

# ===== Inference =====
with torch.no_grad():
    output = model(input_tensor, text_input)  # (1, 1, 16*H_p, 16*W_p)
    output = torch.nn.functional.interpolate(output.float(), size=image_size, mode="bilinear", align_corners=False)  # (1, 1, H, W)
    output = torch.sigmoid(output)            # (1, 1, H, W)
    binary_mask = (output > 0.5).float()[0, 0]  # (H, W)

# ===== Save =====
os.makedirs(os.path.dirname(save_path), exist_ok=True)
mask_img = TF.to_pil_image(binary_mask.cpu())  # convert tensor to PIL image
mask_img.save(save_path)
print(f"✅ Saved binary mask to: {save_path}")