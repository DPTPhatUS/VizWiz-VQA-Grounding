import torch
import torchvision.transforms.functional as TF
import os
from models import GroundingModel

# ===== Settings =====
image_path = "data/vizwiz/test/VizWiz_test_00000006.jpg"
question = "What does it say on here?"
checkpoint_path = "outputs/clip-vit-L-p14-336-wo-crop_epoch3.pt"
save_path = "result/clip-vit-L-p14-336-wo-crop_epoch3/VizWiz_test_00000006_pred_mask.png"
device = torch.device("cpu")
# ===== Model load =====
model = GroundingModel.from_checkpoint(checkpoint_path, device=device)

# predict() selects checkpoint preprocessing: NaFlex exports native size;
# legacy CLIP keeps this example's 336 output convention.
binary_mask = model.predict(image_path, question=question)["pred_mask"]

# ===== Save =====
os.makedirs(os.path.dirname(save_path), exist_ok=True)
mask_img = TF.to_pil_image(binary_mask.cpu())  # convert tensor to PIL image
mask_img.save(save_path)
print(f"✅ Saved binary mask to: {save_path}")
