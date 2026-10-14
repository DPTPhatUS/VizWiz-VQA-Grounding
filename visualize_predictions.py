import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torchvision.transforms as T
import os
from dataset import CLIP_MEAN, CLIP_STD
from models import GroundingModel

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GroundingModel.from_checkpoint("outputs/cross_model_final_epoch100.pt", device=device)

# test data
image_path = "data/vizwiz/test/VizWiz_test_00000006.jpg"
question = "What does it say on here?"
text_input = [f"Q: {question}"]

# preprocess
transform = T.Compose([
    T.Resize((336, 336), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

with Image.open(image_path) as img:
    orig_image = ImageOps.exif_transpose(img).convert("RGB")
if getattr(model, "model_config", None):
    mask = model.predict(orig_image, question=question)["prob_map"]
else:
    image = transform(orig_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image, text_input)
        logits = torch.nn.functional.interpolate(
            logits.float(), size=(orig_image.height, orig_image.width), mode="bilinear", align_corners=False)
        mask = torch.sigmoid(logits)[0, 0].cpu()

# visualize and save
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(orig_image)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(orig_image)
plt.imshow(mask, alpha=0.5, cmap="jet", interpolation='bilinear')
plt.title("Predicted Mask")

os.makedirs("result/cross_model_final_epoch100", exist_ok=True)
save_path = "result/cross_model_final_epoch100/VizWiz_test_00000006_masked.png"  # change path to try other images
plt.savefig(save_path)
plt.close()

print(f"✅ Saved to: {save_path}")
