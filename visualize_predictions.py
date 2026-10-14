import os
import matplotlib.pyplot as plt
from models import GroundingModel

checkpoint = "outputs/naflex_base_clip_512/checkpoint_epoch20.pt"
image_path = "data/vizwiz/test/VizWiz_test_00000006.jpg"
question = "What does it say on here?"
result = GroundingModel.from_checkpoint(checkpoint).predict(image_path, question=question)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(result["image_pil"])
axes[0].set_title("Original image")
axes[1].imshow(result["image_pil"])
axes[1].imshow(result["prob_map"], alpha=0.5, cmap="jet", interpolation="bilinear")
axes[1].set_title("Predicted mask")
for ax in axes:
    ax.axis("off")
os.makedirs("results/preview", exist_ok=True)
fig.savefig("results/preview/prediction.png")
plt.close(fig)
