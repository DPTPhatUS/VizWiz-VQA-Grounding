"""Predict one image/question mask using this branch's checkpoint configuration."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor

from models.model import GroundingModel
from utils import checkpoint_model_kwargs, load_model_weights, read_checkpoint


def prediction_parser(description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--question", required=True, help="Question text without a Q: prefix"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


@torch.no_grad()
def predict_image(checkpoint, image_path, question, device, teacher=False):
    saved = read_checkpoint(checkpoint)
    config = saved["run_config"]
    if teacher:
        raise ValueError("Grounding-first has no teacher inference mode")
    model = GroundingModel(**checkpoint_model_kwargs(saved))
    load_model_weights(model, saved)
    model.to(device).eval()
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    question_text = f"Q: {question}"
    batch = {
        "image": to_tensor(
            image.resize(
                (config.get("image_size", 336), config.get("image_size", 336)),
                Image.Resampling.BICUBIC,
            )
        )
        .unsqueeze(0)
        .to(device),  # (1, 3, H, W)
        "text": [question_text],
        "question_text": [question_text],
        "filename": [Path(image_path).name],
    }
    logits = model(batch)["logits"].float()  # (1, 1, H_dec, W_dec)
    mask = (
        F.interpolate(
            logits,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        > 0
    )  # (H_orig, W_orig)
    return image, Image.fromarray(mask.cpu().numpy().astype("uint8") * 255)


def main(argv=None):
    args = prediction_parser().parse_args(argv)
    _, mask = predict_image(args.checkpoint, args.image, args.question, args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output)
    print(f"Saved binary mask to {output}")


if __name__ == "__main__":
    main()
