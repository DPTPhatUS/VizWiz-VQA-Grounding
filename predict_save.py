"""Predict one image/question mask using this branch's checkpoint configuration."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps

from dataset import image_views
from models.model import GroundingModel
from models.zoom import prediction_at_size
from utils import (
    checkpoint_model_kwargs,
    load_model_weights,
    read_checkpoint,
    to_device,
)


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
    if teacher:
        raise ValueError("This question-only experiment has no teacher inference mode")
    model = GroundingModel(**checkpoint_model_kwargs(saved))
    load_model_weights(model, saved)
    model.to(device).eval()
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    question_text = f"Q: {question}"
    batch = {
        key: [tensor] if key == "native_image" else tensor.unsqueeze(0)
        for key, tensor in image_views(
            image, model.image_size, model.detail_size
        ).items()
    }
    batch.update(
        {
            "question_text": [question_text],
            "filename": [Path(image_path).name],
        }
    )
    output = model(to_device(batch, device))
    mask = prediction_at_size(output, 0, (image.height, image.width))[0, 0] > 0
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
