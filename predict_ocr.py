"""GT-free grounding for one image and question using a previously cached OCR record."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

from dataset import collate_samples
from ocr_cache import CACHE_VERSION, pixel_hash, read_json
from ocr_data import inference_sample
from training.ocr_checkpoint import load_ocr_model
from utils import to_device


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-checkpoint", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--question", required=True)
    evidence = p.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--cache-root")
    evidence.add_argument(
        "--no-ocr",
        action="store_true",
        help="Exact visual fallback without an OCR cache",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    model, saved, _ = load_ocr_model(args.parent_checkpoint, args.checkpoint)
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if args.no_ocr:
        record = {
            "version": CACHE_VERSION,
            "size": list(image.size),
            "pixel_sha256": pixel_hash(image),
            "regions": [],
        }
    else:
        cache = Path(args.cache_root)
        if (
            read_json(cache / "manifest.json")["protocol"]
            != saved["run_config"]["cache_protocol"]
        ):
            raise ValueError("OCR protocol differs from training")
        record = read_json(
            cache / "single" / Path(args.image).with_suffix(".json").name
        )
    sample = inference_sample(
        image,
        args.question,
        record,
        model.image_size,
        model.detail_size,
        saved["run_config"]["max_candidates"],
    )
    model.to(args.device).eval()
    with torch.inference_mode():
        output = model(to_device(collate_samples([sample]), args.device))["logits"]
        native = F.interpolate(
            output.float(),
            (image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )
    target = Path(args.output)
    if target.exists():
        raise ValueError("Prediction output already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((native[0, 0].cpu().numpy() > 0).astype(np.uint8) * 255).save(
        target
    )


if __name__ == "__main__":
    main()
