"""GT-free image + question prediction for a joint/control checkpoint."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps

from dataset import collate_samples
from native_refinement_data import refinement_inputs
from ocr_cache import CACHE_VERSION, pixel_hash, read_json
from ocr_data import inference_sample
from training.joint_checkpoint import read_joint
from utils import to_device


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--question", required=True)
    evidence = p.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--cache-root")
    evidence.add_argument("--no-ocr", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    model, saved = read_joint(args.checkpoint)
    target = Path(args.output)
    if target.exists():
        raise ValueError("Prediction output already exists")
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
            != saved["run_config"]["protocol"]["cache_protocol"]
        ):
            raise ValueError("OCR protocol differs from training")
        record = read_json(
            cache / "single" / Path(args.image).with_suffix(".json").name
        )
    count = saved["run_config"]["max_candidates"]
    sample = inference_sample(
        image, args.question, record, model.image_size, model.detail_size, count
    )
    sample.update(
        refinement_inputs(
            image,
            record,
            count,
            "native_crop" if model.variant == "joint-crop" else "native_point",
        )
    )
    model.to(args.device).eval()
    with torch.inference_mode():
        output = model(to_device(collate_samples([sample]), args.device))
        mask = (output["native_logits"][0][0] > 0).cpu().numpy().astype("uint8") * 255
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(target)


if __name__ == "__main__":
    main()
