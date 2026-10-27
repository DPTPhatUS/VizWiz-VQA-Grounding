"""GT-free grounding for one image and question using a previously cached OCR record."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

from dataset import collate_samples
from evidence_selection import selected_native_mask
from models.native_refinement_model import EXPERIMENT as NATIVE_EXPERIMENT
from models.ocr_pointer_model import EXPERIMENT as POINTER_EXPERIMENT
from models.ocr_routing_model import EXPERIMENT as ROUTING_EXPERIMENT
from models.ocr_scope_model import EXPERIMENT as SCOPE_EXPERIMENT
from models.ocr_selection_model import EXPERIMENT as SELECTION_EXPERIMENT
from native_refinement_data import refinement_inputs
from ocr_cache import CACHE_VERSION, pixel_hash, read_json
from ocr_data import inference_sample
from ocr_routing_data import routing_inputs
from ocr_selection_data import selection_inputs
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
    native = model.experiment_config["experiment"] == NATIVE_EXPERIMENT
    if native:
        sample.update(
            refinement_inputs(
                image,
                record,
                saved["run_config"]["max_candidates"],
                model.experiment_config["architecture"],
            )
        )
    if model.experiment_config["experiment"] == ROUTING_EXPERIMENT:
        sample.update(
            routing_inputs(image, record, saved["run_config"]["max_candidates"])
        )
    selection = model.experiment_config["experiment"] in {
        SELECTION_EXPERIMENT,
        SCOPE_EXPERIMENT,
        POINTER_EXPERIMENT,
    }
    if selection:
        sample.update(
            selection_inputs(image, record, saved["run_config"]["max_candidates"])
        )
        sample["original_size"] = (image.height, image.width)
    model.to(args.device).eval()
    with torch.inference_mode():
        batch = to_device(collate_samples([sample]), args.device)
        output = model(batch)
        if native:
            prediction = (output["native_logits"][0][0] > 0).cpu().numpy()
        elif selection:
            prediction = selected_native_mask(output, batch, 0).cpu().numpy()
        else:
            native = F.interpolate(
                output["logits"].float(),
                (image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )
            prediction = native[0, 0].cpu().numpy() > 0

    target = Path(args.output)
    if target.exists():
        raise ValueError("Prediction output already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction.astype(np.uint8) * 255).save(target)


if __name__ == "__main__":
    main()
