"""GT-free online grounding with a frozen parent and a small evidence adapter."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

from dataset import collate_samples, image_views
from evidence_transport_data import collate_features, encoded_features
from models.evidence_transport import EvidenceTransport, compose_native
from models.text_encoder import masked_mean
from native_refinement_data import refinement_inputs
from ocr_cache import CACHE_VERSION, file_hash, pixel_hash, read_json
from training.joint_checkpoint import read_joint
from utils import to_device


@torch.inference_mode()
def predict(parent, adapter, image, question, record, max_candidates, device):
    inputs = refinement_inputs(image, record, max_candidates, mode="native_point")
    batch = collate_samples(
        [
            {
                **image_views(image, parent.image_size, parent.detail_size),
                "question_text": f"Q: {question.strip()}",
            }
        ]
    )
    batch = to_device(batch, device)
    visual, text = parent.encode_inputs(batch)
    base = parent.decode_features(batch, visual, text)["logits"]
    ids = inputs["ocr_valid"].nonzero().flatten()
    words = (
        parent.text_encoder([inputs["ocr_texts"][j] for j in ids.tolist()])
        if len(ids)
        else None
    )
    words = (
        masked_mean(words.tokens, words.valid_mask)
        if words is not None
        else torch.zeros(0, text.tokens.shape[-1], device=device)
    )
    features = encoded_features(
        "image",
        visual[-1][0],
        text.tokens[0, text.attention_mask[0]],
        words,
        inputs["ocr_boxes"][ids].to(device),
        inputs["refine_boxes"][ids].to(device),
        inputs["ocr_confidence"][ids],
        inputs["native_rgb"][0].to(device),
    )
    packed, _ = collate_features([(features, None)])
    packed = to_device(packed, device)
    output = adapter(packed)
    size = (image.height, image.width)
    evidence = compose_native(output, packed, 0, size)
    probability = (1 - output["gate"][0]) * F.interpolate(
        base.float(), size, mode="bilinear", align_corners=False
    )[0].sigmoid() + output["gate"][0] * evidence
    parent_mask = (
        F.interpolate(base.float(), size, mode="bilinear", align_corners=False)[0, 0]
        > 0
    )
    return torch.where(output["gate"][0] == 0, parent_mask, probability[0] > 0.5)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("checkpoint", "parent-checkpoint", "image", "question", "output"):
        p.add_argument("--" + key, required=True)
    ocr = p.add_mutually_exclusive_group(required=True)
    ocr.add_argument(
        "--cache-root", help="OCR cache with manifest.json and single/IMAGE_STEM.json"
    )
    ocr.add_argument("--no-ocr", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved["config"]["parent_sha256"] != file_hash(args.parent_checkpoint):
        raise ValueError("Parent checkpoint differs from the adapter training cache")
    wrapper, _ = read_joint(args.parent_checkpoint)
    parent = wrapper.pyramid.to(args.device).eval().requires_grad_(False)
    adapter = (
        EvidenceTransport(
            **saved["config"]["dimensions"], width=saved["config"]["width"]
        )
        .to(args.device)
        .eval()
    )
    adapter.load_state_dict(saved["model"], strict=True)
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if args.cache_root:
        cache = Path(args.cache_root)
        if (
            read_json(cache / "manifest.json")["protocol"]
            != saved["config"]["cache_protocol"]
        ):
            raise ValueError("OCR protocol differs from adapter training")
        record = read_json(
            cache / "single" / Path(args.image).with_suffix(".json").name
        )
    else:
        record = {
            "version": CACHE_VERSION,
            "size": list(image.size),
            "pixel_sha256": pixel_hash(image),
            "regions": [],
        }
    prediction = predict(
        parent,
        adapter,
        image,
        args.question,
        record,
        saved["config"]["max_candidates"],
        args.device,
    )
    path = Path(args.output)
    if path.exists():
        raise ValueError("Prediction output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction.cpu().numpy().astype(np.uint8) * 255).save(path)


if __name__ == "__main__":
    main()
