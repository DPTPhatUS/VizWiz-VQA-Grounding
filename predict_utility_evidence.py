"""GT-free application of a frozen, safety-audited polygon-edit policy."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from cache_utility_evidence import encoded_actions
from dataset import collate_samples, image_views
from models.text_encoder import masked_mean
from models.utility_evidence import UtilityEvidence
from native_refinement_data import refinement_inputs
from ocr_cache import CACHE_VERSION, candidates, file_hash, pixel_hash, read_json
from training.joint_checkpoint import read_joint
from utility_evidence import VERSION, choose_actions, collate_utility, render_action
from utils import to_device


@torch.inference_mode()
def predict(parent, head, image, question, record, max_candidates, threshold, device):
    sample = {
        **image_views(image, parent.image_size, parent.detail_size),
        **refinement_inputs(image, record, max_candidates, mode="native_point"),
        "filename": "image",
        "question_text": f"Q: {question.strip()}",
    }
    batch = collate_samples([sample])
    inputs = to_device(batch, device)
    visual, text = parent.encode_inputs(inputs)
    base = parent.decode_features(inputs, visual, text)["logits"]
    regions = candidates(record, max_candidates)
    if regions:
        encoded = parent.text_encoder([r["text"] for r in regions])
        words = masked_mean(encoded.tokens, encoded.valid_mask)
    else:
        words = torch.zeros(0, parent.text_encoder.output_dim)
    record, _ = encoded_actions(parent, visual, text, base, batch, 0, regions, words)
    inputs, _ = collate_utility([(record["features"], None)])
    inputs = to_device(inputs, device)
    action = int(choose_actions(head(inputs), inputs["valid"], threshold)[0])
    return render_action(record, action)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in (
        "checkpoint",
        "policy",
        "parent-checkpoint",
        "image",
        "question",
        "output",
    ):
        p.add_argument("--" + key, required=True)
    ocr = p.add_mutually_exclusive_group(required=True)
    ocr.add_argument("--cache-root")
    ocr.add_argument("--no-ocr", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    policy = read_json(args.policy)
    if policy.get("version") != VERSION or policy["checkpoint_sha256"] != file_hash(
        args.checkpoint
    ):
        raise ValueError("Policy/adapter snapshot mismatch")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    if config["version"] != VERSION or config["parent_sha256"] != file_hash(
        args.parent_checkpoint
    ):
        raise ValueError("Wrong parent checkpoint")
    wrapper, _ = read_joint(args.parent_checkpoint)
    parent = wrapper.pyramid.to(args.device).eval().requires_grad_(False)
    head = (
        UtilityEvidence(**config["dimensions"], width=config["width"])
        .to(args.device)
        .eval()
    )
    head.load_state_dict(saved["model"], strict=True)
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if args.cache_root:
        cache = Path(args.cache_root)
        if read_json(cache / "manifest.json")["protocol"] != config["protocol"]:
            raise ValueError("OCR protocol mismatch")
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
    mask = predict(
        parent,
        head,
        image,
        args.question,
        record,
        config["max_candidates"],
        policy["deployed_threshold"],
        args.device,
    )
    out = Path(args.output)
    if out.exists():
        raise ValueError("Prediction output already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(out)


if __name__ == "__main__":
    main()
