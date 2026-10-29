"""Run a frozen pyramid once; cache ragged evidence features for both adapter runs."""

import argparse
import shutil
import time
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor
from tqdm.auto import tqdm

from dataset import make_loader
from evidence_transport_data import CANVAS, CROP, VERSION, encoded_features, targets
from models.text_encoder import masked_mean
from native_refinement_data import NativeRefinementDataset
from ocr_cache import atomic_json, candidates, file_hash, read_json
from training.joint_checkpoint import read_joint
from utils import save_checkpoint


class EvidenceSource(NativeRefinementDataset):
    """Decode each RGB once; extraction never opens validation/test GT masks."""

    def __getitem__(self, index):
        name = self.names[index]
        record = read_json(
            self.cache_root / self.split / Path(name).with_suffix(".json")
        )
        if record["filename"] != name:
            raise ValueError(f"OCR filename mismatch: {name}")
        with Image.open(self.root / self.split / name) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        result = self._native_inputs(image, record)
        result.update(
            filename=name,
            question_text=f"Q: {str(self.records[name].get('question') or '').strip()}",
            image=to_tensor(
                image.resize(
                    (self.image_size, self.image_size), Image.Resampling.BICUBIC
                )
            ),
        )
        if self.split != "train":
            result["detail_image"] = to_tensor(
                image.resize(
                    (self.detail_size, self.detail_size), Image.Resampling.BICUBIC
                )
            )
        return result


def prepare(args):
    started = time.perf_counter()
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            "Cache output must be new/empty; do not mix parent checkpoints"
        )
    model, saved = read_joint(args.checkpoint)
    if model.variant != "pyramid":
        raise ValueError("Use the matched PYRAMID control checkpoint, not joint-crop")
    parent = model.pyramid.to(args.device).eval().requires_grad_(False)
    datasets = {
        split: EvidenceSource(
            args.data_root,
            split,
            args.cache_root,
            parent.image_size,
            detail_size=parent.detail_size,
            max_candidates=args.max_candidates,
            mode="native_point",
        )
        for split in ("train", "val", "test")
    }
    # Native evaluation retains the exact FP32 parent logit grid. No quantized fallback.
    base_bytes = (
        sum(len(datasets[s]) for s in ("val", "test")) * parent.detail_size**2 * 4
    )
    reserve = (
        base_bytes + sum(len(d) for d in datasets.values()) * 300_000 + 2 * 1024**3
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < reserve:
        raise OSError(
            f"Feature-cache estimate plus 2 GiB reserve requires {reserve / 1024**3:.2f} GiB free"
        )
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": VERSION,
        "complete": False,
        "parent_sha256": file_hash(args.checkpoint),
        "parent_epoch": saved["epoch"],
        "parent_config": model.experiment_config,
        "dimensions": {
            "text_dim": parent.text_encoder.output_dim,
            "visual_dim": parent.image_encoder.out_channels,
        },
        "canvas": CANVAS,
        "crop": CROP,
        "max_candidates": args.max_candidates,
        "feature_precision": "float16-storage-float32-extraction",
        "parent_precision": "float32",
        "splits": {},
        "cache_protocol": datasets["train"].cache_protocol,
        "source_sha256": {
            p: file_hash(p)
            for p in (
                "cache_evidence_transport.py",
                "evidence_transport_data.py",
                "models/model.py",
                "native_refinement_ops.py",
            )
        },
    }
    atomic_json(manifest, root / "manifest.json")
    words = OrderedDict()
    scope_positives = 0
    with torch.inference_mode():
        for split, data in datasets.items():
            folder = root / split
            folder.mkdir()
            if split != "train":
                (folder / "base").mkdir()
            for batch in tqdm(
                make_loader(data, args, args.batch_size), desc=f"Cache {split}"
            ):
                inputs = {
                    k: v.to(args.device, non_blocking=True)
                    if k in {"image", "detail_image", "ocr_boxes", "refine_boxes"}
                    else v
                    for k, v in batch.items()
                }
                if split == "train":
                    features = parent.image_encoder(
                        (inputs["image"] - parent.image_mean) / parent.image_std
                    )
                    text = parent.text_encoder(inputs["question_text"])
                else:
                    features, text = parent.encode_inputs(inputs)
                base = (
                    parent.decode_features(inputs, features, text)["logits"]
                    if split != "train"
                    else None
                )
                for i, name in enumerate(batch["filename"]):
                    valid = batch["ocr_valid"][i]
                    ids = valid.nonzero().flatten()
                    texts = [batch["ocr_texts"][i][j] for j in ids.tolist()]
                    missing = list(dict.fromkeys(t for t in texts if t not in words))
                    if missing:
                        encoded = parent.text_encoder(missing)
                        for t, v in zip(
                            missing,
                            masked_mean(encoded.tokens, encoded.valid_mask).cpu(),
                        ):
                            words[t] = v.half().clone()
                    boxes = inputs["ocr_boxes"][i, ids].double()
                    contexts = inputs["refine_boxes"][i, ids].double()
                    embeddings = (
                        torch.stack([words[t] for t in texts])
                        if texts
                        else torch.zeros(0, manifest["dimensions"]["text_dim"])
                    )
                    for t in texts:
                        words.move_to_end(t)
                    while len(words) > 16384:
                        words.popitem(last=False)
                    record = {
                        "features": encoded_features(
                            name,
                            features[-1][i],
                            text.tokens[i, text.attention_mask[i]],
                            embeddings,
                            boxes,
                            contexts,
                            batch["ocr_confidence"][i, ids],
                            batch["native_rgb"][i][0].to(args.device),
                        )
                    }
                    if split == "train":
                        ocr = read_json(
                            Path(args.cache_root)
                            / split
                            / Path(name).with_suffix(".json")
                        )
                        regions = [
                            candidates(ocr, args.max_candidates)[j]
                            for j in ids.tolist()
                        ]
                        with Image.open(
                            Path(args.data_root)
                            / "binary_masks_png"
                            / split
                            / Path(name).with_suffix(".png")
                        ) as mask:
                            if mask.size != tuple(
                                reversed(record["features"]["native_size"])
                            ):
                                raise ValueError(
                                    f"Upright image/GT mask dimensions differ: {name}"
                                )
                            record["targets"] = targets(
                                mask.convert("L"), regions, contexts.cpu().tolist()
                            )
                            scope_positives += int(record["targets"]["scope"])
                    save_checkpoint(record, folder / f"{Path(name).stem}.pt")
                    if base is not None:
                        save_checkpoint(
                            {"logits": base[i].float().cpu().clone()},
                            folder / "base" / f"{Path(name).stem}.pt",
                        )
            manifest["splits"][split] = {
                "names": data.names,
                "ocr_digest": data.cache_digest,
                "annotation_sha256": file_hash(
                    Path(args.data_root) / f"{split}_grounding.json"
                ),
                "supervision_digest": data.supervision_digest
                if split == "train"
                else None,
            }
            atomic_json(manifest, root / "manifest.json")
    manifest["complete"] = True
    manifest["training_scope_positives"] = scope_positives
    manifest["extraction_seconds"] = time.perf_counter() - started
    atomic_json(manifest, root / "manifest.json")


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for flag in ("checkpoint", "data-root", "cache-root", "output-dir"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--max-candidates", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if (
        args.batch_size < 1
        or args.num_workers < 0
        or not 1 <= args.max_candidates <= 128
    ):
        p.error("Invalid batch/worker/candidate count")
    prepare(args)


if __name__ == "__main__":
    main()
