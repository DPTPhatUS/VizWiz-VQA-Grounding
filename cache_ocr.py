"""Cache OCR from upright RGB images only; PaddleOCR is an optional preprocessing dependency."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from ocr_cache import CACHE_VERSION, atomic_json, pixel_hash, read_json, validate_record


def parse_result(result):
    value = result.json
    if isinstance(value, str):
        value = json.loads(value)
    value = value.get("res", value)
    polygons, texts, scores = (
        value[k] for k in ("rec_polys", "rec_texts", "rec_scores")
    )
    if not len(polygons) == len(texts) == len(scores):
        raise ValueError("OCR output field lengths disagree")
    return [
        {"polygon": np.asarray(p).tolist(), "text": str(t), "confidence": float(s)}
        for p, t, s in zip(polygons, texts, scores)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root")
    source.add_argument(
        "--image", help="Single arbitrary image; writes cache_root/single/name.json"
    )
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-side", type=int, default=1600)
    args = parser.parse_args(argv)
    if args.max_side < 32 or not set(args.splits) <= {"train", "val", "test"}:
        parser.error("Invalid maximum side or dataset split")
    from paddleocr import PaddleOCR

    root, cache = (
        Path(args.data_root) if args.data_root else None,
        Path(args.cache_root),
    )
    protocol = {
        "provider": "paddleocr",
        "detector": "PP-OCRv5_mobile_det",
        "recognizer": "PP-OCRv5_mobile_rec",
        "max_side": args.max_side,
        "orientation": False,
        "unwarp": False,
        "recognition_threshold": 0.0,
        "paddleocr": importlib.metadata.version("paddleocr"),
        "paddlepaddle": importlib.metadata.version("paddlepaddle"),
        "coordinate_system": "exif-transposed-RGB-native-xy",
    }
    manifest_path = cache / "manifest.json"
    if manifest_path.exists() and (
        read_json(manifest_path).get("version") != CACHE_VERSION
        or read_json(manifest_path)["protocol"] != protocol
    ):
        raise ValueError(
            "Existing cache uses another protocol; choose a new cache root"
        )
    atomic_json({"version": CACHE_VERSION, "protocol": protocol}, manifest_path)
    engine = PaddleOCR(
        text_detection_model_name=protocol["detector"],
        text_recognition_model_name=protocol["recognizer"],
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_type="max",
        text_det_limit_side_len=args.max_side,
        text_rec_score_thresh=0.0,
        text_recognition_batch_size=8,
        device=args.device,
    )
    timings = []
    splits = args.splits if root else ["single"]
    for split in splits:
        # Enumeration only: question/answer/mask values never go into OCR.
        names = (
            list(read_json(root / f"{split}_grounding.json"))
            if root
            else [Path(args.image).name]
        )
        for name in tqdm(names, desc=f"OCR {split}"):
            target = cache / split / Path(name).with_suffix(".json")
            with Image.open(root / split / name if root else args.image) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            digest = pixel_hash(image)
            if target.exists():
                record = read_json(target)
                validate_record(record)
                if record["pixel_sha256"] != digest or record["size"] != list(
                    image.size
                ):
                    raise ValueError(f"Existing cache does not match image {name}")
                continue
            start = time.perf_counter()
            # Paddle/OpenCV ndarray convention is BGR.
            prediction = list(engine.predict(np.asarray(image)[:, :, ::-1].copy()))
            if len(prediction) != 1:
                raise ValueError("Expected exactly one OCR result per image")
            seconds = time.perf_counter() - start
            record = {
                "version": CACHE_VERSION,
                "filename": name,
                "size": list(image.size),
                "pixel_sha256": digest,
                "seconds": seconds,
                "regions": parse_result(prediction[0]),
            }
            validate_record(record)
            atomic_json(record, target)
            timings.append(seconds)
    print(
        json.dumps(
            {
                "new_images": len(timings),
                "mean_seconds": np.mean(timings).item() if timings else None,
                "note": "Includes first-call warmup; excludes model load and image decode",
            }
        )
    )


if __name__ == "__main__":
    main()
