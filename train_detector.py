"""Train a YOLO detector on VizWiz grounding bboxes.

Converts polygon ``answer_grounding`` annotations into YOLO-format labels
(single class, normalized ``[cx, cy, w, h]``), symlinks the source images
into the YOLO data directory, writes a ``data.yaml``, and launches
Ultralytics training.

Usage:
    uv run train_detector.py
    uv run train_detector.py --resume-checkpoint outputs/detector/weights/last.pt
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from models import ObjectDetector


def prepare_split(json_path, image_root, label_dir, image_dir):
    """Convert a VizWiz grounding split into YOLO format.

    Writes one ``.txt`` label per image (single class ``0``) and links each
    image into ``image_dir``.  Idempotent: re-running overwrites labels and
    skips images that are already linked.
    """
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    processed = 0
    for filename, meta in data.items():
        vertices = meta.get("answer_grounding")
        if not vertices or "width" not in meta or "height" not in meta:
            continue

        xs = [v["x"] for v in vertices]
        ys = [v["y"] for v in vertices]
        cx = ((min(xs) + max(xs)) / 2) / meta["width"]
        cy = ((min(ys) + max(ys)) / 2) / meta["height"]
        bw = (max(xs) - min(xs)) / meta["width"]
        bh = (max(ys) - min(ys)) / meta["height"]

        # Link if possible, copy as a fallback for filesystems without
        # symlink support (Windows without dev mode, read-only mounts).
        src = os.path.join(image_root, filename)
        dst = os.path.join(image_dir, filename)
        if not os.path.lexists(dst) and os.path.exists(src):
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                shutil.copy2(src, dst)

        label_path = Path(label_dir) / Path(filename).with_suffix(".txt").name
        with open(label_path, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        processed += 1

    return processed


def main():
    parser = argparse.ArgumentParser(description="Train YOLO on VizWiz grounding bboxes")
    parser.add_argument("--data-root", type=str, default="data/vizwiz")
    parser.add_argument("--model-name", type=str, default="yolov8n")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch", type=int, default=None)
    args = parser.parse_args()

    yolo_dir = os.path.join(args.output_dir, "yolo_data")
    data_yaml = os.path.join(yolo_dir, "data.yaml")

    train_count = prepare_split(
        os.path.join(args.data_root, "train_grounding.json"),
        os.path.join(args.data_root, "train"),
        os.path.join(yolo_dir, "labels", "train"),
        os.path.join(yolo_dir, "images", "train"),
    )
    val_count = prepare_split(
        os.path.join(args.data_root, "val_grounding.json"),
        os.path.join(args.data_root, "val"),
        os.path.join(yolo_dir, "labels", "val"),
        os.path.join(yolo_dir, "images", "val"),
    )
    if train_count == 0:
        print("ERROR: No valid training entries found.", file=sys.stderr)
        sys.exit(1)

    with open(data_yaml, "w") as f:
        f.write(f"path: {os.path.abspath(yolo_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write('names: ["grounded_object"]\n')

    print(f"Train: {train_count}, Val: {val_count}")

    model_name = args.resume_checkpoint or args.model_name
    detector = ObjectDetector(model_name=model_name, imgsz=args.image_size)

    print(f"Training YOLO for {args.num_epochs} epochs at image-size={args.image_size}...")
    detector.train_detector(
        data_config=data_yaml,
        epochs=args.num_epochs,
        resume=args.resume_checkpoint,
        save_every=args.save_every,
        output_dir=args.output_dir,
        imgsz=args.image_size,
        device=args.device,
        batch=args.batch,
    )

    print(f"Checkpoints saved to {os.path.join(args.output_dir, 'detector', 'weights')}")


if __name__ == "__main__":
    main()
