"""generate_yolo_box.py — YOLO-detected box dataset at data/vizwiz_yolo_box/.

For each train/val/test image, runs the YOLO detector and draws the
top-k (default 1) highest-confidence boxes on the source image at
original resolution. Writes JPEG to data/vizwiz_yolo_box/. Symlinks
binary_masks_png/, meta_data/, and *_grounding.json from the source —
only the image input changes; masks stay as real GT for downstream
supervision.

If no detection passes the confidence threshold, draws a fallback
full-image box so every output image has exactly one annotation.

Uses ``ultralytics.YOLO`` directly (not ``models.ObjectDetector``)
because the project wrapper assumes a single uniform ``(H, W)`` per
batch, which breaks on variable-sized VizWiz images. YOLO's
``predict()`` handles per-image letterboxing and returns boxes in
the original image's pixel coordinate space.

Usage:
    uv run generate_yolo_box.py --checkpoint outputs/detector/weights/best.pt
    uv run generate_yolo_box.py --checkpoint best.pt --splits val --topk 3
    uv run generate_yolo_box.py --checkpoint best.pt --confidence 0.5 --batch-size 32
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from ultralytics import YOLO


def xywh_to_xyxy_pixel(
    box_xywh: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert YOLO's [cx, cy, w, h] in original-image pixel coords
    to a clamped (x1, y1, x2, y2) pixel tuple inside the image.
    """
    cx, cy, bw, bh = box_xywh
    x1 = int(round(float(cx) - float(bw) / 2.0))
    y1 = int(round(float(cy) - float(bh) / 2.0))
    x2 = int(round(float(cx) + float(bw) / 2.0))
    y2 = int(round(float(cy) + float(bh) / 2.0))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def draw_boxes(
    image: Image.Image,
    bboxes: list[tuple[int, int, int, int]],
    line_width: int,
    color: tuple[int, int, int],
) -> Image.Image:
    """Draw one rectangle per bbox on a copy of the image."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
    return out


def full_image_fallback(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Fallback box covering the entire image."""
    return [(0, 0, width - 1, height - 1)]


def _ensure_symlink(src: Path, dst: Path) -> None:
    """Replace dst with a symlink pointing to src.

    If dst is a real directory/file, raise (do NOT clobber user data).
    If dst is a symlink (even a broken one), unlink and re-create.
    """
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        raise FileExistsError(
            f"{dst} exists and is not a symlink; refusing to clobber. "
            f"Remove it manually or pass --output-root to a fresh location."
        )
    os.symlink(src.resolve(), dst)


def setup_output_dirs(data_root: Path, output_root: Path, splits: list[str]) -> None:
    """Mirror the original generate_box.py setup: symlink masks, meta, JSON."""
    output_root.mkdir(parents=True, exist_ok=True)

    masks_dst = output_root / "binary_masks_png"
    masks_dst.mkdir(exist_ok=True)
    for split in splits:
        src = data_root / "binary_masks_png" / split
        if not src.exists():
            print(f"  [WARN] {src} does not exist; skipping symlink for that split")
            continue
        _ensure_symlink(src, masks_dst / split)

    meta_src = data_root / "meta_data"
    if meta_src.exists():
        _ensure_symlink(meta_src, output_root / "meta_data")

    for split in splits:
        src = data_root / f"{split}_grounding.json"
        if not src.exists():
            print(f"  [WARN] {src} does not exist; skipping symlink for {split}_grounding.json")
            continue
        _ensure_symlink(src, output_root / f"{split}_grounding.json")

    for split in splits:
        (output_root / split).mkdir(exist_ok=True)


@torch.no_grad()
def process_split(
    split: str,
    data_root: Path,
    output_root: Path,
    model: YOLO,
    batch_size: int,
    topk: int,
    confidence: float,
    line_width_frac: float,
    color: tuple[int, int, int],
    jpeg_quality: int,
    imgsz: int,
    device: str,
) -> tuple[int, int, int]:
    """Process one split. Returns (n_processed, n_no_detect, n_skipped_missing)."""
    json_path = data_root / f"{split}_grounding.json"
    img_dir = data_root / split
    out_img_dir = output_root / split
    out_img_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)
    filenames = list(data.keys())

    n_processed = 0
    n_no_detect = 0
    n_skipped_missing = 0

    for start in tqdm(range(0, len(filenames), batch_size), desc=f"{split}"):
        batch_files = filenames[start:start + batch_size]

        pil_imgs: list[Image.Image] = []
        valid_files: list[str] = []
        for fn in batch_files:
            p = img_dir / fn
            if not p.exists():
                n_skipped_missing += 1
                continue
            pil_imgs.append(Image.open(p).convert("RGB"))
            valid_files.append(fn)
        if not pil_imgs:
            continue

        # YOLO handles per-image letterboxing. Boxes come back in
        # original-image pixel coords (not the letterbox 640x640).
        results_list = model.predict(
            pil_imgs,
            conf=confidence,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

        for fn, img, results in zip(valid_files, pil_imgs, results_list):
            W, H = img.size
            line_width = max(2, int(round(max(W, H) * line_width_frac)))

            boxes_xyxy: list[tuple[int, int, int, int]] = []
            if results.boxes is not None and len(results.boxes) > 0:
                xywh = results.boxes.xywh.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                # Highest-confidence first.
                order = confs.argsort()[::-1][:topk]
                for idx in order:
                    boxes_xyxy.append(xywh_to_xyxy_pixel(xywh[idx], W, H))

            if not boxes_xyxy:
                boxes_xyxy = full_image_fallback(W, H)
                n_no_detect += 1

            annotated = draw_boxes(img, boxes_xyxy, line_width, color)
            annotated.save(out_img_dir / fn, format="JPEG", quality=jpeg_quality)
            n_processed += 1

    return n_processed, n_no_detect, n_skipped_missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/vizwiz"))
    parser.add_argument("--output-root", type=Path, default=Path("data/vizwiz_yolo_box"))
    parser.add_argument(
        "--splits", type=str, default="train,val,test",
        help="Comma-separated list of splits to process (default: train,val,test)",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to YOLO .pt checkpoint (e.g. outputs/detector/weights/best.pt) "
             "or a registered model name (e.g. yolov8n).",
    )
    parser.add_argument(
        "--topk", type=int, default=1,
        help="Number of top-confidence boxes to draw per image (default: 1). "
             "If YOLO finds fewer, only those are drawn.",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.25,
        help="YOLO confidence threshold for keeping a detection (default: 0.25)",
    )
    parser.add_argument(
        "--image-size", type=int, default=640,
        help="Square input size for YOLO letterboxing (default: 640)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="How many images to pass to YOLO.predict() per call (default: 16)",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for YOLO inference (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--line-width-frac", type=float, default=0.005,
        help="Line width as fraction of max(W,H); min 2px (default: 0.005)",
    )
    parser.add_argument(
        "--box-color", type=str, default="255,0,0",
        help="RGB color for box outline, comma-separated (default: 255,0,0 = red)",
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=95,
        help="JPEG quality 1-100 (default: 95)",
    )
    args = parser.parse_args()

    try:
        color = tuple(int(c) for c in args.box_color.split(","))
        assert len(color) == 3 and all(0 <= c <= 255 for c in color)
    except (ValueError, AssertionError):
        print("ERROR: --box-color must be 'R,G,B' with each in [0,255]", file=sys.stderr)
        sys.exit(1)

    if not (1 <= args.jpeg_quality <= 100):
        print("ERROR: --jpeg-quality must be in [1, 100]", file=sys.stderr)
        sys.exit(1)

    if not (0.0 < args.confidence <= 1.0):
        print("ERROR: --confidence must be in (0, 1]", file=sys.stderr)
        sys.exit(1)

    if args.topk < 1:
        print("ERROR: --topk must be >= 1", file=sys.stderr)
        sys.exit(1)

    if args.batch_size < 1:
        print("ERROR: --batch-size must be >= 1", file=sys.stderr)
        sys.exit(1)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        print("ERROR: --splits must list at least one split", file=sys.stderr)
        sys.exit(1)

    print(f"Data root:    {args.data_root}")
    print(f"Output root:  {args.output_root}")
    print(f"Splits:       {splits}")
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"top-k:        {args.topk}")
    print(f"Confidence:   {args.confidence}")
    print(f"Image size:   {args.image_size}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Device:       {args.device}")
    print(f"Line width:   max(2, round(max(W,H) * {args.line_width_frac}))")
    print(f"Box color:    RGB{color}")
    print(f"JPEG quality: {args.jpeg_quality}")
    print()

    print(f"Loading YOLO from {args.checkpoint} ...")
    model = YOLO(args.checkpoint)
    # Warm up: trigger model load + device transfer. predict() with one
    # zero-image is enough; results are discarded.
    model.predict(
        [Image.new("RGB", (args.image_size, args.image_size))],
        conf=args.confidence,
        imgsz=args.image_size,
        device=args.device,
        verbose=False,
    )

    setup_output_dirs(args.data_root, args.output_root, splits)

    total_processed = 0
    total_no_detect = 0
    total_skipped = 0
    for split in splits:
        print(f"Processing {split}...")
        n_processed, n_no_detect, n_skipped = process_split(
            split, args.data_root, args.output_root,
            model, args.batch_size,
            args.topk, args.confidence,
            args.line_width_frac, color, args.jpeg_quality,
            args.image_size, args.device,
        )
        print(
            f"  {split}: processed={n_processed}, "
            f"fallback_box={n_no_detect}, skipped_missing={n_skipped}"
        )
        total_processed += n_processed
        total_no_detect += n_no_detect
        total_skipped += n_skipped

    print()
    print(
        f"Done. Total processed: {total_processed}, "
        f"fallback_box (no detection): {total_no_detect}, "
        f"skipped_missing: {total_skipped}"
    )
    print(f"Output: {args.output_root}")
    if total_no_detect > 0:
        print(
            f"NOTE: {total_no_detect} images had no detection above "
            f"confidence={args.confidence} and were drawn with a full-image "
            f"fallback box."
        )


if __name__ == "__main__":
    main()
