"""generate_random_box.py — ablation dataset at data/vizwiz_rand_box/.

For each train/val/test image, samples ONE random AABB (no mask access),
draws it on the source image at original resolution, and writes the
annotated image (JPEG) to a new tree at data/vizwiz_rand_box/ that mirrors
data/vizwiz/. The binary_masks_png/, meta_data/, and *_grounding.json files
are symlinked from the source — masks are still the real GT (downstream
training uses them as supervision; only the image input changes).

This is the null-hypothesis counterpart to generate_box.py: if the
grounding model relies on the drawn boxes rather than grounding from
text+image semantics, training on this dataset should collapse. If it
grounds from text alone, perf should match data/vizwiz_box/.

Usage:
    uv run generate_random_box.py
    uv run generate_random_box.py --splits val
    uv run generate_random_box.py --seed 42 --min-size-frac 0.1 --max-size-frac 0.9
    uv run generate_random_box.py --output-root /tmp/rand_box_dataset
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def derive_random_box(
    width: int,
    height: int,
    rng: np.random.Generator,
    min_size_frac: float,
    max_size_frac: float,
) -> tuple[int, int, int, int]:
    """Sample one random AABB that fully fits inside (width, height).

    Box side lengths are drawn independently from
    Uniform(min_size_frac * max_dim, max_size_frac * max_dim)
    so they scale with image size. Top-left corner is drawn from
    Uniform([0, W - box_w]) x Uniform([0, H - box_h]).

    Returns (x1, y1, x2, y2) inclusive pixel coords.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}")
    if not (0.0 < min_size_frac <= max_size_frac <= 1.0):
        raise ValueError(
            f"Need 0 < min_size_frac <= max_size_frac <= 1, "
            f"got min={min_size_frac}, max={max_size_frac}"
        )

    max_dim = max(width, height)
    box_w = int(rng.integers(
        low=max(1, int(round(min_size_frac * max_dim))),
        high=max(2, int(round(max_size_frac * max_dim)) + 1),
    ))
    box_h = int(rng.integers(
        low=max(1, int(round(min_size_frac * max_dim))),
        high=max(2, int(round(max_size_frac * max_dim)) + 1),
    ))
    # Clamp so the box always fits.
    box_w = min(box_w, width)
    box_h = min(box_h, height)

    x1 = int(rng.integers(low=0, high=max(1, width - box_w + 1)))
    y1 = int(rng.integers(low=0, high=max(1, height - box_h + 1)))
    return x1, y1, x1 + box_w - 1, y1 + box_h - 1


def draw_box(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    line_width: int,
    color: tuple[int, int, int],
) -> Image.Image:
    """Draw the AABB outline on a copy of the image. Returns the new image."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
    return out


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
    """Create the output tree, symlinking ALL non-image parts from the source.

    Symlinks created:
      - binary_masks_png/{split}    -> data_root/binary_masks_png/{split}
      - meta_data/                  -> data_root/meta_data
      - {split}_grounding.json      -> data_root/{split}_grounding.json

    Created new (but empty until process_split fills them):
      - {output_root}/{split}/
    """
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
    meta_dst = output_root / "meta_data"
    if meta_src.exists():
        _ensure_symlink(meta_src, meta_dst)

    for split in splits:
        src = data_root / f"{split}_grounding.json"
        if not src.exists():
            print(f"  [WARN] {src} does not exist; skipping symlink for {split}_grounding.json")
            continue
        _ensure_symlink(src, output_root / f"{split}_grounding.json")

    for split in splits:
        (output_root / split).mkdir(exist_ok=True)


def process_split(
    split: str,
    data_root: Path,
    output_root: Path,
    rng: np.random.Generator,
    line_width_frac: float,
    color: tuple[int, int, int],
    jpeg_quality: int,
    min_size_frac: float,
    max_size_frac: float,
) -> tuple[int, int]:
    """Process one split. Reads the JSON (read-only) to know which images to process.

    Returns (n_processed, n_skipped).
    """
    json_path = data_root / f"{split}_grounding.json"
    img_dir = data_root / split
    out_img_dir = output_root / split
    out_img_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    n_processed = 0
    n_skipped = 0

    for filename in data:
        img_path = img_dir / filename
        if not img_path.exists():
            print(f"  [WARN] {split}: missing image {img_path}, skipping {filename}")
            n_skipped += 1
            continue

        with Image.open(img_path) as image:
            w, h = image.size
            bbox = derive_random_box(w, h, rng, min_size_frac, max_size_frac)
            line_width = max(2, int(round(max(w, h) * line_width_frac)))
            annotated = draw_box(image.convert("RGB"), bbox, line_width, color)
            annotated.save(out_img_dir / filename, format="JPEG", quality=jpeg_quality)
        n_processed += 1

    return n_processed, n_skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/vizwiz"))
    parser.add_argument("--output-root", type=Path, default=Path("data/vizwiz_rand_box"))
    parser.add_argument(
        "--splits", type=str, default="train,val,test",
        help="Comma-separated list of splits to process (default: train,val,test)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducible random boxes (default: 42)",
    )
    parser.add_argument(
        "--min-size-frac", type=float, default=0.2,
        help="Min box side as fraction of max(W,H); default 0.2",
    )
    parser.add_argument(
        "--max-size-frac", type=float, default=0.8,
        help="Max box side as fraction of max(W,H); default 0.8",
    )
    parser.add_argument(
        "--line-width-frac", type=float, default=0.005,
        help="Line width as fraction of max(image.width, image.height); min 2px (default: 0.005)",
    )
    parser.add_argument(
        "--box-color", type=str, default="255,0,0",
        help="RGB color for the box outline, comma-separated (default: 255,0,0 = red)",
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=95,
        help="JPEG quality for output images, 1-100 (default: 95)",
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

    if not (0.0 < args.min_size_frac <= args.max_size_frac <= 1.0):
        print(
            "ERROR: need 0 < --min-size-frac <= --max-size-frac <= 1.0",
            file=sys.stderr,
        )
        sys.exit(1)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        print("ERROR: --splits must list at least one split", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    print(f"Data root:    {args.data_root}")
    print(f"Output root:  {args.output_root}")
    print(f"Splits:       {splits}")
    print(f"Seed:         {args.seed}")
    print(f"Box size:     uniform({args.min_size_frac} * max_dim, {args.max_size_frac} * max_dim)")
    print(f"Line width:   max(2, round(max(W,H) * {args.line_width_frac}))")
    print(f"Box color:    RGB{color}")
    print(f"JPEG quality: {args.jpeg_quality}")
    print()

    setup_output_dirs(args.data_root, args.output_root, splits)

    total_processed = 0
    total_skipped = 0
    for split in splits:
        print(f"Processing {split}...")
        n_processed, n_skipped = process_split(
            split, args.data_root, args.output_root,
            rng,
            args.line_width_frac, color, args.jpeg_quality,
            args.min_size_frac, args.max_size_frac,
        )
        print(f"  {split}: processed={n_processed}, skipped={n_skipped}")
        total_processed += n_processed
        total_skipped += n_skipped

    print()
    print(f"Done. Total processed: {total_processed}, skipped: {total_skipped}")
    print(f"Output: {args.output_root}")
    if total_skipped > 0:
        print(
            f"WARNING: {total_skipped} entries were skipped (missing images). "
            f"Re-run with the same args to retry; symlinks are recreated, "
            f"image outputs are overwritten."
        )


if __name__ == "__main__":
    main()
