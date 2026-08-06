"""bake_exif.py — bake EXIF rotation into VizWiz images, strip EXIF metadata.

Some images in data/vizwiz/{train,val,test}/ carry an EXIF orientation tag
(typically 6 = "rotate 90° CW for display").  When PIL loads such an image
with ``Image.open()``, it returns the raw buffer in pre-rotation pixel
order, while any image viewer (and the human annotators who drew the
binary masks) sees the post-rotation / displayed view.  This causes a
systematic mismatch between the loaded image and the on-disk mask.

This script eliminates the mismatch at the source: for every image whose
EXIF orientation is not 1, it transposes the pixels into displayed-view
order and saves the JPEG with all EXIF metadata stripped.  After
running, ``Image.open()`` returns exactly the view a human sees in any
image viewer, and the on-disk mask is naturally aligned without any
``ImageOps.exif_transpose`` wrapper in the loaders.

This is the image-side analog of ``fix_flipped_masks.py``.  Together they
make the EXIF orientation flag irrelevant to the data pipeline.

JPEG re-encoding uses quality 95 by default, which adds minimal visible
degradation.  The original EXIF (camera, GPS, timestamp) is intentionally
discarded — the script normalizes the data, it does not preserve
provenance.

By default the script writes to a new directory (set with
``--output-root``) so the source tree is never modified.  The new tree
mirrors the source: every EXIF-rotated image is re-saved as a fresh
JPEG in displayed view with no EXIF; every already-correct image is
symlinked from the source.  Pass ``--in-place`` to overwrite the
originals instead.

Usage:
    uv run bake_exif.py --dry-run --output-root data/vizwiz_baked
    uv run bake_exif.py --output-root data/vizwiz_baked
    uv run bake_exif.py --in-place
    uv run bake_exif.py --splits val --output-root data/vizwiz_baked
    uv run bake_exif.py --output-root data/vizwiz_baked --quality 90
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


def read_exif(img: Image.Image) -> int:
    """Return EXIF orientation tag 0x0112; default 1 if missing or unreadable."""
    try:
        exif = img.getexif()
    except Exception:
        return 1
    return exif.get(0x0112, 1) if exif else 1


def find_rotated(
    data_root: Path, splits: list[str]
) -> list[tuple[str, Path, int]]:
    """Yield (split, jpg_path, exif_orientation) for every image that
    has EXIF orientation != 1.  Already-baked images (no EXIF tag) are
    skipped, so the script is idempotent.
    """
    out: list[tuple[str, Path, int]] = []
    for split in splits:
        d = data_root / split
        if not d.exists():
            print(f"  [WARN] {d} does not exist; skipping", file=sys.stderr)
            continue
        for p in sorted(d.glob("*.jpg")):
            try:
                img = Image.open(p)
            except Exception as e:
                print(f"  [WARN] {split}/{p.name}: failed to open ({e})", file=sys.stderr)
                continue
            exif = read_exif(img)
            if exif != 1:
                out.append((split, p, exif))
    return out


def bake_one(src: Path, dst: Path, quality: int) -> int:
    """Load ``src`` with EXIF transpose applied, save as JPEG with
    all EXIF stripped at ``dst``.  Returns the old EXIF orientation.
    """
    img = Image.open(src)
    old_exif = read_exif(img)
    baked = ImageOps.exif_transpose(img)
    # ``exif=b""`` strips all EXIF (no orientation tag remains, so
    # Image.open() will return pixels as-is, which are now the displayed
    # view).
    baked.save(dst, format="JPEG", quality=quality, exif=b"")
    return old_exif


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-root", type=Path, default=Path("data/vizwiz"))
    ap.add_argument(
        "--output-root", type=Path, default=None,
        help="Write baked images to this directory tree (source is never "
             "touched).  Mirrors the source: EXIF=6 images are baked, "
             "EXIF=1 images are symlinked.  Mutually exclusive with --in-place.",
    )
    ap.add_argument(
        "--in-place", action="store_true",
        help="Overwrite the source images instead of writing to --output-root. "
             "Destructive; originals are not backed up.",
    )
    ap.add_argument(
        "--splits", type=str, default="train,val,test",
        help="Comma-separated list of splits to process (default: train,val,test)",
    )
    ap.add_argument(
        "--quality", type=int, default=95,
        help="JPEG quality for re-saved images, 1-100 (default: 95)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Detect and report rotated images, but do not modify any files",
    )
    args = ap.parse_args()

    if args.in_place and args.output_root is not None:
        print("ERROR: --in-place and --output-root are mutually exclusive", file=sys.stderr)
        sys.exit(1)
    if not args.in_place and args.output_root is None:
        print("ERROR: specify either --in-place or --output-root", file=sys.stderr)
        sys.exit(1)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        print("ERROR: --splits must list at least one split", file=sys.stderr)
        sys.exit(1)
    if not (1 <= args.quality <= 100):
        print("ERROR: --quality must be in [1, 100]", file=sys.stderr)
        sys.exit(1)

    mode = (
        "DRY RUN (no changes)" if args.dry_run
        else f"BAKE IN-PLACE -> {args.data_root}" if args.in_place
        else f"BAKE -> {args.output_root}"
    )
    print(f"Data root:    {args.data_root}")
    print(f"Output root:  {args.output_root or '(in-place)'}")
    print(f"Splits:       {splits}")
    print(f"JPEG quality: {args.quality}")
    print(f"Mode:         {mode}")
    print()

    rotated = find_rotated(args.data_root, splits)

    if not rotated:
        print("No rotated images found. Nothing to do.")
        return

    by_split: dict[str, list[tuple[Path, int]]] = {}
    exif_counts: Counter[int] = Counter()
    for split, path, exif in rotated:
        by_split.setdefault(split, []).append((path, exif))
        exif_counts[exif] += 1

    print(f"Found {len(rotated)} images with EXIF orientation != 1:")
    for split, items in sorted(by_split.items()):
        split_exif = Counter(e for _, e in items)
        exif_str = ", ".join(f"EXIF={e}:{n}" for e, n in sorted(split_exif.items()))
        print(f"  {split}: {len(items)} to bake  ({exif_str})")
        for path, exif in items[:3]:
            print(f"    {path.name}  EXIF={exif}  size={Image.open(path).size}")
        if len(items) > 3:
            print(f"    ... and {len(items) - 3} more")
    print()

    if args.dry_run:
        print(
            f"DRY RUN: would bake {len(rotated)} images. "
            f"Re-run without --dry-run to apply."
        )
        return

    # Prepare output dirs (only for --output-root mode)
    if not args.in_place:
        for split in splits:
            (args.output_root / split).mkdir(parents=True, exist_ok=True)

    n_baked = 0
    n_symlinked = 0
    n_skipped = 0
    n_fail = 0

    for split in splits:
        src_dir = args.data_root / split
        if not src_dir.exists():
            continue
        for src in sorted(src_dir.glob("*.jpg")):
            dst = (
                src if args.in_place
                else args.output_root / split / src.name
            )
            if not args.in_place and (dst.exists() or dst.is_symlink()):
                n_skipped += 1
                continue
            exif = read_exif(Image.open(src))
            if exif != 1:
                try:
                    bake_one(src, dst, args.quality)
                    n_baked += 1
                except Exception as e:
                    print(f"  [FAIL] {split}/{src.name}: {e}", file=sys.stderr)
                    n_fail += 1
            else:
                if args.in_place:
                    continue
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    import shutil
                    shutil.copy2(src, dst)
                n_symlinked += 1

    print()
    print(
        f"Baked {n_baked} images "
        f"(EXIF orientation baked into pixels, all EXIF metadata stripped)."
    )
    if not args.in_place:
        print(f"Symlinked {n_symlinked} already-correct images (no rewrite).")
        print(f"Skipped {n_skipped} (already exist in output).")
    if n_fail:
        print(f"  {n_fail} failed.")
    print(
        "Re-run with --dry-run to confirm nothing remains; "
        "loaders that wrap Image.open() with ImageOps.exif_transpose() "
        "are now safe no-ops on this dataset."
    )


if __name__ == "__main__":
    main()

