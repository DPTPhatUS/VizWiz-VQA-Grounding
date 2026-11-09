"""CPU validation diagnostics shared by the public B1 workflow and local analysis.

Reads canonical evaluator IoUs; native PNGs optionally cross-check those scores
and supply extent statistics. All GT-dependent measures are offline analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageOps

Mask = NDArray[np.bool_]


def read_json(path: Path) -> dict:
    def unique(pairs: list[tuple[str, object]]) -> dict:
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(f"Duplicate JSON key {k}: {path}")
            out[k] = v
        return out

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def mask(path: Path) -> Mask:
    with Image.open(path) as im:
        a = np.asarray(im.convert("L"))
    if not np.isin(a, [0, 255]).all():
        raise ValueError(f"Expected native binary 0/255 PNG: {path}")
    return a > 127


def extent(pred: Mask, gt: Mask) -> dict:
    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError("Prediction and GT must have identical native H,W")
    tp = int((pred & gt).sum())
    fp, fn = int((pred & ~gt).sum()), int((~pred & gt).sum())
    p, g = tp + fp, tp + fn
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_area": p,
        "gt_area": g,
        "area_ratio": p / g if g else None,
        "fp_fraction": fp / gt.size,
        "fn_fraction": fn / gt.size,
        "precision": tp / p if p else None,
        "recall": tp / g if g else None,
    }


def polygon_mask(size: tuple[int, int], polygons: list) -> Mask:
    canvas = Image.new("1", size)
    draw = ImageDraw.Draw(canvas)
    for poly in polygons:
        p = np.asarray(poly, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
            raise ValueError("Malformed OCR polygon")
        draw.polygon([tuple(point) for point in p], fill=1)
    return np.asarray(canvas, dtype=bool)


def group_ids(rows: dict) -> dict[str, list[str]]:
    groups = {k: [] for k in ("all", "<1%", "1-5%", "5-10%", "<5%", ">=10%")}
    for name in sorted(rows):
        r, keys = rows[name], ["all"]
        area = r["gt_fraction"]
        keys += [
            "<1%"
            if area < 0.01
            else "1-5%"
            if area < 0.05
            else "5-10%"
            if area < 0.1
            else ">=10%"
        ]
        if area < 0.05:
            keys += ["<5%"]
        if "ocr_count" in r:
            keys += ["ocr_present" if r["ocr_count"] else "ocr_empty"]
            coverage = r["ocr_gt_coverage"]
            if coverage is not None:
                keys += ["ocr_gt_touch" if coverage > 0 else "ocr_gt_no_touch"]
                if coverage >= 0.5:
                    keys += ["ocr_gt_coverage>=0.5"]
        for key in keys:
            groups.setdefault(key, []).append(name)
    return groups


def summarize(values: list[float], rng: np.random.Generator, repeats: int) -> dict:
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    a = np.asarray(values)
    boot = [float(rng.choice(a, len(a), replace=True).mean()) for _ in range(repeats)]
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "quantiles_0_10_50_90_100": np.quantile(a, [0, 0.1, 0.5, 0.9, 1]).tolist(),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
    }


def aggregate(rows: dict, seed: int, repeats: int, paired: bool = False) -> dict:
    rng, out = np.random.default_rng(seed), {}
    for key, ids in group_ids(rows).items():
        selected = [rows[k] for k in ids]
        field = "delta_iou" if paired else "iou"
        out[key] = summarize([r[field] for r in selected], rng, repeats)
        if paired:
            out[key].update(
                improved=sum(r[field] > 0.05 for r in selected),
                degraded=sum(r[field] < -0.05 for r in selected),
                fixed=sum(r["iou_a"] < 0.5 <= r["iou_b"] for r in selected),
                broken=sum(r["iou_b"] < 0.5 <= r["iou_a"] for r in selected),
            )
        for metric in (
            ("delta_fp_fraction", "delta_fn_fraction")
            if paired
            else ("fp_fraction", "fn_fraction", "area_ratio")
        ):
            if selected and all(r.get(metric) is not None for r in selected):
                out[key][metric] = summarize(
                    [r[metric] for r in selected], rng, repeats
                )
    return out


def analyze(
    data: Path, evaluation: Path, cache: Path | None, seed: int, repeats: int
) -> dict:
    metrics = read_json(evaluation / "metrics.json")
    info, scores = metrics["summary"], metrics["per_sample"]
    if (
        info["split"] != "val"
        or info.get("text_mode") != "question"
        or info.get("answer_in_input") is not False
    ):
        raise ValueError("Requires explicit question-only validation metadata")
    records = read_json(data / "val_grounding.json")
    if set(scores) != set(records) or info["num_samples"] != len(records):
        raise ValueError(
            "Evaluation/annotation IDs or counts disagree; partial results forbidden"
        )
    if any(Path(k).name != k for k in records) or len(
        {Path(k).stem for k in records}
    ) != len(records):
        raise ValueError("Unsafe or colliding image IDs")
    if not scores or not all(np.isfinite(v) and 0 <= v <= 1 for v in scores.values()):
        raise ValueError("Invalid IoU values")
    if not np.isclose(
        np.mean(list(scores.values())), info["mean_iou"], atol=1e-7, rtol=0
    ):
        raise ValueError("Saved mean disagrees with per-image scores")
    rows, has_masks = {}, (evaluation / "masks").exists()
    z = ZipFile(cache) if cache else None
    try:
        for name in sorted(records):
            stem = Path(name).stem
            gt_path = data / "binary_masks_png" / "val" / f"{stem}.png"
            gt = mask(gt_path)
            h, w = gt.shape
            row = {
                "iou": scores[name],
                "native_hw": [h, w],
                "gt_area": int(gt.sum()),
                "gt_fraction": float(gt.mean()),
                "gt_sha256": sha256(gt_path),
                "gt_path": str(gt_path.resolve()),
                "question": records[name]["question"],
            }
            if has_masks:
                pred_path = evaluation / "masks" / f"{stem}.png"
                e = extent(mask(pred_path), gt)
                if abs(e["iou"] - scores[name]) > 1e-6:
                    raise ValueError(
                        f"PNG IoU differs from canonical evaluator: {name}"
                    )
                row.update(
                    e,
                    prediction_path=str(pred_path.resolve()),
                    prediction_sha256=sha256(pred_path),
                )
                row["too_large_proxy"] = (
                    0 < e["iou"] < 0.5 and (e["area_ratio"] or 0) > 2
                )
            if z:
                ocr = json.loads(z.read(f"ocr_cache/val/{stem}.json"))
                if ocr["filename"] != name or ocr["size"] != [w, h]:
                    raise ValueError(f"OCR ID/dimensions mismatch: {name}")
                regions = ocr["regions"]
                text = polygon_mask((w, h), [r["polygon"] for r in regions])
                row.update(
                    ocr_count=len(regions),
                    ocr_gt_coverage=float((gt & text).sum() / gt.sum())
                    if gt.any()
                    else None,
                    ocr_image_fraction=float(text.mean()),
                    ocr_mean_rec_score=float(
                        np.mean([r["confidence"] for r in regions])
                    )
                    if regions
                    else None,
                    ocr_empty_strings=sum(not r["text"].strip() for r in regions),
                )
            rows[name] = row
    finally:
        if z:
            z.close()
    return {
        "schema": "ocrg-diagnostics-v1",
        "split": "val",
        "source": info,
        "provenance": {
            "metrics_sha256": sha256(evaluation / "metrics.json"),
            "annotations_sha256": sha256(data / "val_grounding.json"),
            "ocr_archive_sha256": sha256(cache) if cache else None,
        },
        "has_masks": has_masks,
        "seed": seed,
        "bootstrap_repeats": repeats,
        "uncertainty": "Image bootstrap conditional on one fitted checkpoint, not seed uncertainty.",
        "rows": rows,
        "groups": aggregate(rows, seed, repeats),
    }


def pair(a: dict, b: dict, seed: int, repeats: int) -> dict:
    if (
        a["split"] != "val"
        or b["split"] != "val"
        or a["rows"].keys() != b["rows"].keys()
    ):
        raise ValueError("Paired comparison requires identical validation IDs")
    for field in ("annotations_sha256", "ocr_archive_sha256"):
        if a["provenance"][field] != b["provenance"][field]:
            raise ValueError(f"Paired sources differ: {field}")
    rows = {}
    for name, r in a["rows"].items():
        s = b["rows"][name]
        if r["gt_sha256"] != s["gt_sha256"] or r["native_hw"] != s["native_hw"]:
            raise ValueError(f"Paired GT differs: {name}")
        row = {
            **r,
            "iou_a": r["iou"],
            "iou_b": s["iou"],
            "delta_iou": s["iou"] - r["iou"],
        }
        for field in ("fp", "fn", "fp_fraction", "fn_fraction"):
            if field in r and field in s:
                row[f"delta_{field}"] = s[field] - r[field]
        rows[name] = row
    return {
        "schema": "ocrg-paired-v1",
        "split": "val",
        "sources": [a["source"], b["source"]],
        "provenance": [a["provenance"], b["provenance"]],
        "rows": rows,
        "groups": aggregate(rows, seed, repeats, True),
        "seed": seed,
        "bootstrap_repeats": repeats,
        "warning": "Data identity checked; matched training recipes and budgets require review.",
    }


def hierarchical(comparisons: list[dict], seed: int, repeats: int) -> dict:
    if len(comparisons) < 2:
        raise ValueError("Need >=2 paired training seeds; target >=3")
    ids = sorted(comparisons[0]["rows"])
    seeds = [c["sources"][0]["recipe"]["seed"] for c in comparisons]
    if len(set(seeds)) != len(seeds) or any(
        c["sources"][0]["recipe"]["seed"] != c["sources"][1]["recipe"]["seed"]
        for c in comparisons
    ):
        raise ValueError("Require distinct, matched seed pairs")
    for c in comparisons:
        if (
            c["split"] != "val"
            or sorted(c["rows"]) != ids
            or c["provenance"][0]["annotations_sha256"]
            != comparisons[0]["provenance"][0]["annotations_sha256"]
        ):
            raise ValueError("Seed comparisons have incompatible data")
    rng, result = np.random.default_rng(seed), {}
    for group, members in group_ids(comparisons[0]["rows"]).items():
        if not members:
            result[group] = {"n": 0}
            continue
        x = np.array(
            [[c["rows"][k]["delta_iou"] for k in members] for c in comparisons]
        )
        boot = [
            float(
                x[rng.integers(len(x), size=len(x))][
                    :, rng.integers(len(members), size=len(members))
                ].mean()
            )
            for _ in range(repeats)
        ]
        result[group] = {
            "n": len(members),
            "seeds": len(x),
            "per_seed_delta": x.mean(1).tolist(),
            "mean_delta": float(x.mean()),
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        }
    return {
        "groups": result,
        "seed": seed,
        "bootstrap_repeats": repeats,
        "rule": "Resample paired seeds then shared image indices; exploratory if fewer than 3 seeds.",
    }


def gallery(
    report: dict, data: Path, cache: Path, output: Path, seed: int, count: int
) -> None:
    if not report.get("has_masks") or count < 1:
        raise ValueError("Gallery requires native PNGs and positive sample count")
    output.mkdir(exist_ok=False)
    rng, chosen = np.random.default_rng(seed), set()
    groups = group_ids(report["rows"])
    for group in ("<1%", "1-5%", "5-10%", ">=10%"):
        ids = groups[group]
        chosen.update(rng.choice(ids, min(count, len(ids)), replace=False).tolist())
    entries = []
    with ZipFile(cache) as z:
        for name in sorted(chosen):
            row = report["rows"][name]
            with Image.open(data / "val" / name) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
            ocr = json.loads(z.read(f"ocr_cache/val/{Path(name).stem}.json"))
            if (
                list(image.size) != ocr["size"]
                or hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
                != ocr["pixel_sha256"]
            ):
                raise ValueError(f"Upright image differs from OCR: {name}")
            g, p = mask(Path(row["gt_path"])), mask(Path(row["prediction_path"]))
            panels = [image.copy()]
            for m, color in ((g, [0, 255, 0]), (p, [0, 160, 255])):
                a = np.array(image).copy()
                a[m] = (a[m] * 0.45 + np.array(color) * 0.55).astype(np.uint8)
                panels.append(Image.fromarray(a))
            a = np.array(image).copy()
            a[p & ~g], a[~p & g] = [255, 0, 0], [0, 80, 255]
            error = Image.fromarray(a)
            draw = ImageDraw.Draw(error)
            for i, region in enumerate(ocr["regions"]):
                pts = [tuple(pt) for pt in region["polygon"]]
                draw.line(pts + pts[:1], fill="yellow", width=2)
                draw.text(pts[0], str(i), fill="yellow")
            panels.append(error)
            for panel in panels:
                panel.thumbnail((480, 480))
            combined = Image.new(
                "RGB",
                (sum(p.width for p in panels), max(p.height for p in panels)),
                "white",
            )
            x = 0
            for panel in panels:
                combined.paste(panel, (x, 0))
                x += panel.width
            filename = Path(name).stem + ".png"
            combined.save(output / filename)
            texts = " | ".join(
                f"{i}: {r['text']} ({r['confidence']:.3f})"
                for i, r in enumerate(ocr["regions"])
            )
            entries.append(
                f"<h2>{html.escape(name)}</h2><p>{html.escape(row['question'])}</p><img src='{filename}'><p>{html.escape(texts)}</p>"
            )
    (output / "selection.json").write_text(
        json.dumps(
            {
                "policy": "Random per disjoint GT-size bin, offline",
                "seed": seed,
                "ids": sorted(chosen),
            },
            indent=2,
        )
    )
    (output / "index.html").write_text(
        "<meta charset='utf-8'><p>Input | GT green | prediction cyan | FP red, FN blue, OCR yellow. These are spatial proxies, not semantic labels.</p>"
        + "\n".join(entries),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("analyze", "pair", "hierarchical", "gallery"))
    for key in ("data", "eval-dir", "cache"):
        parser.add_argument(f"--{key}", type=Path)
    parser.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--count-per-bin", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    if args.mode == "analyze":
        if not args.data or not args.eval_dir:
            parser.error("analyze requires --data and --eval-dir")
        result = analyze(
            args.data, args.eval_dir, args.cache, args.seed, args.bootstrap
        )
    elif args.mode == "pair":
        if not args.inputs or len(args.inputs) != 2:
            parser.error("pair requires two --inputs")
        result = pair(*(read_json(p) for p in args.inputs), args.seed, args.bootstrap)
    elif args.mode == "hierarchical":
        if not args.inputs:
            parser.error("hierarchical requires --inputs")
        result = hierarchical(
            [read_json(p) for p in args.inputs], args.seed, args.bootstrap
        )
    else:
        if not args.inputs or len(args.inputs) != 1 or not args.data or not args.cache:
            parser.error("gallery requires one --inputs, --data and --cache")
        gallery(
            read_json(args.inputs[0]),
            args.data,
            args.cache,
            args.output,
            args.seed,
            args.count_per_bin,
        )
        return
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
