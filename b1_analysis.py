"""Comprehensive CPU diagnostics after every arm has finished training."""

import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np

import grounding_diagnostics as d
from b1_data import OCRDataset, text_permutation
from models.simple_ocr import ARMS


def selected_epochs(root: Path) -> dict:
    result = {}
    for arm in ARMS:
        history = [
            json.loads(line)
            for line in (root / "training" / arm / "val_curve.jsonl")
            .read_text()
            .splitlines()
        ]
        if [r["epoch"] for r in history] != list(range(1, 11)):
            raise ValueError("Incomplete/duplicate validation curve")
        candidates = [r for r in history if r["epoch"] in (5, 10)]
        result[arm] = max(candidates, key=lambda r: (r["val_miou"], -r["epoch"]))[
            "epoch"
        ]
    return result


def analyze_all(
    root: Path,
    data: Path,
    cache: Path,
    original_reference: dict,
    repeats: int = 2000,
    expected_count: int = 1131,
):
    evaluations = root / "analysis" / "evaluations"
    selected = selected_epochs(root)
    reports = {}
    template = None
    for folder in sorted(p for p in evaluations.iterdir() if p.is_dir()):
        print("CPU diagnostics:", folder.name, flush=True)
        report = d.analyze(
            data, folder, cache if template is None else None, 42, repeats
        )
        if template is None:
            template = report
        else:
            for name, row in report["rows"].items():
                if row["gt_sha256"] != template["rows"][name]["gt_sha256"]:
                    raise ValueError("Changed GT between diagnostic passes")
                row.update(
                    {
                        k: v
                        for k, v in template["rows"][name].items()
                        if k.startswith("ocr_")
                    }
                )
            report["provenance"]["ocr_archive_sha256"] = template["provenance"][
                "ocr_archive_sha256"
            ]
            report["groups"] = d.aggregate(report["rows"], 42, repeats)
        if len(report["rows"]) != expected_count or not report["has_masks"]:
            raise ValueError("Missing complete native-mask outputs")
        if folder.name.endswith("_clean"):
            arm = report["source"]["arm"]
            epoch = report["source"]["epoch"]
            curve = [
                json.loads(l)
                for l in (root / "training" / arm / "val_curve.jsonl")
                .read_text()
                .splitlines()
            ]
            expected = next(r["val_miou"] for r in curve if r["epoch"] == epoch)
            if abs(expected - report["source"]["mean_iou"]) > 1e-6:
                raise ValueError(
                    "Snapshot evaluation does not reproduce its training validation metric"
                )
        if folder.name == "original_b0":
            if set(report["rows"]) != set(original_reference["per_sample"]):
                raise ValueError("Original B0 reference IDs differ")
            if any(
                abs(row["iou"] - original_reference["per_sample"][name]) > 1e-4
                for name, row in report["rows"].items()
            ):
                raise ValueError(
                    "Parent reproduction differs from original validation metrics"
                )
        (folder / "diagnostics.json").write_text(
            json.dumps(report, indent=1, allow_nan=False)
        )
        reports[folder.name] = report
    expected = (
        {"original_b0"}
        | {f"{a}_ep{e}_clean" for a in ARMS for e in (5, 10)}
        | {
            f"{a}_ep{e}_{v}"
            for a in ARMS[1:]
            for e in (5, 10)
            for v in ("empty", "cross_image", "text_shuffle", "geometry")
        }
    )
    if set(reports) != expected:
        raise ValueError(f"Missing/unexpected evaluations: {set(reports) ^ expected}")
    # Include lightweight native targets and OCR records so pixel-level
    # investigation does not depend on private code or unexported per-image data.
    gt_folder = root / "analysis/ground_truth_masks"
    gt_folder.mkdir()
    samples = {}
    with ZipFile(cache) as z:
        ocr = {
            name: json.loads(z.read(f"ocr_cache/val/{Path(name).stem}.json"))
            for name in template["rows"]
        }
    for name, row in template["rows"].items():
        target = gt_folder / Path(row["gt_path"]).name
        shutil.copyfile(row["gt_path"], target)
        samples[name] = {
            "question": row["question"],
            "native_hw": row["native_hw"],
            "gt_mask": "ground_truth_masks/" + target.name,
            "gt_sha256": row["gt_sha256"],
        }
    (root / "analysis/samples.json").write_text(json.dumps(samples, indent=1))
    (root / "analysis/ocr_validation.json").write_text(json.dumps(ocr, indent=1))
    # The synthetic analysis fixture need not build a bank; a real session
    # always has one and exports exact retained indices and intervention maps.
    if (root / "ocr_features.pt").exists():
        import torch

        bank = torch.load(
            root / "ocr_features.pt", map_location="cpu", weights_only=True
        )["val"]
        donor_map = OCRDataset(data, "val", bank, view="cross_image").donors
        details = {}
        for name, r in bank.items():
            order = text_permutation(name, len(r["texts"])).tolist()
            details[name] = {
                "raw_count": r["raw_count"],
                "retained_indices": r["indices"],
                "truncated_text": r["truncated_text"],
                "cross_image_donor": donor_map[name],
                "donor_retained_count": len(bank[donor_map[name]]["texts"]),
                "text_permutation": order,
                "text_strings_changed": sum(
                    r["texts"][i] != r["texts"][j] for i, j in enumerate(order)
                ),
            }
        (root / "analysis/interventions.json").write_text(
            json.dumps(
                {
                    "seed": 42,
                    "per_image": details,
                    "notes": "Cross-image shuffle changes per-image count and evidence. Text shuffle preserves geometry/count but disrupts score-text association. Geometry-only is an inference ablation, not a trained geometry control. Empty is exact OCR-branch off for current trunk.",
                },
                indent=1,
            )
        )
    comparisons = root / "analysis" / "comparisons"
    comparisons.mkdir()
    contrasts = []
    for arm in ARMS:
        for epoch in (5, 10):
            label = f"{arm}_ep{epoch}_clean"
            contrasts.append((label + "_minus_parent", "original_b0", label))
            if arm != "b0_continued":
                contrasts.append(
                    (
                        label + "_minus_b0_same_epoch",
                        f"b0_continued_ep{epoch}_clean",
                        label,
                    )
                )
                contrasts.append(
                    (
                        label + "_minus_selected_b0",
                        f"b0_continued_ep{selected['b0_continued']}_clean",
                        label,
                    )
                )
                for view in ("empty", "cross_image", "text_shuffle", "geometry"):
                    contrasts.append(
                        (
                            f"{arm}_ep{epoch}_{view}_minus_clean",
                            label,
                            f"{arm}_ep{epoch}_{view}",
                        )
                    )
    for name, a, b in contrasts:
        result = d.pair(reports[a], reports[b], 42, repeats)
        (comparisons / f"{name}.json").write_text(
            json.dumps(result, indent=1, allow_nan=False)
        )
    # Identical random IDs for all clean snapshots and the parent, without
    # cherry-picking successes or relying on private local scripts.
    for label, report in reports.items():
        if label.endswith("clean") or label == "original_b0":
            d.gallery(
                report, data, cache, root / "analysis" / f"gallery_{label}", 42, 2
            )
    rows = [
        "# Validation results",
        "",
        "Single exploratory parent/seed42. Image-bootstrap intervals are conditional on fitted weights, not seed uncertainty. All subgroup and intervention analyses are exploratory.",
        "",
        "| Model/view | mIoU % | <1% | <5% | >=10% |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, r in reports.items():
        rows.append(
            "| "
            + label
            + " | "
            + " | ".join(
                f"{100 * r['groups'][g]['mean']:.3f}"
                if r["groups"][g]["mean"] is not None
                else "NA"
                for g in ("all", "<1%", "<5%", ">=10%")
            )
            + " |"
        )
    (root / "analysis/results.md").write_text("\n".join(rows) + "\n")
    extent_rows = [
        "# Native mask extent",
        "",
        "Low-IoU errors use IoU<0.5; zero overlap is separate. Too large: area ratio>2; too small: area ratio<0.5. Precision medians exclude undefined empty predictions.",
        "",
        "| Snapshot | Group | n | Too large | Too small | No overlap | Empty | Median area ratio | Median precision / recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    extent_stats = {}
    for label, r in reports.items():
        if not (label.endswith("clean") or label == "original_b0"):
            continue
        extent_stats[label] = {}
        for group, ids in d.group_ids(r["rows"]).items():
            if not ids:
                continue
            subset = [r["rows"][k] for k in ids]

            def median(field, subset=subset):
                values = [x[field] for x in subset if x[field] is not None]
                return float(np.median(values)) if values else None

            stat = {
                "n": len(subset),
                "too_large": sum(x["too_large_proxy"] for x in subset),
                "too_small": sum(
                    0 < x["iou"] < 0.5
                    and x["area_ratio"] is not None
                    and x["area_ratio"] < 0.5
                    for x in subset
                ),
                "no_overlap": sum(x["iou"] == 0 for x in subset),
                "empty": sum(x["pred_area"] == 0 for x in subset),
                "median_area_ratio": median("area_ratio"),
                "median_precision": median("precision"),
                "median_recall": median("recall"),
            }
            extent_stats[label][group] = stat
            if group in ("all", "<1%", "<5%", ">=10%"):
                fmt = lambda v: "NA" if v is None else f"{v:.3f}"
                extent_rows.append(
                    f"| {label} | {group} | {stat['n']} | {stat['too_large']} | {stat['too_small']} | {stat['no_overlap']} | {stat['empty']} | {fmt(stat['median_area_ratio'])} | {fmt(stat['median_precision'])}/{fmt(stat['median_recall'])} |"
                )
    (root / "analysis/extent.json").write_text(
        json.dumps(extent_stats, indent=2, allow_nan=False)
    )
    (root / "analysis/extent.md").write_text("\n".join(extent_rows) + "\n")
    (root / "analysis/selection.json").write_text(
        json.dumps(
            {
                "selected_epochs": selected,
                "rule": "max overall val among saved5/10, earlier tie",
                "fixed_budget_epoch": 10,
            },
            indent=2,
        )
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for arm in ARMS:
        h = [
            json.loads(line)
            for line in (root / "training" / arm / "history.jsonl")
            .read_text()
            .splitlines()
        ]
        if [r["epoch"] for r in h] != list(range(1, 11)):
            raise ValueError("Incomplete training history")
        axes[0].plot([r["epoch"] for r in h], [r["train_loss"] for r in h], label=arm)
        axes[1].plot(
            [r["epoch"] for r in h], [100 * r["val_miou"] for r in h], label=arm
        )
        axes[2].plot(
            [r["epoch"] for r in h], [r["ocr_beta"] or 0 for r in h], label=arm
        )
    for ax, title in zip(
        axes,
        ("Unsmoothed continuation loss", "Validation mIoU (%)", "OCR residual scale"),
        strict=True,
    ):
        ax.set(title=title, xlabel="Continuation epoch")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "analysis/curves.png", dpi=160)
    fig.savefig(root / "analysis/curves.svg")
    plt.close(fig)
    return {
        "evaluations": len(reports),
        "contrasts": len(contrasts),
        "selected_epochs": selected,
    }
