"""CPU-only paired scope analysis: parent, candidate and selected native masks."""

import argparse
import csv
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", required=True, help="Validation eval directory"
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    saved = json.loads((Path(args.predictions) / "metrics.json").read_text())
    if (
        saved["summary"]["dataset"] != "val"
        or saved["summary"]["metric_resolution"] != "original"
    ):
        raise ValueError("Use native-resolution validation evaluation")
    groups = {
        key: []
        for key in (
            "all",
            "empty",
            "small_lt_1pct",
            "medium_1_to_10pct",
            "large_ge_10pct",
        )
    }
    rows = []
    for name, value in saved["scope_diagnostics"].items():
        area = value["gt_area_fraction"]
        group = (
            "empty"
            if area == 0
            else "small_lt_1pct"
            if area < 0.01
            else "medium_1_to_10pct"
            if area < 0.1
            else "large_ge_10pct"
        )
        row = dict(
            filename=name,
            group=group,
            **value,
            parent_iou=saved["views"]["parent"]["per_sample"][name],
            candidate_iou=saved["views"]["candidate"]["per_sample"][name],
            selected_iou=saved["per_sample"][name],
        )
        rows.append(row)
        groups["all"].append(row)
        groups[group].append(row)

    def summarize(values):
        if not values:
            return {"num_samples": 0}

        def mean(key):
            return sum(row[key] for row in values) / len(values)

        return {
            "num_samples": len(values),
            "parent_mean_iou": mean("parent_iou"),
            "candidate_mean_iou": mean("candidate_iou"),
            "selected_mean_iou": mean("selected_iou"),
            "candidate_gain_pp": 100 * mean("candidate_gain"),
            "selected_gain_pp": 100 * mean("selected_gain"),
            "selected_count": sum(row["selected"] for row in values),
            "beneficial_selection_count": sum(
                row["selected_gain"] > 1e-6 for row in values
            ),
            "harmful_selection_count": sum(
                row["selected_gain"] < -1e-6 for row in values
            ),
            "selected_losses_gt5pp": sum(
                row["selected_gain"] < -0.05 for row in values
            ),
            "initially_good_hurt_gt5pp": sum(
                row["parent_iou"] >= 0.75 and row["selected_gain"] < -0.05
                for row in values
            ),
            "mean_candidate_damage": mean("actual_damage"),
            "mean_selected_damage": sum(
                row["actual_damage"] * row["selected"] for row in values
            )
            / len(values),
        }

    report = {
        "protocol": saved["summary"],
        "by_gt_area": {key: summarize(value) for key, value in groups.items()},
        "interpretation": "Paired native validation metrics; GT grouping never controls inference. One checkpoint, not a multi-seed result.",
    }
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Diagnostic output must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    with (out / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["by_gt_area"], indent=2))


if __name__ == "__main__":
    main()
