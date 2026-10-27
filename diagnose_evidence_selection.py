"""Summarize hypothesis selection, missed gains and harmful switches from exported metrics."""

import argparse
import csv
from pathlib import Path

from models.ocr_selection_model import EXPERIMENT
from ocr_cache import atomic_json, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    metrics = read_json(Path(args.predictions) / "metrics.json")
    summary = metrics["summary"]
    if summary["experiment_config"]["experiment"] != EXPERIMENT or not metrics.get(
        "hypotheses"
    ):
        parser.error("Use a selection checkpoint's normal full-view predictions")
    if summary["view"] != "full":
        parser.error("Use the full view")
    # Native per-image groups are recorded by evaluation, without inferring from predictions.
    groups = metrics["sample_groups"]
    rows = []
    for name, score in metrics["per_sample"].items():
        record = metrics["hypotheses"][name]
        base = metrics["base_per_sample"][name]
        rows.append(
            {
                "filename": name,
                "group": groups[name],
                "base_iou": base,
                "final_iou": score,
                "oracle_iou": record["oracle_iou"],
                "selected_index": record["selected_index"],
                "gain_pp": 100 * (score - base),
                "regret_pp": 100 * (record["oracle_iou"] - score),
            }
        )

    def aggregate(selected):
        n = len(selected)
        return {
            "n": n,
            "mean_iou": sum(r["final_iou"] for r in selected) / n,
            "gain_pp": sum(r["gain_pp"] for r in selected) / n,
            "mean_regret_pp": sum(r["regret_pp"] for r in selected) / n,
            "ocr_selected": sum(r["selected_index"] > 0 for r in selected),
            "harmful_switches_5pp": sum(r["gain_pp"] < -5 for r in selected),
            "missed_gains_5pp": sum(
                r["oracle_iou"] > r["base_iou"] + 0.05 and r["gain_pp"] <= 0
                for r in selected
            ),
        }

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        parser.error("Output must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "overall": aggregate(rows),
        "groups": {
            group: aggregate([r for r in rows if r["group"] == group])
            for group in sorted(set(groups.values()))
        },
        "interpretation": "Oracle/regret use GT for diagnostics only, never for inference.",
    }
    atomic_json(result, out / "summary.json")
    with (out / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(result["overall"])


if __name__ == "__main__":
    main()
