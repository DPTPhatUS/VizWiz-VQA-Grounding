"""Audit native OCR sufficiency decisions; categorical probabilities are never IoU estimates."""

import argparse
import csv
import json
from pathlib import Path

from ocr_cache import atomic_json


def aggregate(rows):
    n = len(rows)
    if not n:
        return {"n": 0}
    mean = lambda key: sum(r[key] for r in rows) / n
    positives = [r for r in rows if r["sufficient_ocr_exists"]]
    switched = [r for r in rows if r["switched"]]
    with_words = [r for r in rows if r["token_purity_mae"] is not None]
    return {
        "n": n,
        "base_mean_iou": mean("base_iou"),
        "mean_iou": mean("final_iou"),
        "gain_pp": 100 * mean("gain"),
        "choice_ce": mean("choice_ce"),
        "sufficient_images": len(positives),
        "switches": len(switched),
        "sufficiency_recall": sum(r["switched"] for r in positives) / len(positives)
        if positives
        else None,
        "retrieval_success_given_sufficient": sum(
            r["retrieved_iou"] >= 0.5 for r in positives
        )
        / len(positives)
        if positives
        else None,
        "switched_mask_IoU_ge_0.5_rate": sum(r["final_iou"] >= 0.5 for r in switched)
        / len(switched)
        if switched
        else None,
        "helpful_switches_5pp": sum(r["gain"] > 0.05 for r in switched),
        "harmful_switches_5pp": sum(r["gain"] < -0.05 for r in switched),
        "retrieved_gain_missed_5pp": sum(
            r["retrieved_iou"] > r["base_iou"] + 0.05 and not r["switched"]
            for r in rows
        ),
        "sufficiency_brier": mean("sufficiency_brier"),
        "token_purity_mae": sum(r["token_purity_mae"] for r in with_words)
        / len(with_words)
        if with_words
        else None,
        "bank_oracle_mean_iou": mean("bank_oracle_iou"),
        "retrieved_branch_oracle_mean_iou": mean("retrieved_branch_oracle_iou"),
        "forced_ocr_mean_iou": mean("retrieved_iou"),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args(argv)
    source = Path(args.predictions) / "metrics.json"
    data = json.loads(source.read_text())
    if (
        data["summary"]["experiment_config"]["experiment"]
        != "ocr-sufficiency-pointer-v1"
    ):
        raise ValueError("Use pointer metrics; quality-regression scores are different")
    rows = []
    for name, h in data["hypotheses"].items():
        if "predicted_probability" not in h:
            raise ValueError("Full pointer diagnostics require a non-base view")
        probability, quality, valid = (
            h["predicted_probability"],
            h["actual_quality"],
            h["valid"],
        )
        ids = [i for i, ok in enumerate(valid) if ok and i]
        best = max(ids, key=lambda i: probability[i]) if ids else 0
        exists = any(quality[i] >= 0.5 for i in ids)
        base = data["base_per_sample"][name]
        final = data["per_sample"][name]
        retrieved = quality[best]
        token_errors = [
            abs(p - t)
            for p, t, v in zip(
                h["token_probability"], h["actual_word_purity"], h["word_valid"]
            )
            if v
        ]
        rows.append(
            {
                "name": name,
                "group": data["sample_groups"][name],
                "base_iou": base,
                "final_iou": final,
                "gain": final - base,
                "retrieved_iou": retrieved,
                "retrieved_index": best,
                "selected_index": h["selected_index"],
                "switched": h["selected_index"] > 0,
                "sufficient_ocr_exists": exists,
                "null_probability": probability[0],
                "sufficiency_brier": ((1 - probability[0]) - float(exists)) ** 2,
                "choice_ce": h["choice_ce"],
                "token_purity_mae": sum(token_errors) / len(token_errors)
                if token_errors
                else None,
                "bank_oracle_iou": h["oracle_iou"],
                "retrieved_branch_oracle_iou": max(base, retrieved),
            }
        )
    if not rows:
        raise ValueError("No pointer hypotheses; evaluate full view first")
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Diagnostic output must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    groups = {"overall": aggregate(rows)}
    groups.update(
        {
            g: aggregate([r for r in rows if r["group"] == g])
            for g in sorted({r["group"] for r in rows})
        }
    )
    atomic_json(
        {
            "source": str(source),
            "score_semantics": "categorical probability, not IoU",
            "oracle_warning": "GT-assisted diagnostic bounds, never deployment scores",
            **groups,
        },
        out / "summary.json",
    )
    with (out / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(groups), flush=True)


if __name__ == "__main__":
    main()
