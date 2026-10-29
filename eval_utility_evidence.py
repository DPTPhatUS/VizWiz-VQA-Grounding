"""Evaluate a previously selected/audited utility policy without training or retuning."""

import argparse
from pathlib import Path

import torch

from models.utility_evidence import UtilityEvidence
from ocr_cache import file_hash, read_json
from run_utility_evidence import evaluate
from utility_evidence import VERSION


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("checkpoint", "policy", "feature-cache", "data-root", "output-dir"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--view", choices=("both", "certified", "proposed"), default="both")
    p.add_argument("--dataset", choices=("both", "val", "test"), default="both")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        p.error("Invalid batch/worker setting")
    policy = read_json(args.policy)
    manifest = read_json(Path(args.feature_cache) / "manifest.json")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    c = saved["config"]
    if (
        manifest.get("version") != VERSION
        or not manifest.get("complete")
        or c["manifest_sha256"] != file_hash(Path(args.feature_cache) / "manifest.json")
    ):
        p.error("Incompatible feature cache")
    if policy.get("version") != VERSION or policy["checkpoint_sha256"] != file_hash(
        args.checkpoint
    ):
        p.error("Policy/adapter checkpoint mismatch")
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        p.error("Evaluation output must be new/empty")
    root.mkdir(parents=True, exist_ok=True)
    model = UtilityEvidence(**c["dimensions"], width=c["width"]).to(args.device).eval()
    model.load_state_dict(saved["model"], strict=True)
    for split in ("val", "test") if args.dataset == "both" else (args.dataset,):
        if (
            file_hash(Path(args.data_root) / f"{split}_grounding.json")
            != manifest["splits"][split]["annotation_sha256"]
        ):
            p.error("Dataset differs from cache")
        for view in ("proposed", "certified") if args.view == "both" else (args.view,):
            threshold = (
                policy["deployed_threshold"]
                if view == "certified"
                else policy["proposed_threshold"]
            )
            evaluate(model, args, manifest, root, split, threshold, view)


if __name__ == "__main__":
    main()
