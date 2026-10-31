"""Forward FLOPs (CPU or GPU) and GPU latency of QGDP-Net for one image.

Usage: uv run --frozen python revision/cost.py OUT_JSON [--detail-input image|zeros|low_res] [--device cuda]
FLOPs use torch.utils.flop_counter (2 FLOPs per multiply-add). Latency is the median of
50 timed forward passes at batch 1 after 10 warm-up passes, in FP32 (the evaluation
precision) and under FP16 autocast.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.model import DETAIL_INPUTS, GroundingModel  # noqa: E402

MODULES = ["image_encoder", "text_encoder", "text_proj", "cross_attn",
           "question_projection", "detail_pyramid", "decoder", "refiner"]


def batch_on(device):
    return {
        "image": torch.rand(1, 3, 336, 336, device=device),
        "detail_image": torch.rand(1, 3, 768, 768, device=device),
        "question_text": ["Q: What is the expiration date on this bottle?"],
    }


def latency(model, device, amp):
    batch = batch_on(device)
    times = []
    with torch.inference_mode(), torch.autocast(device.type, enabled=amp):
        for i in range(60):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(batch)
            torch.cuda.synchronize(device)
            if i >= 10:
                times.append(1000 * (time.perf_counter() - start))
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--detail-input", choices=DETAIL_INPUTS, default="image")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    model = GroundingModel(detail_input=args.detail_input).to(device).eval()
    counter = FlopCounterMode(display=False, depth=None)
    with torch.no_grad(), counter:
        model(batch_on(device))
    counts = counter.get_flop_counts()
    result = {"detail_input": args.detail_input, "device": str(device),
              "total_gflops": sum(counts.get("Global", {}).values()) / 1e9, "modules": {}}
    for name in MODULES:
        key = next((k for k in counts if k.split(".")[-1] == name and k.count(".") == 1), None)
        result["modules"][name] = {
            "gflops": (sum(counts[key].values()) if key else 0) / 1e9,
            "params_m": sum(p.numel() for p in getattr(model, name).parameters()) / 1e6,
        }
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
        result["latency_ms_fp32"] = latency(model, device, amp=False)
        result["latency_ms_amp"] = latency(model, device, amp=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
