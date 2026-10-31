"""Print the saved training configuration of a checkpoint (CPU only, read-only).

Usage (local or in a Kaggle CPU cell):
    python dump_checkpoint_config.py /kaggle/input/.../checkpoint_epoch20.pt
Prints every non-tensor entry, so run_config (lr, batch_size, optimizer, ...) is visible.
"""
import json
import sys

import torch


def strip_tensors(value):
    if torch.is_tensor(value):
        return f"<tensor {tuple(value.shape)}>"
    if isinstance(value, dict):
        if value and all(torch.is_tensor(v) for v in value.values()):
            return f"<state dict with {len(value)} tensors>"
        return {str(k): strip_tensors(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strip_tensors(v) for v in value][:50]
    return value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)


def main(path):
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # older checkpoints may store plain Python objects
        print(f"weights_only load failed ({error}); retrying with weights_only=False", file=sys.stderr)
        saved = torch.load(path, map_location="cpu", weights_only=False)
    summary = {k: strip_tensors(v) for k, v in saved.items() if "state_dict" not in k}
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
