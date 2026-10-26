# Affinity-transport pyramid (`exp/affinity-transport-pyramid`)

Branch: `exp/affinity-transport-pyramid`, based on `exp/question-guided-detail-pyramid` (`ddfdce1`). This experiment trains fresh for 20 epochs from pretrained CLIP using only the image and question (vision encoder fine-tuned, text encoder frozen, with no auxiliary answer head or external teacher).

## Design and limits

A lightweight local head reads 384-scale decoder semantics, 768-scale high-resolution detail features, and the projected question vector to predict 8 directional neighbor affinities plus a spatial gate. Using these weights, the module computes a normalized local transport of coarse logit differences across the 8-connected neighborhood and adds a bounded residual correction (capped at `[-2.0, +2.0]` in logit space). Uniform interior regions with zero local logit differences remain unchanged, and coarse logits with magnitude above `2.0` cannot flip sign on the 768 model grid. Because this transport operates only on local logit differences around existing boundaries, it refines boundary alignment without synthesizing new disconnected regions or clearing large, confident false-positive interiors.

Ground-truth masks supervise whether adjacent pixels share the same binary label, using per-image class balancing between same-label and cross-boundary neighbor pairs so interior pixels do not dominate the loss. Simultaneously, the main segmentation loss supervises the post-transport mask logits. While [SegFix (ECCV 2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570477.pdf) predicts boundary-to-interior offset fields from an independent boundary network, this module reuses the grounder's question-conditioned features to learn soft 8-neighbor logit transport end-to-end.

```text
Image + Q -> pyramid -> raw final logits -----------------------+
                |                                              |
                +-> decoder semantics + high-res detail + Q      |
                                      |                         |
                              8 affinities + gate               |
                                      |                         |
                                      +-> bounded local transport
                                                    |
                                               final mask
```

The affinity-transport module adds **1,153 trainable parameters**. Its constructor is parameterized by feature channel width and question dimension and supports arbitrary rectangular spatial grids without CLIP-specific imports or fixed patch counts; on this branch it is wired to the 336/768 CLIP detail pyramid.

Training combines the standard final and auxiliary pyramid segmentation losses with an explicit coarse pre-transport segmentation loss (`coarse_weight=0.2`) and the class-balanced neighbor affinity loss (`affinity_weight=0.05`). Supervising both coarse and final logits allows direct paired comparison between pre-transport and post-transport predictions within the same model, in addition to comparing against the standalone parent pyramid run.

## Kaggle setup

Enable Internet and two GPUs, and use a clean `outputs` directory.

Python cell:

```python
!git clone --single-branch --branch exp/affinity-transport-pyramid https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen
```

Python cell:

```python
import json
import os
from pathlib import Path

os.environ["DATA_ROOT"] = "/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz"
os.environ["RUN_ROOT"] = "outputs"
os.environ["RUN_NAME"] = "affinity_transport_pyramid_ep20_seed42"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
root = Path(os.environ["RUN_ROOT"])
(root / "configs").mkdir(parents=True, exist_ok=True)
config = {'overlap_loss': 'dice', 'overlap_weight': 0.5, 'fp_weight': 0.7, 'aux96_weight': 0.1, 'aux192_weight': 0.2, 'aux384_weight': 0.3, 'coarse_weight': 0.2, 'affinity_weight': 0.05}
(root / "configs/spatial.json").write_text(json.dumps(config, indent=2))
```

Precedence follows **JSON config > explicit CLI flags > defaults**. Loss weights are configured in JSON (`fp_weight` applies only when `overlap_loss="tversky"`), while optimizer and encoder-freezing settings can be passed via CLI flags or JSON.

## Preflight

```bash
%%bash
set -euo pipefail
uv run --frozen torchrun --standalone --nproc_per_node=2 preflight.py \
    --data-root "$DATA_ROOT" \
    --batch-size 2 --num-workers 0 --steps 3 \
    --freeze-vision-encoder False --freeze-text-encoder True \
    --gradient-checkpointing \
    --config "$RUN_ROOT/configs/spatial.json"
```

Both ranks must pass preflight, which runs 3 real forward/backward steps under AMP and DDP and discards parameter updates without writing checkpoints.

## Train

A global microbatch of 2 with `--gradient-accumulation-steps 2` gives an effective batch size of 4 across two GPUs (if memory permits, verify `--batch-size 4` in preflight and run with `--batch-size 4 --gradient-accumulation-steps 1`).

```bash
%%bash
set -euo pipefail
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py \
    --data-root "$DATA_ROOT" --output-dir "$RUN_ROOT" \
    --num-epochs 20 \
    --batch-size 2 --gradient-accumulation-steps 2 \
    --num-workers 2 --seed 42 --save-every 10 \
    --lr 0.00001 --optimizer adam --weight-decay 0 \
    --lr-scheduler none --warmup-ratio 0 \
    --freeze-vision-encoder False --freeze-text-encoder True \
    --gradient-checkpointing \
    --config "$RUN_ROOT/configs/spatial.json"
```

## Four evaluations

Select checkpoints and hyperparameters on the validation split only (scoring the test split requires local test ground-truth masks). `eval.py` evaluates native-resolution masks and reports per-image `mean_iou` alongside dataset-wide `global_iou`.

```bash
%%bash
set -euo pipefail
for epoch in 10 20; do
    for split in val test; do
        uv run --frozen eval.py         \
    --checkpoint "$RUN_ROOT/checkpoint_epoch${epoch}.pt"         \
    --dataset "$split" --data-root "$DATA_ROOT"         \
    --batch-size 4 --num-workers 2         \
    --output-dir "$RUN_ROOT/eval_${split}_ep${epoch}"
    done
done
```

## Diagnostics

`diagnose.py` runs standard CPU mask diagnostics on the exported PNG predictions. `diagnose_spatial.py` runs a single forward pass per validation image to compare the model's pre-transport coarse logits and post-transport final logits at native resolution, reporting IoU by ground-truth size bucket, newly added foreground hits, and removed true-positive pixels.

```bash
%%bash
set -euo pipefail
for epoch in 10 20; do
    uv run --frozen diagnose.py     \
    --predictions "$RUN_ROOT/eval_val_ep${epoch}"     \
    --data-root "$DATA_ROOT"     \
    --output-dir "$RUN_ROOT/diagnostic_ep${epoch}"
    uv run --frozen diagnose_spatial.py     \
    --checkpoint "$RUN_ROOT/checkpoint_epoch${epoch}.pt"     \
    --data-root "$DATA_ROOT" --batch-size 4 --num-workers 2     \
    --output-dir "$RUN_ROOT/spatial_diagnostic_ep${epoch}"
done
```

## Download ZIP without checkpoints

```python
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(os.environ["RUN_ROOT"])
archive = root / (os.environ["RUN_NAME"] + ".zip")
with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() not in {".pt", ".zip", ".tmp"}:
            bundle.write(path, path.relative_to(root))
print(archive)
```

Download `.pt` checkpoints separately from Kaggle Outputs.

## Checkpoints and experimental controls

Use `--resume-checkpoint` only with full checkpoints generated by this branch, keeping the original training settings while setting `--num-epochs` to the final target epoch count. Checkpoints from the parent pyramid or answer-supervision branches are not interchangeable with this branch.
