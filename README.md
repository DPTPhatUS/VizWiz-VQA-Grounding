# area-calibrated-pyramid

Branch `exp/area-calibrated-pyramid`; parent pyramid `ddfdce1`. Fresh training runs for 20 epochs using image and question inputs (`I+Q`) only. Both pretrained encoders come from CLIP; the vision encoder fine-tunes while the text encoder stays frozen. The new spatial module is small, while the full pyramid and backbones remain at their standard scale. This branch does not use a teacher/reader module or an answer loss.

## Design and limits

The main decoder predicts pixel scores, while an independent small area controller reads a 4x4 layout of decoder features along with the question feature (`Q`). It predicts the logit foreground area directly without receiving current mask probabilities or area statistics, preventing it from learning a trivial identity correction on training masks. A differentiable scalar-bias solver enforces `mean(sigmoid(final_logits)) = predicted_area` on the model output grid while preserving within-image pixel ordering. Its backward pass uses implicit differentiation rather than unrolling the solver iterations.

The zero-initialized final area layer starts with a neutral 0.5 soft-area prior, so segmentation and area supervision learn the image-specific area from scratch. The area loss applies smooth L1 on logit area, which emphasizes multiplicative errors on small regions. Ground-truth area is used only as a training target; inference predicts area directly from `I+Q`.

This formulation separates pixel ranking from total foreground mass prediction. Because the scalar shift preserves pixel ordering on the model grid, it adjusts threshold calibration rather than re-ranking spatial locations, and the soft-area constraint applies prior to native-resolution resizing and binary thresholding.

Related work: [RankSEG, JMLR 2023](https://arxiv.org/html/2206.13086v3) separates spatial ranking from segmentation thresholding. Rather than computing RankIoU/RankDice, our module predicts foreground area and applies a differentiable mass projection. Unlike the earlier evidence-extent experiment (which used learned support/extent tokens on a compact model), this branch applies an implicit mass solver to the full pyramid.

```text
Image + Q -> pyramid -> raw final logits --------------------+
                |                                          |
                +-> decoder feature + Q (no mask statistics)    |
                                  |                         |
                         small area controller              |
                                  |                         |
                           predicted soft area              |
                                  |                         |
                                  +-> implicit mass solve <-+
                                              |
                                         final mask
```

The added module introduces **18,081 parameters** in this pyramid adapter. Its constructor takes the feature-channel and question dimensions and accepts rectangular spatial grids without hardcoded CLIP imports, patch counts, or fixed image sizes. In this branch, the adapter is wired to the 336/768 CLIP pyramid.

Training combines the original final and auxiliary segmentation losses with a coarse segmentation loss (weight 0.2) and the logit-area loss (weight 0.1). The coarse loss supervises the uncorrected logits during joint training. When analyzing results, compare both final versus coarse predictions within this run and final predictions against the independent parent baseline.

## Kaggle setup

Enable Internet and two GPUs, and use a fresh `outputs` directory. Run preflight before starting full training.

Python cell:

```python
!git clone --single-branch --branch exp/area-calibrated-pyramid https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
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
os.environ["RUN_NAME"] = "area_calibrated_pyramid_ep20_seed42"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
root = Path(os.environ["RUN_ROOT"])
(root / "configs").mkdir(parents=True, exist_ok=True)
config = {'overlap_loss': 'dice', 'overlap_weight': 0.5, 'fp_weight': 0.7, 'aux96_weight': 0.1, 'aux192_weight': 0.2, 'aux384_weight': 0.3, 'coarse_weight': 0.2, 'area_weight': 0.1}
(root / "configs/spatial.json").write_text(json.dumps(config, indent=2))
```

Configuration values in JSON override explicit CLI flags, which in turn override defaults. Loss weights are configured via JSON (`fp_weight` applies only when using Tversky loss, not the default Dice). Optimizer and encoder freezing controls can be set via CLI flags or JSON.

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

Both ranks must pass. Preflight runs forward and backward passes through the real backbone, discards parameter updates, and writes no checkpoints.

## Train

A global microbatch of 2 with gradient accumulation of 2 gives an effective batch size of 4 across two GPUs. If GPU memory permits, you can preflight with `--batch-size 4` and train with batch size 4 and accumulation 1. Per-epoch validation is disabled during training.

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

Use validation metrics to select checkpoints and configurations. Local test IoU evaluation requires ground-truth test masks. Evaluation resizes predictions to native image resolution and reports per-image mean IoU alongside global foreground IoU.

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

The standard mask diagnostic (`diagnose.py`) runs on CPU. The paired spatial diagnostic (`diagnose_spatial.py`) runs the model once per validation image and compares the uncorrected coarse predictions against the area-calibrated final predictions at native resolution. Ground truth is used only to compute metrics after inference. The report includes size-group IoU, newly added foreground hits, and removed true positives. Note that the coarse predictions come from this trained model's uncorrected head rather than the separate parent checkpoint.

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

Inspect small, medium, and large IoU, precision/recall, predicted area, and paired coarse-vs-final deltas, alongside `area_loss` and `area_absolute_error`.

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

Download `.pt` checkpoints separately if you want to keep them.

## Checkpoints and experimental controls

To resume an interrupted run on this branch, pass `--resume-checkpoint` with the original configuration; `--num-epochs` specifies the final total epoch count. Checkpoints from the parent branch or answer-supervision branches remain valid on their respective branches but cannot be used to resume training here.

Run the default fresh configuration first without changing the backbone or optimizer at the same time, and evaluate the spatial experiments independently before combining them.
