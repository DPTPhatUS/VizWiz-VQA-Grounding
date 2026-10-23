# area-calibrated-pyramid

Branch `exp/area-calibrated-pyramid`; parent pyramid `ddfdce1`. Fresh training, 20 epochs, I+Q only. Both pretrained encoders come from CLIP; vision fine-tunes and text stays frozen. The new spatial module is small; the full pyramid/backbones remain large. There is no teacher/reader and no answer loss.

## Design and limits

The main decoder learns pixel scores. An independent small area controller reads a 4x4 layout of decoder features and the Q feature. It predicts logit foreground area directly, without receiving current mask probabilities or area statistics. This avoids learning an identity correction on overly accurate training masks. A differentiable scalar-bias solve enforces mean(sigmoid(final_logits)) = predicted_area on the model output grid. The solver preserves within-image pixel ordering. Its backward uses implicit derivatives, not an unrolled solve. The zero-initialized last area layer starts with a neutral 0.5 soft-area prior, not a dataset-derived prior; segmentation and area supervision learn the image-specific area from scratch. The area loss is smooth L1 on logit area; it emphasizes multiplicative errors on small regions. GT area is a training target only. Inference predicts area from I+Q.

This separates *which pixels rank higher* from *how much foreground to predict*. It cannot rescue a wrong spatial ranking, and predicted soft area is not the binary mask area. The constraint is on the model grid, not guaranteed after native resizing. A wrong area estimate can hurt a large region; there is no accuracy-preservation guarantee.

Related work: [RankSEG, JMLR 2023](https://arxiv.org/html/2206.13086v3) separates spatial ranking and segmentation decisions. This implementation does not implement RankIoU/RankDice or inherit their consistency theorem. It instead learns area and applies a differentiable mass constraint. The older evidence-extent experiment used learned support/extent tokens on a compact model; it did not use this mass projection.


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

The added module has **18,081 parameters** in this pyramid adapter. The module constructor takes feature-channel/question dimensions and accepts rectangular spatial grids. It has no CLIP-specific imports, patch count or fixed image size. The experiment adapter is currently wired to the existing 336/768 CLIP pyramid. Trying a different backbone still requires its feature/preprocessing adapter; multiple trained backbones are not validated.

Training uses the original final/auxiliary segmentation losses plus coarse segmentation (weight 0.2) and logit-area loss (weight 0.1). The coarse loss protects the uncorrected predictor during joint training; it does not ensure no regression. Compare both final-vs-coarse within this run and final-vs-independent parent. These are different comparisons.

## Kaggle setup

Enable Internet and two GPUs. Use fresh `outputs`. No notebook is generated. CPU checks do not establish T4 VRAM fit, runtime or accuracy; preflight must pass.

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

Config overrides explicit flags, then defaults. Loss weights are JSON controls. `fp_weight` affects Tversky only, not the default Dice. Optimizer/freeze controls remain available as flags and JSON.

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

Both ranks must pass. Preflight exercises real backbone forward/backward; updates are discarded, no checkpoint is written.

## Train

Global microbatch 2 / accumulation 2 gives effective batch 4 over two GPUs. If memory permits, preflight with batch 4 and use batch 4 / accumulation 1. No internal per-epoch validation.

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

Select checkpoints/settings by validation, not test. Local test IoU requires test GT. Evaluation uses native-resolution masks and reports mean IoU separately from global foreground IoU.

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

The standard mask diagnostic is CPU-only. The paired diagnostic runs the model once per image, then compares raw and corrected predictions on native validation masks. GT is used only for metrics after predictions. It reports size-group IoU, added foreground hits and destroyed true positives. Coarse is from this same trained model, not the old parent checkpoint.

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

Inspect small/medium/large IoU, precision/recall, predicted area and paired damage. `area_loss` and `area_absolute_error` describe predicted mass accuracy; low mass error does not imply correct localization.

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

Download `.pt` checkpoints separately. Total session time is not measured; training plus diagnostics might exceed the available runtime.

## Checkpoints and experimental controls

Only this branch's full checkpoints are valid resumes. Use `--resume-checkpoint` with original settings; `--num-epochs` specifies the final total. Parent and answer-experiment checkpoints remain valid on their own branches, not as resumes here. No old run is modified.

The first useful result is the default fresh run. Do not change backbone/optimizer simultaneously. The two spatial experiments should first be evaluated separately, not combined. A favorable single-seed result motivates a matched control and more seeds; it does not establish novelty or a reliable improvement.
