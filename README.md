# Dense answer supervision with hard background points

Branch: `exp/dense-answer-supervision`, based on `exp/question-guided-detail-pyramid` (`ddfdce1`). This experiment adds a lightweight training-only point classifier that predicts the training answer class on 64 ground-truth foreground points and background class `0` on 64 outside-GT points (32 high-logit hard negatives plus 32 uniform random negatives). The head adds `12,480 + 65 * (number_of_train_answers + 1)` parameters (for example, **77,545 parameters** for 1,000 training answers). Points are sampled on the 768-resolution ground-truth grid before bilinearly interpolating shared 192-scale decoder features. Class `0` denotes points outside the annotated grounding region for the question. The answer vocabulary is built strictly from the training split and verified via checkpoint hash checks; validation samples with unseen answers are skipped in auxiliary answer diagnostics.

## Architecture and supervision

The base `I336` CLIP + `I768` detail pyramid, question conditioning, and multi-scale segmentation losses are unchanged. During training, the point classifier reads shared 192-scale decoder features after mask prediction, so ground-truth answers and masks never enter the grounding forward pass while auxiliary gradients update the shared decoder representation. Standard evaluation removes the answer head before transferring the model to the GPU, so inference uses only the image and question.

New `--config` JSON keys:
- `answer_weight`: `0.05`
- `answer_warmup_epochs`: `2`
- `answer_ramp_epochs`: `3`

All existing optimizer, encoder-freezing, and segmentation loss flags remain available via `train.py --help` (`fp_weight` applies only when `overlap_loss="tversky"`). Runs train fresh for 20 epochs from pretrained CLIP weights.

## Kaggle workflow

Enable Internet and 2 T4 GPUs, and start with an empty `outputs` directory.

### 1. Setup

```python
!git clone --single-branch --branch exp/dense-answer-supervision https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen
```

### 2. Paths and configuration

All artifacts are written under `outputs/`. Configuration precedence follows **JSON config > explicit CLI flags > defaults**.

```python
import json
import os
from pathlib import Path

os.environ['DATA_ROOT'] = '/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz'
os.environ['RUN_ROOT'] = 'outputs'
os.environ['RUN_NAME'] = 'dense_answer_supervision_ep20_seed42'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
root = Path(os.environ['RUN_ROOT'])
(root / 'configs').mkdir(parents=True, exist_ok=True)
config = {'overlap_loss': 'dice', 'overlap_weight': 0.5, 'fp_weight': 0.7, 'aux96_weight': 0.1, 'aux192_weight': 0.2, 'aux384_weight': 0.3, 'answer_weight': 0.05, 'answer_warmup_epochs': 2, 'answer_ramp_epochs': 3}
(root / 'configs/answer.json').write_text(json.dumps(config, indent=2))
```

### 3. Preflight

Both ranks must pass preflight, which verifies forward/backward execution, gradient flow, and peak GPU memory before launching the full run (preflight updates are discarded).

```python
%%bash
set -euo pipefail
uv run --frozen torchrun --standalone --nproc_per_node=2 preflight.py \
    --data-root "$DATA_ROOT" \
    --batch-size 2 --num-workers 0 --steps 3 \
    --gradient-checkpointing \
    --freeze-vision-encoder False --freeze-text-encoder True \
    --config "$RUN_ROOT/configs/answer.json"

```

### 4. Train 20 epochs

A global microbatch of 2 places 1 image per GPU, and `--gradient-accumulation-steps 2` gives an effective batch size of 4 (use `--batch-size 4 --gradient-accumulation-steps 1` if memory allows). The vision encoder is fine-tuned while the text encoder stays frozen, using Adam at `1e-5` with constant learning rate. The auxiliary answer loss weight is `0.0` during epochs 1 and 2, ramps linearly over epochs 3 to 5, and stays at `0.05` from epoch 5 onward.

```python
%%bash
set -euo pipefail
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$RUN_ROOT" \
    --num-epochs 20 \
    --batch-size 2 --gradient-accumulation-steps 2 \
    --num-workers 2 \
    --lr 0.00001 --optimizer adam --weight-decay 0 \
    --lr-scheduler none --warmup-ratio 0 \
    --freeze-vision-encoder False --freeze-text-encoder True \
    --gradient-checkpointing \
    --seed 42 --save-every 10 \
    --config "$RUN_ROOT/configs/answer.json"

```

### 5. Four evaluations

`eval.py` strips the auxiliary answer head before moving the model to the GPU, predicts native-resolution masks from image + question only, and reports both per-image `mean_iou` and dataset-wide `global_iou`. Select checkpoints using the validation split only.

```python
%%bash
set -euo pipefail
uv run --frozen eval.py \
    --checkpoint "$RUN_ROOT/checkpoint_epoch10.pt" \
    --dataset val --data-root "$DATA_ROOT" \
    --batch-size 4 --num-workers 2 \
    --output-dir "$RUN_ROOT/eval_val_ep10"

```

```python
%%bash
set -euo pipefail
uv run --frozen eval.py \
    --checkpoint "$RUN_ROOT/checkpoint_epoch10.pt" \
    --dataset test --data-root "$DATA_ROOT" \
    --batch-size 4 --num-workers 2 \
    --output-dir "$RUN_ROOT/eval_test_ep10"

```

```python
%%bash
set -euo pipefail
uv run --frozen eval.py \
    --checkpoint "$RUN_ROOT/checkpoint_epoch20.pt" \
    --dataset val --data-root "$DATA_ROOT" \
    --batch-size 4 --num-workers 2 \
    --output-dir "$RUN_ROOT/eval_val_ep20"

```

```python
%%bash
set -euo pipefail
uv run --frozen eval.py \
    --checkpoint "$RUN_ROOT/checkpoint_epoch20.pt" \
    --dataset test --data-root "$DATA_ROOT" \
    --batch-size 4 --num-workers 2 \
    --output-dir "$RUN_ROOT/eval_test_ep20"

```

### 6. Diagnostics

`diagnose.py` analyzes the exported validation masks on CPU. `diagnose_answers.py` reuses a single image encoding pass and identical sampled point coordinates to compare point classification cross-entropy and accuracy across original features, zeroed visual features, and spatially rotated features. Foreground answer accuracy, background accuracy, and in-vocabulary validation coverage are reported separately.

```python
%%bash
set -euo pipefail
for epoch in 10 20; do
    uv run --frozen diagnose.py \
        --predictions "$RUN_ROOT/eval_val_ep${epoch}" \
        --data-root "$DATA_ROOT" \
        --output-dir "$RUN_ROOT/diagnostic_ep${epoch}"
done
uv run --frozen diagnose_answers.py \
    --checkpoint "$RUN_ROOT/checkpoint_epoch20.pt" \
    --data-root "$DATA_ROOT" \
    --batch-size 4 --num-workers 2 \
    --output-dir "$RUN_ROOT/answer_diagnostic_ep20"

```

### 7. Download ZIP without checkpoints

This bundles configs, training history, evaluation metrics, exported masks, and diagnostics into a single archive while excluding `.pt` checkpoint files (download the selected `.pt` checkpoint separately from Kaggle Outputs).

```python
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(os.environ['RUN_ROOT'])
archive = root / (os.environ['RUN_NAME'] + '.zip')
with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() not in {'.pt', '.zip', '.tmp'}:
            bundle.write(path, path.relative_to(root))
print(archive)

```

## Controls and resume

To run a matched no-answer control on this branch, set `answer_weight` to `0` in a fresh output directory. Use `--resume-checkpoint` only with full checkpoints produced by this branch, keeping the original training flags, annotation hashes, architecture, and answer vocabulary while increasing `--num-epochs` to the target total.

For loss reduction, point cross-entropy is averaged over valid sampled points within each image and then averaged across all images in the batch (with skipped/invalid answers contributing `0.0` and `answer_valid_fraction` tracking valid sample coverage). This keeps loss scaling identical across DDP ranks and partial gradient-accumulation windows.
