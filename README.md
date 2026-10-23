# Dense answer supervision with hard background points

A small shared point classifier predicts a train-answer class on64 GT foreground points and background class0 on64 outside-GT points (32 high-logit hard negatives +32 uniform negatives). Added parameters:12,480 +65*(number_of_train_answers+1), e.g.77,545 for1,000 answers. Points are sampled on the768 GT grid before feature interpolation. Class0 means outside the annotated evidence for this Q, not that no answer can be read there. Vocabulary comes only from train and is stored/hash-checked in checkpoints; unseen validation answers are skipped in auxiliary diagnostics. This is discrete semantic classification, not cosine interaction or embedding regression.

## Architecture and supervision

The original I336 CLIP + I768 detail pyramid, Q conditioning and mask losses are unchanged. The head reads shared192-scale features after mask prediction. GT Answer/region never enter the grounding path. Auxiliary gradients update shared features. There is no extra VLM, image encoder, latent answer or scope selector.

New config keys: answer_weight=0.05, answer_warmup_epochs=2, answer_ramp_epochs=3. Other optimizer/freeze/config options remain available through train.py --help. fp_weight is inactive for Dice. Standard eval drops the answer head. Both new experiments require fresh training; old experiment artifacts stay unchanged.

## Complete Kaggle guide

# Dense answer supervision with hard background points

Branch `exp/dense-answer-supervision`, based on pyramid commit `ddfdce1`. Enable Internet and2 T4 GPUs. Train20 fresh epochs from pretrained CLIP; no parent/teacher checkpoint or external VLM is required. The grounder remains the original large pyramid; only the additional head is small. T4 VRAM/runtime have not been measured. Start with an empty outputs directory.

## 1. Setup

```python
!git clone --single-branch --branch exp/dense-answer-supervision https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen
```

## 2. Paths and one config

All files remain directly under outputs. JSON overrides flags, then defaults.

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

## 3. Preflight

Both ranks must pass. This checks execution/gradients, not accuracy. Preflight updates are discarded. GPU memory is reported here before full training.

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

## 4. Train20 epochs

Global microbatch2 is1 image/GPU; accumulation2 gives effective batch4. Use batch4/accumulation1 if preflight fits. Vision fine-tunes; text stays frozen. Adam1e-5/no schedule matches the pyramid recipe. Answer loss is disabled for epochs1–2 and ramps in epochs3–5. No internal validation runs every epoch.

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

## 5. Four evaluations

Native mean per-image IoU is primary; global foreground IoU is separate. Masks are native size. Eval removes the auxiliary head before GPU transfer. Inputs are I+Q only. Select checkpoints on validation, never test.

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

## 6. Diagnostics

Mask analysis is CPU-only. Grounding is computed first using I+Q. The auxiliary diagnostic reuses one image encoding and identical sampled coordinates to compare original, zeroed and spatially rotated features. Foreground and background CE/class accuracy are reported separately. Inspect vocabulary coverage: unseen validation answers are not forced into a known class. These are feature sensitivity probes, not free-generation VQA accuracy.

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

## 7. Download ZIP without checkpoints

Includes config, run_config, history, four evaluations and diagnostics. Download .pt checkpoints separately from Kaggle Outputs; keep the selected checkpoint.

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

For a matched no-answer control, change only answer_weight to0 and use a fresh run/session. Resume only this branch's full checkpoint with --resume-checkpoint; repeat original settings and set a larger total num-epochs. Annotation hashes, architecture and vocabulary must match. Old pyramid/latent/scope checkpoints are not valid resumes.

Loss reduction: mean valid targets within each image, then mean original images. Invalid answers contribute zero; answer_valid_fraction reports coverage. This matches DDP and partial accumulation-window sample weighting.

Answer loss supervises shared representations, not the predicted mask directly. Improvements and visual dependence are hypotheses. Compare size groups and an answer_weight0 control before attributing gains to Answer. CPU synthetic tests do not establish T4 performance or accuracy.
