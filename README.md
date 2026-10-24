# per-image-lovasz

Branch `exp/per-image-lovasz`. Input: image+question. Original 336/768 pyramid architecture/state-dict keys remain unchanged. The objective/projection does not depend on CLIP channels or resolution; changing the backbone still requires its normal feature/preprocessing adapter.

## Method and scope

Keep the original pyramid, BCE and three auxiliary BCE+Dice losses. Replace ONLY the final-mask Dice term with binary Lovasz hinge, sorted independently per image, then average original images. No reader, area head, hard mining or new inference rule. Empty foreground is supervised, not skipped. This is an IoU-motivated surrogate, not a guarantee of improved generalization.

Train fresh20epochs from pretrained CLIP. Vision fine-tunes; text stays frozen. To reproduce the original-loss control, change overlap_loss to dice in a fresh run. Auxiliary overlap stays Dice even when main overlap is Lovasz.

References: [Berman et al., CVPR2018](https://arxiv.org/abs/1705.08790). Independent implementation of the binary Lovasz construction.

## Setup

Enable Internet and two GPUs. CPU tests do not establish CUDA memory fit, speed or accuracy. Start with fresh outputs and run preflight first. No notebook is generated.

```python
!git clone --single-branch --branch exp/per-image-lovasz https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen
```

```python
import json
import os
from pathlib import Path

os.environ['DATA_ROOT'] = '/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz'
os.environ['RUN_ROOT'] = 'outputs'
os.environ['RUN_NAME'] = 'per_image_lovasz_ep20_seed42'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
root = Path(os.environ['RUN_ROOT'])
(root / 'configs').mkdir(parents=True, exist_ok=True)
config = {'overlap_loss': 'lovasz', 'overlap_weight': 0.5, 'fp_weight': 0.7, 'aux96_weight': 0.1, 'aux192_weight': 0.2, 'aux384_weight': 0.3}
(root / 'configs/objective.json').write_text(json.dumps(config, indent=2))
```

JSON > explicit flags > defaults. Loss/optimizer/freezing options can share the config. fp_weight is inactive for Dice/Lovasz.

## Preflight

```bash
%%bash
set -euo pipefail
uv run --frozen torchrun --standalone --nproc_per_node=2 preflight.py \
    --data-root "$DATA_ROOT" --batch-size 2 --num-workers 0 --steps 3 \
    --freeze-vision-encoder False --freeze-text-encoder True \
    --gradient-checkpointing \
    --config "$RUN_ROOT/configs/objective.json"
```

Both ranks must pass; no checkpoint is written and updates are discarded.

## Train20epochs

Global microbatch2 plus accumulation2 gives effective batch4. If memory permits, preflight with batch4 then use batch4/accumulation1. Full-resolution sorting adds training cost, no inference cost.

```bash
%%bash
set -euo pipefail
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py \
    --data-root "$DATA_ROOT" --output-dir "$RUN_ROOT" --num-epochs 20 \
    --batch-size 2 --gradient-accumulation-steps 2 --num-workers 2 \
    --lr 0.00001 --optimizer adam --weight-decay 0 \
    --lr-scheduler none --warmup-ratio 0 \
    --freeze-vision-encoder False --freeze-text-encoder True \
    --gradient-checkpointing --seed 42 --save-every 10 \
    --validate-every 5 --val-batch-size 1 --no-save-final-weights \
    --config "$RUN_ROOT/configs/objective.json"
```

## Evaluation and diagnostic

Training exports validation masks/metrics at5/10/15/20 without saving weights at5/15. Full checkpoints are ONLY epoch10/20; no duplicate final raw weights. Validation batch size is per GPU. Run CPU diagnostics on those existing folders; test10/20 is descriptive, not a selection criterion. Test metrics require local GT. Native mean IoU and global foreground IoU are separate. Reduce eval batch size if needed.

```bash
%%bash
set -euo pipefail
for epoch in 5 10 15 20; do
    uv run --frozen diagnose.py --predictions "$RUN_ROOT/eval_val_ep${epoch}" \
        --data-root "$DATA_ROOT" --output-dir "$RUN_ROOT/diagnostic_ep${epoch}"
done
for epoch in 10 20; do
    uv run --frozen eval.py --checkpoint "$RUN_ROOT/checkpoint_epoch${epoch}.pt" \
        --dataset test --data-root "$DATA_ROOT" --batch-size 4 --num-workers 2 \
        --output-dir "$RUN_ROOT/eval_test_ep${epoch}"
done
```

Epoch5/15 have masks/metrics but no saved weights. Select a deployable checkpoint from epoch10/20; validation does not automatically save a best model. Report <1%,1-10%,>=10%, overall mean/global IoU, precision/recall and area ratio; do not hide either smaller group behind a pooled score.

## ZIP without checkpoints

```python
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(os.environ['RUN_ROOT'])
archive = root / (os.environ['RUN_NAME'] + '.zip')
with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() not in {'.pt', '.zip', '.tmp'}:
            bundle.write(path, path.relative_to(root))
print(archive)
```

Download checkpoints separately. Session duration is unmeasured.

## Compatibility

Old parent weights remain valid for evaluation with identical predictions. Other architectures stay on their branches. Full resume requires this objective/protocol, original optimizer/sampler/batch/world-size/mode and `--resume-checkpoint`. num-epochs is the final total. Initialization from weights is not optimizer resume. No existing experiment or result archive is changed.

## Recover the disk-full run without restarting

The earlier guide saved at5/10/15/20; that storage schedule was unsuitable for
this Kaggle run. Use the latest COMPLETE full checkpoint that was successfully
saved (for example epoch15). An incomplete .tmp is not a checkpoint. Existing
successful checkpoints are not changed by this update. Put the chosen .pt in an
attached Kaggle dataset and use a fresh session/output directory with free space.
Do not delete your only recoverable checkpoint.

Update to this branch's latest code before resuming. Keep the old objective,
optimizer, freezing, batch/accumulation, seed and GPU count. Add to the train
command above, replacing the path with the actual saved file:

```bash
--resume-checkpoint /kaggle/input/YOUR_LOVASZ_CHECKPOINT/checkpoint_epoch15.pt
```

Keep `--num-epochs 20`: epoch15 resumes at16, with only five training epochs
remaining. Do not pass a pyramid or scale checkpoint. For a fresh output folder,
history starts at the resumed epoch+1. Checkpoint settings must match; do not
change loss to Dice on resume. In this recovered session only future validation
folders are generated (for epoch15 ->20, only eval_val_ep20). Run diagnostic and
test on those available epochs instead of the fresh-run loops above. Earlier
mask exports can be kept separately. Save spacing may change without resetting
the optimizer or altering the training objective.

Full checkpoints include Adam moments and are much larger than weights-only
files. Even two full checkpoints require enough free disk for the NEXT atomic
write; model caches and the environment also consume storage. The writer checks
estimated tensor storage plus a safety margin before writing, retains previous
checkpoints, and removes a partial .tmp on a failed write. Available space may
still change during the operation. Disk errors are shared over a CPU control
group so another GPU does not sit in NCCL waiting for a rank-zero save failure.
CPU tests do not establish Kaggle disk fit or GPU performance.
