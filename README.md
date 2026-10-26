# scale-constrained-pyramid

Branch `exp/scale-constrained-pyramid`. Input: image + question. The original 336/768 pyramid architecture and state-dict keys remain unchanged. The group-constrained objective and update projection do not depend on CLIP channel dimensions or input resolution, and no extra architecture or inference parameters are added.

## Method and scope

Both the projected run and the unprojected control start from the same trained 20-epoch parent pyramid checkpoint. Pretrained vision and text encoders stay frozen while the CNN pyramid and decoder continue training. Each step collects gradients separately for three ground-truth foreground area groups: `<1%`, `1-10%`, and `>=10%`.

Adam/AdamW proposes a parameter step weighted equally across the two smaller groups (`<1%` and `1-10%`), and the projection step projects the actual parameter displacement (including momentum and weight decay) onto the non-increase half-spaces of all three size groups. If the constraints conflict strongly, zero movement is a valid solution, so monitor the projection logs for stalls or rejected updates.

Each run trains for 20 additional epochs from the parent checkpoint. Because balanced group sampling and frozen encoders also affect training dynamics, compare the projected run against the matched 20-epoch continuation control (`--no-project-updates`) in addition to the original 20-epoch parent. Computing separate per-group backward passes increases wall-clock training time per step.

References: [GEM](https://papers.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf), [Gradient Surgery](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).

## Setup

Enable Internet and two GPUs. Start with a fresh `outputs` directory and run preflight before launching training.

```python
!git clone --single-branch --branch exp/scale-constrained-pyramid https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen
```

```python
import json
import os
from pathlib import Path

os.environ['DATA_ROOT'] = '/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz'
os.environ['RUN_ROOT'] = 'outputs'
os.environ['RUN_NAME'] = 'scale_constrained_pyramid_ep20_seed42'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
root = Path(os.environ['RUN_ROOT'])
(root / 'configs').mkdir(parents=True, exist_ok=True)
config = {'overlap_loss': 'dice', 'overlap_weight': 0.5, 'fp_weight': 0.7, 'aux96_weight': 0.1, 'aux192_weight': 0.2, 'aux384_weight': 0.3}
(root / 'configs/objective.json').write_text(json.dumps(config, indent=2))
os.environ['PYRAMID_CKPT'] = '/kaggle/input/YOUR_PYRAMID_DATASET/checkpoint_epoch20.pt'
# Replace the placeholder with your trained parent pyramid checkpoint.
checkpoint = Path(os.environ['PYRAMID_CKPT'])
if not checkpoint.is_file():
    print("Available .pt files under /kaggle/input:")
    for candidate in sorted(Path('/kaggle/input').rglob('*.pt')):
        print(candidate)
    raise FileNotFoundError(
        f"Set PYRAMID_CKPT to the exact trained parent checkpoint path: {checkpoint}"
    )
```

Precedence is JSON config > explicit CLI flags > defaults. Loss, optimizer, and encoder-freezing options can all be specified in the JSON config. `fp_weight` is only active when using Tversky loss.

## Preflight

```bash
%%bash
set -euo pipefail
uv run --frozen torchrun --standalone --nproc_per_node=2 preflight_group.py \
    --data-root "$DATA_ROOT" --init-checkpoint "$PYRAMID_CKPT" \
    --batch-size 6 --gradient-accumulation-steps 1 --num-workers 0 --steps 3 \
    --freeze-vision-encoder True --freeze-text-encoder True \
    --gradient-checkpointing \
    --config "$RUN_ROOT/configs/objective.json"
```

Both ranks must pass; updates are discarded and no checkpoint is written.

## Train 20 epochs

A global batch size of 6 allocates 2 images per size group across 2 GPUs (1 image per group per GPU before gradient accumulation). All three training size groups must be non-empty. The balanced sampler cycles through shuffled indices within each group, repeating smaller groups and undersampling the large group relative to natural dataset frequencies. The same sampler is used for the unprojected control.

```bash
%%bash
set -euo pipefail
uv run --frozen torchrun --standalone --nproc_per_node=2 train_group.py \
    --data-root "$DATA_ROOT" --init-checkpoint "$PYRAMID_CKPT" \
    --output-dir "$RUN_ROOT" --num-epochs 20 \
    --batch-size 6 --gradient-accumulation-steps 1 --num-workers 2 \
    --lr 0.00001 --optimizer adam --weight-decay 0 \
    --lr-scheduler none --warmup-ratio 0 \
    --freeze-vision-encoder True --freeze-text-encoder True \
    --gradient-checkpointing --seed 42 --save-every 10 --no-save-final-weights \
    --config "$RUN_ROOT/configs/objective.json"
```

Use `train_group.py` and `preflight_group.py` for this experiment (`train.py` runs standard ungrouped training). Training size groups are indexed on the nearest-resized training-mask grid, so boundary samples near a size cutoff can fall into a neighboring bin during native-resolution evaluation.

To run the matched control in a separate session, start from the same parent checkpoint and settings, append `--no-project-updates`, and set `RUN_NAME` to `scale_control_ep20_seed42`. If `project_updates` is set in the JSON config, it overrides the CLI flag. Keep the projection mode unchanged when resuming a run.

During projected training, Adam moments track the proposed update's gradients while the projection step adjusts the realized parameter displacement. If a numerical constraint violation occurs, the trainer restores the pre-step weights and logs the event while optimizer moments and scheduler state advance. In `history.jsonl`, inspect `gradient_cosine_small_medium`, `gradient_cosine_small_large`, `gradient_cosine_medium_large`, `conflict_fraction`, `projection_applied`, `projection_rejected`, `update_ratio`, and `min_group_cosine`.

## Evaluation and diagnostic

Training runs without internal validation. Afterward, evaluate the saved epoch 10 and epoch 20 checkpoints. Use validation metrics for checkpoint selection; test evaluation is descriptive and requires local test ground truth. Evaluation reports per-image native mean IoU and global foreground IoU separately. Reduce `--batch-size` if evaluation exceeds GPU memory.

```bash
%%bash
set -euo pipefail
for epoch in 10 20; do
    uv run --frozen eval.py --checkpoint "$RUN_ROOT/checkpoint_epoch${epoch}.pt" \
        --dataset val --data-root "$DATA_ROOT" --batch-size 4 --num-workers 2 \
        --output-dir "$RUN_ROOT/eval_val_ep${epoch}"
    uv run --frozen diagnose.py --predictions "$RUN_ROOT/eval_val_ep${epoch}" \
        --data-root "$DATA_ROOT" --output-dir "$RUN_ROOT/diagnostic_ep${epoch}"
done
for epoch in 10 20; do
    uv run --frozen eval.py --checkpoint "$RUN_ROOT/checkpoint_epoch${epoch}.pt" \
        --dataset test --data-root "$DATA_ROOT" --batch-size 4 --num-workers 2 \
        --output-dir "$RUN_ROOT/eval_test_ep${epoch}"
done
```

This schedule saves checkpoints at epochs 10 and 20 (skipping epochs 5 and 15) for both projected and control runs. When reporting results, include `<1%`, `1-10%`, `>=10%`, overall mean IoU, global IoU, precision/recall, and area ratio.

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

Download `.pt` checkpoints separately if needed.

## Compatibility

Parent pyramid checkpoints remain compatible with `eval.py` and produce identical predictions. Full training resume requires `--resume-checkpoint` with matching objective, optimizer, sampler, batch size, world size, and projection mode (`--num-epochs` sets the total continuation epochs). Note that `--init-checkpoint` loads model weights for a new continuation run rather than resuming optimizer state.

## Checkpoint storage and diagnostics

The command above saves only `checkpoint_epoch10.pt` and `checkpoint_epoch20.pt` (`--save-every 10 --no-save-final-weights`). Avoid passing `--save-every 5` or enabling final raw weight exports so the run stays within Kaggle disk limits. Full checkpoints still include weights for both frozen backbones (freezing only omits their optimizer states), so start in a fresh output directory with sufficient free disk space. Before writing, the checkpoint saver estimates tensor storage plus a safety margin, retains prior checkpoints, and coordinates disk-full errors across ranks over a CPU process group so workers do not hang in NCCL.

When analyzing training logs, check `optimizer_update` and `overflow` alongside `update_ratio`, `projection_rejected`, and `qp_zero_fallback`. A near-zero `update_ratio` during otherwise valid optimizer steps indicates that conflicting group constraints are stalling parameter movement. Always compare against the unprojected continuation control trained from the same parent for the same number of epochs.
