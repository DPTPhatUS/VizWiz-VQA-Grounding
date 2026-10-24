# scale-constrained-pyramid

Branch `exp/scale-constrained-pyramid`. Input: image+question. Original 336/768 pyramid architecture/state-dict keys remain unchanged. The objective/projection does not depend on CLIP channels or resolution; changing the backbone still requires its normal feature/preprocessing adapter.

## Method and scope

Start both projected and control runs from the SAME trained pyramid checkpoint. Freeze pretrained vision/text; CNN and decoder still train. Collect gradients for <1%, 1-10%, >=10% GT-area groups. Adam/AdamW proposes an update focused equally on the two smaller groups. Project its ACTUAL parameter displacement (including momentum/decay) onto all three non-increase half-spaces. This is a local first-order constraint on sampled training losses, not a validation-IoU guarantee. Zero movement is feasible; monitor stalls/rejections. No architecture or inference parameters are added.

A run is20 additional epochs from the parent. Its comparison must be the matched continuation control, not only the original20epoch parent. Multiple backward passes cost training time. This is different from freezing pretrained encoders before the parent was trained.

References: [GEM](https://papers.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf), [Gradient Surgery](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).

## Setup

Enable Internet and two GPUs. CPU tests do not establish CUDA memory fit, speed or accuracy. Start with fresh outputs and run preflight first. No notebook is generated.

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

JSON > explicit flags > defaults. Loss/optimizer/freezing options can share the config. fp_weight is inactive for Dice/Lovasz.

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

Both ranks must pass; no checkpoint is written and updates are discarded.

## Train20epochs

Global batch6 has2 images/group across2 GPUs. Each GPU forwards one image at a time, separately per group. Accumulation multiplies this count. All three training groups must be nonempty. Balanced sampling cycles shuffled indices: small groups repeat and large groups can be undersampled relative to standard epochs. This sampling also applies to the control.

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

Use `train_group.py` / `preflight_group.py` for this protocol. The standard `train.py` entrypoint does not project updates. Size groups are indexed on the nearest-resized training-mask grid; native evaluation bins can differ slightly near a cutoff.

Run the matched control in a separate session using the SAME parent and settings; append `--no-project-updates` and rename RUN_NAME to `scale_control_ep20_seed42`. JSON project_updates, if supplied, outranks the flag. Do not change this mode on resume.

Adam moments retain the proposed update's gradients; projection controls realized parameter displacement. A numerical constraint violation restores original weights and is logged; optimizer moments/schedule still advance for that successful proposal. Rejection or zero movement is not scientific success. Inspect gradient_cosine_small_medium/small_large/medium_large, conflict_fraction, projection_applied/rejected, update_ratio and min_group_cosine in history.jsonl.

## Evaluation and diagnostic

No validation runs internally. Evaluate the saved epoch10/20 checkpoints only; test10/20 is descriptive, not a selection criterion. Test metrics require local GT. Native mean IoU and global foreground IoU are separate. Reduce eval batch size if needed.

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

No epoch5/15 checkpoints are created by this recipe. Both projected/control runs use identical save/evaluation schedules. Report <1%,1-10%,>=10%, overall mean/global IoU, precision/recall and area ratio; do not hide either smaller group behind a pooled score.

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

## Checkpoint storage and execution risks

The guide saves ONLY full checkpoint_epoch10.pt and checkpoint_epoch20.pt.
Default save interval is10; extra final raw weights are disabled by default.
Do not reuse an old command/config setting save_every5 or save_final_weights=true.
This policy does not delete old files already present. Full checkpoints still
contain both frozen backbones; freezing only reduces optimizer-state storage.
Use a fresh output/session with enough free disk for the next atomic write.
The writer estimates tensor storage plus a safety margin, retains previous
checkpoints, and coordinates disk failures on a CPU group rather than leaving
peers waiting in NCCL. Available disk can change; fit is not guaranteed.
Checkpoint spacing/output changes do not alter gradients, projection, optimizer
updates or the resume protocol.

The method itself may not improve validation. Sampled first-order constraints
do not guarantee validation IoU, and strongly conflicting groups can leave
very small updates. Inspect optimizer_update/overflow alongside update_ratio,
projection_rejected and qp_zero_fallback. A near-zero update_ratio with otherwise
successful optimizer updates suggests stalled learning, not a successful model.
Compare against the same-parent, same-duration unprojected control.
