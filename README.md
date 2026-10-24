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
    --gradient-checkpointing --seed 42 --save-every 5 \
    --config "$RUN_ROOT/configs/objective.json"
```

## Evaluation and diagnostic

No validation runs internally. Check val at5/10/15/20; test10/20 is descriptive, not a selection criterion. Test metrics require local GT. Native mean IoU and global foreground IoU are separate. Reduce eval batch size if needed.

```bash
%%bash
set -euo pipefail
for epoch in 5 10 15 20; do
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

If validation selects epoch5/15, evaluate that checkpoint on test separately. Report <1%,1-10%,>=10%, overall mean/global IoU, precision/recall and area ratio; do not hide either smaller group behind a pooled score.

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
