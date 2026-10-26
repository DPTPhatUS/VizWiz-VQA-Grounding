# per-image-lovasz

Branch `exp/per-image-lovasz`. Input: image + question. The original 336/768 pyramid architecture and state-dict keys remain unchanged. The loss function does not depend on CLIP channel dimensions or resolution, and no extra architecture or inference parameters are introduced.

## Method and scope

This branch retains the original pyramid architecture, binary cross-entropy (BCE), and the three auxiliary BCE + Dice losses, replacing only the final-mask Dice term with a binary Lovasz hinge loss. Pixel errors are sorted independently per image and averaged across images in the batch, without adding a reader module, area head, hard mining, or modified inference rule. Images with empty foreground are supervised normally.

Training runs for 20 fresh epochs from pretrained CLIP encoders, with the vision encoder fine-tuning and the text encoder frozen. Auxiliary supervision remains BCE + Dice even when `overlap_loss` is set to `lovasz`. To run the original-loss baseline from this branch, set `overlap_loss` to `dice` in a fresh run.

Reference: [Berman et al., CVPR 2018](https://arxiv.org/abs/1705.08790).

## Setup

Enable Internet and two GPUs. Start with a fresh `outputs` directory and run preflight before launching training.

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

Precedence is JSON config > explicit CLI flags > defaults. Loss, optimizer, and encoder-freezing options can share the JSON config. `fp_weight` is inactive for Dice and Lovasz losses.

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

Both ranks must pass; updates are discarded and no checkpoint is written.

## Train 20 epochs

A global microbatch of 2 with gradient accumulation of 2 gives an effective batch size of 4. If GPU memory permits, preflight with batch size 4 and train with `--batch-size 4 --gradient-accumulation-steps 1`. Sorting pixel errors at full resolution adds compute during training only and has no inference overhead.

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

Training exports validation masks and metrics at epochs 5, 10, 15, and 20 (`--validate-every 5`, where `--val-batch-size` is per GPU), while full model checkpoints are written only at epochs 10 and 20 (`--save-every 10 --no-save-final-weights`). Run CPU diagnostics on all four validation folders, and run test evaluation on the saved epoch 10 and 20 checkpoints. Local test evaluation requires ground-truth test masks. Evaluation reports per-image native mean IoU and global foreground IoU separately.

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

Because weights are saved at epochs 10 and 20, select the deployable checkpoint from those two epochs using validation metrics. Report `<1%`, `1-10%`, `>=10%`, overall mean IoU, global IoU, precision/recall, and area ratio.

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

Existing parent pyramid checkpoints remain valid for evaluation and produce identical predictions. Resuming an interrupted training run requires `--resume-checkpoint` with the same objective, optimizer, sampler, batch size, world size, and `--num-epochs` set to the final total epoch count.

## Recover the disk-full run without restarting

An earlier storage schedule saved full checkpoints at epochs 5, 10, 15, and 20, which exceeded Kaggle's working disk quota after epoch 15. To finish training from epoch 15 to 20 without restarting from scratch, attach the latest complete checkpoint (`checkpoint_epoch15.pt`; ignore any partial `.tmp` file) as a Kaggle dataset and start a fresh session with a clean `outputs` folder.

Pull the latest code on this branch, keep all objective, optimizer, freezing, batch/accumulation, seed, and GPU settings identical, and append `--resume-checkpoint` to the training command:

```bash
--resume-checkpoint /kaggle/input/YOUR_LOVASZ_CHECKPOINT/checkpoint_epoch15.pt
```

Keep `--num-epochs 20` so `checkpoint_epoch15.pt` resumes at epoch 16 and runs the remaining 5 epochs. In the fresh output directory, `history.jsonl` logs epochs 16 through 20 and generates `eval_val_ep20` plus `checkpoint_epoch20.pt`. Run `diagnose.py` and `eval.py --dataset test` on epoch 20 (earlier validation exports from epochs 5, 10, and 15 can be kept from the first session). Changing `--save-every` on resume does not alter optimizer state or training gradients.

Full checkpoints include Adam state and are much larger than weights-only files. Before writing, the checkpoint saver checks estimated tensor storage plus a safety margin, retains prior checkpoints, removes any partial `.tmp` file on failure, and broadcasts disk errors over a CPU control group so peer GPUs do not hang in NCCL.

## Analyze the existing epoch-5/10/15 checkpoints (no more training)

To evaluate the existing complete checkpoints from epochs 5, 10, and 15 without running further training, execute Cells E1 through E6 in a fresh single-GPU Kaggle session. Keep the `.pt` files mounted under `/kaggle/input` rather than copying them into `/kaggle/working`.

### Cell E1: Clone the evaluation branch

```python
!git clone --single-branch --branch exp/per-image-lovasz https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen
```

If this branch is already cloned, run `!git pull --ff-only` followed by `!uv sync --frozen`.

### Cell E2: Dataset, three checkpoint paths and output directory

Replace `YOUR_LOVASZ_CHECKPOINTS` with the mounted dataset directory containing all three `.pt` files. If `EVAL_BATCH = '4'` exceeds GPU memory, lower it to `'1'` (predictions and thresholds are unaffected by evaluation batch size).

```python
import json
import os
from pathlib import Path

os.environ['DATA_ROOT'] = '/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz'
os.environ['LOVASZ_CKPT_ROOT'] = '/kaggle/input/YOUR_LOVASZ_CHECKPOINTS'
os.environ['RUN_ROOT'] = 'outputs'
os.environ['RUN_NAME'] = 'per_image_lovasz_ep5_10_15_seed42'
os.environ['EVAL_BATCH'] = '4'
os.environ['EVAL_WORKERS'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

root = Path(os.environ['RUN_ROOT'])
checkpoints = {
    epoch: Path(os.environ['LOVASZ_CKPT_ROOT']) / f'checkpoint_epoch{epoch}.pt'
    for epoch in (5, 10, 15)
}
missing = [str(path) for path in checkpoints.values() if not path.is_file()]
if missing:
    print('Available .pt files:')
    for path in sorted(Path('/kaggle/input').rglob('*.pt')):
        print(path)
    raise FileNotFoundError(f'Correct LOVASZ_CKPT_ROOT; missing files: {missing}')
for epoch in checkpoints:
    for folder in (root / f'eval_val_ep{epoch}', root / f'eval_test_ep{epoch}',
                   root / f'diagnostic_ep{epoch}'):
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f'Use fresh outputs; folder is not empty: {folder}')
root.mkdir(parents=True, exist_ok=True)
protocol = {
    'branch': 'exp/per-image-lovasz',
    'epochs_evaluated': list(checkpoints),
    'data_root': os.environ['DATA_ROOT'],
    'eval_batch_size': int(os.environ['EVAL_BATCH']),
    'checkpoints': {
        str(epoch): {'path': str(path), 'size_bytes': path.stat().st_size}
        for epoch, path in checkpoints.items()
    },
}
(root / 'analysis_protocol.json').write_text(json.dumps(protocol, indent=2))
print(json.dumps(protocol, indent=2))
```

### Cell E3: All three validation evaluations and CPU diagnostics

```bash
%%bash
set -euo pipefail
for epoch in 5 10 15; do
    uv run --frozen python eval.py \
        --checkpoint "$LOVASZ_CKPT_ROOT/checkpoint_epoch${epoch}.pt" \
        --dataset val --data-root "$DATA_ROOT" --device cuda \
        --batch-size "$EVAL_BATCH" --num-workers "$EVAL_WORKERS" \
        --output-dir "$RUN_ROOT/eval_val_ep${epoch}"
    uv run --frozen python diagnose.py \
        --predictions "$RUN_ROOT/eval_val_ep${epoch}" \
        --data-root "$DATA_ROOT" \
        --output-dir "$RUN_ROOT/diagnostic_ep${epoch}"
done
```

Predicted masks are resized to each image's original ground-truth resolution before thresholding. Each validation folder writes PNG masks and `metrics.json` (mean IoU, global IoU, and per-image IoUs), and each diagnostic folder writes `summary.json` and `per_image.csv` (including `<1%`, `1-10%`, `>=10%`, precision/recall, and area ratio).

### Cell E4: Optional test evaluation for all three checkpoints

Run this cell only if local test annotations and ground-truth test masks are available; otherwise skip to Cell E5. Select checkpoints using validation metrics rather than test scores.

```bash
%%bash
set -euo pipefail
for epoch in 5 10 15; do
    uv run --frozen python eval.py \
        --checkpoint "$LOVASZ_CKPT_ROOT/checkpoint_epoch${epoch}.pt" \
        --dataset test --data-root "$DATA_ROOT" --device cuda \
        --batch-size "$EVAL_BATCH" --num-workers "$EVAL_WORKERS" \
        --output-dir "$RUN_ROOT/eval_test_ep${epoch}"
done
```

### Cell E5: Compare the three epochs

```python
import csv

root = Path(os.environ['RUN_ROOT'])
rows = []
def percent(value):
    return round(100 * value, 4) if value is not None else None

for epoch in (5, 10, 15):
    metrics = json.loads((root / f'eval_val_ep{epoch}/metrics.json').read_text())
    diagnostic = json.loads((root / f'diagnostic_ep{epoch}/summary.json').read_text())
    summary = metrics['summary']
    test_path = root / f'eval_test_ep{epoch}/metrics.json'
    test_summary = json.loads(test_path.read_text())['summary'] if test_path.is_file() else {}
    rows.append({
        'epoch': epoch,
        'val_num_samples': summary['num_samples'],
        'val_miou_pct': percent(summary['mean_iou']),
        'val_global_iou_pct': percent(summary['global_iou']),
        'small_miou_pct': percent(diagnostic['by_gt_area']['small_lt_1pct']['mean_iou']),
        'medium_miou_pct': percent(diagnostic['by_gt_area']['medium_1_to_10pct']['mean_iou']),
        'large_miou_pct': percent(diagnostic['by_gt_area']['large_ge_10pct']['mean_iou']),
        'test_miou_pct': percent(test_summary.get('mean_iou')),
        'test_global_iou_pct': percent(test_summary.get('global_iou')),
    })
comparison = root / 'checkpoint_comparison.csv'
with comparison.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(comparison.read_text())
```

Inspect the `<1%`, `1-10%`, and `>=10%` groups alongside overall mean IoU, and compare against baselines at matching epochs (5, 10, and 15).

### Cell E6: ZIP results, excluding every checkpoint

```python
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(os.environ['RUN_ROOT'])
archive = Path('/kaggle/working') / f"{os.environ['RUN_NAME']}.zip"
with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() not in {'.pt', '.tmp', '.zip'}:
            bundle.write(path, path.relative_to(root))
print('Download:', archive)
print('Checkpoint files are excluded; keep the original .pt files separately.')
```

The archive packages `analysis_protocol.json`, `checkpoint_comparison.csv`, validation masks and metrics, all three CPU diagnostic folders, and any test masks and metrics, while excluding `.pt`, `.tmp`, and nested `.zip` files.
