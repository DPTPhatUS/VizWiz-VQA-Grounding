# Size-weighted pyramid: resume A/B from +5 to +10

Branch: `exp/size-weighted-pyramid`. This workflow resumes the completed `natural` (A) and `mild_focus` (B) continuation runs from epoch +5 to epoch +10 independently, preserving each run's optimizer state, AMP scaler, frozen encoders, loss weights, balanced sampler, seed, and constant learning rate. Use `train_group.py` rather than `train.py`.

Passing `--num-epochs 10` sets the final continuation counter so each run trains epochs 6 through 10 only (reaching 30 total epochs when counting the original 20-epoch parent pyramid). Because both runs resume from their own `checkpoint_epoch5.pt` files via `--resume-checkpoint`, the original 20-epoch parent checkpoint is not needed for this session.

## Cell 1: Setup

Start a fresh Kaggle session with Internet and two GPUs enabled. Attach the VizWiz dataset and both full A and B `checkpoint_epoch5.pt` files (the previous results ZIP excludes `.pt` files and cannot be used to resume training).

```python
!git clone --single-branch --branch exp/size-weighted-pyramid https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen
```

If the repository is already cloned on this branch, run `!git pull --ff-only` and `!uv sync --frozen`.

## Cell 2: Locate A/B and restore their configurations

This script identifies the A (`natural`, `focus_alpha = 0.0`) and B (`mild_focus`, `focus_alpha = 0.25`) checkpoints from their saved metadata rather than folder names. If multiple matching checkpoints are mounted under `/kaggle/input`, narrow `INPUT_ROOT` or set explicit paths in `selected`.

```python
import gc
import json
import os
from pathlib import Path
from utils import read_checkpoint

os.environ['DATA_ROOT'] = '/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz'
os.environ['INPUT_ROOT'] = '/kaggle/input'
os.environ['RUN_ROOT'] = 'outputs'
os.environ['RUN_NAME'] = 'size_weighted_pyramid_two_runs_ep10_seed42'
os.environ['EVAL_BATCH'] = '4'
os.environ['EVAL_TEST'] = '1'  # Use0 if local test GT is unavailable.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

selected = {'natural': '', 'mild_focus': ''}  # Optional exact checkpoint paths.
candidates = {run: [] for run in selected}
explicit = {str(Path(path)) for path in selected.values() if path}
paths = sorted(set(Path(os.environ['INPUT_ROOT']).rglob('checkpoint_epoch5.pt'))
               | {Path(path) for path in explicit})
metadata = {}
for path in paths:
    saved = read_checkpoint(path)
    run_config = saved.get('run_config', {})
    alpha = run_config.get('focus_alpha')
    if (saved.get('epoch') == 5
            and run_config.get('objective_version') == 'size-weighted-updates-v1'
            and alpha in (0.0, 0.25)):
        run = 'natural' if alpha == 0.0 else 'mild_focus'
        candidates[run].append(path)
        metadata[str(path)] = {
            'epoch': saved['epoch'],
            'has_optimizer': bool(saved.get('optimizer_state_dict', {}).get('state')),
            'has_scaler': 'scaler_state_dict' in saved,
            'run_config': run_config,
            'controls': saved.get('training_controls', {}).get('config'),
        }
    del saved
    gc.collect()

root = Path(os.environ['RUN_ROOT'])
for run, expected_alpha in (('natural', 0.0), ('mild_focus', 0.25)):
    choices = [Path(selected[run])] if selected[run] else candidates[run]
    if len(choices) != 1:
        raise ValueError(f'Select exactly one {run} checkpoint: {choices}')
    path = choices[0]
    info = metadata.get(str(path))
    if info is None:
        raise ValueError(f'Not a compatible A/B epoch5 checkpoint: {path}')
    config = info['run_config']
    assert info['has_optimizer'] and info['has_scaler'], 'Use full training checkpoints.'
    assert config['focus_alpha'] == expected_alpha, f'Wrong checkpoint for {run}'
    assert config['batch_size'] == 6 and config['gradient_accumulation_steps'] == 1
    assert config['world_size'] == 2 and config['seed'] == 42
    assert config['lr'] == 0.00001 and config['gradient_checkpointing'] is True
    assert config['project_updates'] is False
    controls = info['controls']
    assert controls is not None, 'Missing saved optimizer/freezing controls.'
    for key, expected in {
        'optimizer': 'adam', 'lr': 0.00001, 'weight_decay': 0.0,
        'lr_scheduler': 'none', 'warmup_ratio': 0.0,
        'freeze_vision_encoder': True, 'freeze_text_encoder': True,
    }.items():
        assert controls[key] == expected, f'{run}: unexpected {key}'
    folder = root / run
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f'Use a fresh output folder: {folder}')
    os.environ[f'{run.upper()}_CKPT'] = str(path)
    selected[run] = str(path)

left, right = [metadata[selected[run]]['run_config'] for run in selected]
for key in ('training_config', 'optimization_config', 'train_annotation_sha256',
            'experiment_config'):
    assert left[key] == right[key], f'A/B mismatch: {key}'
for key in ('initial_checkpoint', 'mask_group_sha256', 'group_counts',
            'sampling', 'updates_per_epoch'):
    assert left['group_protocol'][key] == right['group_protocol'][key], key
(root / 'configs').mkdir(parents=True, exist_ok=True)
for run in selected:
    config = metadata[selected[run]]['run_config']
    settings = {**config['training_config'], 'focus_alpha': config['focus_alpha'],
                'project_updates': False}
    (root / 'configs' / f'{run}.json').write_text(json.dumps(settings, indent=2))
(root / 'continuation.json').write_text(json.dumps({
    'start_counter': 5, 'target_counter': 10, 'total_parent_epochs': 20,
    'checkpoints': selected, 'saved_metadata': metadata,
}, indent=2))
print(json.dumps(selected, indent=2))
```

JSON config values take precedence over CLI flags, which take precedence over defaults. Here the loss settings are restored directly from each checkpoint's metadata, and the trainer verifies that all resume settings and dataset hashes match. Each run continues its own optimizer state and epoch-indexed sampling with both encoders frozen.

## Cell 3: One session: A then B, train and evaluate

Each command runs as a separate process. Preflight tests a temporary in-memory copy of the resumed state and discards its updates, after which `train_group.py` reloads the mounted checkpoint from `/kaggle/input`.

```bash
%%bash
set -euo pipefail
for run in natural mild_focus; do
    checkpoint_var="${run^^}_CKPT"
    checkpoint="${!checkpoint_var}"
    run_dir="$RUN_ROOT/$run"
    common=(
        --data-root "$DATA_ROOT"
        --resume-checkpoint "$checkpoint"
        --config "$RUN_ROOT/configs/${run}.json"
        --batch-size 6
        --gradient-accumulation-steps 1
        --lr 0.00001
        --optimizer adam
        --weight-decay 0
        --lr-scheduler none
        --warmup-ratio 0
        --freeze-vision-encoder True
        --freeze-text-encoder True
        --no-project-updates
        --gradient-checkpointing
        --seed 42
    )
    uv run --frozen torchrun --standalone --nproc_per_node=2 preflight_group.py \
        "${common[@]}" --num-workers 0 --steps 3
    uv run --frozen torchrun --standalone --nproc_per_node=2 train_group.py \
        "${common[@]}" --output-dir "$run_dir" \
        --num-epochs 10 --num-workers 2 \
        --save-every 10 --no-save-final-weights

    # Evaluate both counters using the same checkout and native-resolution rules.
    for epoch in 5 10; do
        eval_checkpoint="$checkpoint"
        if [[ "$epoch" == '10' ]]; then
            eval_checkpoint="$run_dir/checkpoint_epoch10.pt"
        fi
        uv run --frozen python eval.py \
            --checkpoint "$eval_checkpoint" --dataset val \
            --data-root "$DATA_ROOT" --device cuda \
            --batch-size "$EVAL_BATCH" --num-workers 2 \
            --output-dir "$run_dir/eval_val_ep${epoch}"
        uv run --frozen python diagnose.py \
            --predictions "$run_dir/eval_val_ep${epoch}" --data-root "$DATA_ROOT" \
            --output-dir "$run_dir/diagnostic_ep${epoch}"
        if [[ "$EVAL_TEST" == '1' ]]; then
            uv run --frozen python eval.py \
                --checkpoint "$eval_checkpoint" --dataset test \
                --data-root "$DATA_ROOT" --device cuda \
                --batch-size "$EVAL_BATCH" --num-workers 2 \
                --output-dir "$run_dir/eval_test_ep${epoch}"
        fi
    done
done
```

Validation runs after training finishes, evaluating both +5 and +10 checkpoints in FP32 at native mask resolution. Use validation metrics for checkpoint selection; lower `EVAL_BATCH` to `1` if batch size 4 exceeds GPU memory.

This workflow writes two new checkpoint files:

- `outputs/natural/checkpoint_epoch10.pt`
- `outputs/mild_focus/checkpoint_epoch10.pt`

Intermediate epochs (6 through 9), duplicate epoch-5 copies, and extra final weight files are not written. Keep the mounted epoch-5 checkpoints separately. If a run is interrupted before epoch 10, restart that run from epoch 5 in a clean output folder. If `natural` (A) completes and `mild_focus` (B) fails, rerun Cell 3 with `for run in mild_focus; do` after clearing any partial files in `outputs/mild_focus`.

## Cell 4: Compare +5 versus +10

```python
import csv

root = Path(os.environ['RUN_ROOT'])
rows = []
for run in ('natural', 'mild_focus'):
    baseline = None
    for epoch in (5, 10):
        folder = root / run
        summary = json.loads((folder / f'eval_val_ep{epoch}/metrics.json').read_text())['summary']
        diagnostic = json.loads((folder / f'diagnostic_ep{epoch}/summary.json').read_text())
        protocol = summary['training_protocol']
        assert summary['objective_version'] == 'size-weighted-updates-v1'
        assert protocol['focus_alpha'] == (0.0 if run == 'natural' else 0.25)
        assert protocol['projection'] is False
        source = metadata[selected[run]]['run_config']['group_protocol']
        for key in ('initial_checkpoint', 'mask_group_sha256', 'proposal_weights',
                    'group_counts', 'updates_per_epoch'):
            assert protocol[key] == source[key], key
        values = {
            'val_miou_pct': 100 * summary['mean_iou'],
            'val_global_iou_pct': 100 * summary['global_iou'],
            **{f'{name}_miou_pct': 100 * diagnostic['by_gt_area'][group]['mean_iou']
               for name, group in [('small', 'small_lt_1pct'),
                                   ('medium', 'medium_1_to_10pct'),
                                   ('large', 'large_ge_10pct')]},
        }
        if baseline is None:
            baseline = values
        row = {'run': run, 'counter': epoch, 'total_training_epochs': 20 + epoch,
               **values,
               **{f'{key}_delta_vs_ep5_pp': value - baseline[key]
                  for key, value in values.items()}}
        test_path = folder / f'eval_test_ep{epoch}/metrics.json'
        row['test_miou_pct'] = (100 * json.loads(test_path.read_text())['summary']['mean_iou']
                                if test_path.is_file() else None)
        rows.append(row)
comparison = root / 'continuation_comparison.csv'
with comparison.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(comparison.read_text())
```

Compare +5 and +10 across small, medium, and large ground-truth area groups as well as overall validation mean IoU. For reference, the seed-42 20-epoch parent pyramid scores 65.7663% validation mean IoU (19.4521% small, 46.2369% medium, 79.5730% large).

## Cell 5: ZIP results, exclude checkpoints

```python
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(os.environ['RUN_ROOT'])
archive = Path('/kaggle/working') / f"{os.environ['RUN_NAME']}.zip"
with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() not in {'.pt', '.tmp', '.zip'}:
            bundle.write(path, path.relative_to(root))
print('Download results:', archive)
print('Keep both checkpoint_epoch10.pt files separately for future resume.')
```

The archive contains configuration files, continuation metadata, training history for epochs 6 through 10, validation and test predictions and metrics for counters 5 and 10, diagnostics, and `continuation_comparison.csv`, while excluding `.pt`, `.tmp`, and nested `.zip` files.

## Objective and compatibility

Group proposal weights are interpolated from the training split's mask-size frequencies `p`:

    weights = (1 - focus_alpha) * p + focus_alpha * (0.5, 0.5, 0)

| Run | `focus_alpha` | Small / medium / large weights (current train set) |
|---|---:|---|
| `natural` (A) | 0 | 3.83% / 25.81% / 70.36% |
| `mild_focus` (B) | 0.25 | 15.38% / 31.86% / 52.77% |

Both runs use the balanced group sampler, so `natural` (A) pairs natural-frequency loss weighting with stratified group sampling. On this branch, `--focus-alpha` defaults to `0.25` and gradient projection defaults to `False`. Model architecture and inference parameters are unchanged. Full optimizer resume requires a `size-weighted-updates-v1` checkpoint with matching loss configuration, `focus_alpha`, dataset hash, optimizer, encoder freezing policy, seed, world size, batch size, and gradient accumulation steps.

To start a fresh 5-epoch A/B experiment from the original 20-epoch parent checkpoint instead of continuing from epoch 5, pass `--init-checkpoint` pointing to the parent `checkpoint_epoch20.pt` along with `--num-epochs 5 --save-every 5`.
