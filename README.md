# Size-weighted pyramid: resume A/B from +5 to +10

Branch: `exp/size-weighted-pyramid`. This session continues the completed
`natural` (A) and `mild_focus` (B) runs independently, preserving their optimizer,
AMP scaler, frozen encoders, loss, sampling, seed and learning rate.
Use `train_group.py`, not `train.py`.

`--num-epochs 10` is the final continuation counter: train epochs6-10 only.
Including the original pyramid's20epochs, this reaches30total epochs.
Do not pass30, use `--init-checkpoint`, or initialize B from A.
The original parent20 file is not needed for this session.

The existing resume implementation already supports this constant-LR extension;
no model, objective, checkpoint-format or default behavior change is required.

## Cell 1 — Setup

Use a fresh Kaggle session with Internet and two GPUs. Attach VizWiz and the
TWO FULL A/B `checkpoint_epoch5.pt` files, keeping their folder names if possible.
The previous results ZIP intentionally excludes checkpoints; it cannot resume
training. Raw model weights and C/D checkpoints are also not valid substitutes.

```python
!git clone --single-branch --branch exp/size-weighted-pyramid https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen
```

If already cloned on this branch, use `!git pull --ff-only` and `!uv sync --frozen`.
No notebook file or separate personal guide is required.

## Cell 2 — Locate A/B and restore their configurations

This identifies A/B from checkpoint metadata, not folder names. Set INPUT_ROOT
to a narrower mounted dataset folder if several copies are attached. If multiple
valid A or B files remain, set the corresponding path in `selected` explicitly.
Only load checkpoints you trust.

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

Config overrides CLI, then defaults. Here loss settings are copied from each
checkpoint, and the trainer verifies that resume settings/data match. Each run
continues its OWN optimizer and epoch-indexed sampling; nothing is reset to A/B's
initial parent. Both encoders remain frozen; only CNN/fusion/decoder train.

## Cell 3 — One session: A then B, train and evaluate

Each command is a separate process. Preflight uses a temporary in-memory copy
of the resumed state; its updates are discarded. Training reloads the mounted
checkpoint. No checkpoint is copied to outputs.

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

No validation runs inside training. This evaluates after training, at+5 and+10,
with FP32 inference. Test metrics require local GT; select recipes/checkpoints
using validation only. Set EVAL_BATCH=1 if evaluation batch4 exceeds VRAM.

Only TWO new checkpoint files are written:

- `outputs/natural/checkpoint_epoch10.pt`
- `outputs/mild_focus/checkpoint_epoch10.pt`

No checkpoints6/7/8/9, copied epoch5 or duplicate final weights are saved.
No old checkpoints are deleted. Keep the mounted epoch5 files separately.
If interrupted before10, that run restarts from5. If A completes but B fails,
rerun Cell3 with `for run in mild_focus; do`, using a fresh B output folder if
partial history exists. Do not rerun completed A into its existing folder.
The session budget/VRAM/disk requirements still depend on Kaggle hardware;
CPU verification does not establish that it fits every session limit.

## Cell 4 — Compare +5 versus +10

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

Judge whether more training helps each run, including both smaller groups and
large-region regressions. Keep parent20 as a fixed reference: val65.7663%,
small19.4521%, medium46.2369%, large79.5730% for the supplied seed42 run.
Do not automatically extend to+20 or select the best test score. A/B are close
at+5; neither is a proven multi-seed winner.

## Cell 5 — ZIP results, exclude checkpoints

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

Includes configs, continuation provenance, history6-10, val/test predictions and
metrics for5/10, diagnostics and comparison CSV. No .pt/.tmp/nested.zip files.

## Objective and compatibility

Weights are derived from TRAIN mask frequencies p:

    weights = (1 - focus_alpha) * p + focus_alpha * (0.5, 0.5, 0)

| Run | focus_alpha | Small / medium / large weights (current train set) |
|---|---:|---|
| natural (A) |0|3.83% /25.81% /70.36%|
| mild_focus (B) |0.25|15.38% /31.86% /52.77%|

The balanced sampler is unchanged; A represents the natural-frequency objective
with stratified sampling, not the original pyramid training recipe.
`--focus-alpha` defaults0.25; projection defaultsfalse on this branch. JSON takes
precedence over CLI. Architecture and inference parameters are unchanged.
Old parent/scale weights remain usable for initialization/inference, but optimizer
resume requires `size-weighted-updates-v1` with matching loss, alpha, data,
optimizer, freeze policy, seed, world size, batch and accumulation.

For a fresh A/B experiment, use the same configs/common settings with
`--init-checkpoint` pointing to original parent20 and `--num-epochs 5 --save-every 5`.
That is independent initialization with a fresh optimizer, not this continuation.
Do not use C/D full checkpoints on this branch.