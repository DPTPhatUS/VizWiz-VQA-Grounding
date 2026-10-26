# Context-crop pyramid: one additional Kaggle run

Branch: `exp/context-crop-pyramid`, branched from `exp/size-weighted-pyramid` at `e89dc65`. This experiment evaluates training-only scale and context augmentation without adding new network modules, loss terms, inference crops, answer inputs, or inference parameters.

Training resumes from the full `mild_focus` (B) checkpoint at counter 5 (`B5`), retaining its model weights, optimizer state, AMP scaler, frozen vision and text encoders, group-weighted BCE + Dice loss, balanced sampler, learning rate of `1e-5`, seed 42, and update count. Training runs counters 6 through 10 (5 additional epochs, or 30 total epochs including the original 20-epoch parent). Compare the resulting model (`crop10`) against `mild_focus` continued to counter 10 without crops (`B10` from `exp/size-weighted-pyramid`) as well as `B5`.

## What changes

For each sampled training image, the dataset attempts a context crop with probability `0.25`; otherwise it uses the full image. When a crop is attempted and a valid crop window exists:

1. Compute the foreground bounding box from the training ground-truth mask.
2. Select a crop window matching the source image aspect ratio that contains all ground-truth foreground components plus surrounding context.
3. Jitter the window position while keeping the padded foreground extent inside the window.
4. Crop the native upright RGB image and ground-truth mask using identical coordinates before constructing the 336 RGB and 768 detail views. Standard RGB normalization, nearest-neighbor mask resizing, and question text remain unchanged.

Width and height share the same scale factor (up to integer pixel rounding). Let `r` be `max(bbox_width / image_width, bbox_height / image_height)`. The bounding box is padded by 25% of its width and height on available sides, and the window side fraction is sampled uniformly between `max(0.35, 1.5 * r, padded_bbox_extent)` and `min(0.9, max(0.5, 3 * r))`. Window position is sampled uniformly across valid integer coordinates rather than centering the target. All ground-truth foreground pixels are retained prior to mask resizing.

If the ground-truth mask is empty, if disjoint foreground components span too wide an extent, or if the padded context window is infeasible, the loader falls back to the uncropped full image. Because large foreground regions frequently trigger this fallback, the realized crop fraction is lower than the requested `0.25` attempt probability.

Size-group membership and loss weights are determined from the uncropped nearest-resized training mask, so a small target that occupies a larger fraction of a cropped view still contributes to its original size group's objective. Training data order, group frequency weights, and the number of forward passes per step match run B (each crop replaces the full-image training view rather than adding an extra forward pass).

Validation and test evaluation always run on uncropped full images at native mask resolution. Background reference on multi-scale context and detail: [HRDA](https://arxiv.org/abs/2204.13132).

## Cell 1: Setup

Start a fresh Kaggle session with Internet and two GPUs enabled, and attach the VizWiz dataset along with the full `B5` (`mild_focus/checkpoint_epoch5.pt`) checkpoint.

```python
!git clone --single-branch --branch exp/context-crop-pyramid https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen
```

If already cloned on this branch, run `!git pull --ff-only` and `!uv sync --frozen`.

## Cell 2: Select B5 and write the crop config

The script locates the `B5` checkpoint via its saved metadata. If multiple `checkpoint_epoch5.pt` files are mounted, set `B_CKPT` explicitly or narrow `INPUT_ROOT`.

```python
import gc
import json
import os
from pathlib import Path
from utils import read_checkpoint

os.environ['DATA_ROOT'] = '/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz'
os.environ['INPUT_ROOT'] = '/kaggle/input'
os.environ['B_CKPT'] = ''  # Optional exact full B/mild_focus checkpoint_epoch5.pt.
os.environ['RUN_ROOT'] = 'outputs'
os.environ['RUN_NAME'] = 'context_crop_pyramid_ep10_seed42'
os.environ['EVAL_BATCH'] = '4'
os.environ['EVAL_TEST'] = '1'  # Set0 when local test GT is unavailable.
os.environ['CONTROL_RESULTS_ZIP'] = ''  # Optional completed A/B+10 results ZIP.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

paths = ([Path(os.environ['B_CKPT'])] if os.environ['B_CKPT'] else
         sorted(Path(os.environ['INPUT_ROOT']).rglob('checkpoint_epoch5.pt')))
choices = []
for path in paths:
    saved = read_checkpoint(path)
    config = saved.get('run_config', {})
    if (saved.get('epoch') == 5 and config.get('focus_alpha') == 0.25
            and config.get('objective_version') == 'size-weighted-updates-v1'):
        assert saved.get('optimizer_state_dict', {}).get('state'), 'Use a full checkpoint.'
        assert 'scaler_state_dict' in saved
        choices.append((path, config, saved.get('training_controls', {}).get('config')))
    del saved
    gc.collect()
if len(choices) != 1:
    raise ValueError(f'Select exactly one B5 checkpoint using B_CKPT: {[str(x[0]) for x in choices]}')
path, source_config, controls = choices[0]
assert controls is not None, 'Missing saved optimizer/freezing controls.'
for key, expected in {
    'batch_size': 6, 'gradient_accumulation_steps': 1, 'world_size': 2,
    'seed': 42, 'lr': 0.00001, 'gradient_checkpointing': True,
    'project_updates': False,
}.items():
    assert source_config[key] == expected, key
for key, expected in {
    'optimizer': 'adam', 'lr': 0.00001, 'weight_decay': 0.0,
    'lr_scheduler': 'none', 'warmup_ratio': 0.0,
    'freeze_vision_encoder': True, 'freeze_text_encoder': True,
}.items():
    assert controls[key] == expected, key
os.environ['B_CKPT'] = str(path)
root = Path(os.environ['RUN_ROOT'])
folder = root / 'context_crop'
if folder.exists() and any(folder.iterdir()):
    raise ValueError(f'Use a fresh run folder: {folder}')
(root / 'configs').mkdir(parents=True, exist_ok=True)
config = {
    **source_config['training_config'], 'focus_alpha': 0.25,
    'project_updates': False, 'crop_probability': 0.25,
}
(root / 'configs/context_crop.json').write_text(json.dumps(config, indent=2))
(root / 'continuation.json').write_text(json.dumps({
    'source_checkpoint': str(path), 'source_counter': 5, 'target_counter': 10,
    'source_config': source_config, 'optimizer_controls': controls,
}, indent=2))
print('B5 checkpoint:', path)
print('Training config:', config)
```

This branch adds a single option, `--crop-probability` (default `0.25`), which can also be set via JSON (`crop_probability`). Because JSON overrides CLI flags, set `"crop_probability": 0` in the JSON config if you want to run a no-crop control on this branch. All other crop geometry bounds are fixed and recorded in checkpoint metadata.

## Cell 3: Preflight, train, evaluate and diagnostic

```bash
%%bash
set -euo pipefail
run_dir="$RUN_ROOT/context_crop"
common=(
    --data-root "$DATA_ROOT"
    --resume-checkpoint "$B_CKPT"
    --config "$RUN_ROOT/configs/context_crop.json"
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
for epoch in 5 10; do
    checkpoint="$B_CKPT"
    if [[ "$epoch" == '10' ]]; then
        checkpoint="$run_dir/checkpoint_epoch10.pt"
    fi
    uv run --frozen python eval.py \
        --checkpoint "$checkpoint" --dataset val --data-root "$DATA_ROOT" \
        --device cuda --batch-size "$EVAL_BATCH" --num-workers 2 \
        --output-dir "$run_dir/eval_val_ep${epoch}"
    uv run --frozen python diagnose.py \
        --predictions "$run_dir/eval_val_ep${epoch}" --data-root "$DATA_ROOT" \
        --output-dir "$run_dir/diagnostic_ep${epoch}"
    if [[ "$EVAL_TEST" == '1' ]]; then
        uv run --frozen python eval.py \
            --checkpoint "$checkpoint" --dataset test --data-root "$DATA_ROOT" \
            --device cuda --batch-size "$EVAL_BATCH" --num-workers 2 \
            --output-dir "$run_dir/eval_test_ep${epoch}"
    fi
done
```

Preflight discards its temporary updates, and `train_group.py` reloads `B5` (including optimizer and scaler state) from disk. Training uses AMP, while evaluation runs in FP32 on uncropped full images at native resolution. Set `EVAL_BATCH = '1'` if evaluation batch size 4 exceeds GPU memory.

Only one new checkpoint is written: `outputs/context_crop/checkpoint_epoch10.pt` (intermediate epochs 6 through 9 and duplicate final weights are skipped). If training is interrupted before counter 10, restart from `B5` in a fresh output directory.

## Cell 4: Compare with B5 and optionally matched B10

If you attach `size_weighted_pyramid_two_runs_ep10_seed42.zip` and set `CONTROL_RESULTS_ZIP`, this script reads the matched `B10` control metrics directly from the archive so you can compare `crop10` against `B10` at the same 30-epoch budget.

```python
import csv
from zipfile import ZipFile

root = Path(os.environ['RUN_ROOT'])
folder = root / 'context_crop'
rows = []
entries = []
for name, epoch in [('B5_reference', 5), ('crop10', 10)]:
    metrics = json.loads((folder / f'eval_val_ep{epoch}/metrics.json').read_text())
    diagnostic = json.loads((folder / f'diagnostic_ep{epoch}/summary.json').read_text())
    test_path = folder / f'eval_test_ep{epoch}/metrics.json'
    test = json.loads(test_path.read_text())['summary'] if test_path.is_file() else {}
    entries.append((name, epoch, metrics['summary'], diagnostic, test))
if os.environ['CONTROL_RESULTS_ZIP']:
    with ZipFile(os.environ['CONTROL_RESULTS_ZIP']) as archive:
        def read_result(suffix):
            matches = [name for name in archive.namelist() if name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f'Expected one {suffix}, found {matches}')
            return json.loads(archive.read(matches[0]))
        summary = read_result('mild_focus/eval_val_ep10/metrics.json')['summary']
        diagnostic = read_result('mild_focus/diagnostic_ep10/summary.json')
        test_names = [name for name in archive.namelist()
                      if name.endswith('mild_focus/eval_test_ep10/metrics.json')]
        test = json.loads(archive.read(test_names[0]))['summary'] if len(test_names) == 1 else {}
        assert summary['objective_version'] == 'size-weighted-updates-v1'
        entries.append(('B10_control', 10, summary, diagnostic, test))
source = source_config['group_protocol']
for name, epoch, summary, diagnostic, test in entries:
    protocol = summary['training_protocol']
    for key in ('initial_checkpoint', 'mask_group_sha256', 'proposal_weights',
                'group_counts', 'updates_per_epoch', 'focus_alpha'):
        assert protocol[key] == source[key], (name, key)
    assert summary['training_config'] == source_config['training_config']
    assert summary['optimization_config'] == source_config['optimization_config']
    if name == 'crop10':
        assert summary['objective_version'] == 'context-crop-updates-v1'
        assert protocol['context_crop']['probability'] == config['crop_probability']
    rows.append({
        'run': name, 'counter': epoch, 'total_training_epochs': 20 + epoch,
        'val_miou_pct': 100 * summary['mean_iou'],
        'val_global_iou_pct': 100 * summary['global_iou'],
        **{f'{label}_miou_pct': 100 * diagnostic['by_gt_area'][group]['mean_iou']
           for label, group in [('small', 'small_lt_1pct'),
                                ('medium', 'medium_1_to_10pct'),
                                ('large', 'large_ge_10pct')]},
        'test_miou_pct': 100 * test['mean_iou'] if test else None,
    })
comparison = root / 'crop_comparison.csv'
with comparison.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(comparison.read_text())
```

Use validation metrics to evaluate the effect of cropping across small, medium, and large ground-truth regions, precision, and recall. For reference, the seed-42 `B5` checkpoint scores 66.1693% validation mean IoU (19.5605% small, 47.8303% medium, 79.4138% large).

## Cell 5: ZIP results without .pt files

```python
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(os.environ['RUN_ROOT'])
archive = Path('/kaggle/working') / f"{os.environ['RUN_NAME']}.zip"
with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() not in {'.pt', '.tmp', '.zip'}:
            bundle.write(path, path.relative_to(root))
print('Download results:', archive)
print('Keep outputs/context_crop/checkpoint_epoch10.pt separately.')
```

The ZIP archive includes configs, continuation metadata, training history for epochs 6 through 10, full-image validation and test masks and metrics for counters 5 and 10, diagnostics, and `crop_comparison.csv`, while excluding `.pt`, `.tmp`, and nested `.zip` files.

## Logs and compatibility

In `history.jsonl`, `train_loss` records the group-weighted segmentation objective across both full-image and cropped training views. Additional per-group crop statistics are logged for each source size group:

- `crop_requested_<group>`: fraction of draws in `<group>` where a crop was attempted (target ~`0.25`).
- `crop_applied_<group>`: fraction of draws in `<group>` where a valid crop window was applied (a low applied rate when `crop_requested` is near `0.25` reflects geometry/context fallbacks on large masks).
- `crop_input_area_fraction_<group>`: mean area of the selected crop window relative to the full image (`1.0` indicates an uncropped full image).
- `crop_applied_fraction`: overall fraction of balanced-sampler draws that were cropped.

Checkpoint metadata stores the versioned crop policy, source `B5` file hash, original 20-epoch parent identity, source group mask digest, and resolved group weights. Importing `B5` retains its optimizer, AMP scaler, and epoch counter (`5`) to apply the augmentation intervention mid-stream; any subsequent resume from a `context-crop-updates-v1` checkpoint requires matching crop probability, geometry policy, and training settings. Model state-dict keys and inference behavior remain unchanged.
