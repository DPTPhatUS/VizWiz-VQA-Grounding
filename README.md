# Context-crop pyramid: one additional Kaggle run

Branch: `exp/context-crop-pyramid`, from `exp/size-weighted-pyramid` at `e89dc65`.
This tests TRAIN-ONLY scale/context augmentation, with no new network modules,
loss terms, inference crops, answer input or inference parameters.

Start from the FULL B/mild_focus checkpoint at counter5. Keep its model,
optimizer, AMP scaler, frozen vision/text, weighted BCE+Dice, balanced sampler,
LR1e-5, seed42 and update count. Train counters6-10 (five more epochs;30total
including the original parent20). Compare with B continued to10 WITHOUT crops
on `exp/size-weighted-pyramid`, not only with B5 or original parent20.

## What changes

For each sampled TRAIN image, attempt a crop with probability0.25. Other samples
use the complete image. If a valid crop exists:

1. Derive the foreground bounding extent from the existing TRAIN mask.
2. Choose a source-aspect-ratio window, retaining ALL GT components and context.
3. Jitter the window position while retaining the padded foreground extent.
4. Crop native upright RGB and mask using the SAME coordinates, BEFORE creating
   the336RGB and768detail views. Existing RGB normalization and mask-nearest
   resizing stay unchanged. Question text is unchanged.

The scale is shared between width/height (integer rounding aside). Let r be the
larger of bbox-width/image-width and bbox-height/image-height. The lower side
fraction is max(0.35,1.5*r, padded bbox extent); the upper is min(0.9,max(0.5,3*r)).
Sample uniformly between these limits. Pad the bbox by25% of its width/height
on available sides. Position is random among valid integer windows, not forced
to put the object at the center. Retain all GT pixels BEFORE normal mask resize.

Empty GT, a foreground extent too wide for this policy, or infeasible context
falls back to the full image. Disjoint masks must ALL fit; do not crop one
component and silently remove the others. Large images often fall back, so25%
is the attempted rate, not a promise that25% of all draws are cropped.

Group membership and weights use the UNCROPPED nearest-resized TRAIN mask.
A crop can make an originally small object occupy a larger fraction, but it
still contributes to its original group's objective. Train data order, frequency
weights and number of model forwards per update remain the same as B.
A crop REPLACES one training view; there is no extra crop model or extra forward.

Val/test always use full images with their normal native-resolution metrics.
GT-dependent crop selection is training augmentation only. The risk is learning
crop-specific context/position shortcuts; unchanged full-image validation is the
test of transfer. This does not promise recovery of detail missing in the photo.
The general context/detail motivation is related to
[HRDA](https://arxiv.org/abs/2204.13132), but this is not the HRDA architecture.

## Cell 1 — Setup

Fresh Kaggle session: Internet, two GPUs, VizWiz dataset and the FULL B epoch5
checkpoint attached. The results ZIP excludes .pt and cannot resume training.
The original parent20, A/natural, C/D and raw final weights are not substitutes.

```python
!git clone --single-branch --branch exp/context-crop-pyramid https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen
```

If already cloned on this branch, use `!git pull --ff-only` and `!uv sync --frozen`.

## Cell 2 — Select B5 and write the crop config

Auto-detect by checkpoint metadata. Set B_CKPT explicitly if multiple B5 files
are mounted. Only load checkpoints you trust. INPUT_ROOT can be narrowed to the
attached checkpoint dataset, avoiding unrelated files.

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

Only one new flag: `--crop-probability` (default0.25), also available in JSON.
Config > explicit CLI > defaults. To run a no-crop control on this branch,
change crop_probability in the JSON to0; an explicit flag cannot override JSON.
The prescribed run uses0.25. Remaining crop geometry is fixed, versioned and
saved in metadata rather than exposed as many new flags.

## Cell 3 — Preflight, train, evaluate and diagnostic

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

Preflight updates are discarded; the next process reloads B5 from disk, including
its optimizer/scaler. Each command exits before the next, releasing GPU state.
Training uses AMP; evaluation FP32, full images and original mask resolution.
No validation inside training. Eval batch4 can be changed to1 if it exceeds VRAM.

Only ONE new checkpoint is saved: `outputs/context_crop/checkpoint_epoch10.pt`.
No intermediate6/7/8/9, copied5 or duplicate final weights. No files are deleted.
The mounted B5 checkpoint stays separate. ZIP excludes checkpoints.
If interrupted before10, restart from B5 into a fresh output folder. To resume
this branch's own later full checkpoint, retain the same probability/geometry,
flags and data; set num-epochs to the final counter. A changed crop policy is
rejected on resume. Scheduled runs cannot extend a fixed scheduler budget.

Intended budget: one5epoch continuation plus evaluation/diagnostics. Actual Kaggle
runtime, memory and disk fit still require measurement on the user's hardware.

## Cell 4 — Compare with B5 and optionally matched B10

CONTROL_RESULTS_ZIP may point to the completed
`size_weighted_pyramid_two_runs_ep10_seed42.zip`. This reads B10 metrics directly
from the ZIP, without extracting masks or loading another checkpoint/model.
If it is unavailable, the table is provisional: B5 versus crop10 changes both
training length and augmentation. B10 is required to isolate augmentation.

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

Choose using validation. Inspect both small groups, large regressions, precision,
recall and native visualizations; do not select using test. B5's supplied seed42
reference is val66.1693%, small19.5605%, medium47.8303%, large79.4138%.
A crop10 gain over B5 alone is not evidence that crop augmentation helps.

## Cell 5 — ZIP results without .pt files

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

Contains configs, source metadata, history6-10, full-image val/test5/10 masks and
metrics, diagnostics and comparison. No checkpoints or nested ZIPs.

## Logs and compatibility

`train_loss` remains the B weighted segmentation objective (all existing heads).
It now mixes full/crop targets; its value is not directly a full-image loss.
`crop_requested_<group>`, `crop_applied_<group>` and
`crop_input_area_fraction_<group>` report draw-averaged statistics per SOURCE
size group. `crop_applied_fraction` is the fraction of actual training draws
cropped across the balanced sampler, not the natural dataset population.
Low applied rate with requested~0.25 means geometry/context fallback, not that
probability was ignored. An input-area fraction1 means a full image.

Checkpoint metadata stores the complete versioned crop policy, original B5 file
hash, original parent identity, source group mask digest and resolved weights.
The first B5 import deliberately retains optimizer/scaler and counter5; this is
an augmentation intervention, NOT a fresh optimizer initialization. Subsequent
resumes require this branch's policy and exact training settings. Crops0 replay
the original B update path; C/D optimizer checkpoints are not accepted.

Model/state-dict keys, preprocessing defaults and inference architecture are
unchanged. Old B checkpoints evaluate with full images as before. The augmentation
uses only PIL RGB/masks, not CLIP-specific features; another dataset/backbone still
needs its normal adapter and independent validation. CPU geometry/gradient/DDP
checks do not establish GPU fit, accuracy gains or cross-backbone performance.