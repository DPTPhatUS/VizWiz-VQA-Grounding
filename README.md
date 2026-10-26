# Crop support: four controlled continuations in two Kaggle sessions

Branch: `exp/crop-recall-preservation`, branched from `exp/context-crop-pyramid` at `2b40528`. All four runs start independently from the full `mild_focus` (B) counter-5 checkpoint (`B5`), retaining its Adam and AMP scaler state, learning rate of `1e-5`, frozen vision and text encoders, group weights, balanced sampler, seed 42, and global batch size of 6. Each run trains counters 6 through 10 (5 additional epochs, or 30 total epochs including the original 20-epoch parent).

No new inference parameters, inputs, or test-time crops are introduced. In the retention variants, the frozen `B5` reference state exists only during training and inside full training checkpoints.

## Why these experiments

In `exp/context-crop-pyramid`, the `crop25` continuation improved tiny and medium validation IoU by 2.51 pp and 2.45 pp over the matched `B10` control, but lost 3.17 pp on large regions as large-region recall dropped from 88.28% to 83.42%. Conversely, continuing `B5` to `B10` without crops reduced small and medium IoU. These four runs test whether a smaller crop dose (`0.10`) or targeted large-region support on uncropped large images can preserve large-object recall while keeping the small and medium gains:

| Session | Folder / `--variant` | Crop probability | Large/full retention | Large/full final overlap |
|---|---|---:|---|---|
| 1 | `low_crop` | 0.10 | None | Dice (unchanged) |
| 1 | `retain` | 0.25 | One-sided `B5` foreground penalty, weight 1.0 | Dice (unchanged) |
| 2 | `recall` | 0.25 | None | Tversky: FP 0.3 / FN 0.7 |
| 2 | `retain_recall` | 0.25 | Same retention, weight 1.0 | Same Tversky |

An optional `control` preset reproduces the `crop25` recipe and is omitted from the two sessions because its results already exist.

### Scope and exact objectives

Size-group labels and weights use the original uncropped nearest-resized training mask. The large-region support scope `S` consists of training samples whose source foreground area is `>=10%` and that are not cropped (`crop_applied == 0`). Thus, an originally tiny target that fills a cropped view is never placed in `S`, and large images that do get cropped use the standard objective.

Let `L_B` be the baseline per-image loss: final `BCE + 0.5 * Dice` plus the three auxiliary `BCE + Dice` heads. In the `recall` variants, only the final overlap term on `S` is replaced with asymmetric [Tversky loss](https://arxiv.org/abs/1706.05721) (`FP = 0.3`, `FN = 0.7`):

    T = (TP+1)/(TP + 0.3*FP + 0.7*FN + 1)
    D = (2*TP+1)/(2*TP+FP+FN+1)
    L_recall = L_B + 0.5*(D-T) on S; L_B elsewhere

All auxiliary heads remain unchanged. Because weighting false negatives more heavily (`0.7` vs. `0.3`) can increase false positives, inspect large-region precision and IoU alongside recall.

For the `retain` variants, let `K` be the pixels on an image in `S` that are ground-truth foreground and where the frozen `B5` logit is positive (`> 0`):

    keep(image) = mean_K relu(sigmoid(B5)-sigmoid(student))^2
    L_retain = L_B + 1.0*keep
    L_retain_recall = L_recall + 1.0*keep

If `K` is empty, `keep(image)` is `0`. The penalty is normalized per image. Teacher false positives and false negatives are not penalized, and the student is not penalized for exceeding `B5` confidence (see [Learning without Forgetting](https://arxiv.org/abs/1606.09282)). While a similar foreground penalty in `exp/parent-preserving-pyramid` did not improve uncropped training from the 20-epoch parent, here it specifically targets large-region recall drift induced by context cropping.

The `B5` reference copies only the post-encoder modules (~6.65M parameters), sharing the frozen CLIP vision and text encoder outputs computed during the student's forward pass. The reference runs in `eval()` / `no_grad` mode outside DDP, the optimizer, and `model_state_dict`. Resuming a `retain` checkpoint restores the original `B5` body from `support_state` rather than re-anchoring to the current student. `low_crop` and `recall` do not allocate a reference module.

### Crop policy

All variants use the same crop geometry as `exp/context-crop-pyramid`: a source-aspect-ratio window on the native upright RGB image and mask that retains all ground-truth foreground components plus 25% bbox padding and jittered positioning, with window side fraction sampled between 35% and 90% before building the 336 and 768 views. Infeasible or empty masks fall back to the full image. Only the crop attempt probability changes for `low_crop` (`0.10` vs. `0.25`).

## Cell 1: Setup (both sessions)

Start a fresh Kaggle session with Internet and two GPUs enabled, and attach the VizWiz dataset and the full `B5` checkpoint (`mild_focus/checkpoint_epoch5.pt`). Optionally attach the completed `size_weighted_pyramid_two_runs_ep10_seed42` and `context_crop_pyramid_ep10_seed42` results (as extracted directories or `.zip` archives) for comparison in Cell 4. Use `train_group.py` rather than `train.py`.

```python
!git clone --single-branch --branch exp/crop-recall-preservation https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen
```

If already cloned on this branch, run `!git pull --ff-only` followed by `!uv sync --frozen`.

## Cell 2: Choose session and B5

Run `SESSION = 1` (`low_crop` and `retain`) in the first Kaggle session, then run `SESSION = 2` (`recall` and `retain_recall`) in a second fresh session using Cells 1 through 5. The script verifies `B5` from its checkpoint metadata; set `B_CKPT` explicitly if multiple copies are mounted.

```python
import gc
import json
import os
from pathlib import Path
from training.crop_support import RECIPES
from utils import read_checkpoint

SESSION = 1  # Session2: change this to2; all other steps stay the same.
os.environ['DATA_ROOT'] = '/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz'
os.environ['INPUT_ROOT'] = '/kaggle/input'
os.environ['B_CKPT'] = '/kaggle/input/datasets/khoahoang0512/vizwiz-size-weighted-pyramid-cp/mild_focus/checkpoint_epoch5.pt'
os.environ['RUN_ROOT'] = 'outputs'
os.environ['EVAL_BATCH'] = '4'
os.environ['EVAL_TEST'] = '1'  # Use0 if local test GT is unavailable.
os.environ['BASELINE_RESULTS'] = ''  # Optional exact result directory OR .zip; empty auto-detects below.
os.environ['CROP_RESULTS'] = ''  # Optional exact result directory OR .zip; empty auto-detects below.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
# Kaggle may expose uploaded ZIP contents as directories, as in vizwiz-exp.
for variable, name in (
    ('BASELINE_RESULTS', 'size_weighted_pyramid_two_runs_ep10_seed42'),
    ('CROP_RESULTS', 'context_crop_pyramid_ep10_seed42'),
):
    if not os.environ[variable]:
        candidates = sorted(set(Path(os.environ['INPUT_ROOT']).rglob(name))
                            | set(Path(os.environ['INPUT_ROOT']).rglob(name + '.zip')))
        candidates = [path for path in candidates if path.is_dir() or path.suffix == '.zip']
        if len(candidates) > 1:
            raise ValueError(f'Set {variable} to one result directory/ZIP: {candidates}')
        os.environ[variable] = str(candidates[0]) if candidates else ''
    print(variable, os.environ[variable] or '(not attached; comparison will skip it)')
assert SESSION in (1, 2)
variants = ('low_crop', 'retain') if SESSION == 1 else ('recall', 'retain_recall')
os.environ['RUN_VARIANTS'] = ' '.join(variants)
os.environ['RUN_NAME'] = f'crop_support_s{SESSION}_ep10_seed42'
paths = ([Path(os.environ['B_CKPT'])] if os.environ['B_CKPT'] else
         sorted(Path(os.environ['INPUT_ROOT']).rglob('checkpoint_epoch5.pt')))
choices = []
for path in paths:
    saved = read_checkpoint(path)
    config = saved.get('run_config', {})
    if (saved.get('epoch') == 5 and config.get('focus_alpha') == 0.25
            and config.get('objective_version') == 'size-weighted-updates-v1'):
        assert saved.get('optimizer_state_dict', {}).get('state'), 'Use a FULL checkpoint.'
        assert 'scaler_state_dict' in saved
        choices.append((path, config, saved.get('training_controls', {}).get('config')))
    del saved
    gc.collect()
if len(choices) != 1:
    raise ValueError(f'Select one B5 using B_CKPT: {[str(x[0]) for x in choices]}')
path, source_config, controls = choices[0]
assert controls is not None
for key, expected in {
    'batch_size': 6, 'gradient_accumulation_steps': 1, 'world_size': 2,
    'seed': 42, 'lr': 0.00001, 'gradient_checkpointing': True, 'project_updates': False,
}.items():
    assert source_config[key] == expected, key
for key, expected in {
    'optimizer': 'adam', 'lr': 0.00001, 'weight_decay': 0.0,
    'lr_scheduler': 'none', 'warmup_ratio': 0.0,
    'freeze_vision_encoder': True, 'freeze_text_encoder': True,
}.items():
    assert controls[key] == expected, key
assert source_config['training_config']['overlap_loss'] == 'dice'
os.environ['B_CKPT'] = str(path)
root = Path(os.environ['RUN_ROOT'])
for variant in variants:
    folder = root / variant
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f'Use a fresh output folder: {folder}')
(root / 'configs').mkdir(parents=True, exist_ok=True)
for variant in variants:
    config = {**source_config['training_config'], 'focus_alpha': 0.25,
              'project_updates': False, 'variant': variant}
    (root / 'configs' / f'{variant}.json').write_text(json.dumps(config, indent=2))
(root / 'session.json').write_text(json.dumps({
    'session': SESSION, 'variants': list(variants), 'B5_checkpoint': str(path),
    'source_config': source_config, 'optimizer_controls': controls,
    'recipes': {name: RECIPES[name].config for name in variants},
}, indent=2))
print('B5:', path)
print('Runs:', os.environ['RUN_VARIANTS'])
```

The `--variant` flag (and `"variant"` in JSON, with JSON taking precedence over CLI) selects the versioned recipe preset (`crop_probability`, `retention_weight`, and `large_fp_weight`). Keep all optimizer, learning rate, and freezing settings identical across variants.

## Cell 3: Two independent runs, then evaluate

```bash
%%bash
set -euo pipefail
for variant in $RUN_VARIANTS; do
    run_dir="$RUN_ROOT/$variant"
    common=(
        --data-root "$DATA_ROOT"
        --resume-checkpoint "$B_CKPT"
        --config "$RUN_ROOT/configs/${variant}.json"
        --variant "$variant"
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
        --num-epochs 10 --num-workers 2 --save-every 10 --no-save-final-weights
    uv run --frozen python eval.py \
        --checkpoint "$run_dir/checkpoint_epoch10.pt" --dataset val \
        --data-root "$DATA_ROOT" --device cuda \
        --batch-size "$EVAL_BATCH" --num-workers 2 --output-dir "$run_dir/eval_val_ep10"
    uv run --frozen python diagnose.py \
        --predictions "$run_dir/eval_val_ep10" --data-root "$DATA_ROOT" \
        --output-dir "$run_dir/diagnostic_ep10"
    if [[ "$EVAL_TEST" == '1' ]]; then
        uv run --frozen python eval.py \
            --checkpoint "$run_dir/checkpoint_epoch10.pt" --dataset test \
            --data-root "$DATA_ROOT" --device cuda \
            --batch-size "$EVAL_BATCH" --num-workers 2 --output-dir "$run_dir/eval_test_ep10"
    fi
done
# Evaluate the unchanged B5 reference once per session, not once per variant.
uv run --frozen python eval.py \
    --checkpoint "$B_CKPT" --dataset val --data-root "$DATA_ROOT" --device cuda \
    --batch-size "$EVAL_BATCH" --num-workers 2 --output-dir "$RUN_ROOT/reference/eval_val_ep5"
uv run --frozen python diagnose.py \
    --predictions "$RUN_ROOT/reference/eval_val_ep5" --data-root "$DATA_ROOT" \
    --output-dir "$RUN_ROOT/reference/diagnostic_ep5"
```

Each command runs in a fresh subprocess. Preflight discards its temporary updates, and `train_group.py` reloads the untouched `B5` checkpoint for each variant (`--num-epochs 10` runs counters 6 through 10). Full-image FP32 validation and diagnostics run after training completes. Lower `EVAL_BATCH` to `1` if batch size 4 exceeds GPU memory.

Each session writes two checkpoint files (`outputs/<variant>/checkpoint_epoch10.pt`). In `retain` and `retain_recall`, the checkpoint includes the post-encoder `B5` reference (~26.6 MB in FP32) alongside the student model and frozen encoders. If a run is interrupted before counter 10, clear its output directory and restart that variant from `B5` (if the first variant in the session already finished, set `RUN_VARIANTS` to run only the second variant).

## Cell 4: Compare both runs with B5 and existing B10/crop25

If the previous `B10` and `crop25` result folders or ZIP files are attached, Cell 2 auto-detects them (or you can set `BASELINE_RESULTS` and `CROP_RESULTS` explicitly) so this script can include them directly in the comparison table.

```python
import csv
from contextlib import contextmanager
from zipfile import ZipFile

root = Path(os.environ['RUN_ROOT'])
entries = []
def local_result(folder, epoch):
    summary = json.loads((folder / f'eval_val_ep{epoch}/metrics.json').read_text())['summary']
    diag = json.loads((folder / f'diagnostic_ep{epoch}/summary.json').read_text())
    path = folder / f'eval_test_ep{epoch}/metrics.json'
    test = json.loads(path.read_text())['summary'] if path.is_file() else {}
    return summary, diag, test
entries.append(('B5_reference', 5, *local_result(root / 'reference', 5)))
for variant in variants:
    summary, diag, test = local_result(root / variant, 10)
    assert summary['objective_version'] == 'crop-support-updates-v1'
    assert summary['training_protocol']['support_recipe'] == RECIPES[variant].config
    entries.append((variant, 10, summary, diag, test))
@contextmanager
def result_reader(location):
    path = Path(location)
    if path.is_dir():
        def read_result(suffix, required=True):
            matches = [file for file in path.rglob(Path(suffix).name)
                       if file.is_file() and file.as_posix().endswith(suffix)]
            if not matches and not required:
                return {}
            if len(matches) != 1:
                raise ValueError(f'Expected one {suffix} under {path}: {matches}')
            return json.loads(matches[0].read_text())
        yield read_result
    else:
        with ZipFile(path) as archive:
            def read_result(suffix, required=True):
                matches = [name for name in archive.namelist() if name.endswith(suffix)]
                if not matches and not required:
                    return {}
                if len(matches) != 1:
                    raise ValueError(f'Expected one {suffix}: {matches}')
                return json.loads(archive.read(matches[0]))
            yield read_result

for variable, label, prefix, expected_version in [
    ('BASELINE_RESULTS', 'B10_control', 'mild_focus', 'size-weighted-updates-v1'),
    ('CROP_RESULTS', 'crop25_control', 'context_crop', 'context-crop-updates-v1'),
]:
    if not os.environ[variable]:
        continue
    with result_reader(os.environ[variable]) as read_result:
        summary = read_result(f'{prefix}/eval_val_ep10/metrics.json')['summary']
        diag = read_result(f'{prefix}/diagnostic_ep10/summary.json')
        test = read_result(f'{prefix}/eval_test_ep10/metrics.json', required=False).get('summary', {})
        assert summary['objective_version'] == expected_version
        entries.append((label, 10, summary, diag, test))
rows = []
for label, epoch, summary, diag, test in entries:
    protocol = summary['training_protocol']
    for key in ('initial_checkpoint', 'mask_group_sha256', 'group_counts',
                'proposal_weights', 'updates_per_epoch', 'focus_alpha'):
        assert protocol[key] == source_config['group_protocol'][key], (label, key)
    assert summary['optimization_config'] == source_config['optimization_config']
    assert summary['training_config'] == source_config['training_config']
    recipe = protocol.get('support_recipe', {})
    rows.append({
        'run': label, 'counter': epoch, 'total_training_epochs': 20 + epoch,
        'crop_probability': protocol.get('context_crop', {}).get('probability', 0),
        'retention_weight': recipe.get('retention_weight', 0),
        'large_fp_weight': recipe.get('large_fp_weight'),
        'val_miou_pct': 100 * summary['mean_iou'],
        'val_global_iou_pct': 100 * summary['global_iou'],
        **{f'{name}_miou_pct': 100 * diag['by_gt_area'][group]['mean_iou']
           for name, group in [('small', 'small_lt_1pct'),
                               ('medium', 'medium_1_to_10pct'),
                               ('large', 'large_ge_10pct')]},
        'large_recall_pct': 100 * diag['by_gt_area']['large_ge_10pct']['mean_recall'],
        'large_precision_pct': 100 * diag['by_gt_area']['large_ge_10pct']['mean_precision'],
        'test_miou_pct': 100 * test['mean_iou'] if test else None,
    })
comparison = root / 'support_comparison.csv'
with comparison.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(comparison.read_text())
```

### Decision criteria

Compare `low_crop` against `B10` and `crop25` to evaluate the crop-probability tradeoff, and compare `retain`, `recall`, and `retain_recall` against `crop25` at the matched 30-epoch budget (`B5` serves as the 25-epoch starting reference). Seed-42 validation reference numbers:

| Reference | Val mean | Tiny | Medium | Large | Large recall |
|---|---:|---:|---:|---:|---:|
| `B5` | 66.1693 | 19.5605 | 47.8303 | 79.4138 | 87.5861 |
| `B10` | 66.0497 | 18.8439 | 46.8101 | 79.7930 | 88.2752 |
| `crop25+10` | 64.9561 | 21.3502 | 49.2588 | 76.6254 | 83.4239 |

Look for candidates that recover the large-region recall and IoU drop relative to `crop25+10` while preserving the tiny and medium IoU gains. Select recipes using validation metrics only.

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
print('Keep the two full checkpoint_epoch10.pt files separately.')
```

Each session archive (`crop_support_s1_ep10_seed42.zip` and `crop_support_s2_ep10_seed42.zip`) packages the session plan, configs, training histories for counters 6 through 10, validation and test masks and metrics, diagnostics, the `B5` reference evaluation, and `support_comparison.csv`, excluding `.pt`, `.tmp`, and nested `.zip` files.

## Logging and compatibility

In `history.jsonl`, `train_loss` and `proposal_loss` include the active support terms, while the unadjusted mixed full/crop segmentation loss is logged as `base_segmentation_objective` and `loss_<group>`:

    train_loss = base_segmentation_objective
                 + retention_objective_contribution + recall_objective_delta

Note that `recall_objective_delta` is signed because the Tversky term can be smaller than the Dice term it replaces. Retention variants log `retention_shortfall_large`, `protected_coverage_large`, and `support_fraction_large` (`protected_coverage_large` is zero for teacher-free recipes). Per-group crop request, applied, and input-area fractions retain the same definitions as in `exp/context-crop-pyramid`. Checkpoints record the source mask-group digest, recipe constants, `B5` file hash, and `support_state` so subsequent resumes verify the exact recipe and restore the original `B5` reference.
