# Pyramid weight interpolation: export all models, then evaluate (no training)

Branch: `exp/pyramid-weight-interpolation`. One Kaggle evaluation session.
Use the FULL B/mild_focus epoch5 and FULL low_crop epoch10 checkpoints.
Do NOT use original parent20, A/natural, crop25, retain or retain_recall.

    body(alpha) = (1-alpha)*body(B5) + alpha*body(low_crop10)

The default grid is0,0.1,0.25,0.5,1: FIVE independent weights-only files are saved,
including exact endpoints. No optimizer, AMP scaler, frozen reference or epoch
counter is stored. Every file is a complete single inference model, with the
same number of parameters as the original pyramid. These are not full training
resumes. Frozen encoders and ALL non-trainable buffers must match bit-for-bit;
normalization buffers are copied, not averaged. Source hashes and provenance
are included in every exported model and in interpolation.json.

This is an established parameter-averaging direction, motivated by
[Model Soups](https://proceedings.mlr.press/v162/wortsman22a.html), not a claim of
novelty or guaranteed VizWiz improvement. The goal is to test whether part of
low_crop's small/medium benefit can be retained nearer the stronger B5 model.
It does not by itself solve the persistent hard localization cases.

## Storage and checkpoint policy

All five models are saved; the script does NOT keep only the validation winner.
Each still includes both encoders, so weights-only is not a tiny decoder file.
The exporter measures actual tensor storage and free disk AFTER uv sync,
with an explicit14GiB budget cap (about15GB) even if the filesystem reports
more space than the Kaggle working-directory quota. It uses the LOWER of
physical free space and that cap, reserving2GiB for masks, metrics, ZIP and serialization overhead. It prints the
estimated size/file and maximum file count under this conservative budget.
It rejects a grid that will not fit BEFORE writing any .pt files.

With roughly15GB available, the default five-model grid is the starting setup;
actual fit depends on the measured checkpoint size. To save six coefficients,
add0.75 to ALPHAS in Cell2 (endpoints are always included). No files are deleted
or silently dropped. Do not lower the reserve just to fill the disk completely.
Existing output folders are never overwritten. Input full checkpoints stay in
/kaggle/input and are not copied into outputs; the exported endpoint files are
weights-only versions of them.

## Cell 1 — Setup

Attach VizWiz, full B5 and the low_crop checkpoint from session1. Result ZIPs
contain no .pt and cannot serve as sources. Enable Internet and a GPU; evaluation
uses one GPU and does not need torchrun. Export itself runs only on CPU.

```python
!git clone --single-branch --branch exp/pyramid-weight-interpolation https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen
```

Already cloned on this branch: `!git pull --ff-only`, then `!uv sync --frozen`.

## Cell 2 — Paths and the predeclared alpha grid

B5 uses your existing mount. Low_crop is auto-detected under a low_crop folder;
paste its full path into LOW_CROP_CKPT if the upload uses another name/layout.
Only load trusted checkpoint files.

```python
import json
import os
from pathlib import Path

os.environ['DATA_ROOT'] = '/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz'
os.environ['B_CKPT'] = '/kaggle/input/datasets/khoahoang0512/vizwiz-size-weighted-pyramid-cp/mild_focus/checkpoint_epoch5.pt'
os.environ['LOW_CROP_CKPT'] = ''  # Full low_crop/checkpoint_epoch10.pt from session1.
os.environ['RUN_ROOT'] = 'outputs'
os.environ['RUN_NAME'] = 'pyramid_weight_interpolation_seed42'
os.environ['ALPHAS'] = '0.1 0.25 0.5'  # Automatically includes0/1. Add0.75 for six files.
os.environ['EVAL_BATCH'] = '4'
os.environ['EVAL_TEST'] = '1'  # Use0 if local test GT is unavailable.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

if not Path(os.environ['B_CKPT']).is_file():
    raise FileNotFoundError('Set B_CKPT to the original full B/mild_focus epoch5 file.')
if not os.environ['LOW_CROP_CKPT']:
    candidates = [path for path in Path('/kaggle/input').rglob('checkpoint_epoch10.pt')
                  if path.parent.name == 'low_crop']
    if len(candidates) != 1:
        raise ValueError(f'Set LOW_CROP_CKPT explicitly; candidates: {candidates}')
    os.environ['LOW_CROP_CKPT'] = str(candidates[0])
if not Path(os.environ['LOW_CROP_CKPT']).is_file():
    raise FileNotFoundError(os.environ['LOW_CROP_CKPT'])
root = Path(os.environ['RUN_ROOT'])
if root.exists() and any(root.iterdir()):
    raise ValueError('Use a fresh output folder. Existing files are not deleted.')
print('Base:', os.environ['B_CKPT'])
print('Candidate:', os.environ['LOW_CROP_CKPT'])
```

The tool validates model config, source B5 file hash, original parent/data/group
identity, trainable-name layout, frozen policy, tensor shapes/dtypes and exact
frozen tensor equality. It blends only finite FP32 post-encoder parameters.
A mismatch is rejected; there is no force flag that changes frozen encoders.
Loading both source checkpoints briefly needs CPU RAM; only one complete frozen
weight set plus two small trainable bodies is retained for exporting the grid.

## Cell 3 — Export ALL weights-only models, then full validation/diagnostics

```bash
%%bash
set -euo pipefail
uv run --frozen python interpolate.py \
    --base-checkpoint "$B_CKPT" \
    --candidate-checkpoint "$LOW_CROP_CKPT" \
    --alphas $ALPHAS \
    --storage-budget-gib 14 \
    --disk-reserve-gib 2 \
    --output-dir "$RUN_ROOT"

for checkpoint in "$RUN_ROOT"/alpha_*.pt; do
    tag="$(basename "$checkpoint" .pt)"
    run_dir="$RUN_ROOT/$tag"
    uv run --frozen python eval.py \
        --checkpoint "$checkpoint" --dataset val --data-root "$DATA_ROOT" \
        --device cuda --batch-size "$EVAL_BATCH" --num-workers 2 \
        --output-dir "$run_dir/eval_val"
    uv run --frozen python diagnose.py \
        --predictions "$run_dir/eval_val" --data-root "$DATA_ROOT" \
        --output-dir "$run_dir/diagnostic_val"
done
```

Default files: alpha_0.pt, alpha_0p1.pt, alpha_0p25.pt, alpha_0p5.pt, alpha_1.pt.
The loop discovers actual exported files, so adding0.75 needs no loop edits.
No gradients or training epochs. FP32/native-resolution evaluation rules and
question-only inference stay unchanged. Each evaluation is a new process and
releases GPU allocations afterward. Set EVAL_BATCH=1 if needed.

Export finishes BEFORE any evaluation. If a later evaluation fails, all exported
models remain usable with the ordinary eval.py command; do not rerun export into
that nonempty directory. Run only the pending evaluation/diagnostic commands,
with a new output directory if partial predictions already exist. No automatic
cleanup or checkpoint deletion is performed.

## Cell 4 — Compare and select using validation only

This selects highest validation mean IoU; ties favor smaller alpha (closer to
B5). This is a numerical selection convenience, not a multi-seed conclusion.
Inspect tiny/medium and large regressions before adopting it as the research
candidate. The primary choice does not use test, global IoU or GT area gating.

```python
import csv

root = Path(os.environ['RUN_ROOT'])
manifest = json.loads((root / 'interpolation.json').read_text())
assert manifest['complete'] and len(manifest['checkpoints']) == len(manifest['alphas'])
rows = []
for tag, entry in manifest['checkpoints'].items():
    metrics = json.loads((root / tag / 'eval_val/metrics.json').read_text())
    diagnostic = json.loads((root / tag / 'diagnostic_val/summary.json').read_text())
    summary = metrics['summary']
    info = summary['weight_interpolation']
    assert info['alpha'] == entry['alpha']
    assert info['base']['sha256'] == manifest['provenance']['base']['sha256']
    assert info['candidate']['sha256'] == manifest['provenance']['candidate']['sha256']
    rows.append({
        'tag': tag, 'alpha': entry['alpha'], 'checkpoint': entry['filename'],
        'val_miou_pct': 100 * summary['mean_iou'],
        'val_global_iou_pct': 100 * summary['global_iou'],
        **{f'{name}_miou_pct': 100 * diagnostic['by_gt_area'][group]['mean_iou']
           for name, group in [('small', 'small_lt_1pct'),
                               ('medium', 'medium_1_to_10pct'),
                               ('large', 'large_ge_10pct')]},
        'large_recall_pct': 100 * diagnostic['by_gt_area']['large_ge_10pct']['mean_recall'],
        'test_miou_pct': None,
    })
rows.sort(key=lambda row: row['alpha'])
best = max(rows, key=lambda row: (row['val_miou_pct'], -row['alpha']))
base = next(row for row in rows if row['alpha'] == 0)
for row in rows:
    for key in ('val_miou_pct', 'small_miou_pct', 'medium_miou_pct', 'large_miou_pct'):
        row[key + '_delta_vs_B5_pp'] = row[key] - base[key]
selection = {
    'criterion': 'validation_mean_per_image_iou',
    'best_tag': best['tag'], 'best_alpha': best['alpha'],
    'best_checkpoint': str(root / best['checkpoint']),
    'test_tags': [row['tag'] for row in rows
                  if row['alpha'] in (0, 1) or row['tag'] == best['tag']],
    'all_weights_retained': True,
}
(root / 'selection.json').write_text(json.dumps(selection, indent=2))
with (root / 'interpolation_comparison.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print((root / 'interpolation_comparison.csv').read_text())
print('Validation-selected model:', selection['best_checkpoint'])
```

The endpoint scores should reproduce B5 and low_crop10 under this evaluation
setup. Supplied seed42 validation references: B5 66.1693%, low_crop10 65.3050%.
Small/medium/large: B5 19.5605/47.8303/79.4138%; low_crop21.3001/49.1806/77.2213%.
Tiny GPU/batch/version rounding differences near threshold are possible; a material
endpoint mismatch requires investigation before interpreting interior coefficients.

## Cell 5 — Test the two endpoints and validation-selected model

At most three test evaluations; if an endpoint wins, only the two endpoints run.
This saves evaluation time and keeps test out of alpha selection. To evaluate
ALL retained files on test instead, replace the tag source with all manifest
checkpoint keys. Keep selection.json unchanged.

```bash
%%bash
set -euo pipefail
if [[ "$EVAL_TEST" == '1' ]]; then
    mapfile -t tags < <(uv run --frozen python -c 'import json,os;from pathlib import Path;print("\n".join(json.loads((Path(os.environ["RUN_ROOT"])/"selection.json").read_text())["test_tags"]))')
    for tag in "${tags[@]}"; do
        uv run --frozen python eval.py \
            --checkpoint "$RUN_ROOT/$tag.pt" --dataset test --data-root "$DATA_ROOT" \
            --device cuda --batch-size "$EVAL_BATCH" --num-workers 2 \
            --output-dir "$RUN_ROOT/$tag/eval_test"
    done
fi
```

```python
root = Path(os.environ['RUN_ROOT'])
for row in rows:
    path = root / row['tag'] / 'eval_test/metrics.json'
    if path.is_file():
        row['test_miou_pct'] = 100 * json.loads(path.read_text())['summary']['mean_iou']
with (root / 'interpolation_comparison.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print((root / 'interpolation_comparison.csv').read_text())
```

Missing test entries mean not evaluated, not zero accuracy. Do not change the
chosen alpha using these test scores. A validation improvement from several
coefficients is exploratory and needs additional seeds before a strong claim.

## Cell 6 — ZIP results, keep every weights file separately

```python
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(os.environ['RUN_ROOT'])
archive = Path('/kaggle/working') / f"{os.environ['RUN_NAME']}.zip"
with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() not in {'.pt', '.tmp', '.zip'}:
            bundle.write(path, path.relative_to(root))
print('Download results:', archive)
print('All retained weights-only models:')
for path in sorted(root.glob('alpha_*.pt')):
    print(path, f'{path.stat().st_size / 1024**3:.2f} GiB')
```

The ZIP includes all validation masks, diagnostics, tested masks, source
provenance, disk plan, selection and comparison. It excludes .pt/.tmp/nested.zip.
Every model remains in outputs for Kaggle output retention/download; the script
never removes losing models. Do not zip the weights too and duplicate their disk
usage. No best-model duplicate is saved: selection.json points to its alpha file.

## Compatibility and limits

Every alpha file works with existing eval.py and GroundingModel.from_checkpoint.
They contain model_state_dict and architecture/provenance metadata, with NO
optimizer_state_dict, scaler_state_dict, training_controls, support_state or
training epoch. They cannot resume the prior Adam trajectory. A new fine-tuning
experiment would need its own initialization/optimizer protocol.

Frozen encoders/buffers are compared tensor-by-tensor before exporting; only the
recorded compatible FP32 body parameters are interpolated. Alpha0/1 copy exact
endpoint tensors instead of relying on floating-point arithmetic. Inputs remain
unchanged. Artifacts are atomically written with disk-space checks, and existing
outputs are rejected rather than overwritten. There is no multi-model inference
ensemble: each file is used as one normal pyramid model.

This branch does not change training recipes, architecture or losses. Only the
evaluation metadata adds weight_interpolation provenance. CPU arithmetic,
checkpoint and tiny end-to-end checks do not establish Kaggle runtime/VRAM fit
or segmentation improvements. No actual GPU training is required by this plan.