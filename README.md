# Pyramid weight interpolation: export all models, then evaluate (no training)

Branch: `exp/pyramid-weight-interpolation`. This workflow runs in a single Kaggle evaluation session without any training. It interpolates the trainable post-encoder weights between the full `mild_focus` epoch-5 checkpoint (`B5`) and the full `low_crop` epoch-10 checkpoint (`low_crop10`):

    body(alpha) = (1-alpha)*body(B5) + alpha*body(low_crop10)

The default grid `alpha in {0, 0.1, 0.25, 0.5, 1}` exports five standalone weights-only `.pt` files (including both exact endpoints). Optimizer state, AMP scaler state, frozen reference bodies, and epoch counters are omitted so each file is a standard single-model inference checkpoint with the same parameter count as the base pyramid. Frozen encoders and all non-trainable buffers must match bit-for-bit between the two source checkpoints and are copied directly rather than averaged. Each exported file and `interpolation.json` record the SHA256 hashes and provenance of both source checkpoints.

Background reference on weight-space averaging across fine-tuned checkpoints: [Model Soups](https://proceedings.mlr.press/v162/wortsman22a.html). Here we test whether interpolating between `B5` and `low_crop10` retains `low_crop10`'s small and medium target gains while staying closer to `B5` on large regions.

## Storage and checkpoint policy

All five models in the grid are retained on disk (`alpha_0.pt`, `alpha_0p1.pt`, `alpha_0p25.pt`, `alpha_0p5.pt`, and `alpha_1.pt`). Because each weights-only checkpoint still contains both frozen CLIP encoders, `interpolate.py` checks actual tensor storage and available disk space after `uv sync` against the lower of physical free space and a `14 GiB` budget cap (~15 GB), reserving `2 GiB` for predicted masks, metrics, the results ZIP, and serialization overhead. It prints the per-file size and maximum supported file count and aborts before writing any `.pt` files if the requested grid exceeds the budget.

To include `0.75` for a six-point grid if disk space permits, add `0.75` to `ALPHAS` in Cell 2 (the `0` and `1` endpoints are always included automatically). Source checkpoints stay mounted under `/kaggle/input` and are not copied into `outputs/`.

## Cell 1: Setup

Attach the VizWiz dataset, the full `B5` checkpoint, and the full `low_crop/checkpoint_epoch10.pt` checkpoint from Session 1 of `exp/crop-recall-preservation`. Enable Internet and a single GPU (`interpolate.py` runs on CPU, and `eval.py` uses one GPU without `torchrun`).

```python
!git clone --single-branch --branch exp/pyramid-weight-interpolation https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen
```

If already cloned on this branch, run `!git pull --ff-only` followed by `!uv sync --frozen`.

## Cell 2: Paths and the predeclared alpha grid

The script auto-detects `low_crop/checkpoint_epoch10.pt` under `/kaggle/input`; if your dataset uses a different folder name, set `LOW_CROP_CKPT` explicitly.

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

Before exporting, `interpolate.py` verifies that both checkpoints share the same model configuration, `B5` lineage hash, original 20-epoch parent and mask-group digests, trainable parameter layout, encoder-freezing policy, tensor shapes/dtypes, and bit-identical frozen encoder weights, then blends only the finite FP32 post-encoder parameters.

## Cell 3: Export all weights-only models, then full validation/diagnostics

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

By default, this exports `alpha_0.pt`, `alpha_0p1.pt`, `alpha_0p25.pt`, `alpha_0p5.pt`, and `alpha_1.pt` and evaluates each checkpoint on the validation split in FP32 at native resolution. Set `EVAL_BATCH = '1'` if batch size 4 exceeds GPU memory. Because all `.pt` files are exported before the evaluation loop begins, if an evaluation command is interrupted you can run the remaining `eval.py` / `diagnose.py` commands directly without re-running `interpolate.py`.

## Cell 4: Compare and select using validation only

This script ranks the grid by validation per-image mean IoU (breaking ties in favor of smaller `alpha`, closer to `B5`) and writes `selection.json` and `interpolation_comparison.csv`. Also inspect the small, medium, and large group breakdowns before choosing a model.

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

Verify that the `alpha = 0` and `alpha = 1` endpoints match the standalone `B5` and `low_crop10` validation numbers:

- `B5` (`alpha = 0`): 66.1693% val mean IoU (19.5605% small, 47.8303% medium, 79.4138% large)
- `low_crop10` (`alpha = 1`): 65.3050% val mean IoU (21.3001% small, 49.1806% medium, 77.2213% large)

## Cell 5: Test the two endpoints and validation-selected model

To save runtime while keeping test data out of `alpha` selection, this cell evaluates at most three checkpoints on the test split: the two endpoints (`alpha_0` and `alpha_1`) and the validation-selected `best_tag` (two runs total if an endpoint wins).

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

Untested interior `alpha` rows leave `test_miou_pct` empty (`None`). Keep `selection.json` fixed to the validation choice.

## Cell 6: ZIP results, keep every weights file separately

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

The ZIP archive bundles all validation masks, diagnostics, evaluated test masks, `interpolation.json`, `selection.json`, and `interpolation_comparison.csv`, while excluding `.pt`, `.tmp`, and nested `.zip` files. All `alpha_*.pt` checkpoints remain in `outputs/` for direct download.

## Compatibility

Every exported `alpha_*.pt` checkpoint loads directly via `eval.py` and `GroundingModel.from_checkpoint`. Because these are weights-only inference checkpoints (containing `model_state_dict` and provenance metadata without `optimizer_state_dict`, `scaler_state_dict`, `training_controls`, or `support_state`), they cannot be passed to `--resume-checkpoint` to continue a prior Adam trajectory. Endpoints (`alpha = 0` and `alpha = 1`) copy the exact source tensors without floating-point rounding.
