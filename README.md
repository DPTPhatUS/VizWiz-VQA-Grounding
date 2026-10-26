# Parent-preserving pyramid: C and D in one Kaggle session

Branch: `exp/parent-preserving-pyramid`, branched from `exp/size-weighted-pyramid` at `3cbe644`. This workflow runs experiments C (`foreground`) and D (`parameters`) sequentially in a single Kaggle session using `train_group.py` and `preflight_group.py`. Both runs use run B's mild-focus weighting (`focus_alpha = 0.25`), initialize from the same 20-epoch parent pyramid checkpoint, keep the vision and text encoders frozen, and train with BCE + Dice and Adam without gradient projection. Each run initializes independently from the 20-epoch parent rather than from run B or from one another.

## Two different preservation hypotheses

| Run folder | Mode | Strength | Additional training-only state |
|---|---|---:|---|
| `foreground` (C) | `foreground` | 1.0 | Frozen copy of the post-encoder body |
| `parameters` (D) | `parameters` | 0.01 | Initial values of trainable parameters |

Both runs use group weights `w = (1 - alpha) * p + alpha * (0.5, 0.5, 0)` with `alpha = 0.25`, where `p` is the training split frequency across `<1%`, `1-10%`, and `>=10%` ground-truth foreground masks (249 / 1,676 / 4,569 images, giving weights of approximately 15.38% / 31.86% / 52.77%). Balanced group sampling and data ordering match run B.

### C: retain correct parent foreground on large-GT images

For each training image with `>=10%` foreground on the resized training mask, let `K` be the set of ground-truth foreground pixels where the frozen parent's logit is positive (`> 0`). Student probabilities `p` are penalized only when they drop below the parent's probabilities `t`:

    keep(image) = mean over K of relu(t-p)^2
    L_C = w_small*L_small + w_medium*L_medium
          + w_large*(L_large + 1.0*keep)

If `K` is empty, `keep(image)` is zero. The penalty is averaged over `K` per image before averaging across images in the batch. Teacher false positives, teacher false negatives, and images in the smaller ground-truth size groups are not penalized, and the student is free to increase confidence above the parent. Standard ground-truth segmentation supervision remains active at all final and auxiliary heads.

During training in mode C, the frozen CLIP vision and text encoders run once per image. Their outputs are shared with a frozen copy of the parent's post-encoder modules (CNN pyramid, fusion, decoder, and refiner, totaling about 6.65M parameters). Because the student's CNN pyramid updates during training, the frozen parent retains its own separate CNN pyramid branch. The frozen parent copy requires no gradients and is excluded from the student's `model_state_dict`, DDP wrapper, and optimizer state.

### D: L2-SP anchor around the parent

Mode D stores the initial parent values `theta0` for all trainable student parameters and adds a quadratic anchor penalty around `theta0`:

    anchor = 0.5 * sum over trainable parameters of (theta-theta0)^2
    L_D = L_B + 0.01*anchor

This term sums squared parameter differences across trainable tensors (rather than taking a per-parameter mean). Its exact gradient `0.01 * (theta - theta0)` is added once per optimizer step prior to gradient clipping and the Adam update, with proper AMP unscaling, independent of the number of size groups, accumulation steps, or GPUs. Mode D does not construct a second model copy or run a parent forward pass.

Note that C and D operate in different units (probability-space MSE vs. parameter-space L2 sum), so their strength coefficients (`1.0` and `0.01`) have different scales. Background references: [Learning without Forgetting](https://arxiv.org/abs/1606.09282) and [L2-SP](https://proceedings.mlr.press/v80/li18a.html).

At inference time, both C and D export a single standard student pyramid with unchanged state-dict keys, architecture, and preprocessing. All training-only reference states are ignored during evaluation.

## Cell 1: Fresh session and environment

Enable Internet and two GPUs (such as 2x T4). Attach both the VizWiz dataset and the original 20-epoch pyramid checkpoint dataset under `/kaggle/input`.

```python
!git clone --single-branch --branch exp/parent-preserving-pyramid https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen
```

If the repository is already cloned on this branch, run `!git pull --ff-only` followed by `!uv sync --frozen`.

## Cell 2: Paths and C/D configs

```python
import json
import os
from pathlib import Path

os.environ['DATA_ROOT'] = '/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz'
os.environ['PYRAMID_CKPT'] = '/kaggle/input/datasets/khoahoang0512/vizwiz-question-guided-detail-pyramid/checkpoint_epoch20.pt'
os.environ['RUN_ROOT'] = 'outputs'
os.environ['RUN_NAME'] = 'parent_preserving_pyramid_CD_ep5_seed42'
os.environ['EVAL_BATCH'] = '4'
os.environ['EVAL_TEST'] = '1'  # Set0 to skip test evaluation if local GT is unavailable.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

parent = Path(os.environ['PYRAMID_CKPT'])
if not parent.is_file():
    print('Available checkpoints:')
    for path in sorted(Path('/kaggle/input').rglob('*.pt')):
        print(path)
    raise FileNotFoundError(f'Set PYRAMID_CKPT to the original pyramid epoch20: {parent}')
root = Path(os.environ['RUN_ROOT'])
for run in ('foreground', 'parameters'):
    folder = root / run
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f'Use a fresh run folder: {folder}')
(root / 'configs').mkdir(parents=True, exist_ok=True)
common = {
    'overlap_loss': 'dice',
    'overlap_weight': 0.5,
    'aux96_weight': 0.1,
    'aux192_weight': 0.2,
    'aux384_weight': 0.3,
    'project_updates': False,
    'focus_alpha': 0.25,
}
for run, strength in (('foreground', 1.0), ('parameters', 0.01)):
    config = {**common, 'preservation_mode': run, 'preservation_strength': strength}
    path = root / 'configs' / f'{run}.json'
    path.write_text(json.dumps(config, indent=2))
    print(path, config)
```

Precedence is JSON config > explicit CLI flags > defaults. Both configs keep `focus_alpha = 0.25` and differ only in `preservation_mode` and `preservation_strength`. Ensure `PYRAMID_CKPT` points to the original 20-epoch `question-guided-detail-pyramid` checkpoint; each run initializes its student weights from this parent with a fresh optimizer, AMP scaler, and epoch counter.

## Cell 3: C then D: preflight, train, evaluate and diagnostic

This cell runs `foreground` (C) followed by `parameters` (D) in separate subprocesses, releasing GPU memory between commands. Training uses two GPUs with a global batch size of 6 (1 image per size group per GPU), and evaluation runs on a single GPU.

```bash
%%bash
set -euo pipefail
for run in foreground parameters; do
    config="$RUN_ROOT/configs/${run}.json"
    run_dir="$RUN_ROOT/$run"
    common=(
        --data-root "$DATA_ROOT"
        --init-checkpoint "$PYRAMID_CKPT"
        --config "$config"
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
        --num-epochs 5 --num-workers 2 \
        --save-every 5 --no-save-final-weights
    uv run --frozen python eval.py \
        --checkpoint "$run_dir/checkpoint_epoch5.pt" \
        --dataset val --data-root "$DATA_ROOT" --device cuda \
        --batch-size "$EVAL_BATCH" --num-workers 2 \
        --output-dir "$run_dir/eval_val_ep5"
    uv run --frozen python diagnose.py \
        --predictions "$run_dir/eval_val_ep5" --data-root "$DATA_ROOT" \
        --output-dir "$run_dir/diagnostic_ep5"
    if [[ "$EVAL_TEST" == '1' ]]; then
        uv run --frozen python eval.py \
            --checkpoint "$run_dir/checkpoint_epoch5.pt" \
            --dataset test --data-root "$DATA_ROOT" --device cuda \
            --batch-size "$EVAL_BATCH" --num-workers 2 \
            --output-dir "$run_dir/eval_test_ep5"
    fi
done
```

Preflight must pass on both ranks; its updates are discarded before `train_group.py` re-initializes from `PYRAMID_CKPT`. Training runs under AMP while evaluation runs in FP32 at native resolution (set `EVAL_BATCH = '1'` if batch size 4 exceeds VRAM). In mode C, the reference branch reuses the training-autocast encoder features and computes the probability penalty in FP32.

Each run trains for 5 continuation epochs after the 20-epoch parent (25 total training epochs) and writes a single checkpoint:

- `outputs/foreground/checkpoint_epoch5.pt`
- `outputs/parameters/checkpoint_epoch5.pt`

Each full checkpoint also stores `preservation_state` (the post-encoder reference tensors for C or the initial trainable parameter anchors for D) alongside the model and frozen backbones, using the same pre-write disk-space check and coordinated CPU-group error handling as the parent branch.

## Cell 4: Compare validation and verify experiment identity

```python
import csv

root = Path(os.environ['RUN_ROOT'])
rows = []
protocols = []
def percent(value):
    return round(100 * value, 4) if value is not None else None

for run in ('foreground', 'parameters'):
    folder = root / run
    metrics = json.loads((folder / 'eval_val_ep5/metrics.json').read_text())
    diagnostic = json.loads((folder / 'diagnostic_ep5/summary.json').read_text())
    summary = metrics['summary']
    protocol = summary['training_protocol']
    assert summary['objective_version'] == 'parent-preserving-updates-v1'
    assert protocol['projection'] is False
    assert protocol['focus_alpha'] == 0.25
    preservation = protocol['preservation']
    assert preservation['mode'] == run
    expected_strength = 1.0 if run == 'foreground' else 0.01
    assert preservation['strength'] == expected_strength
    protocols.append(protocol)
    weights = protocol['proposal_weights']
    test_path = folder / 'eval_test_ep5/metrics.json'
    test = json.loads(test_path.read_text())['summary'] if test_path.is_file() else {}
    rows.append({
        'run': run,
        'focus_alpha': protocol['focus_alpha'],
        'preservation_mode': preservation['mode'],
        'preservation_strength': preservation['strength'],
        'reference_parameters_training_only': preservation['training_only_reference_parameters'],
        'small_weight': weights[0],
        'medium_weight': weights[1],
        'large_weight': weights[2],
        'val_miou_pct': percent(summary['mean_iou']),
        'val_global_iou_pct': percent(summary['global_iou']),
        'small_miou_pct': percent(diagnostic['by_gt_area']['small_lt_1pct']['mean_iou']),
        'medium_miou_pct': percent(diagnostic['by_gt_area']['medium_1_to_10pct']['mean_iou']),
        'large_miou_pct': percent(diagnostic['by_gt_area']['large_ge_10pct']['mean_iou']),
        'large_recall_pct': percent(diagnostic['by_gt_area']['large_ge_10pct']['mean_recall']),
        'small_precision_pct': percent(diagnostic['by_gt_area']['small_lt_1pct']['mean_precision']),
        'medium_precision_pct': percent(diagnostic['by_gt_area']['medium_1_to_10pct']['mean_precision']),
        'test_miou_pct': percent(test.get('mean_iou')),
    })
for key in ('initial_checkpoint', 'mask_group_sha256', 'group_counts', 'updates_per_epoch', 'proposal_weights'):
    assert protocols[0][key] == protocols[1][key], f'Run mismatch: {key}'
comparison = root / 'two_run_comparison.csv'
with comparison.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(comparison.read_text())
```

Compare C and D against run B (`mild_focus` at +5 from `exp/size-weighted-pyramid`), which shares the exact parent, frozen encoders, balanced sampler, group weights, and optimizer without a preservation term (whereas the earlier `scale-constrained-pyramid` control used 50/50/0 group weights). Use validation metrics to compare runs, and inspect large-region recall alongside small and medium IoU and precision.

## Cell 5: ZIP both result folders, excluding all checkpoints

```python
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(os.environ['RUN_ROOT'])
archive = Path('/kaggle/working') / f"{os.environ['RUN_NAME']}.zip"
with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() not in {'.pt', '.tmp', '.zip'}:
            bundle.write(path, path.relative_to(root))
print('Download results:', archive)
print('Keep the two full checkpoint files separately if needed.')
```

The ZIP archive bundles the configuration files, training histories, resolved weights and provenance in `metrics.json`, validation masks, CPU diagnostics, optional test masks, and `two_run_comparison.csv`, excluding `.pt`, `.tmp`, and nested `.zip` files.

## Logs, references and compatibility

The preservation controls are exposed as `preservation_mode` (`--preservation-mode`) and `preservation_strength` (`--preservation-strength`). Supported modes are `none`, `foreground`, and `parameters`. By default, `preservation_mode` is `foreground` with strength `1.0` (`parameters` defaults to `0.01`, and `none` defaults to `0`). Setting strength to `0` disables reference allocation and reproduces the unregularized run B update path. Both `foreground` and `parameters` modes require frozen encoders and `--no-project-updates`.

Key fields logged in `history.jsonl`:

- `loss_small`, `loss_medium`, `loss_large`: per-group segmentation losses (including auxiliary heads).
- `segmentation_objective`: group-weighted segmentation loss.
- `train_loss` / `proposal_loss`: total objective including the weighted preservation term.
- `preservation_loss`: raw per-image mean `keep` loss on large images in mode C, or `0.5 * sum((theta - theta0)^2)` in mode D.
- `preservation_weighted_loss`: scaled contribution added to `segmentation_objective`.
- `protected_foreground_fraction`: fraction of large-image ground-truth foreground pixels selected by the parent mask in mode C.
- `preservation_active_fraction`: fraction of large images with non-empty `K` in mode C.
- `parameter_distance_l2` and `preservation_gradient_ratio`: L2 distance from initial parent parameters and the ratio of anchor gradient norm to segmentation gradient norm before clipping in mode D.

Full checkpoints store `preservation_state` separately from `model_state_dict` so resuming a completed C or D checkpoint restores the original parent reference rather than re-anchoring to the student, even if the original 20-epoch parent file is no longer mounted. Standard evaluation (`eval.py`) loads `model_state_dict` only and ignores `preservation_state`.

If `foreground` (C) finishes and `parameters` (D) fails, keep `outputs/foreground`, clear any partial files in `outputs/parameters`, and rerun Cell 3 with `for run in parameters; do`. To extend a completed C or D run to more epochs later, replace `--init-checkpoint` with `--resume-checkpoint` pointing to that run's `checkpoint_epoch5.pt`, keep its config unchanged, and set `--num-epochs` to the new total continuation epoch target.
