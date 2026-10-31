# ACIIDS revision controls: Kaggle guide

Branch: `exp/aciids-revision-controls` (from `exp/size-weighted-pyramid`). This branch adds two opt-in controls requested by the manuscript revision. Default behaviour, checkpoint keys and the published runs are unchanged.

| Option | Where | What it does |
|---|---|---|
| `--sampling uniform` | `train_group.py` | Size-agnostic control for the continuation stage. Same 1,083 steps per epoch, global batch 6, frozen encoders, Adam, clipping and evaluation as the stratified runs, but the 6 images of each step are drawn uniformly from the whole training set and averaged into one gradient. |
| `--detail-input zeros` | `train.py` (also `preflight.py`) | Detail-pathway ablation. Every QGDP-Net module is kept, but the five detail maps are zero. The 0.31M pyramid weights stay in the checkpoint, are never used and are excluded from optimisation. This removes the 768 px pathway's information; it does **not** isolate input resolution. |
| `--detail-input low_res` | `train.py` | Optional resolution-controlled ablation. The CNN sees the 336 px view upsampled to 768 px, so only image resolution changes. |

Helper scripts in `revision/`:
- `dump_checkpoint_config.py` prints a checkpoint's saved `run_config` on CPU.
- `cost.py` reports FLOPs and GPU latency.

Local checks (kept outside Git, like other tests in this repo): `tests/test_revision_controls.py`, which covers the new options with the tiny model, including a CPU preflight of both sampling modes.

Select checkpoints by validation only; test metrics are reported, never used for selection. Do not modify or delete existing checkpoints or result archives.

## Runs

| ID | Session | Command | Purpose |
|---|---|---|---|
| R1 | A | `train_group.py --sampling uniform --focus-alpha 0`, seeds 42, 43, 44 | Uniform continuation control |
| R2–R5 | A | `train_group.py --sampling balanced --focus-alpha {0, 0.25}`, seeds 43, 44 | Extra seeds for natural and mild-focus continuation |
| R6 | B | `train.py --detail-input zeros`, 20 epochs, seed 42 | Detail-pathway ablation |
| R6b (optional) | B | `train.py --detail-input low_res`, 20 epochs, seed 42 | Resolution-only ablation |
| R7 | C | `eval.py` on the published epoch-20 checkpoint | Test Global IoU and masks for the base model |
| R8 | C | `revision/cost.py` | FLOPs and T4 latency |
| G2 | C | `revision/dump_checkpoint_config.py` | Direct record of the base model's training configuration |

Each continuation run takes about as long as one run of the previous one-session A/B guide, so plan two or three runs per Session A. Run the seven continuation runs over several sessions by changing `RUNS` in Cell A2.

## Cell 1 (all sessions): setup

Enable Internet and two T4 GPUs. Attach the VizWiz dataset and the published 20-epoch QGDP-Net checkpoint (`vizwiz-question-guided-detail-pyramid/checkpoint_epoch20.pt`).

```python
!git clone --single-branch --branch exp/aciids-revision-controls https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen
```

```python
import hashlib, os
from pathlib import Path

os.environ['DATA_ROOT'] = '/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz'
os.environ['RUN_ROOT'] = 'outputs'
os.environ['EVAL_BATCH'] = '4'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

EXPECTED = '6117b074a9b5c93c53c222d74d23a08a9a761621f2534ef1bb01da531f9e99aa'  # parent of all continuation runs
matches = []
for path in Path('/kaggle/input').rglob('checkpoint_epoch20.pt'):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            digest.update(block)
    if digest.hexdigest() == EXPECTED:
        matches.append(path)
assert len(matches) == 1, f'Attach exactly one published epoch-20 checkpoint: {matches}'
os.environ['PARENT_CKPT'] = str(matches[0])
print('Parent checkpoint:', matches[0])
```

## Session A: continuation controls (R1–R5)

### Cell A2: choose this session's runs

Each entry is `name:sampling:alpha:seed`. Use a different subset in each session. Never reuse an output folder.

```python
os.environ['RUN_NAME'] = 'continuation_controls_session1'
os.environ['RUNS'] = ' '.join([
    'uniform_s42:uniform:0:42',
    'natural_s43:balanced:0:43',
    'mild_focus_s43:balanced:0.25:43',
])
# Remaining sessions, for example:
#   'uniform_s43:uniform:0:43 natural_s44:balanced:0:44 mild_focus_s44:balanced:0.25:44'
#   'uniform_s44:uniform:0:44'
```

### Cell A3: preflight, train 5 epochs, evaluate

```bash
%%bash
set -euo pipefail
for spec in $RUNS; do
    IFS=: read -r name sampling alpha seed <<< "$spec"
    run_dir="$RUN_ROOT/$name"
    if [[ -e "$run_dir" ]]; then echo "Refusing to reuse $run_dir"; exit 1; fi
    common=(
        --data-root "$DATA_ROOT" --init-checkpoint "$PARENT_CKPT"
        --sampling "$sampling" --focus-alpha "$alpha" --no-project-updates
        --batch-size 6 --gradient-accumulation-steps 1
        --lr 0.00001 --optimizer adam --weight-decay 0
        --lr-scheduler none --warmup-ratio 0
        --freeze-vision-encoder True --freeze-text-encoder True
        --gradient-checkpointing --seed "$seed"
    )
    uv run --frozen torchrun --standalone --nproc_per_node=2 preflight_group.py \
        "${common[@]}" --num-workers 0 --steps 3
    uv run --frozen torchrun --standalone --nproc_per_node=2 train_group.py \
        "${common[@]}" --output-dir "$run_dir" \
        --num-epochs 5 --num-workers 2 --save-every 5 --no-save-final-weights
    for split in val test; do
        uv run --frozen python eval.py \
            --checkpoint "$run_dir/checkpoint_epoch5.pt" --dataset "$split" \
            --data-root "$DATA_ROOT" --device cuda \
            --batch-size "$EVAL_BATCH" --num-workers 2 \
            --output-dir "$run_dir/eval_${split}_ep5"
    done
    uv run --frozen python diagnose.py \
        --predictions "$run_dir/eval_val_ep5" --data-root "$DATA_ROOT" \
        --output-dir "$run_dir/diagnostic_ep5"
done
```

The protocol is identical to the published `natural` and `mild_focus` runs: the same parent, 1,083 updates per epoch, global batch 6, Adam with lr $10^{-5}$, no weight decay, frozen encoders and no projection. Only `--sampling` and `--seed` vary. Uniform runs record `"sampling": "uniform-shuffled-control"` in `group_protocol`.

Then run Cell Z to download the results.

## Session B: detail-pathway ablation (R6, optional R6b)

### Cell B2: preflight and 20-epoch training

These are the published base-model settings (Adam, lr $10^{-5}$, no weight decay or schedule, global microbatch 2 × accumulation 2, seed 42), plus `--detail-input`.

```python
os.environ['RUN_NAME'] = 'detail_input_zeros_ep20_seed42'
os.environ['DETAIL_INPUT'] = 'zeros'  # or 'low_res' for the optional R6b
```

```bash
%%bash
set -euo pipefail
run_dir="$RUN_ROOT/detail_$DETAIL_INPUT"
uv run --frozen torchrun --standalone --nproc_per_node=2 preflight.py \
    --data-root "$DATA_ROOT" --batch-size 2 --num-workers 0 --steps 3 \
    --gradient-checkpointing --detail-input "$DETAIL_INPUT"
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py \
    --data-root "$DATA_ROOT" --output-dir "$run_dir" \
    --num-epochs 20 --batch-size 2 --gradient-accumulation-steps 2 \
    --gradient-checkpointing --num-workers 2 --lr 0.00001 --seed 42 \
    --optimizer adam --weight-decay 0 --lr-scheduler none --warmup-ratio 0 \
    --save-every 10 --detail-input "$DETAIL_INPUT"
```

If the session ends early, resume in a new session from the last full checkpoint with the **same** flags, replacing `--output-dir` by the same folder and adding `--resume-checkpoint "$run_dir/checkpoint_epoch10.pt"` (`--num-epochs` stays 20).

### Cell B3: evaluate epochs 10 and 20

```bash
%%bash
set -euo pipefail
run_dir="$RUN_ROOT/detail_$DETAIL_INPUT"
for epoch in 10 20; do
    for split in val test; do
        uv run --frozen python eval.py \
            --checkpoint "$run_dir/checkpoint_epoch${epoch}.pt" --dataset "$split" \
            --data-root "$DATA_ROOT" --device cuda \
            --batch-size "$EVAL_BATCH" --num-workers 2 \
            --output-dir "$run_dir/eval_${split}_ep${epoch}"
    done
    uv run --frozen python diagnose.py \
        --predictions "$run_dir/eval_val_ep${epoch}" --data-root "$DATA_ROOT" \
        --output-dir "$run_dir/diagnostic_ep${epoch}"
done
```

Then run Cell Z.

## Session C: base-model records and cost (R7, R8, G2)

```python
os.environ['RUN_NAME'] = 'base_records_and_cost'
```

```bash
%%bash
set -euo pipefail
out="$RUN_ROOT/base_records"
mkdir -p "$out"
uv run --frozen python revision/dump_checkpoint_config.py "$PARENT_CKPT" > "$out/parent_checkpoint_config.json"
for split in val test; do
    uv run --frozen python eval.py \
        --checkpoint "$PARENT_CKPT" --dataset "$split" \
        --data-root "$DATA_ROOT" --device cuda \
        --batch-size "$EVAL_BATCH" --num-workers 2 \
        --output-dir "$out/eval_${split}_ep20"
done
for mode in image zeros; do
    uv run --frozen python revision/cost.py "$out/cost_${mode}.json" --detail-input "$mode" --device cuda
done
```

`parent_checkpoint_config.json` is the direct record of the original 20-epoch training configuration (optimizer, learning rate, batch size). The `eval_*_ep20/metrics.json` files add the test Global IoU, which the original archive lacks. Then run Cell Z.

## Cell Z: ZIP results, excluding checkpoints

```python
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(os.environ['RUN_ROOT'])
archive = Path('/kaggle/working') / f"{os.environ['RUN_NAME']}.zip"
with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() not in {'.pt', '.tmp', '.zip'}:
            bundle.write(path, path.relative_to(root))
print('Download results:', archive)
```

Keep checkpoints separately if you may resume. Return the ZIPs; they are analysed read-only.
