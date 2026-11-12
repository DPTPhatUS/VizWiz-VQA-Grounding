# ACIIDS revision controls: Kaggle guide

Branch: `exp/aciids-revision-controls` (from `exp/size-weighted-pyramid`). This branch adds two opt-in controls requested by the manuscript revision. Default behaviour, checkpoint keys and the published runs are unchanged.

| Option | Where | What it does |
|---|---|---|
| `--sampling uniform` | `train_group.py` | Size-agnostic control for the continuation stage. Same 1,083 steps per epoch, global batch 6, frozen encoders, Adam, clipping and evaluation as the stratified runs, but the 6 images of each step are drawn uniformly from the whole training set and averaged into one gradient. |
| `--detail-input zeros` | `train.py` (also `preflight.py`) | Detail-pathway ablation. Every QGDP-Net module is kept, but the five detail maps are zero. The 0.31M pyramid weights stay in the checkpoint, are never used and are excluded from optimisation. This removes the 768 px pathway's information; it does **not** isolate input resolution. |
| `--detail-input low_res` | `train.py` | Optional resolution-controlled ablation. The CNN sees the 336 px view upsampled to 768 px, so only image resolution changes. |
| `--refiner-mode none` | `train.py`, `preflight.py` | Decoder ablation (Session I). No 768 px refiner: the output is the 384 px decoder logits upsampled bilinearly, which is the refiner's own starting point. The refiner weights stay in the checkpoint, unused and untrained. |
| `--no-skip-film` | `train.py`, `preflight.py` | Decoder ablation (Session I). The CLIP skips from blocks 4, 7 and 9 are used without question modulation. The FiLM layers stay in the checkpoint, unused and untrained. The question still enters through cross-attention, the gates and the refiner. |
| `--expect-overlap-loss dice` | `train.py` | Guard: stop before training unless the resolved main loss is the given one. |
| `--val-every N` | `train.py` | Also log validation mIoU every N epochs to `val_curve.jsonl`, sharded over both GPUs (same metric as `eval.py`). Logging only: no extra checkpoint, no effect on training. |

**Default main loss is Dice since 2026-11-01.** That is the published QGDP-Net objective. Sessions B, E and F ran with the earlier Lovász default (see Session F).

Helper scripts in `revision/`:
- `dump_checkpoint_config.py` prints a checkpoint's saved `run_config` on CPU.
- `cost.py` reports FLOPs and GPU latency.

Local checks (kept outside Git, like other tests in this repo):
- `tests/test_revision_controls.py` covers the new options with the tiny model, including a CPU preflight of both sampling modes.
- `tests/test_session_g.py` covers the Dice default, the loss guard, the two decoder ablations (checkpoint round trip, frozen unused weights, resume), and validation logging. For the logging it checks that weights are bit-identical with and without it, and that the logged mIoU equals `eval.py`'s on the same checkpoint.

A two-process CPU `torchrun` run (gloo backend) also confirmed the sharded validation: the two shards cover the split without overlap and match a single-process evaluation exactly.

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
| A2 | D | Single-scale baseline commit `2079c8a`, `--training-recipe grounding-only` + `"question_only": true`, 20 epochs, seed 42 | Single-scale baseline without the answer-alignment loss |
| A5 | D | `revision/question_sensitivity.py` on the published epoch-20 QGDP-Net | How much the predicted mask depends on the question |
| E (priority) | E | `train.py --detail-input {image,zeros}`, seeds 43 and 44, 20 epochs | Seed repeats for the detail-pathway conclusion |
| A6 | C or D (CPU) | `revision/mask_resolution_audit.py` on the training masks | Target information lost by resizing; grid IoU ceilings |
| F | — | superseded by G (its runs used Lovász) | — |
| G | G | `train.py`, Dice, `--detail-input {image, zeros}`, seeds 42–44 | Multi-seed headline and Dice-matched detail-pathway comparison |
| H | H | Baseline commit `2079c8a`, published recipe, seeds 43, 44 | Multi-seed single-scale baseline |
| I (optional) | I | `train.py --detail-input zeros` with `--refiner-mode none`, `--no-skip-film` or zero auxiliary weights, seeds 42, 43 | Which decoder components matter |
| J | J | QGDP-Net (Session G cells) and the baseline (Session H cells), seeds 45, 46 | Two more seeds for the headline comparison |
| L1 | L | Baseline commit `2079c8a`, question-only recipe (as A2), seeds 43, 44 | Answer-alignment loss on three seeds; cleanest single-scale baseline |
| L2 | L | `train.py --freeze-vision-encoder --freeze-text-encoder`, seeds 43, 44 | Encoder adaptation on three seeds |
| L3 | L | `clipseg_baseline.py` (CLIPSeg, frozen CLIP, trained decoder), learning rate chosen on seed 42, then seeds 43, 44 | External baseline |
| P | P | `revision/detail_intervention.py` on the selected checkpoint of each trained full QGDP-Net (inference only) | Does the trained model use its detail pathway? |

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

> **Loss caveat (found 2026-11-01).** On this branch `losses.py` defaults to `overlap_loss: "lovasz"` (inherited from `exp/per-image-lovasz`). The published QGDP-Net used Dice. Cells B2 and E2 below pass no `--config`, so the Session B and E runs used the Lovász main loss. They are matched with each other, not with the published model. Session F gives Dice-matched commands.

These are the published base-model optimizer settings (Adam, lr $10^{-5}$, no weight decay or schedule, global microbatch 2 × accumulation 2, seed 42), plus `--detail-input`. The main overlap loss is **not** matched; see the caveat above.

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

## Session D: strict single-scale baseline (A2), question sensitivity (A5), mask audit (A6)

**Why this session uses two copies of the repository**

| Folder | Code | Used for |
|---|---|---|
| `/kaggle/working/VizWiz-VQA-Grounding` (from Cell 1) | this branch | A5, A6 and the ZIP (Cell Z) |
| `/kaggle/working/baseline_code` (Cell D1) | old commit `2079c8a` | A2 training and evaluation |

The single-scale baseline in the paper was trained with commit `2079c8a`. That model is not in the current branch, so A2 checks out that exact commit in a separate folder and changes only one thing: it trains in the commit's built-in **question-only** mode, with no answer-alignment loss.

A2 is still not architecture-matched to QGDP-Net. Its decoder is different, and it has no 768 px view, detail pyramid, auxiliary heads or refiner.

Run Cell 1 first (setup, `DATA_ROOT`, `PARENT_CKPT`), then the cells below.

### Cell D1: get the baseline code

```bash
%%bash
set -euo pipefail
cd /kaggle/working
rm -rf baseline_code
git clone https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git baseline_code
cd baseline_code
git checkout --detach 2079c8a   # commit that trained the published single-scale baseline
uv sync --frozen
```

### Cell D2: train A2 (20 epochs, same recipe as the baseline)

```bash
%%bash
set -euo pipefail
cd /kaggle/working/baseline_code
echo '{"freeze_text_encoder": true, "question_only": true, "dice_weight": 0.5}' > a2_config.json
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py \
    --data-root "$DATA_ROOT" --num-epochs 20 --batch-size 4 --num-workers 2 \
    --lr 0.00001 --seed 42 --save-every 10 \
    --training-recipe grounding-only --config a2_config.json \
    --output-dir /kaggle/working/VizWiz-VQA-Grounding/outputs/a2_single_scale_question_only
```

The output goes into the main repository's `outputs/` folder, so Cell Z includes it. If the session stops early, rerun the same command with `--resume-checkpoint <output-dir>/checkpoint_epoch10.pt` added.

### Cell D3: evaluate A2 (still with the baseline code)

```bash
%%bash
set -euo pipefail
cd /kaggle/working/baseline_code
run_dir=/kaggle/working/VizWiz-VQA-Grounding/outputs/a2_single_scale_question_only
for epoch in 10 20; do
    for split in val test; do
        uv run --frozen python eval.py \
            --checkpoint "$run_dir/checkpoint_epoch${epoch}.pt" --dataset "$split" \
            --data-root "$DATA_ROOT" --device cuda --batch-size 4 --num-workers 2 \
            --output-dir "$run_dir/eval_${split}_ep${epoch}"
    done
done
```

Each `metrics.json` should say `"grounding_mode": "question"`.

### Cell D4: A5 and A6 (back in the main repository)

```bash
%%bash
set -euo pipefail
cd /kaggle/working/VizWiz-VQA-Grounding
for split in val test; do
    uv run --frozen python revision/question_sensitivity.py \
        --checkpoint "$PARENT_CKPT" --data-root "$DATA_ROOT" --dataset "$split" \
        --output-dir "$RUN_ROOT/a5_question_sensitivity_${split}" \
        --batch-size 4 --device cuda --seed 2027
done
uv run --frozen python revision/mask_resolution_audit.py \
    "$DATA_ROOT/binary_masks_png/train" "$RUN_ROOT/a6_mask_audit_train.json"
```

- **A5** keeps each image fixed and changes only the question (the same question again, a random donor question, a size-matched donor question, "What is this?", an empty question). It measures how much the predicted mask changes; it is a sensitivity measure, not an accuracy measure.
- **A6** is CPU-only: it checks how much target detail survives mask resizing on the training split.

### Cell D5: name the ZIP, then run Cell Z

```python
os.environ['RUN_NAME'] = 'session_d_a2_a5_a6'
```

## Session E: seed repeats of QGDP-Net vs. zeroed detail pathway (priority)

> **Loss caveat (found 2026-11-01).** On this branch `losses.py` defaults to `overlap_loss: "lovasz"` (inherited from `exp/per-image-lovasz`). The published QGDP-Net used Dice. Cells B2 and E2 pass no `--config`, so the Session B and E runs used the Lovász main loss. They are matched with each other, not with the published model. Session F gives Dice-matched commands.


**Why.** With seed 42, zeroing the detail maps did not lower accuracy (validation-selected: +0.15 test points, CI −0.65 to +0.95), but checkpoint-to-checkpoint noise is about 1.3 points. The paper's claim about the detail pathway needs repeated seeds, run as pairs that share everything except `--detail-input`.

| Order | `DETAIL_INPUT` | `SEED` | Output folder |
|---|---|---|---|
| 1 | `image` | 43 | `outputs/seed_image_s43` |
| 2 | `zeros` | 43 | `outputs/seed_zeros_s43` |
| 3 | `image` | 44 | `outputs/seed_image_s44` |
| 4 | `zeros` | 44 | `outputs/seed_zeros_s44` |

Run one row per session, in this order. Rows 1–2 are the minimum useful pair. The protocol is identical to the published 20-epoch model: same flags, checkpoints and evaluation at epochs 10 and 20, selection by validation only.

### Cell E1: choose the run

```python
os.environ['DETAIL_INPUT'] = 'image'   # 'image' or 'zeros'
os.environ['SEED'] = '43'              # '43' or '44'
os.environ['RUN_NAME'] = f"seed_{os.environ['DETAIL_INPUT']}_s{os.environ['SEED']}"
```

### Cell E2: train and evaluate

```bash
%%bash
set -euo pipefail
run_dir="$RUN_ROOT/$RUN_NAME"
if [[ -e "$run_dir" ]]; then echo "Refusing to reuse $run_dir"; exit 1; fi
uv run --frozen torchrun --standalone --nproc_per_node=2 preflight.py \
    --data-root "$DATA_ROOT" --batch-size 2 --num-workers 0 --steps 3 \
    --gradient-checkpointing --detail-input "$DETAIL_INPUT"
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py \
    --data-root "$DATA_ROOT" --output-dir "$run_dir" \
    --num-epochs 20 --batch-size 2 --gradient-accumulation-steps 2 \
    --gradient-checkpointing --num-workers 2 --lr 0.00001 --seed "$SEED" \
    --optimizer adam --weight-decay 0 --lr-scheduler none --warmup-ratio 0 \
    --save-every 10 --detail-input "$DETAIL_INPUT"
for epoch in 10 20; do
    for split in val test; do
        uv run --frozen python eval.py \
            --checkpoint "$run_dir/checkpoint_epoch${epoch}.pt" --dataset "$split" \
            --data-root "$DATA_ROOT" --device cuda --batch-size "$EVAL_BATCH" --num-workers 2 \
            --output-dir "$run_dir/eval_${split}_ep${epoch}"
    done
    uv run --frozen python diagnose.py \
        --predictions "$run_dir/eval_val_ep${epoch}" --data-root "$DATA_ROOT" \
        --output-dir "$run_dir/diagnostic_ep${epoch}"
done
```

If the session ends early, resume with the same command plus `--resume-checkpoint "$run_dir/checkpoint_epoch10.pt"` (same `--num-epochs 20`) and then run the evaluation loop. Then run Cell Z.

## Session F: superseded, do not run

Session F was meant to repeat the detail-pathway runs with the Dice loss. Its four runs (`dice_zeros_s42`, `dice_image_s43`, `dice_zeros_s43`, `dice_low_res_s42`) recorded `"overlap_loss": "lovasz"` in `history.jsonl`: the `--config` flag never reached `train.py`. They are kept as same-seed Lovász repeats. **Session G replaces Session F.** Three changes make the mistake impossible to repeat:

- Dice is now the default main loss of `train.py`. It is the published QGDP-Net objective: pixel-averaged BCE plus 0.5 × per-image Dice at 768 px, with Dice auxiliary terms at 96, 192 and 384 px, weights 0.1, 0.2 and 0.3.
- Every new cell passes `--expect-overlap-loss dice`. `train.py` stops **before training** if the resolved loss is anything else.
- After training, each cell checks `history.jsonl` again and stops before evaluation if any epoch is not Dice.

Old Lovász checkpoints still resume with Lovász, because a resume always uses the loss saved in the checkpoint.

## Sessions G, H and I: multi-seed, Dice-matched experiments

The paper's headline (QGDP-Net vs. the single-scale baseline) and its detail-pathway result each rest on one run, or on Lovász runs. Same-seed repeats showed that identical runs can differ by up to 2.9 validation points. These sessions repeat the key comparison over three seeds, all with the published recipe.

| Priority | `RUN` | Model | Seed | Session | Pairs with |
|---|---|---|---|---|---|
| 1 | `g_image_s43` | QGDP-Net (full) | 43 | G | `h_single_s43`, `g_zeros_s43` |
| 1 | `h_single_s43` | single-scale baseline | 43 | H | `g_image_s43` |
| 2 | `g_image_s44` | QGDP-Net (full) | 44 | G | `h_single_s44`, `g_zeros_s44` |
| 2 | `h_single_s44` | single-scale baseline | 44 | H | `g_image_s44` |
| 3 | `g_zeros_s42` | QGDP-Net, detail maps zeroed | 42 | G | published QGDP-Net (seed 42) |
| 3 | `g_zeros_s43` | QGDP-Net, detail maps zeroed | 43 | G | `g_image_s43` |
| 3 | `g_zeros_s44` | QGDP-Net, detail maps zeroed | 44 | G | `g_image_s44` |
| 4 (optional) | `i_norefiner_s42`, `i_norefiner_s43` | zeroed model, no refiner | 42, 43 | I | `g_zeros_s42/43` |
| 4 (optional) | `i_nofilm_s42`, `i_nofilm_s43` | zeroed model, no FiLM on the skips | 42, 43 | I | `g_zeros_s42/43` |
| 4 (optional) | `i_noaux_s42`, `i_noaux_s43` | zeroed model, no auxiliary losses | 42, 43 | I | `g_zeros_s42/43` |
| 5 (optional) | `g_low_res_s42` | QGDP-Net, low-res detail input | 42 | G | published QGDP-Net (seed 42) |

Seed 42 of QGDP-Net and of the baseline already exist (the published runs), so priorities 1–3 complete three seeds of every arm. A third seed of each Session I ablation (`_s44`) is supported, but only after `g_zeros_s44`.

Each row is one Kaggle session: about 8 h of training, plus validation logging and evaluation (budget 9–10 h; Kaggle's limit is 12 h). Rows are independent and can run in parallel in separate sessions. Run priorities 1 and 2 first.

**Protocol (identical for every run and seed).** 20 epochs; checkpoints at epochs 10 and 20, both evaluated on validation and test; the reported checkpoint is the one with the higher **validation** mIoU. `--val-every 2` also logs validation mIoU every two epochs to `val_curve.jsonl`. That log never selects a checkpoint, saves nothing extra and does not change training (tested). The analysis is fixed in advance in `analyze_sessions_ghi.py` (kept with the paper's analysis scripts).

### Cell G1 (Sessions G and I): choose the run

Run Cell 1 first. Then set `RUN` to one row of the table:

```python
RUN = 'g_image_s43'   # one row of the table above

SPEC = {  # RUN: (detail input, seed, model flags, loss config)
    'g_image_s43': ('image', 43, '', ''),
    'g_image_s44': ('image', 44, '', ''),
    'g_zeros_s42': ('zeros', 42, '', ''),
    'g_zeros_s43': ('zeros', 43, '', ''),
    'g_zeros_s44': ('zeros', 44, '', ''),
    'g_low_res_s42': ('low_res', 42, '', ''),
    **{f'g_image_s{s}': ('image', s, '', '') for s in (45, 46)},  # Session J
    **{f'i_norefiner_s{s}': ('zeros', s, '--refiner-mode none', '') for s in (42, 43, 44)},
    **{f'i_nofilm_s{s}': ('zeros', s, '--no-skip-film', '') for s in (42, 43, 44)},
    **{f'i_noaux_s{s}': ('zeros', s, '', 'noaux_config.json') for s in (42, 43, 44)},
}
detail, seed, model_flags, loss_config = SPEC[RUN]
Path('noaux_config.json').write_text('{"aux96_weight": 0, "aux192_weight": 0, "aux384_weight": 0}\n')
os.environ.update(RUN_NAME=RUN, DETAIL_INPUT=detail, SEED=str(seed), MODEL_FLAGS=model_flags,
                  LOSS_CONFIG=loss_config, RESUME_FROM='')
print(RUN, '| detail input:', detail, '| seed:', seed, '| model flags:', model_flags or '-',
      '| loss config:', loss_config or 'default (Dice)')
```

### Cell G2 (Sessions G and I): train, check the loss, evaluate

Do not edit this cell; everything run-specific comes from Cell G1.

```bash
%%bash
set -euo pipefail
run_dir="$RUN_ROOT/$RUN_NAME"
if [[ -e "$run_dir" && -z "$RESUME_FROM" ]]; then echo "Refusing to reuse $run_dir"; exit 1; fi
uv run --frozen torchrun --standalone --nproc_per_node=2 preflight.py \
    --data-root "$DATA_ROOT" --batch-size 2 --num-workers 0 --steps 3 \
    --gradient-checkpointing --detail-input "$DETAIL_INPUT" $MODEL_FLAGS
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py \
    --data-root "$DATA_ROOT" --output-dir "$run_dir" \
    --num-epochs 20 --batch-size 2 --gradient-accumulation-steps 2 \
    --gradient-checkpointing --num-workers 2 --lr 0.00001 --seed "$SEED" \
    --optimizer adam --weight-decay 0 --lr-scheduler none --warmup-ratio 0 \
    --save-every 10 --detail-input "$DETAIL_INPUT" $MODEL_FLAGS \
    ${LOSS_CONFIG:+--config "$LOSS_CONFIG"} --expect-overlap-loss dice \
    --val-every 2 --val-batch-size 2 \
    ${RESUME_FROM:+--resume-checkpoint "$RESUME_FROM"}
# Stop before evaluation unless every epoch used Dice (and, for no-aux runs, zero auxiliary weights).
uv run --frozen python - "$run_dir/history.jsonl" "$LOSS_CONFIG" <<'PY'
import json, sys
hist = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
losses = {h["training_config"]["overlap_loss"] for h in hist}
aux = {tuple(h["training_config"][k] for k in ("aux96_weight", "aux192_weight", "aux384_weight")) for h in hist}
want_aux = {(0, 0, 0)} if sys.argv[2] else {(0.1, 0.2, 0.3)}
print("epochs:", len(hist), "| overlap_loss:", losses, "| auxiliary weights:", aux)
assert len(hist) == 20 and losses == {"dice"} and aux == want_aux, "WRONG RECIPE: do not use this run"
PY
for epoch in 10 20; do
    for split in val test; do
        uv run --frozen python eval.py \
            --checkpoint "$run_dir/checkpoint_epoch${epoch}.pt" --dataset "$split" \
            --data-root "$DATA_ROOT" --device cuda --batch-size "$EVAL_BATCH" --num-workers 2 \
            --output-dir "$run_dir/eval_${split}_ep${epoch}"
    done
    uv run --frozen python diagnose.py \
        --predictions "$run_dir/eval_val_ep${epoch}" --data-root "$DATA_ROOT" \
        --output-dir "$run_dir/diagnostic_ep${epoch}"
done
cat "$run_dir/val_curve.jsonl"
```

What to look for:
- Before training, `train.py` prints a JSON line containing `"training_config": {"overlap_loss": "dice", ...}`. With the wrong loss it stops with "nothing was trained".
- A `{"epoch": N, "val_miou": ...}` line every two epochs.
- After training, the line `overlap_loss: {'dice'}`. If you see `WRONG RECIPE` instead, do not send that run.

**If the session ends early**, start a new session with Cell 1, keep the same `RUN` in Cell G1, and before Cell G2 run:

```python
os.environ['RESUME_FROM'] = f"{os.environ['RUN_ROOT']}/{os.environ['RUN_NAME']}/checkpoint_epoch10.pt"
```

This needs the epoch-10 checkpoint, `history.jsonl` and `val_curve.jsonl` from the first session in the same folder. If the first session got past epoch 10, first cut both logs back to epoch 10, because `train.py` resumes only where its history ends:

```python
import json
run_dir = Path(os.environ['RUN_ROOT']) / os.environ['RUN_NAME']
for name in ('history.jsonl', 'val_curve.jsonl'):
    path = run_dir / name
    if path.exists():
        kept = [l for l in path.read_text().splitlines() if l.strip() and json.loads(l)['epoch'] <= 10]
        path.write_text(''.join(l + '\n' for l in kept))
        print(name, 'now ends at epoch', json.loads(kept[-1])['epoch'] if kept else None)
```

The loss, seed and ablation switches are restored from the checkpoint, and a mismatch stops the run. Then run Cell Z with `RUN_NAME` unchanged.

### Session H: single-scale baseline, seeds 43 and 44

The baseline must be trained with its own code, commit `2079c8a` (as in Session D), using exactly the recipe of the published seed-42 run (`spatial_alignment_frozen`). From that run's history:
- `training_recipe` `alignment` and `loss_version` `stable`;
- Dice weight 0.5 and answer-alignment weight 0.5;
- frozen text encoder;
- batch 4 on two GPUs, learning rate $10^{-5}$, 20 epochs.

Only the seed changes. Run Cell 1 first (for `DATA_ROOT` and `RUN_ROOT`).

#### Cell H1: baseline code and run choice

```python
SEED = 43   # 43 or 44 (Session H); 45 or 46 (Session J)
os.environ.update(SEED=str(SEED), RUN_NAME=f'h_single_s{SEED}')
```

```bash
%%bash
set -euo pipefail
cd /kaggle/working
rm -rf baseline_code
git clone https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git baseline_code
cd baseline_code
git checkout --detach 2079c8a   # commit that trained the published single-scale baseline
uv sync --frozen
```

#### Cell H2: train, check the recipe, evaluate

```bash
%%bash
set -euo pipefail
cd /kaggle/working/baseline_code
run_dir="/kaggle/working/VizWiz-VQA-Grounding/$RUN_ROOT/$RUN_NAME"
if [[ -e "$run_dir" ]]; then echo "Refusing to reuse $run_dir"; exit 1; fi
echo '{"freeze_text_encoder": true, "dice_weight": 0.5, "latent_weight": 0.5}' > spatial_alignment_frozen.json
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py \
    --data-root "$DATA_ROOT" --num-epochs 20 --batch-size 4 --num-workers 2 \
    --lr 0.00001 --seed "$SEED" --save-every 10 \
    --training-recipe alignment --loss-version stable --config spatial_alignment_frozen.json \
    --output-dir "$run_dir"
# Stop before evaluation unless the saved recipe equals the published seed-42 baseline's.
uv run --frozen python - "$run_dir" "$SEED" <<'PY'
import json, sys, torch
run_dir, seed = sys.argv[1], int(sys.argv[2])
expected = {"training_recipe": "alignment", "loss_version": "stable", "detach_predicted_answer": False,
            "dice_weight": 0.5, "latent_weight": 0.5, "oracle_weight": 0.5, "question_weight": 0.5,
            "kd_weight": 1.0, "attn_weight": 0.2, "gain_mix": 0.75, "warmup_epochs": 0,
            "answer_transition_epochs": 0, "freeze_text_encoder": True, "question_only": False}
hist = [json.loads(line) for line in open(f"{run_dir}/history.jsonl") if line.strip()]
run = torch.load(f"{run_dir}/checkpoint_epoch20.pt", map_location="cpu", weights_only=False)["run_config"]  # file written by this run; the baseline code loads it the same way
checks = {"20 epochs": len(hist) == 20,
          "training_config": all(h["training_config"] == expected for h in hist),
          "seed": run["seed"] == seed, "lr": run["lr"] == 1e-5, "batch_size": run["batch_size"] == 4,
          "world_size": run["world_size"] == 2, "spatial_evidence": run.get("spatial_evidence") is True}
print(checks)
assert all(checks.values()), "WRONG RECIPE: do not use this run"
PY
for epoch in 10 20; do
    for split in val test; do
        uv run --frozen python eval.py \
            --checkpoint "$run_dir/checkpoint_epoch${epoch}.pt" --dataset "$split" \
            --data-root "$DATA_ROOT" --device cuda --batch-size 4 --num-workers 2 \
            --output-dir "$run_dir/eval_${split}_ep${epoch}"
    done
done
```

Each `metrics.json` should say `"grounding_mode": "predicted_answer"`, as for the published baseline: it predicts an answer embedding from the image and question and never receives the answer. Then run Cell Z (the output folder is in the main repository, so Cell Z includes it). If the session ends early, rerun the training command with `--resume-checkpoint "$run_dir/checkpoint_epoch10.pt"` added (remove the `Refusing to reuse` line for that one run), then the check and the evaluation.

### After each run

Run Cell Z and send the ZIP, named `RUN_NAME.zip`. Checkpoints are not in the ZIP. Keep `checkpoint_epoch20.pt` of every G run (Kaggle dataset or download), because a later threshold sweep or figure may need it.

## Session J: two more seeds for the headline (45, 46)

Why: the seed-42 pair has the largest gap (+2.16 test, against +0.91 and +0.78 at seeds 43 and 44). Code, recipe and evaluation are the same for all three seeds (checked 2026-11-02), so seed 42 was a favourable draw, and a rerun of seed 42 would not settle anything (same-seed reruns differ by up to 2.9 points). More seeds make the headline less dependent on any single draw. These runs use the existing cells; only the run choice changes.

| Priority | Run | Cells | Setting |
|---|---|---|---|
| 1 | `g_image_s45` | G1, G2 | `RUN = 'g_image_s45'` |
| 1 | `h_single_s45` | H1, H2 | `SEED = 45` |
| 2 | `g_image_s46` | G1, G2 | `RUN = 'g_image_s46'` |
| 2 | `h_single_s46` | H1, H2 | `SEED = 46` |

Run the two rows of a seed together; a seed without its partner is not used in the paired comparison. Session J and Session I are independent, so they can run in parallel. If sessions are limited, use this order: `g_image_s45` and `h_single_s45`, then Session I seeds 42 and 43, then `g_image_s46` and `h_single_s46`. The protocol, recipe checks and Cell Z are unchanged, and the analysis was fixed before these runs (`analyze_sessions_ghi.py`, section "Session J").

## Session L: extra controls (question-only baseline, frozen encoder, CLIPSeg)

Each control below exists for seed 42 only, or not at all. These sessions give three seeds of each, with the analysis fixed in advance (`analyze_session_l.py`). Run Cell 1 first in every session, and Cell Z at the end with `os.environ['RUN_NAME']` set as shown.

| Priority | Run(s) | Cells | Kaggle sessions | Time (estimate) |
|---|---|---|---|---|
| 1 | `l_qonly_s43`, `l_qonly_s44` | L1a, L1b | one per seed | about 8–9 h (same model as Session H) |
| 2 | `l_frozen_s43`, `l_frozen_s44` | L2a, L2b | one per seed | about 5–8 h (no backward pass through the ViT) |
| 3 | all four CLIPSeg runs | L3a, L3b | one session (both GPUs), or one per phase | about 2–4 h per phase |

The removed-detail-pathway model (item 7) needs no GPU. With zeroed detail maps, every weight that touches the detail pathway multiplies zeros and receives exactly zero gradient; changing those weights changes the output by exactly 0.0 (`tests/test_zero_equivalence.py`). The trained zeroed models (`g_zeros_s42/43/44`) are therefore the model without the pathway.

### Session L1: question-only baseline, seeds 43 and 44

Same code (`2079c8a`) and command as the seed-42 question-only run (Cell D2); only the seed changes.

#### Cell L1a

```python
SEED = 43   # 43 or 44
os.environ.update(SEED=str(SEED), RUN_NAME=f'l_qonly_s{SEED}')
```

#### Cell L1b: train, check the recipe, evaluate

```bash
%%bash
set -euo pipefail
cd /kaggle/working
rm -rf baseline_code
git clone https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git baseline_code
cd baseline_code
git checkout --detach 2079c8a   # commit that trained the seed-42 question-only baseline
uv sync --frozen
run_dir="/kaggle/working/VizWiz-VQA-Grounding/$RUN_ROOT/$RUN_NAME"
if [[ -e "$run_dir" ]]; then echo "Refusing to reuse $run_dir"; exit 1; fi
echo '{"freeze_text_encoder": true, "question_only": true, "dice_weight": 0.5}' > a2_config.json
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py \
    --data-root "$DATA_ROOT" --num-epochs 20 --batch-size 4 --num-workers 2 \
    --lr 0.00001 --seed "$SEED" --save-every 10 \
    --training-recipe grounding-only --config a2_config.json \
    --output-dir "$run_dir"
# Stop before evaluation unless the recipe equals the seed-42 question-only run's.
uv run --frozen python - "$run_dir" "$SEED" <<'PY'
import json, sys, torch
run_dir, seed = sys.argv[1], int(sys.argv[2])
expected = {"training_recipe": "grounding-only", "loss_version": "legacy", "detach_predicted_answer": False,
            "dice_weight": 0.5, "latent_weight": 0.5, "oracle_weight": 0.5, "question_weight": 0.5,
            "kd_weight": 1.0, "attn_weight": 0.2, "gain_mix": 0.75, "warmup_epochs": 0,
            "answer_transition_epochs": 0, "freeze_text_encoder": True, "question_only": True}
hist = [json.loads(line) for line in open(f"{run_dir}/history.jsonl") if line.strip()]
run = torch.load(f"{run_dir}/checkpoint_epoch20.pt", map_location="cpu", weights_only=False)["run_config"]  # file written by this run
checks = {"20 epochs": len(hist) == 20,
          "training_config": all(h["training_config"] == expected for h in hist),
          "seed": run["seed"] == seed, "lr": run["lr"] == 1e-5, "batch_size": run["batch_size"] == 4,
          "world_size": run["world_size"] == 2}
print(checks)
assert all(checks.values()), "WRONG RECIPE: do not use this run"
PY
for epoch in 10 20; do
    for split in val test; do
        uv run --frozen python eval.py \
            --checkpoint "$run_dir/checkpoint_epoch${epoch}.pt" --dataset "$split" \
            --data-root "$DATA_ROOT" --device cuda --batch-size 4 --num-workers 2 \
            --output-dir "$run_dir/eval_${split}_ep${epoch}"
    done
done
```

### Session L2: QGDP-Net with frozen encoders, seeds 43 and 44

The seed-42 run (`question_guided_detail_pyramid_frozen_backbone_ep20_seed42`) recorded: Adam, lr $10^{-5}$, no weight decay or schedule, vision and text encoders frozen, batch 2 × accumulation 2 on two GPUs, gradient checkpointing, Dice, 6,649,101 trainable parameters. The current code reproduces exactly this with the flags below (the same count is printed before training as `"trainable_parameters": 6649101`).

#### Cell L2a

```python
SEED = 43   # 43 or 44
os.environ.update(SEED=str(SEED), RUN_NAME=f'l_frozen_s{SEED}')
```

#### Cell L2b: train, check the recipe, evaluate

```bash
%%bash
set -euo pipefail
run_dir="$RUN_ROOT/$RUN_NAME"
if [[ -e "$run_dir" ]]; then echo "Refusing to reuse $run_dir"; exit 1; fi
FREEZE="--freeze-vision-encoder --freeze-text-encoder"
uv run --frozen torchrun --standalone --nproc_per_node=2 preflight.py \
    --data-root "$DATA_ROOT" --batch-size 2 --num-workers 0 --steps 3 \
    --gradient-checkpointing $FREEZE
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py \
    --data-root "$DATA_ROOT" --output-dir "$run_dir" \
    --num-epochs 20 --batch-size 2 --gradient-accumulation-steps 2 \
    --gradient-checkpointing --num-workers 2 --lr 0.00001 --seed "$SEED" \
    --optimizer adam --weight-decay 0 --lr-scheduler none --warmup-ratio 0 $FREEZE \
    --save-every 10 --expect-overlap-loss dice --val-every 2 --val-batch-size 2
# Stop before evaluation unless every epoch used Dice and both encoders were frozen.
uv run --frozen python - "$run_dir" <<'PY'
import json, sys, torch
run_dir = sys.argv[1]
hist = [json.loads(line) for line in open(f"{run_dir}/history.jsonl") if line.strip()]
run = torch.load(f"{run_dir}/checkpoint_epoch20.pt", map_location="cpu", weights_only=False)["run_config"]  # file written by this run
opt = run["optimization_config"]
cfg = {(h["training_config"]["overlap_loss"], h["training_config"]["aux96_weight"], h["training_config"]["aux192_weight"],
        h["training_config"]["aux384_weight"]) for h in hist}
checks = {"20 epochs": len(hist) == 20, "dice and default auxiliary weights": cfg == {("dice", 0.1, 0.2, 0.3)},
          "vision frozen": opt["freeze_vision_encoder"] is True, "text frozen": opt["freeze_text_encoder"] is True,
          "adam, lr 1e-5, no weight decay": (opt["optimizer"], opt["lr"], opt["weight_decay"]) == ("adam", 1e-5, 0.0)}
print(checks)
assert all(checks.values()), "WRONG RECIPE: do not use this run"
PY
for epoch in 10 20; do
    for split in val test; do
        uv run --frozen python eval.py \
            --checkpoint "$run_dir/checkpoint_epoch${epoch}.pt" --dataset "$split" \
            --data-root "$DATA_ROOT" --device cuda --batch-size "$EVAL_BATCH" --num-workers 2 \
            --output-dir "$run_dir/eval_${split}_ep${epoch}"
    done
done
cat "$run_dir/val_curve.jsonl"
```

### Session L3: CLIPSeg external baseline (one session, both GPUs)

CLIPSeg (`CIDAS/clipseg-rd64-refined`, CLIP ViT-B/16) as published: the CLIP encoders stay frozen and only the decoder (1.13M parameters, from the released weights) is trained, with the question as the prompt. Recipe as close to ours as the method allows: full-frame resize, BCE + 0.5 Dice at 768 px, Adam, constant learning rate, clipping 1.0, batch 4, 20 epochs, evaluation at epochs 10 and 20 with our metric code, selection by validation. The protocol is fixed in the header of `clipseg_baseline.py`. Phase 1 trains seed 42 with learning rates $10^{-4}$ and $10^{-5}$ in parallel (one per GPU); the cell then picks the rate with the higher validation mIoU; phase 2 trains seeds 43 and 44 with that rate in parallel.

#### Cell L3a: phase 1, learning rate chosen on validation (seed 42)

```bash
%%bash
set -euo pipefail
root="$RUN_ROOT"; py="$PWD/.venv/bin/python"
for d in l_clipseg_lr1e-4_s42 l_clipseg_lr1e-5_s42; do
    if [[ -e "$root/$d" ]]; then echo "Refusing to reuse $root/$d"; exit 1; fi
done
"$py" -c "from transformers import CLIPSegForImageSegmentation as M, CLIPSegProcessor as P; M.from_pretrained('CIDAS/clipseg-rd64-refined'); P.from_pretrained('CIDAS/clipseg-rd64-refined')"
run() {  # GPU, run name, seed, learning rate
    mkdir -p "$root/$2"
    CUDA_VISIBLE_DEVICES=$1 "$py" clipseg_baseline.py train --data-root "$DATA_ROOT" \
        --output-dir "$root/$2" --seed "$3" --lr "$4" > "$root/$2/train.log" 2>&1 &&
    CUDA_VISIBLE_DEVICES=$1 "$py" clipseg_baseline.py eval --data-root "$DATA_ROOT" \
        --run-dir "$root/$2" > "$root/$2/eval.log" 2>&1
}
run 0 l_clipseg_lr1e-4_s42 42 0.0001 & a=$!
run 1 l_clipseg_lr1e-5_s42 42 0.00001 & b=$!
wait $a; wait $b
"$py" clipseg_baseline.py choose-lr "$root/l_clipseg_lr1e-4_s42" "$root/l_clipseg_lr1e-5_s42" > "$root/l_clipseg_chosen_lr.txt"
echo "Learning rate chosen on validation: $(cat "$root/l_clipseg_chosen_lr.txt")"
for d in l_clipseg_lr1e-4_s42 l_clipseg_lr1e-5_s42; do echo "== $d"; tail -n 4 "$root/$d/eval.log"; done
```

#### Cell L3b: phase 2, seeds 43 and 44 with the chosen rate

In the same session this cell needs no edit. In a new session, first run `os.environ['CLIPSEG_LR'] = '...'` with the rate Cell L3a printed.

```bash
%%bash
set -euo pipefail
root="$RUN_ROOT"; py="$PWD/.venv/bin/python"
LR="${CLIPSEG_LR:-$(cat "$root/l_clipseg_chosen_lr.txt" 2>/dev/null || true)}"
if [[ "$LR" != "0.0001" && "$LR" != "1e-05" && "$LR" != "1e-5" && "$LR" != "0.00001" ]]; then
    echo "Set CLIPSEG_LR to the rate printed by Cell L3a (got '$LR')"; exit 1
fi
for d in l_clipseg_s43 l_clipseg_s44; do
    if [[ -e "$root/$d" ]]; then echo "Refusing to reuse $root/$d"; exit 1; fi
done
run() {  # GPU, run name, seed, learning rate
    mkdir -p "$root/$2"
    CUDA_VISIBLE_DEVICES=$1 "$py" clipseg_baseline.py train --data-root "$DATA_ROOT" \
        --output-dir "$root/$2" --seed "$3" --lr "$4" > "$root/$2/train.log" 2>&1 &&
    CUDA_VISIBLE_DEVICES=$1 "$py" clipseg_baseline.py eval --data-root "$DATA_ROOT" \
        --run-dir "$root/$2" > "$root/$2/eval.log" 2>&1
}
run 0 l_clipseg_s43 43 "$LR" & a=$!
run 1 l_clipseg_s44 44 "$LR" & b=$!
wait $a; wait $b
for d in l_clipseg_s43 l_clipseg_s44; do echo "== $d"; tail -n 4 "$root/$d/eval.log"; done
```

After Cell L3a, run `os.environ['RUN_NAME'] = 'l_clipseg_phase1'` and Cell Z; after Cell L3b, `os.environ['RUN_NAME'] = 'l_clipseg_phase2'` and Cell Z. Send both ZIPs. While a phase runs, `!tail -n 3 outputs/l_clipseg_s43/train.log` (or the phase-1 run name) shows one line per epoch with its duration; about 10 minutes per epoch means about 4 h per phase.

## Session T: mask-threshold sweep (inference only, about 30–60 min)

**Why.** On small targets QGDP-Net's masks are tighter than the single-scale baseline's (paper Table 3). This session tests whether the baseline closes the gap when its own threshold is tuned. Every model's logits are thresholded at 19 values from −4 to 4. The threshold is chosen on validation only; test is reported at that threshold. No training.

`revision/threshold_sweep.py` runs inside whichever repository holds the model's code: this branch for QGDP-Net, commit `2079c8a` for the baseline. It reproduces `eval.py` exactly at threshold 0; this was checked on CPU in both codebases.

**Attach:**
- the VizWiz dataset;
- the published QGDP-Net checkpoint (Cell 1);
- the **single-scale baseline checkpoint**: `checkpoint_epoch20.pt` of the `spatial_alignment_frozen` run, commit `2079c8a`;
- optional: the A2 checkpoint (`a2_single_scale_question_only/checkpoint_epoch20.pt`), if you kept it.

Run Cell 1 first.

### Cell T1: checkpoint paths

```python
os.environ['SINGLE_CKPT'] = '/kaggle/input/REPLACE/checkpoint_epoch20.pt'  # single-scale baseline (required)
os.environ['A2_CKPT'] = ''                                                # optional, '' to skip
os.environ['RUN_NAME'] = 'threshold_sweep'
assert Path(os.environ['SINGLE_CKPT']).is_file(), 'Set SINGLE_CKPT to the attached baseline checkpoint'
```

### Cell T2: sweep validation and test

```bash
%%bash
set -euo pipefail
repo=/kaggle/working/VizWiz-VQA-Grounding
SWEEP="$repo/revision/threshold_sweep.py"
out="$repo/outputs/threshold_sweep"
mkdir -p "$out"
cd "$repo"
for split in val test; do
    uv run --frozen python "$SWEEP" --checkpoint "$PARENT_CKPT" --data-root "$DATA_ROOT" \
        --dataset "$split" --device cuda --batch-size "$EVAL_BATCH" --output "$out/qgdp_${split}.json"
done
if [[ ! -d /kaggle/working/baseline_code ]]; then
    git clone https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git /kaggle/working/baseline_code
    (cd /kaggle/working/baseline_code && git checkout --detach 2079c8a && uv sync --frozen)
fi
cd /kaggle/working/baseline_code
for spec in "single:$SINGLE_CKPT" "a2:${A2_CKPT:-}"; do
    name=${spec%%:*}; ckpt=${spec#*:}
    [[ -z "$ckpt" ]] && continue
    for split in val test; do
        uv run --frozen python "$SWEEP" --checkpoint "$ckpt" --data-root "$DATA_ROOT" \
            --dataset "$split" --device cuda --batch-size 4 --output "$out/${name}_${split}.json"
    done
done
```

**Check the printed `mean_iou_at_0`.** It must equal the published numbers:

| Model | Validation | Test |
|---|---|---|
| QGDP-Net | 0.6577 | 0.6722 |
| Single-scale baseline | 0.6316 | 0.6506 |
| A2 | 0.6347 | 0.6491 |

If a value differs, the wrong checkpoint is attached. Then run Cell Z (the ZIP is named `threshold_sweep.zip`).

## Session P: does the trained model use its detail pathway? (inference only, about 3–5 h)

**Why.** The paper retrains QGDP-Net with zeroed detail maps and finds no loss. That shows the rest of the model does not need the pathway; it does not show whether the *trained* full model uses it. This session evaluates each trained full QGDP-Net under four inference-time conditions (`revision/detail_intervention.py`): `original`, `zeros` (all detail maps zero), `misaligned` (detail view rotated 180°, so the maps no longer line up with the ViT features) and `low_res` (the detail CNN sees the upsampled 336 px view). It also records the learned residual scales, the mean gates and the size of each detail residual relative to the decoder features. No training. The analysis is fixed in advance in `analyze_detail_intervention.py` (with the paper's analysis scripts).

**Attach:**
- the VizWiz dataset and the published epoch-20 checkpoint (Cell 1; this is run 1);
- if you kept them, the selected checkpoints of the other full QGDP-Net runs: `g_image_s43/checkpoint_epoch20.pt`, `g_image_s44/checkpoint_epoch10.pt` (epoch 10 was selected for this run), `g_image_s45/checkpoint_epoch20.pt`, `g_image_s46/checkpoint_epoch20.pt`.

Runs whose checkpoint is missing are skipped; the analysis reports how many runs it used. Run Cell 1 first.

### Cell P1: checkpoint paths

```python
CKPTS = {  # run name: selected checkpoint ('' if not attached)
    'image_s42': os.environ['PARENT_CKPT'],
    'image_s43': '',   # .../g_image_s43/checkpoint_epoch20.pt
    'image_s44': '',   # .../g_image_s44/checkpoint_epoch10.pt
    'image_s45': '',   # .../g_image_s45/checkpoint_epoch20.pt
    'image_s46': '',   # .../g_image_s46/checkpoint_epoch20.pt
}
EXPECTED = {  # published (val, test) mean IoU of each selected checkpoint
    'image_s42': (0.6577, 0.6722), 'image_s43': (0.6504, 0.6681), 'image_s44': (0.6499, 0.6635),
    'image_s45': (0.6424, 0.6654), 'image_s46': (0.6513, 0.6662),
}
CKPTS = {k: v for k, v in CKPTS.items() if v}
for name, path in CKPTS.items():
    assert Path(path).is_file(), f'{name}: {path} not found'
Path('p_runs.txt').write_text(''.join(f'{k} {v}\n' for k, v in CKPTS.items()))
os.environ['RUN_NAME'] = 'p_detail_intervention'
print('attached runs:', list(CKPTS))
if list(CKPTS) == ['image_s42']:
    print('Only run 1 is attached: the analysis will cover run 1 only (image sampling, no run-to-run variation).')
```

### Cell P2: verify every checkpoint before any intervention (both GPUs, about 1 h)

Each checkpoint is first evaluated without intervention on validation and test. Only checkpoints that reproduce their published scores go on to Cell P3.

```bash
%%bash
set -euo pipefail
out="$RUN_ROOT/$RUN_NAME/verify"
mkdir -p "$out"
work() {  # $1: GPU, then the runs for that GPU
    gpu=$1; shift
    for run in "$@"; do
        ckpt=$(awk -v r="$run" '$1 == r {print $2}' p_runs.txt)
        for split in val test; do
            [[ -f "$out/${run}_${split}.json" ]] && continue
            CUDA_VISIBLE_DEVICES=$gpu uv run --frozen python revision/detail_intervention.py \
                --checkpoint "$ckpt" --data-root "$DATA_ROOT" --dataset "$split" --device cuda \
                --conditions original --batch-size "$EVAL_BATCH" --num-workers 2 \
                --output "$out/${run}_${split}.json" > "$out/${run}_${split}.log" 2>&1
        done
    done
}
mapfile -t runs < <(awk '{print $1}' p_runs.txt)
work 0 "${runs[@]:0:3}" &
work 1 "${runs[@]:3}" &
wait
```

```python
import json
verified, bad = [], []
for name in CKPTS:
    ok = True
    for split, expected in zip(('val', 'test'), EXPECTED[name]):
        got = json.loads(Path(os.environ['RUN_ROOT'], os.environ['RUN_NAME'], 'verify', f'{name}_{split}.json')
                         .read_text())['summary']['mean_iou']['original']
        print(f'{name} {split}: {got:.4f} (published {expected:.4f})')
        ok &= abs(got - expected) <= 5e-5
    (verified if ok else bad).append(name)
Path('p_verified.txt').write_text(''.join(f'{k} {CKPTS[k]}\n' for k in verified))
print('verified:', verified, '| NOT verified (excluded):', bad)
assert verified, 'No checkpoint reproduces its published scores: stop here'
```

A run that is not verified is the wrong file for that run; it is excluded and listed in the ZIP (`verify/`).

### Cell P3: interventions on the verified checkpoints (both GPUs, about 3–4 h for five runs)

```bash
%%bash
set -euo pipefail
out="$RUN_ROOT/$RUN_NAME"
work() {
    gpu=$1; shift
    for run in "$@"; do
        ckpt=$(awk -v r="$run" '$1 == r {print $2}' p_verified.txt)
        for split in val test; do
            [[ -f "$out/${run}_${split}.json" ]] && continue
            CUDA_VISIBLE_DEVICES=$gpu uv run --frozen python revision/detail_intervention.py \
                --checkpoint "$ckpt" --data-root "$DATA_ROOT" --dataset "$split" --device cuda \
                --batch-size "$EVAL_BATCH" --num-workers 2 --output "$out/${run}_${split}.json" \
                > "$out/${run}_${split}.log" 2>&1
            tail -1 "$out/${run}_${split}.log"
        done
    done
}
mapfile -t runs < <(awk '{print $1}' p_verified.txt)
work 0 "${runs[@]:0:3}" &
work 1 "${runs[@]:3}" &
wait
```

```python
for name in Path('p_verified.txt').read_text().split('\n'):
    if not name.strip():
        continue
    run = name.split()[0]
    for split in ('val', 'test'):
        a = json.loads(Path(os.environ['RUN_ROOT'], os.environ['RUN_NAME'], f'{run}_{split}.json').read_text())
        b = json.loads(Path(os.environ['RUN_ROOT'], os.environ['RUN_NAME'], 'verify', f'{run}_{split}.json').read_text())
        assert a['per_sample']['original'] == b['per_sample']['original'], f'{run} {split}: original changed between cells'
        print(run, split, {c: round(v, 4) for c, v in a['summary']['mean_iou'].items()})
```

Then run Cell Z (the ZIP is named `p_detail_intervention.zip`).

## Session Q: are the intervened masks the same masks? (inference only, about 4 h)

**Why.** Session P showed that zeroing the detail maps barely changes mean IoU. Similar means can hide different masks. This session compares the masks directly (`revision/mask_agreement.py`): for each image, the IoU between the unchanged and the intervened mask, the share of pixels where they differ, and IoU against the ground truth. Under the unchanged model it also measures how much of each detail residual depends on the detail map, by recomputing each fusion block (and the refiner) with the same inputs but a zero map. No training; Session P's files are not used or changed. The analysis is fixed in advance in `analyze_mask_agreement.py` (with the paper's analysis scripts).

**Attach** the same checkpoints as Session P: the dataset, the published epoch-20 checkpoint (run 1, Cell 1), and `g_image_s43/checkpoint_epoch20.pt`, `g_image_s44/checkpoint_epoch10.pt`, `g_image_s45/checkpoint_epoch20.pt`, `g_image_s46/checkpoint_epoch20.pt`. Run Cell 1 first.

### Cell Q1: checkpoint paths

```python
CKPTS = {  # run name: selected checkpoint ('' if not attached)
    'image_s42': os.environ['PARENT_CKPT'],
    'image_s43': '',   # .../g_image_s43/checkpoint_epoch20.pt
    'image_s44': '',   # .../g_image_s44/checkpoint_epoch10.pt
    'image_s45': '',   # .../g_image_s45/checkpoint_epoch20.pt
    'image_s46': '',   # .../g_image_s46/checkpoint_epoch20.pt
}
EXPECTED = {  # published (val, test) mean IoU of each selected checkpoint
    'image_s42': (0.6577, 0.6722), 'image_s43': (0.6504, 0.6681), 'image_s44': (0.6499, 0.6635),
    'image_s45': (0.6424, 0.6654), 'image_s46': (0.6513, 0.6662),
}
CKPTS = {k: v for k, v in CKPTS.items() if v}
for name, path in CKPTS.items():
    assert Path(path).is_file(), f'{name}: {path} not found'
Path('q_runs.txt').write_text(''.join(f'{k} {v}\n' for k, v in CKPTS.items()))
os.environ['RUN_NAME'] = 'q_mask_agreement'
print('attached runs:', list(CKPTS))
if not set(CKPTS) - {'image_s42'}:
    print('Only run 1 is attached: Session P excluded run 1, so the analysis will be descriptive only.')
```

### Cell Q2: verify every checkpoint before any intervention (both GPUs, about 1 h)

```bash
%%bash
set -euo pipefail
out="$RUN_ROOT/$RUN_NAME/verify"
mkdir -p "$out"
work() {
    gpu=$1; shift
    for run in "$@"; do
        ckpt=$(awk -v r="$run" '$1 == r {print $2}' q_runs.txt)
        for split in val test; do
            [[ -f "$out/${run}_${split}.json" ]] && continue
            CUDA_VISIBLE_DEVICES=$gpu uv run --frozen python revision/detail_intervention.py \
                --checkpoint "$ckpt" --data-root "$DATA_ROOT" --dataset "$split" --device cuda \
                --conditions original --batch-size "$EVAL_BATCH" --num-workers 2 \
                --output "$out/${run}_${split}.json" > "$out/${run}_${split}.log" 2>&1
        done
    done
}
mapfile -t runs < <(awk '{print $1}' q_runs.txt)
work 0 "${runs[@]:0:3}" &
work 1 "${runs[@]:3}" &
wait
```

```python
import json
verified, bad = [], []
for name in CKPTS:
    ok = True
    for split, expected in zip(('val', 'test'), EXPECTED[name]):
        got = json.loads(Path(os.environ['RUN_ROOT'], os.environ['RUN_NAME'], 'verify', f'{name}_{split}.json')
                         .read_text())['summary']['mean_iou']['original']
        print(f'{name} {split}: {got:.4f} (published {expected:.4f})')
        ok &= abs(got - expected) <= 5e-5
    (verified if ok else bad).append(name)
Path('q_verified.txt').write_text(''.join(f'{k} {CKPTS[k]}\n' for k in verified))
print('verified:', verified, '| NOT verified (excluded):', bad)
assert verified, 'No checkpoint reproduces its published scores: stop here'
```

### Cell Q3: mask agreement on the verified checkpoints (both GPUs, about 3 h for five runs)

```bash
%%bash
set -euo pipefail
out="$RUN_ROOT/$RUN_NAME"
work() {
    gpu=$1; shift
    for run in "$@"; do
        ckpt=$(awk -v r="$run" '$1 == r {print $2}' q_verified.txt)
        for split in val test; do
            [[ -f "$out/${run}_${split}.json" ]] && continue
            CUDA_VISIBLE_DEVICES=$gpu uv run --frozen python revision/mask_agreement.py \
                --checkpoint "$ckpt" --data-root "$DATA_ROOT" --dataset "$split" --device cuda \
                --batch-size "$EVAL_BATCH" --num-workers 2 --output "$out/${run}_${split}.json" \
                > "$out/${run}_${split}.log" 2>&1
            tail -1 "$out/${run}_${split}.log"
        done
    done
}
mapfile -t runs < <(awk '{print $1}' q_verified.txt)
work 0 "${runs[@]:0:3}" &
work 1 "${runs[@]:3}" &
wait
```

```python
for line in Path('q_verified.txt').read_text().splitlines():
    run = line.split()[0]
    for split in ('val', 'test'):
        a = json.loads(Path(os.environ['RUN_ROOT'], os.environ['RUN_NAME'], f'{run}_{split}.json').read_text())
        b = json.loads(Path(os.environ['RUN_ROOT'], os.environ['RUN_NAME'], 'verify', f'{run}_{split}.json').read_text())
        mine = {n: r['iou_original'] for n, r in a['per_sample'].items()}
        assert mine == b['per_sample']['original'], f'{run} {split}: original changed between cells'
        s = a['summary']
        print(run, split, {c: round(s[c]['mask_iou'], 4) for c in ('zeros', 'misaligned', 'low_res')})
```

Then run Cell Z (the ZIP is named `q_mask_agreement.zip`).

## Session R: which stage uses its detail map, and does amplifying the maps help? (inference only, about 2–3 h)

**Why.** Session P zeroed all detail maps at once. This session (`revision/stagewise_detail.py`) asks three finer questions on the validation split only, with the trained models unchanged:
1. Zeroing one map at a time (`zero_p0` … `zero_p4`): does any single decoder stage, or the refiner, use its map?
2. Multiplying all maps by 2 or 4 (`scale_x2`, `scale_x4`): do the maps carry information the trained model weights too weakly?
3. Gradient × input attribution of each map against the decoder features it is fused with, plus the gradient reaching each parameter group: how much of the training signal reaches the pathway?

No training. The test split is not used. Session P and Q files are not used or changed. The analysis is fixed in advance in `analyze_stagewise_detail.py` (with the paper's analysis scripts).

**Attach** the same checkpoints as Sessions P and Q: the dataset, the published epoch-20 checkpoint (run 1, Cell 1), and `g_image_s43/checkpoint_epoch20.pt`, `g_image_s44/checkpoint_epoch10.pt`, `g_image_s45/checkpoint_epoch20.pt`, `g_image_s46/checkpoint_epoch20.pt`. Run Cell 1 first.

### Cell R1: checkpoint paths

```python
CKPTS = {  # run name: selected checkpoint ('' if not attached)
    'image_s42': os.environ['PARENT_CKPT'],
    'image_s43': '',   # .../g_image_s43/checkpoint_epoch20.pt
    'image_s44': '',   # .../g_image_s44/checkpoint_epoch10.pt
    'image_s45': '',   # .../g_image_s45/checkpoint_epoch20.pt
    'image_s46': '',   # .../g_image_s46/checkpoint_epoch20.pt
}
EXPECTED_VAL = {  # published validation mean IoU of each selected checkpoint
    'image_s42': 0.6577, 'image_s43': 0.6504, 'image_s44': 0.6499, 'image_s45': 0.6424, 'image_s46': 0.6513,
}
CKPTS = {k: v for k, v in CKPTS.items() if v}
for name, path in CKPTS.items():
    assert Path(path).is_file(), f'{name}: {path} not found'
Path('r_runs.txt').write_text(''.join(f'{k} {v}\n' for k, v in CKPTS.items()))
os.environ['RUN_NAME'] = 'r_stagewise'
print('attached runs:', list(CKPTS))
if not set(CKPTS) - {'image_s42'}:
    print('Only run 1 is attached: Sessions P and Q excluded run 1, so the analysis will be descriptive only.')
```

### Cell R2: verify every checkpoint before any intervention (both GPUs, about 20 min)

```bash
%%bash
set -euo pipefail
out="$RUN_ROOT/$RUN_NAME/verify"
mkdir -p "$out"
work() {
    gpu=$1; shift
    for run in "$@"; do
        ckpt=$(awk -v r="$run" '$1 == r {print $2}' r_runs.txt)
        [[ -f "$out/${run}_val.json" ]] && continue
        CUDA_VISIBLE_DEVICES=$gpu uv run --frozen python revision/detail_intervention.py \
            --checkpoint "$ckpt" --data-root "$DATA_ROOT" --dataset val --device cuda \
            --conditions original --batch-size "$EVAL_BATCH" --num-workers 2 \
            --output "$out/${run}_val.json" > "$out/${run}_val.log" 2>&1
    done
}
mapfile -t runs < <(awk '{print $1}' r_runs.txt)
work 0 "${runs[@]:0:3}" &
work 1 "${runs[@]:3}" &
wait
```

```python
import json
verified, bad = [], []
for name in CKPTS:
    got = json.loads(Path(os.environ['RUN_ROOT'], os.environ['RUN_NAME'], 'verify', f'{name}_val.json')
                     .read_text())['summary']['mean_iou']['original']
    print(f'{name} val: {got:.4f} (published {EXPECTED_VAL[name]:.4f})')
    (verified if abs(got - EXPECTED_VAL[name]) <= 5e-5 else bad).append(name)
Path('r_verified.txt').write_text(''.join(f'{k} {CKPTS[k]}\n' for k in verified))
print('verified:', verified, '| NOT verified (excluded):', bad)
assert verified, 'No checkpoint reproduces its published score: stop here'
```

### Cell R3: stagewise interventions and gradients on the verified checkpoints (both GPUs, about 2 h for five runs)

The batch size is the same as in Cell R2, so the unchanged model's per-image IoU can be compared exactly. The gradient pass does not backpropagate through the encoders, so it fits on a T4 at this batch size. If a run still runs out of memory, stop and report it rather than changing the batch size.

```bash
%%bash
set -euo pipefail
out="$RUN_ROOT/$RUN_NAME"
work() {
    gpu=$1; shift
    for run in "$@"; do
        ckpt=$(awk -v r="$run" '$1 == r {print $2}' r_verified.txt)
        [[ -f "$out/${run}_val.json" ]] && continue
        CUDA_VISIBLE_DEVICES=$gpu uv run --frozen python revision/stagewise_detail.py \
            --checkpoint "$ckpt" --data-root "$DATA_ROOT" --dataset val --device cuda \
            --batch-size "$EVAL_BATCH" --num-workers 2 --output "$out/${run}_val.json" \
            > "$out/${run}_val.log" 2>&1
        tail -1 "$out/${run}_val.log" | cut -c1-400
    done
}
mapfile -t runs < <(awk '{print $1}' r_verified.txt)
work 0 "${runs[@]:0:3}" &
work 1 "${runs[@]:3}" &
wait
```

```python
for line in Path('r_verified.txt').read_text().splitlines():
    run = line.split()[0]
    a = json.loads(Path(os.environ['RUN_ROOT'], os.environ['RUN_NAME'], f'{run}_val.json').read_text())
    b = json.loads(Path(os.environ['RUN_ROOT'], os.environ['RUN_NAME'], 'verify', f'{run}_val.json').read_text())
    mine = {n: r['iou_original'] for n, r in a['per_sample'].items()}
    assert mine == b['per_sample']['original'], f'{run}: original changed between cells'
    s = a['summary']
    print(run, {c: round(s[c]['iou'] - s['mean_iou_original'], 4) for c in a['conditions'][1:]})
```

Then run Cell Z (the ZIP is named `r_stagewise.zip`) and return it.

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
