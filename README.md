# Question-guided high-resolution detail pyramid

Branch: `exp/question-guided-detail-pyramid`, created from `exp/spatial-detail-decoder` at `63977d2`. This experiment introduces a 768-pixel convolutional detail pyramid, question-conditioned multi-scale fusion, and a signed residual mask refiner, trained fresh from pretrained CLIP weights.

## Architecture

```text
Original upright RGB image
  +-- resize 336 --> CLIP vision --> image/text cross-attention + visual residual
  |                                      |
  |                 3 CLIP skips: 1024 -> 256/GN/GELU -> Q-FiLM -> 128
  |                                      |
  |                           three skip fusions at 24x24
  |                                      |
  |                         standard-conv upsampling stages
  |                               48 -> 96 -> 192 -> 384
  |                                ^     ^      ^      ^
  +-- resize 768 --> CNN encoder --> top-down detail pyramid, 16 channels
                                  48 -> 96 -> 192 -> 384 -> 768
                                                         |
  Question --> frozen CLIP text --> pooled Q --> spatial guidance at all stages
                                                         |
  Decoder feature + logits384 + detail768 + projected Q --> residual refiner
                                                         |
                        logits768 = resize(logits384) + signed correction
```

* Both RGB views are generated directly from the original `[0, 1]` image and normalized with CLIP mean and std. Aspect ratios are resized to squares without crop, flip, or color augmentation, matching the parent CLIP setup.
* CLIP is `openai/clip-vit-large-patch14-336`. The text encoder is frozen and kept in `eval` mode. The vision encoder is trainable except for the unused pooled-CLS `post_layernorm` (whose state-dict keys are retained for checkpoint compatibility).
* Vision skip connections tap hidden states 4, 7, and 9, which share shape `(B, 1024, 24, 24)` with the final bottleneck. Each skip has its own nonlinear channel compression and zero-initialized FiLM head (identity modulation at step 0), sharing a common question projection.
* Decoder channels progress through `128 -> 64 -> 32 -> 16 -> 16`. Each upsampling block uses bilinear resize followed by **one standard 3x3 convolution**, GroupNorm, and GELU. CLIP skip-fusion and detail-fusion blocks use depthwise-separable convolutions.
* The high-resolution CNN encoder runs at `768 / 384 / 192 / 96 / 48` with channel widths `8 / 16 / 32 / 64 / 128`. Each stage consists of two standard 3x3 Conv/GN/GELU blocks followed by max-pooling downsampling. Top-down pyramid levels combine a 1x1 lateral projection with bilinear upsampling and a 3x3 Conv/GN/GELU smoothing block into 16 channels.
* At each fusion stage, a nonlinear spatial guide combines decoder semantic features, 16-channel detail features, and 16-channel projected question features. Fusion adds a gated residual with a zero-initialized scalar weight. At 768 resolution, the residual refiner concatenates 41 channels (decoder features, upsampled 384-scale logits, 768-scale detail features, and projected question features), reduces them to 16 channels through a 3x3 block, and predicts a 1-channel signed logit correction over the full grid. Because the final correction head is zero-initialized, initial `logits768` match bilinearly upsampled `logits384`.
* Auxiliary mask heads at 96 and 192 are used only during training, while the 384 head feeds the 768 refiner. Evaluation exports only the final 768 logits. Both training and inference use only the image and question.

The non-backbone parameter count is **6,649,101**, including text projection and cross-attention.

## Objective

Each mask head is supervised with `BCE + 0.5 * soft Dice`:

```text
L = L768 + 0.3 * L384 + 0.2 * L192 + 0.1 * L96
```

The main 768 target uses nearest-neighbor resized binary masks. Auxiliary targets at 96, 192, and 384 use area interpolation from the 768 target to preserve fractional coverage on small regions. Loss computations accumulate in float32 under CUDA FP16 AMP.

Optional `--config` JSON (showing default values):

```json
{
  "overlap_loss": "dice",
  "overlap_weight": 0.5,
  "fp_weight": 0.7,
  "aux96_weight": 0.1,
  "aux192_weight": 0.2,
  "aux384_weight": 0.3
}
```

`fp_weight` applies only when `overlap_loss="tversky"`, which uses numerator `TP + 1` and denominator `TP + fp_weight * FP + (1 - fp_weight) * FN + 1`. Setting an auxiliary weight to `0` keeps a zero-multiplied term in the computation graph for DDP reducer consistency.

## Kaggle setup and 20-epoch run

Enable Internet and two T4 GPUs. Clone the branch and sync the locked dependencies before launching `torchrun`:

```bash
git clone --single-branch --branch exp/question-guided-detail-pyramid https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
cd VizWiz-VQA-Grounding
uv sync --frozen
```

With `DATA_ROOT` pointing to the VizWiz dataset directory, run the multi-GPU preflight check:

```bash
uv run --frozen torchrun --standalone --nproc_per_node=2 preflight.py \
  --data-root "$DATA_ROOT" --batch-size 2 --num-workers 0 --steps 3 \
  --gradient-checkpointing
```

Preflight verifies input shapes, forward/backward passes, frozen and trainable parameter gradients, AMP, and DDP synchronization without writing checkpoints or history files. If an initial FP16 step overflows, `GradScaler` backs off and retries the same batch (up to 8 scale reductions per batch); backoff events are logged and excluded from completed step counts, while missing gradients, nonfinite forward losses, or persistent overflow raise an error.

Run the 20-epoch training job:

```bash
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py \
  --data-root "$DATA_ROOT" \
  --output-dir outputs \
  --num-epochs 20 \
  --batch-size 2 \
  --gradient-accumulation-steps 2 \
  --gradient-checkpointing \
  --num-workers 2 \
  --lr 0.00001 \
  --seed 42 \
  --save-every 10
```

`--batch-size 2` is the **global microbatch** (1 image per GPU across 2 GPUs), and `--gradient-accumulation-steps 2` yields an effective global batch size of 4 with exact sample weighting on the final partial accumulation window. If GPU memory allows, `--batch-size 4 --gradient-accumulation-steps 1` produces the same effective batch size of 4.

Default training uses Adam (`lr=1e-5`, zero weight decay, no LR scheduler), gradient clipping at global norm 1.0, CUDA FP16 `GradScaler`, deterministic epoch-keyed seeding, and nonreentrant activation checkpointing. Periodic validation is disabled by default and can be enabled with `--validate-every`. Existing output files in `--output-dir` cause an immediate error to prevent accidental overwrites.

Full resumable checkpoints are written to `outputs/checkpoint_epoch10.pt` and `outputs/checkpoint_epoch20.pt` alongside final raw model weights. Full checkpoints store the model, optimizer, AMP scaler, resolved configuration, architecture metadata, source file hashes, and environment details. Use `--resume-checkpoint` to resume a run from this branch with matching batch size, accumulation steps, learning rate, seed, and GPU count, where `--num-epochs` sets the total target epoch count.

## Evaluate and diagnose

Run evaluation in a single process. Both image views are constructed from the original image, and predicted 768-scale logits are bilinearly resized to native ground-truth resolution **before** thresholding at zero.

```bash
uv run --frozen python eval.py \
  --checkpoint outputs/checkpoint_epoch20.pt --dataset val \
  --data-root "$DATA_ROOT" --output-dir outputs/eval_val_ep20 \
  --batch-size 1 --num-workers 2 --device cuda

uv run --frozen python diagnose.py \
  --predictions outputs/eval_val_ep20 --data-root "$DATA_ROOT" \
  --output-dir outputs/diagnostic_ep20
```

Run the same commands for `checkpoint_epoch10.pt` and for the `test` split when test ground-truth masks are available. Select checkpoints and hyperparameters using validation metrics only. `diagnose.py` reads the exported PNG masks on CPU, cross-checks per-image IoU against `metrics.json`, and breaks down native-resolution precision, recall, predicted-to-GT area ratios, and GT size buckets. Output directories for `eval.py` and `diagnose.py` must be new or empty.

## Encoder freezing and optimization

Omitting the options below keeps the branch's default training configuration: Adam with LR `1e-5`, zero weight decay, constant LR, and the default encoder freezing policy. Encoder freezing is opt-in, while the detail CNN and decoder always remain trainable. Model state-dict keys and evaluation preprocessing are unchanged.

| Option | Values / default |
| --- | --- |
| `--freeze-backbone [BOOL]` | Set both pretrained encoders; omitted preserves existing policy |
| `--freeze-vision-encoder [BOOL]` | Set vision encoder only |
| `--freeze-text-encoder [BOOL]` | Set text encoder only |
| `--optimizer` | `adam` (default), `adamw` |
| `--lr` | Main learning rate; default `0.00001` |
| `--backbone-lr` | Optional separate LR for unfrozen encoders; default uses `--lr` |
| `--weight-decay` | Nonnegative; default `0`. Biases, norm vectors and token/position embeddings receive no decay when parameter groups are enabled |
| `--lr-scheduler` | `none` (default), `cosine`, `linear`, `poly` |
| `--warmup-ratio` | Fraction of planned optimizer updates, in `[0, 1)`; default `0` |
| `--min-lr-ratio` | Final LR / starting LR for decay schedules; default `0.1` |
| `--poly-power` | Positive exponent for `poly`; default `0.9` |
| `--config` | Flat JSON with training options; optional |

`BOOL` accepts `True` or `False`; passing a freeze flag without a value sets it to `True`. The negative aliases `--no-freeze-backbone`, `--no-freeze-vision-encoder`, and `--no-freeze-text-encoder` set the corresponding flag to `False`. Within the same configuration source, a specific encoder flag overrides `freeze_backbone`. Precedence follows **JSON config > explicit CLI > branch defaults**, so a JSON shorthand also overrides component flags passed on the CLI. Recipe and loss settings can share the same JSON file, and unrecognized keys raise an error immediately. Run `train.py --help` to see all branch-specific flags.

Example CLI flags to freeze the backbone and use AdamW with a cosine schedule:

```bash
--freeze-backbone True \
--optimizer adamw \
--lr 0.0001 \
--weight-decay 0.01 \
--lr-scheduler cosine \
--warmup-ratio 0.05 \
--min-lr-ratio 0.1
```

To freeze only the vision encoder while keeping the default text policy, pass `--freeze-vision-encoder True`. To train both encoders explicitly, pass `--freeze-backbone False`. Branches that rely on a fixed reference stage (such as gain-guided refinement) reject encoder unfreezing. Unused CLIP/SigLIP pooled-output parameters remain frozen when token encoders are unfrozen.

Equivalent JSON configuration (`--config path/to/train.json`):

```json
{
  "freeze_backbone": true,
  "optimizer": "adamw",
  "lr": 0.0001,
  "weight_decay": 0.01,
  "lr_scheduler": "cosine",
  "warmup_ratio": 0.05,
  "min_lr_ratio": 0.1
}
```

Frozen encoders stay in `eval` mode during training and are excluded from the optimizer. Learning-rate schedules step once per completed optimizer update (accounting for gradient accumulation) and do not advance when AMP skips an update due to gradient overflow. Checkpoints record the resolved optimization config, trainable parameter layout, and scheduler state; resuming restores those settings and rejects conflicting overrides. When resuming a scheduled run, keep `--num-epochs`, dataloader length, and gradient accumulation unchanged so the LR schedule trajectory stays consistent.

Existing checkpoints remain compatible with evaluation and legacy constant-LR resume. Changing encoder freezing, optimizer type, weight decay, or LR scheduling requires starting a new run (or using weights-only initialization where supported).

## Foreground global IoU

`eval.py` writes both dataset-level IoU metrics to `metrics.json` under `summary`, computed from the final native-resolution binary masks:

- `mean_iou`: arithmetic mean of per-image foreground IoUs (weighting each image equally).
- `global_iou`: total foreground intersection pixels divided by total foreground union pixels across the evaluated split.
- `total_intersection` and `total_union`: exact integer pixel counts across the split.

Values are reported as fractions in `[0, 1]` (multiply by 100 for percentages). Logits are resized to each image's original resolution before thresholding at zero, and an empty union evaluates to an IoU of 1.0.

`diagnose.py` also records `global_iou` under `overall` and `by_gt_area` in `summary.json` directly from the exported PNG masks and native ground truth on CPU:

```bash
uv run diagnose.py --predictions outputs/eval_val_ep20 \
    --data-root /path/to/vizwiz --output-dir outputs/diagnostic_global_iou_ep20
```

## Continue the original pyramid: epoch 20 to 40

The workflow below extends the vision-trainable, text-frozen pyramid run on `exp/question-guided-detail-pyramid` from epoch 20 to epoch 40 using the original natural sampling and BCE + Dice objective.

### Cell R1: Fresh Kaggle session

Enable Internet and two GPUs, and attach the dataset containing the full `checkpoint_epoch20.pt` (the weights-only `model_final_epoch20.pt` does not contain optimizer or GradScaler state). Start with a clean `outputs` directory.

```python
!git clone --single-branch --branch exp/question-guided-detail-pyramid https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen
```

If the repository is already cloned on this branch, run `!git pull --ff-only` and `!uv sync --frozen`.

### Cell R2: Read the saved training settings

Update `RESUME_CKPT` if your Kaggle input mount path differs. This cell inspects the checkpoint metadata on CPU and writes `continuation.json`.

```python
import gc
import json
import os
from pathlib import Path
from utils import read_checkpoint, checkpoint_model_kwargs

os.environ['DATA_ROOT'] = '/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz'
os.environ['RUN_ROOT'] = 'outputs'
os.environ['RUN_NAME'] = 'question_guided_detail_pyramid_ep40_seed42'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['RESUME_CKPT'] = '/kaggle/input/datasets/khoahoang0512/vizwiz-question-guided-detail-pyramid/checkpoint_epoch20.pt'

checkpoint_path = Path(os.environ['RESUME_CKPT'])
if not checkpoint_path.is_file():
    print('Available checkpoints:')
    for candidate in sorted(Path('/kaggle/input').rglob('*.pt')):
        print(candidate)
    raise FileNotFoundError(f'Set RESUME_CKPT to the original pyramid checkpoint: {checkpoint_path}')

saved = read_checkpoint(checkpoint_path)
checkpoint_model_kwargs(saved)
assert saved.get('epoch') == 20, 'This continuation starts at epoch 20.'
assert 'optimizer_state_dict' in saved, 'Use a full checkpoint, not final raw weights.'
run = saved['run_config']
controls = saved.get('training_controls') or {}
optimization = controls.get('config', run.get('optimization_config', {}))
assert optimization.get('lr_scheduler', 'none') == 'none', 'This guide extends a constant-LR run only.'
assert optimization.get('warmup_ratio', 0.0) == 0.0, 'Do not extend a fixed warmup/schedule budget here.'
assert run['world_size'] == 2, 'Use the same GPU count as the original run.'

for variable, value in {
    'RESUME_BATCH': run['batch_size'],
    'RESUME_ACCUMULATION': run['gradient_accumulation_steps'],
    'RESUME_LR': run['lr'],
    'RESUME_SEED': run['seed'],
}.items():
    os.environ[variable] = str(value)

root = Path(os.environ['RUN_ROOT'])
root.mkdir(parents=True, exist_ok=True)
assert not (root / 'history.jsonl').exists() and not any(root.glob('*.pt')), 'Use a fresh session/output directory.'
metadata = {
    'resume_checkpoint': str(checkpoint_path),
    'start_epoch': saved['epoch'],
    'target_epoch': 40,
    'batch_size': run['batch_size'],
    'gradient_accumulation_steps': run['gradient_accumulation_steps'],
    'lr': run['lr'],
    'seed': run['seed'],
    'world_size': run['world_size'],
    'training_config': run['training_config'],
    'optimization_config': optimization,
    'gradient_checkpointing': run.get('gradient_checkpointing', False),
}
(root / 'continuation.json').write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata, indent=2))
del saved, run, controls, optimization
gc.collect()
```

Resuming restores the saved loss configuration, encoder freezing policy, optimizer state, and gradient-checkpointing setting directly from `checkpoint_epoch20.pt`.

### Cell R3: Train epochs 21 to 40

```bash
%%bash
set -euo pipefail
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$RUN_ROOT" \
    --resume-checkpoint "$RESUME_CKPT" \
    --num-epochs 40 \
    --batch-size "$RESUME_BATCH" \
    --gradient-accumulation-steps "$RESUME_ACCUMULATION" \
    --lr "$RESUME_LR" \
    --seed "$RESUME_SEED" \
    --num-workers 2 \
    --save-every 10 \
    --validate-every 5 \
    --val-batch-size 1 \
    --no-save-final-weights
```

Setting `--num-epochs 40` runs 20 additional epochs (epochs 21 through 40), restoring the model, optimizer, AMP scaler, and epoch-keyed RNG/sampler ordering from epoch 20. Validation runs every 5 epochs in FP32 with batch size 1 per GPU and exports masks and metrics at epochs 25, 30, 35, and 40 without padding or duplicating samples across ranks. Full checkpoints are saved at epochs 30 and 40 under `outputs/`, while `--no-save-final-weights` skips writing a duplicate raw-weights file at the end. Keep the new `history.jsonl` (epochs 21 to 40) alongside the original epochs 1 to 20 history.

### Cell R4: Validation and size-group diagnostics

```bash
%%bash
set -euo pipefail
for epoch in 25 30 35 40; do
    uv run --frozen python diagnose.py \
        --predictions "$RUN_ROOT/eval_val_ep${epoch}" \
        --data-root "$DATA_ROOT" \
        --output-dir "$RUN_ROOT/diagnostic_ep${epoch}"
done
```

Because Cell R3 already exported validation predictions for epochs 25, 30, 35, and 40, Cell R4 runs CPU diagnostics directly on those folders. Compare `mean_iou`, `global_iou`, and the three GT-size buckets against epoch 20 to select a checkpoint on validation.

### Cell R5: Test exports at epochs 30 and 40

```bash
%%bash
set -euo pipefail
for epoch in 30 40; do
    uv run --frozen python eval.py \
        --checkpoint "$RUN_ROOT/checkpoint_epoch${epoch}.pt" \
        --dataset test --data-root "$DATA_ROOT" \
        --batch-size 4 --num-workers 2 --device cuda \
        --output-dir "$RUN_ROOT/eval_test_ep${epoch}"
done
```

Standalone test evaluation uses batch size 4 (or 1 if memory is constrained). Local test scoring requires test annotations and ground-truth masks; skip this step if test masks are unavailable.

### Cell R6: Download results without checkpoint files

```python
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(os.environ['RUN_ROOT'])
archive = Path('/kaggle/working') / f"{os.environ['RUN_NAME']}.zip"
with ZipFile(archive, 'w', ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() not in {'.pt', '.tmp', '.zip'}:
            bundle.write(path, path.relative_to(root))
print('Download results:', archive)
print('Keep full .pt checkpoints separately for later resume.')
```

If the session stops before epoch 40, resume from the latest saved full checkpoint (`checkpoint_epoch30.pt`) with `--num-epochs 40` and the same batch size, accumulation steps, learning rate, seed, and GPU count.

### Optional validation and storage controls

- `--validate-every N`: Exports native-resolution validation masks and metrics every `N` completed epochs (`0` disables in-training validation, the default).
- `--val-batch-size`: Per-GPU validation batch size during training (defaults to `1` since optimizer state stays in VRAM).
- `--save-every`: Interval for saving full resumable checkpoints.
- `--no-save-final-weights`: Skips writing the extra raw-weights file `model_final_epoch<N>.pt` while keeping scheduled full checkpoints.
