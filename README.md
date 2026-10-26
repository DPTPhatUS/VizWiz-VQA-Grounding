# Token-aware detail pyramid with region ranking

Branch: `exp/pyramid-token-region-ranking`, created from `exp/question-guided-detail-pyramid` at `3eaa010`. This experiment adds full-token cross-attention at 96x96 resolution and within-image hard-negative region ranking to the high-resolution detail pyramid, trained fresh from pretrained CLIP weights.

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

## Token interaction and region ranking

Building on the parent pyramid, a residual token-pixel interaction module sits after detail fusion at **96x96** resolution. It projects the `(B, 32, 96, 96)` feature map to 64-dimensional pixel queries and projects all non-padding CLIP question tokens (including SOT/EOT) from 768 to 64 dimensions as keys and values for 4-head cross-attention. Both pixel queries and text keys/values use LayerNorm. A 64-to-32 output projection and a zero-initialized residual scale preserve the parent features at initialization.

During training, a shared nonlinear scorer evaluates pooled **post-interaction F96** features alongside the pooled question vector. The positive region support is the area-interpolated ground-truth mask at 96x96. Negative supports are rigid in-bounds translations of that exact soft support mask that do not wrap, clip, alter support area, or overlap the 1-cell dilated ground-truth mask. Candidate translations are drawn from a 5x5 placement grid plus 8 high-confidence prediction peaks, and the 3 candidates with the highest detached prediction confidence are selected as hard negatives.

If a ground-truth mask is too large or dispersed to admit any valid non-overlapping translation, it still receives full segmentation supervision and contributes **zero ranking loss**. The scorer receives only pooled feature vectors and no area or coordinate inputs, and translating the exact positive support keeps positive and negative support areas identical.

```text
Detail fusion96 -> token-pixel interaction -> F96 -> decoder192/384 -> refiner768
                                              |
                  GT mask -> positive pooling + translated negative pooling
                                              |
                             shared region/Q scorer -> margin ranking loss
```

Hard-negative mining uses detached final logits and ground-truth masks. Ranking gradients update the scorer and the network up to `F96` without flowing through the mining step or downstream 192/384/768 decoder stages. During evaluation, the scorer and mining path are bypassed so inference uses only the RGB image and question.

Non-backbone parameter count: **6,770,798** total (including 70,305 parameters for token interaction and 51,392 for the training-only region scorer). Because the new modules are initialized after all parent modules, shared parent layers receive identical initial weights across same-seed ablation variants.

## Objective

Each mask head uses `BCE + 0.5 * soft Dice`, combined with the margin ranking loss:

```text
L = L768 + 0.3 * L384 + 0.2 * L192 + 0.1 * L96 + effective_ranking_weight * Lrank
Lrank = mean_images(mean_valid_pairs(relu(margin - score_positive + score_negative)))
```

The main 768 target uses nearest-neighbor resized binary masks, while auxiliary targets at 96, 192, and 384 use area interpolation from the 768 target to preserve fractional coverage on small regions. Loss calculations accumulate in float32 under CUDA FP16 AMP.

Base segmentation `--config` JSON options (defaults shown):

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

This branch adds four configuration keys:

```json
{
  "token_interaction": true,
  "ranking_weight": 0.1,
  "ranking_margin": 0.2,
  "ranking_warmup_epochs": 2
}
```

With the default settings, `effective_ranking_weight` is `0.0` during epochs 1 and 2 and switches to `0.1` starting at epoch 3, while the token-interaction block trains through the segmentation losses from epoch 1. Images without valid negative translations contribute `0.0` to the batch mean; dividing by the full batch size rather than the count of valid images preserves exact equivalence across DDP ranks and gradient-accumulation steps. Training logs record ranking coverage, score gap, and raw/weighted ranking losses. Setting `ranking_weight > 0` builds the scorer head; setting `ranking_weight = 0` omits it.

Controlled ablation variants on this branch:

| Variant | token_interaction | ranking_weight |
|---|---|---:|
| Parent-architecture control | false | 0.0 |
| Token interaction only | true | 0.0 |
| Full experiment (default) | true | 0.1 |

Keep all other training settings matched across variants. Resuming requires the same configuration and architecture, and checkpoints from the parent branch are rejected.

`fp_weight` applies only when `overlap_loss="tversky"`, which uses numerator `TP + 1` and denominator `TP + fp_weight * FP + (1 - fp_weight) * FN + 1`. Setting an auxiliary weight to `0` keeps a zero-multiplied term in the computation graph for DDP reducer consistency.

## Kaggle setup and 20-epoch run

Enable Internet and two T4 GPUs. Clone the branch and sync the locked dependencies before launching `torchrun`:

```bash
git clone --single-branch --branch exp/pyramid-token-region-ranking https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
cd VizWiz-VQA-Grounding
uv sync --frozen
```

With `DATA_ROOT` pointing to the VizWiz dataset directory, run the multi-GPU preflight check:

```bash
uv run --frozen torchrun --standalone --nproc_per_node=2 preflight.py \
  --data-root "$DATA_ROOT" --batch-size 2 --num-workers 0 --steps 3 \
  --gradient-checkpointing
```

Preflight verifies input shapes, forward/backward passes, frozen and trainable parameter gradients, AMP, and DDP synchronization without writing checkpoints or history files. When running a non-default variant, pass the same `--config` file to both `preflight.py` and `train.py` (preflight exercises the active ranking objective directly rather than waiting for warmup). If an initial FP16 step overflows, `GradScaler` backs off and retries the same batch (up to 8 scale reductions per batch); backoff events are logged and excluded from completed step counts, while missing gradients, nonfinite forward losses, or persistent overflow raise an error.

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

Default training uses Adam (`lr=1e-5`, zero weight decay, no LR scheduler), gradient clipping at global norm 1.0, CUDA FP16 `GradScaler`, deterministic epoch-keyed seeding, and nonreentrant activation checkpointing. No per-epoch validation runs during training, and existing files in `--output-dir` cause an immediate error to prevent accidental overwrites.

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
