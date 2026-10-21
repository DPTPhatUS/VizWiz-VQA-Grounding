# Token-aware detail pyramid with region ranking

Branch: `exp/pyramid-token-region-ranking`, created from
`exp/question-guided-detail-pyramid` at `3eaa010`. This is a new experiment requiring
fresh training from pretrained CLIP, not a checkpoint-compatible refactor.

The hypothesis is that full-token interaction and within-image hard-negative
region supervision can improve evidence selection beyond high-resolution detail alone. The
implementation and synthetic checks do not establish an accuracy improvement.

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

* Both RGB views come directly from the same original image, in `[0,1]`. The
  model applies CLIP mean/std to both. No crop/flip/color augmentation is added.
  Aspect ratios are warped to squares, as in the parent CLIP experiments.
* CLIP is `openai/clip-vit-large-patch14-336`. Text is frozen and kept in eval
  mode. Vision is trainable except pooled-CLS `post_layernorm`, which patch
  grounding does not use. Its checkpoint keys are retained.
* Vision skips are hidden states 4/7/9. They and the final bottleneck have shape
  `(B,1024,24,24)`. Each skip has its own nonlinear compression and FiLM head;
  Q reduction is shared. FiLM heads start at zero (identity modulation).
* Decoder channels are `128 -> 64 -> 32 -> 16 -> 16`. Every upsampling block is
  bilinear resize followed by **one standard 3x3 convolution**, GroupNorm and
  GELU. CLIP skip-fusion and detail-fusion blocks remain depthwise separable.
* CNN encoder channels at `768/384/192/96/48` are `8/16/32/64/128`. Each level
  has two standard 3x3/GN/GELU blocks. Max pooling downsamples. Top-down pyramid
  levels use 1x1 lateral projection, addition and 3x3/GN/GELU smoothing.
* Each spatial guide is a nonlinear function of semantic features, 16-channel
  detail features and 16-channel projected Q. Fusion adds a residual whose
  scalar starts at zero. The refiner reads 41 concatenated channels, reduces
  to 16 with a standard 3x3 block, then predicts a one-channel correction.
  The final correction head starts at zero, so initial logits768 exactly match
  bilinear-resized logits384. Refinement is not restricted to a thresholded mask.
* Mask heads at 96/192 are training-only; the 384 mask feeds the refiner. Only
  final 768 logits are exported for evaluation. There is no latent answer,
  teacher, oracle view, answer reader, or GT answer input on this branch.

## Token interaction and region ranking

The parent architecture above is retained. After detail fusion at **96x96**, a
residual token-pixel module projects `(B,32,96,96)` to 64-dimensional pixel
queries. All CLIP Q tokens are projected from 768 to 64 dimensions; a four-head
cross-attention reads them as keys/values. Padding is excluded; SOT/EOT remain
available, like the existing bottleneck cross-attention. Pixel queries and text
keys/values each have LayerNorm. A 64-to-32 projection and zero-initialized
residual scale preserve the parent feature at initialization.

During training, a shared nonlinear scorer reads pooled **post-interaction F96**
and pooled Q. The positive support is the area-resized GT mask. Negatives are
translations of exactly the same soft support: no wrap, clipping, area change,
or overlap with the GT's one-cell dilation is allowed. A 5x5 placement grid and
eight high-prediction centers supply candidates; the three highest detached
prediction-confidence candidates are selected. These are task-negative regions
outside the annotated GT, not guaranteed semantically different objects.

Some GT regions (especially large/dispersed ones) have no valid translation.
They retain all segmentation supervision and contribute **zero ranking loss**.
The scorer sees no area/coordinates. Equal support area prevents a direct
small-region-size shortcut, but does not rule out appearance/context shortcuts.

```text
Detail fusion96 -> token-pixel interaction -> F96 -> decoder192/384 -> refiner768
                                              |
                  GT mask -> positive pooling + translated negative pooling
                                              |
                             shared region/Q scorer -> margin ranking loss
```

Mining uses detached final logits and GT. Ranking gradients update the scorer
and the feature-producing network up to F96; they do not flow through mining or
directly update downstream decoder192/384/refiner. No segmentation logits are
thresholded in the differentiable path. Scorer/mining do not execute in eval;
inference still requires only RGB image and Q. The scorer's small weights remain
in the checkpoint even though mask inference does not call it.

Parameters outside CLIP: **6,770,798** in the full model, including 70,305 for
token interaction and 51,392 for the training-only ranker. Parent modules keep
identical initial weights across same-seed variants because new modules are
initialized after all parent modules. These counts are from implementation;
GPU memory/time and accuracy are not inferred from parameter count.

## Objective

The default is `BCE + 0.5 * soft Dice` for each head:

```text
L = L768 + 0.3 * L384 + 0.2 * L192 + 0.1 * L96 + effective_ranking_weight * Lrank
Lrank = mean_images(mean_valid_pairs(relu(margin - score_positive + score_negative)))
```

Main targets are nearest-neighbor binary masks at 768. Auxiliary targets use
area interpolation from this target to preserve fractional coverage of tiny
regions. This cannot recover an object already lost during the native-to-768
mask conversion. Loss calculations accumulate in float32 under CUDA FP16 AMP.
Weights above are initial experimental settings, not tuned coefficients.

Optional `--config` JSON (all defaults shown):

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

The experiment adds four JSON keys:

```json
{
  "token_interaction": true,
  "ranking_weight": 0.1,
  "ranking_margin": 0.2,
  "ranking_warmup_epochs": 2
}
```

Ranking weight is zero for epochs1/2, then 0.1 from epoch3. Token interaction is
trained through segmentation from the start. Warmup does not pretrain the
scorer. Margin/weight/warmup are initial settings, not validated hyperparameters.
Invalid images contribute zero to the batch mean; reduction is deliberately
not normalized by a batch-dependent count of valid images, preserving DDP and
gradient-accumulation equivalence. Logs include ranking coverage, gap and raw/
weighted loss. Positive ranking weight constructs the ranker; zero omits it.

For controlled fresh runs on this branch:

| Variant | token_interaction | ranking_weight |
|---|---|---:|
| Parent-architecture control | false | 0.0 |
| Token interaction only | true | 0.0 |
| Full experiment (default) | true | 0.1 |

Use the same remaining training settings. Resume requires the same config and
architecture; do not change variants in the middle of a run. Parent experiment
checkpoints are rejected, including raw weights without the new version marker.

`fp_weight` is used only by `overlap_loss="tversky"`: its denominator is
`TP + fp_weight*FP + (1-fp_weight)*FN + 1`, with numerator `TP + 1`.
This is an optional separate ablation, not enabled in the default Dice run.
Zero auxiliary weights retain zero-valued graph dependencies for DDP.

## Kaggle setup and 20-epoch run

Enable Internet and two T4 GPUs. Clone the branch, change into the repository,
and install the locked environment before launching multiple processes:

```bash
git clone --single-branch --branch exp/pyramid-token-region-ranking https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
cd VizWiz-VQA-Grounding
uv sync --frozen
```

With `DATA_ROOT` pointing at the VizWiz folder, run the short real-backbone check:

```bash
uv run --frozen torchrun --standalone --nproc_per_node=2 preflight.py \
  --data-root "$DATA_ROOT" --batch-size 2 --num-workers 0 --steps 3 \
  --gradient-checkpointing
```

This checks real input shapes, several backward iterations, frozen/trainable
gradients, AMP and DDP; it writes no checkpoints or training history. CPU checks
cannot replace this GPU check. It does not establish accuracy or full-session runtime.
Pass the same `--config` to preflight/train for non-default variants. Preflight
exercises the active ranking objective rather than the warmup coefficient.
Initial FP16 gradient overflow is handled by GradScaler backoff and a retry of
the same batch (at most eight scale reductions per batch). Backoff events are
printed and do not count as successful updates. Missing gradients, nonfinite
forward loss, FP32 nonfinite gradients or persistent overflow still fail.

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

Batch size is **global microbatch**: one image/GPU here. Accumulation gives
effective global batch 4, including correct sample weighting in the final
partial window. If memory permits, `--batch-size 4` with accumulation 1 also
gives effective batch 4. Memory/runtime have not been measured on Kaggle T4.

Adam, no weight decay or LR scheduler, clipping at global norm 1, CUDA FP16
GradScaler, deterministic epoch-keyed seeding and sampler shuffling follow the
simple existing trainer. Activation checkpointing uses nonreentrant recomputation;
it changes memory/time, not the intended forward computation. No per-epoch
validation is run. Output collisions fail rather than overwrite another run.

Full checkpoints are `outputs/checkpoint_epoch10.pt` and `checkpoint_epoch20.pt`.
Raw final weights are also saved. Full checkpoints include optimizer/scaler,
resolved config, architecture, source hashes and environment version. Resume
with `--resume-checkpoint` from this experiment only; `--num-epochs` is the
total target epoch. Batch/accumulation/LR/seed/world size must match.

## Evaluate and diagnose

Run each evaluation with one process. Both views are reconstructed from the
original image, and logits are resized to native GT size **before** thresholding.
The metric is mean per-image foreground IoU, not foreground/background class mIoU.

```bash
uv run --frozen python eval.py \
  --checkpoint outputs/checkpoint_epoch20.pt --dataset val \
  --data-root "$DATA_ROOT" --output-dir outputs/eval_val_ep20 \
  --batch-size 1 --num-workers 2 --device cuda

uv run --frozen python diagnose.py \
  --predictions outputs/eval_val_ep20 --data-root "$DATA_ROOT" \
  --output-dir outputs/diagnostic_ep20
```

Repeat evaluation for epoch10 and for test if native test GT masks are available.
Select checkpoints/hyperparameters by validation, never by the test results.
Diagnostic reads exported masks on CPU, verifies each mask against stored IoU,
and reports native precision/recall, predicted/GT area ratio and size groups.
Evaluation/diagnostic output directories must be new or empty.

Personal notebook instructions and local tests remain under ignored `personal/`.
Older answer-latent probes/checkpoints belong to their original branches.
