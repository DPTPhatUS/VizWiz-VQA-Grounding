# Question-guided high-resolution detail pyramid

Branch: `exp/question-guided-detail-pyramid`, created from
`exp/spatial-detail-decoder` at `63977d2`. This is a new experiment requiring
fresh training from pretrained CLIP, not a checkpoint-compatible refactor.

The hypothesis is that high-resolution local features, question-conditioned
fusion, and signed mask refinement can improve small-region precision. The
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

The implemented non-backbone parameter count is **6,649,101**, including text
projection and cross-attention. This differs from the earlier 6.09M sketch
because skip conditioning, Q-guided detail fusion and auxiliary heads changed.

## Objective

The default is `BCE + 0.5 * soft Dice` for each head:

```text
L = L768 + 0.3 * L384 + 0.2 * L192 + 0.1 * L96
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

`fp_weight` is used only by `overlap_loss="tversky"`: its denominator is
`TP + fp_weight*FP + (1-fp_weight)*FN + 1`, with numerator `TP + 1`.
This is an optional separate ablation, not enabled in the default Dice run.
Zero auxiliary weights retain zero-valued graph dependencies for DDP.

## Kaggle setup and 20-epoch run

Enable Internet and two T4 GPUs. Clone the branch, change into the repository,
and install the locked environment before launching multiple processes:

```bash
git clone --single-branch --branch exp/question-guided-detail-pyramid https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
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

## Encoder freezing and optimization

Omitting the options below preserves this branch's existing training behavior:
Adam, LR `1e-5`, zero weight decay, no LR schedule, and the branch's original
encoder freezing policy. Freezing is opt-in; the detail CNN and decoder remain
trainable. No model/state-dict keys or evaluation preprocessing change.

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

`BOOL` accepts `True` or `False`; a freeze flag without a value means `True`.
The aliases `--no-freeze-backbone`, `--no-freeze-vision-encoder`, and
`--no-freeze-text-encoder` mean `False`. An explicit component setting overrides
`freeze_backbone` within the same source. **JSON config > explicit CLI > branch
defaults**: a config shorthand also overrides component flags on the CLI.
Existing recipe/loss settings can share the same JSON; unsupported keys fail
early. Existing branch-specific flags remain available through `train.py --help`.

Append, for example, to this branch's existing training command:

```bash
--freeze-backbone True \
--optimizer adamw \
--lr 0.0001 \
--weight-decay 0.01 \
--lr-scheduler cosine \
--warmup-ratio 0.05 \
--min-lr-ratio 0.1
```

This is a configurable experiment example, not a validated best hyperparameter
setting. To freeze only vision while retaining the text policy, append
`--freeze-vision-encoder True`. To explicitly train both encoders, use
`--freeze-backbone False`. Models with a deliberately fixed reference stage
(gain-guided refinement) reject encoder unfreezing to preserve that algorithm.
Unused CLIP/SigLIP pooled-output parameters stay frozen when unfreezing token encoders.

Equivalent JSON (`--config path/to/train.json`):

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

Frozen encoders stay in eval mode during training and are excluded from the
optimizer when a freezing option is set. LR schedules advance after successful
optimizer updates, including gradient accumulation; AMP overflow skips do not
advance them. Checkpoints save the resolved optimization settings, trainable
parameter layout and schedule progress. Resume restores these settings and
rejects conflicting options. For scheduled runs, keep the original total
`--num-epochs`, loader length and accumulation so the LR trajectory is unchanged.

Old checkpoints still load for evaluation with the same architecture and
weights. Full legacy resume keeps its original optimizer layout and constant
LR behavior; changing freezing, optimizer, decay or enabling a new LR schedule
requires a new run (or an explicit weights-only initialization where supported).
Resume retains each branch's existing RNG/data-order policy; legacy branches
that do not save or reconstruct RNG state do not guarantee bitwise replay.
No freezing/optimization change retroactively changes a previously trained
checkpoint or its reported results.
