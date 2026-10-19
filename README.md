# Latent-answer grounding: Spatial Evidence-Decoder Interaction (`exp/spatial-evidence-decoder`)

This branch extends the latent-answer grounding architecture with **Spatial Evidence-Decoder Interaction** (inspired by DaVI / dual visual-evidence principles adapted for dense segmentation). Alongside the shared CLIP image/question encoders, latent answer predictor, and conditioned mask decoder, it introduces direct, continuous 2D spatial compatibility gating between dense visual representations and answer evidence ($A$ or predicted $A'$).

Student inference requires only image + question. CLIP input is fixed at 336 pixels.

## Architecture: Spatial Evidence Interaction

### 1. Motivation and Mechanism

In the baseline architecture, answer conditioning in the decoder relies primarily on global scalar affine modulation (FiLM) inside the `SkipConditioner`. While effective for global feature recalibration, it lacks direct spatial localization of answer evidence across the decoder hierarchy.

`SpatialEvidenceInteraction` introduces continuous 2D position-wise compatibility gating:
1. **L2-Normalized Projection**: Dense visual features $V \in \mathbb{R}^{B \times C_v \times H \times W}$ and answer evidence vector $A \in \mathbb{R}^{B \times D_t}$ (or tokens $(B, K, D_t)$) are projected into a common space and normalized:
   $$\widehat{V} = \frac{\text{Conv}_{1\times 1}(V)}{\|\text{Conv}_{1\times 1}(V)\|_2}, \quad \widehat{A} = \frac{\text{Linear}(A)}{\|\text{Linear}(A)\|_2}$$
2. **Continuous 2D Spatial Gate**:
   $$S(b, 1, h, w) = \sum_c \widehat{V}(b, c, h, w) \cdot \widehat{A}(b, c)$$
   $$\text{Gate}(b, 1, h, w) = \sigma(\text{gate\_scale} \cdot S + \text{gate\_bias})$$
   Position-wise sigmoid gating avoids the single-token softmax degeneracy inherent in standard cross-attention.
3. **Depthwise-Separable Feature Refinement & Zero-Init Residual**:
   $$\Delta V = \text{Refine}([V \odot \text{Gate}, \text{Gate}])$$
   $$V_{\text{out}} = V + \text{residual\_scale} \cdot \Delta V$$
   Where `residual_scale` is initialized to $0.0$, guaranteeing **exact numerical identity** to the baseline at step 0.
4. **Question-Only View Bypass**: When `answer_vec` is `None` (in question-only evaluation or distillation reference views), the interaction is cleanly bypassed.

### 2. Multi-Level Decoder Hierarchy & Configurable Upsampling Stages

The spatial interaction is applied hierarchically across the decoder:
- **Bottleneck Stage**: Always enabled after skip fusion at the lowest resolution ($24 \times 24$, 128 channels).
- **Progressive Upsampling Stages**: Modularized and configurable across 4 upsampling resolutions:
  - **Stage 0**: $48 \times 48$, 64 channels
  - **Stage 1**: $96 \times 96$, 32 channels
  - **Stage 2**: $192 \times 192$, 16 channels
  - **Stage 3**: $384 \times 384$, 16 channels

#### Configuration & Defaults
- **Default configuration**: Strictly stages `(0, 1)` ($48 \times 48$ and $96 \times 96$) alongside the bottleneck.
- **Arbitrary stage configuration**: Any subset of stages (or all 4 stages `(0, 1, 2, 3)`) can be configured:
  ```python
  from models.model import GroundingModel

  # Default: stages (0, 1) -> bottleneck + 2 upsampling stages
  model_default = GroundingModel()

  # All 4 stages: bottleneck + 4 upsampling stages (up to 384x384)
  model_4stages = GroundingModel(spatial_upsample_stages=(0, 1, 2, 3))

  # Bottleneck only:
  model_bottleneck_only = GroundingModel(spatial_upsample_stages=())
  ```
- **CLI Flag**: Specify via `--spatial-upsample-stages 0 1 2 3` in `train.py`.

### 3. Checkpoint Compatibility & Auto-Detection

- **Strict backward compatibility**: Checkpoints trained with the default 2 stages `(0, 1)` (including legacy checkpoints without stage metadata) load seamlessly with `strict=True`.
- **Automatic detection**: `checkpoint_model_kwargs(saved)` inspects checkpoint metadata and state dict keys to automatically instantiate the exact architecture (2 stages, 4 stages, bottleneck only, or baseline plain model).
- **Mismatch protection**: Incompatible stage configurations (e.g. attempting to load a 4-stage checkpoint into a 2-stage model) are explicitly rejected during weight validation.

## Training

The dataset root contains `train_grounding.json`, `val_grounding.json`, image
folders `train/`, `val/`, and masks in `binary_masks_png/<split>/`.

The original single-stage training behavior is `full + legacy`. Both are fresh-run
defaults; writing them explicitly makes the configuration clear:

```bash
uv run torchrun --nproc_per_node=2 train.py \
    --data-root /path/to/vizwiz \
    --num-epochs 20 \
    --batch-size 4 \
    --num-workers 2 \
    --lr 0.00001 \
    --seed 42 \
    --save-every 10 \
    --training-recipe full \
    --loss-version legacy \
    --output-dir outputs/latent_full_legacy
```

To train with custom stages (e.g. all 4 upsampling stages enabled):

```bash
uv run torchrun --nproc_per_node=2 train.py \
    --data-root /path/to/vizwiz \
    --num-epochs 20 \
    --batch-size 4 \
    --training-recipe alignment \
    --loss-version stable \
    --spatial-upsample-stages 0 1 2 3 \
    --output-dir outputs/spatial_4stages
```


Batch size is **global across GPUs**: 4 on two GPUs means 2 per GPU. Training
chooses CUDA automatically, uses AMP on CUDA, and does not run validation.
Training is single-stage: real answers supervise the same model's oracle view.
There is no external teacher checkpoint or separate teacher-training command.

| `--training-recipe` | Objective |
|---|---|
| `full` | Predicted-answer mask + oracle/Q-reference masks + gain-weighted mask KD + answer alignment + attention guidance; no warm-up |
| `grounding-only` | Predicted-answer mask supervision only; the latent predictor remains present |
| `alignment` | Predicted-answer mask supervision + answer-vector alignment |
| `warmup` | Answer alignment throughout; mask supervision transitions from real A to predicted A' |

For new matched comparisons, explicitly select `--loss-version stable`. This
uses the audited KL calculations and valid-answer reductions. `legacy` preserves
the previous objective. Grounding-only segmentation is identical in both versions.
New training objectives require new runs to obtain results for those objectives.

Warmup defaults to 5 real-answer epochs followed by a 5-epoch transition. For a
20-epoch run, predicted-mask weights on answer-available samples are 0 in epochs
1-5, then 0.2/0.4/0.6/0.8/1.0 in epochs 6-10, and 1 thereafter. Missing-answer
samples always use predicted grounding. The schedule mixes per-image losses,
not answer vectors. Full has no warm-up.

`--help` lists the complete CLI. Normal runs need no configuration file.
For an advanced experiment, `--config /path/to/settings.json` accepts a flat JSON
object with only the settings to override, for example:

```json
{
    "latent_weight": 0.3,
    "detach_predicted_answer": true
}
```

Supported keys are `training_recipe`, `loss_version`, `dice_weight`,
`latent_weight`, `detach_predicted_answer`, `freeze_text_encoder`, `warmup_epochs`,
`answer_transition_epochs`, and the full-recipe weights `oracle_weight`,
`question_weight`, `kd_weight`, `attn_weight`, `gain_mix`. Schedule settings
require the warmup recipe. Explicit recipe/loss flags must agree with the file.
Resolved settings are recorded in checkpoints and training history.

For frozen-text controls, use `{"freeze_text_encoder": true}` in the JSON.
This freezes the shared CLIP question/answer encoder and keeps it in eval mode
during training. Vision, text projection, latent predictor and decoder still
train. The default is false, preserving previous runs. Resume restores the
saved policy before constructing the optimizer; conflicting overrides fail.

The former individual loss/schedule/detachment flags, `--help-advanced`,
`--teacher-checkpoint`, and the duplicate `--supervised-only` alias are removed.
Use `--training-recipe grounding-only` for the latter. Unsupported flags fail
explicitly; they are never silently ignored.

## Checkpoints and evaluation

`--save-every 10` writes resumable `checkpoint_epoch10.pt`, etc. Training also
writes final raw weights as `model_final_epoch<N>.pt`. Periodic checkpoints
include optimizer/scaler state, epoch, and training configuration; raw weights
contain model tensors only.

Use `--resume-checkpoint` for a periodic checkpoint. Recipe, loss weights,
detachment and schedule are restored; conflicting overrides are rejected.
The original JSON file is not needed when resuming. Historical external-teacher
training runs require their original runner rather than this single-stage trainer.
Repeat the original batch size, LR and seed. `--num-epochs` is the final total
count on resume. Use `--init-checkpoint` for a new run initialized from weights,
with a new optimizer and epoch counter. Keep different experiments in separate
output directories.

```bash
uv run python eval.py \
    --checkpoint outputs/latent_full_legacy/checkpoint_epoch20.pt \
    --dataset val \
    --data-root /path/to/vizwiz \
    --batch-size 4 \
    --num-workers 2 \
    --output-dir outputs/eval_val_ep20
```

Evaluation scores and exports masks at native resolution by default. Use
validation for checkpoint selection. Complete existing latent checkpoints retain
matching parameter names/shapes and load for inference. Student predictions do
not use ground-truth answers. `predict_save.py` and `visualize_predictions.py`
provide single-image prediction and overlays. Historical standalone teacher
checkpoint formats are detected automatically; no `--teacher` flag is needed.

## Diagnostics

All diagnostic tools are included at the repository root. `diagnose.py`
compares Q-only, predicted-answer and real-answer masks. For Q/A interventions,
embedding retrieval and conditioning traces, run:

```bash
uv run python probe_latent_conditioning.py \
    --checkpoint outputs/checkpoint_epoch20.pt \
    --data-root /path/to/vizwiz \
    --dataset val \
    --batch-size 4 \
    --num-workers 2 \
    --device cuda \
    --output-dir outputs/probe_val_ep20
```

Use one process, without torchrun, and a new or empty output directory. The
probe leaves weights and annotations unchanged. It exports summaries, per-image
measurements, embeddings and a gallery. Intervention scores use the original
GT and measure sensitivity; they are not accuracy for a newly posed question.
The pretrained text reference runs on CPU; `--skip-pretrained-reference` omits
that comparison when needed. Reports, data and local tests remain untracked.

To test predicted-answer content directly, use `probe_predicted_answer.py`:

```bash
uv run python probe_predicted_answer.py \
    --checkpoint outputs/alignment_frozen/checkpoint_epoch20.pt \
    --data-root /path/to/vizwiz \
    --batch-size 4 \
    --num-workers 2 \
    --device cuda \
    --seed 42 \
    --output-dir outputs/latent_alignment_frozen_ep20
```

This validation-only probe caches predicted vectors in a first pass, then keeps
each image/question fixed while replaying, swapping, averaging or zeroing the
answer vector. Replacements preserve the answer-token slot and affect both
attention and skip conditioning; a separate Q-only view removes the slot.
It exports JSON/CSV summaries, donor mappings and a gallery without creating PT
files or changing weights. Check cached-replay differences before interpreting
effects. The mean uses the label-free validation cache as a diagnostic control;
it is not a deployable learned constant. Distinct-answer donors are selected by
annotation strings, not verified semantic negatives. Use the original GT to
measure sensitivity, not accuracy for the donor question/answer.
