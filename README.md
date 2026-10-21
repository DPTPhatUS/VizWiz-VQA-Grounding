# Latent-answer grounding

This branch trains a compact grounding model with CLIP image/question encoders,
a latent answer predictor, and a mask decoder. Training can use real answers as
privileged supervision. Student inference requires only image + question.
CLIP input is fixed at 336 pixels.

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
