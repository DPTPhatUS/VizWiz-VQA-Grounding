# Grounding first with local answer supervision

Branch: `exp/grounding-first-answer`, based on `exp/spatial-detail-decoder` at `63977d2`.
This is a new experiment requiring fresh training. It tests whether answer supervision
can improve the predicted evidence mask when the answer reader must use local RGB
features selected by that mask.

## Forward path

```text
Image + Q --> CLIP vision/text --> question-conditioned detail decoder --> mask logits
RGB image --> local reader features -- masked pooling by predicted mask --> answer embedding
```

The mask decoder never receives a real or predicted answer. It retains three
question-conditioned visual skips and RGB detail fusion at decoder stages 2/3.
There is no cosine spatial interaction, pre-grounding answer predictor, teacher
checkpoint, oracle-mask decoder, or self-distillation recipe on this branch.
CLIP text is frozen and stays in eval mode. CLIP vision patch features are trainable.
The vision `post_layernorm` is frozen because it only normalizes pooled CLS,
which this decoder does not use. Its weights remain in checkpoints; freezing it
does not change patch features and avoids unused trainable parameters in DDP.

The auxiliary reader has two bias-free 3x3 convolutions, question-gated visual
channels, and a small bias-free projection. Its visual receptive field is 5x5,
including at most a two-pixel context halo around a selected location. It has no
CLIP visual features, global visual skip, spatial normalization, or direct
question-to-answer residual. Question/appearance priors and soft-mask shortcuts
remain possible; this architecture is not a proof of causal grounding.

The reader predicts a **CLIP answer embedding**, not a free-form answer string.
A real answer is encoded as `A: ...` using frozen CLIP token features, with masked
mean pooling excluding padding and SOT/EOT. Cosine alignment is not answer accuracy.
The full-size auxiliary reader adds 121,696 parameters and is skipped during mask-only inference.

## Loss and gradient routing

The same reader is evaluated on two regions during training:

1. **GT region:** updates the reader only, supervised by the real-answer embedding.
2. **Predicted soft region:** reader features/weights and question features are detached;
   the answer loss differentiates through the soft mask into the grounding network.

The predicted-region path does **not** detach the mask, run under `no_grad`, or
threshold/crop it with a nondifferentiable operation. Pixel masks come from sigmoid
of logits resized to the RGB feature resolution. The GT-region path does not feed
back into mask inference. No GT mask/answer is required at deployment.

The default objective is:

```text
L = BCE + 0.5 * Dice + 0.5 * reader_GT + effective_answer_weight * mask_answer
reader_GT and mask_answer = cosine error + mean SmoothL1 against frozen answer embeddings
```

`effective_answer_weight` is 0 for epochs 1-2 and 0.5 from epoch 3 onward. This
calibrates the reader before using it to guide masks, within one training run and
one optimizer. These are initial research settings, not tuned coefficients.
Missing answers and empty GT regions are excluded from both answer terms, while
segmentation supervision is retained. Valid-answer reduction is global across DDP ranks.
Grounder and reader gradients are clipped separately to norm 1, so reader calibration
does not change the grounder's update through a shared clipping norm.

Optional JSON contains only:

```json
{
  "dice_weight": 0.5,
  "reader_weight": 0.5,
  "answer_weight": 0.5,
  "answer_warmup_epochs": 2
}
```

No config file is needed to use these defaults. The text-freezing and decoder
architecture are fixed for this experiment. Old flags such as `--training-recipe`,
`--loss-version`, `--no-spatial-evidence`, and latent alignment configs do not apply.

## Train

```bash
uv run torchrun --nproc_per_node=2 train.py \
    --data-root /path/to/vizwiz \
    --output-dir outputs \
    --num-epochs 20 \
    --batch-size 4 \
    --num-workers 2 \
    --lr 0.00001 \
    --seed 42 \
    --save-every 10
```

Batch size is global: 4 means 2 images/GPU on two GPUs. AMP FP16 is enabled on
CUDA. The trainer does not run validation. Full checkpoints are saved at epochs
10/20 and final raw weights are saved separately. Source hashes and resolved
training/model settings are stored in full checkpoints.

Resume with `--resume-checkpoint outputs/checkpoint_epoch10.pt`, keeping the same
batch size, LR, seed, world size and loss settings, with `--num-epochs 20` meaning
20 total epochs. Raw weights cannot resume training. Previous latent/spatial
checkpoints belong to their original branches and are rejected here.

## Evaluate and diagnose

```bash
uv run python eval.py \
    --checkpoint outputs/checkpoint_epoch20.pt \
    --data-root /path/to/vizwiz \
    --dataset val --batch-size 4 --num-workers 2 --device cuda \
    --output-dir outputs/eval_val_ep20

uv run python diagnose.py \
    --checkpoint outputs/checkpoint_epoch20.pt \
    --data-root /path/to/vizwiz \
    --batch-size 4 --num-workers 2 --device cuda \
    --output-dir outputs/diagnostic_ep20
```

Mask evaluation bypasses the auxiliary reader entirely. IoU and exported masks
use native GT resolution; logits are resized before thresholding. Local test
scoring requires test masks. Select checkpoints and hyperparameters on validation.

Diagnostics report per-image/size-group IoU and embedding alignment when the reader
uses GT, predicted, or full-image support. It does not generate answers or prove
that a correct answer requires the annotated region. Legacy A-prime intervention
probes are retired on this branch because answers do not condition the mask.

## Required comparison and evidence limits

A matched control sets only `answer_weight` to 0, retaining the same architecture,
reader GT training and other settings. That control isolates whether the indirect
answer gradient helps the mask. Comparing only with the old latent alignment run
would also change architecture and training objective.

The detail input remains 336x336, so lost source-image detail is not restored.
If the reader cannot learn useful GT-region representations, its mask gradient may
be unhelpful or harmful. Inspect the reader diagnostics alongside mask accuracy.
CPU/synthetic gradient, locality, DDP and checkpoint checks do not establish VizWiz
accuracy, seed robustness, GPU memory requirements, or Kaggle runtime.

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
