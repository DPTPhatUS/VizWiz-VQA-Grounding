# VizWiz VQA Grounding — residual-fusion

Adds a learned residual connection around cross-attention to preserve visual features. This checkout runs the `exp/residual-fusion` model through `train.py` and `eval.py`.

Given an image, question, and answer, predict the supporting binary mask. Evaluation reports mean Intersection over Union (IoU). Inputs use the fixed 336 × 336 CLIP resolution.

## Setup

```bash
uv sync --locked
```

Place `train_grounding.json`, `val_grounding.json`, and `test_grounding.json` under `data/vizwiz/`, alongside `train/`, `val/`, `test/`, and `binary_masks_png/{train,val,test}/`. Use `--data-root` for another location. The first model run downloads the pretrained CLIP encoders if they are not cached.

## Train and evaluate

```bash
uv run train.py --data-root data/vizwiz --num-epochs 100 --batch-size 4 --output-dir outputs/residual-fusion
uv run eval.py --checkpoint outputs/residual-fusion/checkpoint_epoch100.pt --dataset val --data-root data/vizwiz --output-dir results/residual-fusion
```

Training also supports `--lr`, `--num-workers`, `--seed`, `--resume-checkpoint`, `--validate-every`, and `--save-every`. Set `--save-every 0` to save only the final model; validation defaults to disabled. Use `torchrun` for distributed training. Evaluation requires a checkpoint and supports `--device`, `--batch-size`, and `--num-workers`; use `--dataset test` for the test split.

Checkpoints retain the branch architecture identity and load strictly. Use a checkpoint from this experiment.

## Project

Developed for the [VizWiz VQA Grounding Challenge](https://vizwiz.org/tasks-and-datasets/visual-qa/) 2025. The work was selected as a workshop spotlight at CVPR 2025.

Licensed under [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/).

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
