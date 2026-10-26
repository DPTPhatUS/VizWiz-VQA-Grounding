# VizWiz VQA Grounding: separate-qa-skips

Adds residual fusion and separate question/answer pooling for skip conditioning. This checkout runs the `exp/separate-qa-skips` model through `train.py` and `eval.py`.

Given an image, question, and answer, predict the supporting binary mask. Evaluation reports mean Intersection over Union (IoU). Inputs use the fixed 336 × 336 CLIP resolution.

## Setup

```bash
uv sync --locked
```

Place `train_grounding.json`, `val_grounding.json`, and `test_grounding.json` under `data/vizwiz/`, alongside `train/`, `val/`, `test/`, and `binary_masks_png/{train,val,test}/`. Use `--data-root` for another location. The first model run downloads the pretrained CLIP encoders if they are not cached.

## Train and evaluate

```bash
uv run train.py --data-root data/vizwiz --num-epochs 100 --batch-size 4 --output-dir outputs/separate-qa-skips
uv run eval.py --checkpoint outputs/separate-qa-skips/checkpoint_epoch100.pt --dataset val --data-root data/vizwiz --output-dir results/separate-qa-skips
```

Training also supports `--lr`, `--num-workers`, `--seed`, `--resume-checkpoint`, `--validate-every`, and `--save-every`. Set `--save-every 0` to save only the final model; validation defaults to disabled. Use `torchrun` for distributed training. Evaluation requires a checkpoint and supports `--device`, `--batch-size`, and `--num-workers`; use `--dataset test` for the test split.

Checkpoints record the branch architecture and load with strict key matching. Use a checkpoint trained on this branch.

## Project

Developed for the [VizWiz VQA Grounding Challenge](https://vizwiz.org/tasks-and-datasets/visual-qa/) 2025. The work was selected as a workshop spotlight at CVPR 2025.

Licensed under [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/).

## Encoder freezing and optimization

Omitting the options below keeps this branch's default training setup: Adam, learning rate `1e-5`, zero weight decay, constant learning rate, and the default encoder freezing policy. Freezing is opt-in, and the detail CNN and decoder always remain trainable. Model state-dict keys and evaluation preprocessing are unchanged.

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

`BOOL` accepts `True` or `False`; passing a freeze flag without a value sets it to `True`. The negative flags `--no-freeze-backbone`, `--no-freeze-vision-encoder`, and `--no-freeze-text-encoder` set the corresponding option to `False`. Within the same source, a specific encoder flag overrides `freeze_backbone`. Precedence follows JSON config, then explicit CLI flags, then branch defaults, so a JSON config value also overrides CLI flags. Existing recipe and loss settings can share the same JSON file, and unrecognized keys raise an error immediately. See `train.py --help` for branch-specific flags.

For example, append these flags to the training command:

```bash
--freeze-backbone True \
--optimizer adamw \
--lr 0.0001 \
--weight-decay 0.01 \
--lr-scheduler cosine \
--warmup-ratio 0.05 \
--min-lr-ratio 0.1
```

To freeze only the vision encoder while keeping the default text policy, pass `--freeze-vision-encoder True`. To train both encoders, pass `--freeze-backbone False`. Gain-guided refinement models keep their reference stage fixed and reject encoder unfreezing. Unused CLIP and SigLIP pooled-output parameters stay frozen when token encoders are unfrozen.

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

Frozen encoders stay in `eval` mode during training and are excluded from the optimizer. Learning rate schedules step after each completed optimizer update (accounting for gradient accumulation) and do not advance when AMP skips an overflowing step. Checkpoints store the resolved optimization config, trainable parameter layout, and scheduler state; resuming restores these settings and rejects conflicting flags. When resuming a scheduled run, keep `--num-epochs`, dataset length, and gradient accumulation unchanged to preserve the learning rate schedule.

Existing checkpoints remain compatible with evaluation. Resuming a legacy checkpoint keeps its original optimizer layout and constant learning rate; changing encoder freezing, optimizer type, weight decay, or learning rate schedule requires starting a new run (or initializing from weights only where supported). Resuming follows each branch's RNG and data-order handling.
