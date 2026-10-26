# Grounding first with local answer supervision

Branch `exp/grounding-first-answer` (branched from `exp/spatial-detail-decoder` at `63977d2`) tests whether auxiliary answer supervision improves predicted evidence masks when an auxiliary answer reader is restricted to local RGB features pooled inside the predicted mask.

## Forward path

```text
Image + Q --> CLIP vision/text --> question-conditioned detail decoder --> mask logits
RGB image --> local reader features -- masked pooling by predicted mask --> answer embedding
```

The mask decoder receives only the image and question, retaining three question-conditioned visual skips and RGB detail fusion at decoder stages 2 and 3. It omits cosine spatial interaction, pre-grounding answer prediction, external teachers, oracle-mask decoding, and self-distillation. The CLIP text encoder stays frozen in `eval` mode while CLIP vision patch features are trained. The vision `post_layernorm` layer is kept frozen in checkpoints because it only normalizes the unused pooled CLS token, avoiding unused trainable parameters under DDP.

The auxiliary answer reader consists of two bias-free 3x3 convolutions, question-gated visual channels, and a bias-free linear projection. Its 5x5 visual receptive field covers at most a two-pixel context border around any selected pixel, without access to CLIP visual features, global visual skips, spatial normalization, or a direct question-to-answer residual path. Because the mask is soft during training, question and appearance priors can still influence the pooled representation.

The reader predicts a continuous CLIP answer embedding. Ground-truth answers are formatted as `A: ...` and encoded with frozen CLIP token features via masked mean pooling (excluding padding, SOT, and EOT tokens). The full-size auxiliary reader adds 121,696 parameters and is bypassed completely during mask-only inference.

## Loss and gradient routing

During training, the auxiliary reader is evaluated on two spatial regions per image:

1. **GT region:** updates the reader parameters only, supervised against the frozen ground-truth answer embedding.
2. **Predicted soft region:** detaches the reader features, reader weights, and question features so the answer loss backpropagates exclusively through the soft mask probabilities into the grounding network.

The predicted-region path resizes mask logits to the RGB feature resolution and applies `sigmoid` directly, preserving differentiability without hard thresholding or cropping. At evaluation and deployment, neither ground-truth masks nor answers are used.

The training loss is:

```text
L = BCE + 0.5 * Dice + 0.5 * reader_GT + effective_answer_weight * mask_answer
reader_GT and mask_answer = cosine error + mean SmoothL1 against frozen answer embeddings
```

`effective_answer_weight` is `0` during epochs 1-2 and `0.5` from epoch 3 onward, allowing the reader to calibrate on ground-truth regions before sending gradients into the mask decoder. Samples with missing answers or empty ground-truth masks are excluded from both answer loss terms (with valid-sample reductions synchronized across DDP ranks) while still contributing to the segmentation loss. Grounder and reader gradients are clipped independently to a max norm of `1.0` so reader updates do not rescale the grounder's gradient norm.

Optional loss overrides can be passed via `--config`:

```json
{
  "dice_weight": 0.5,
  "reader_weight": 0.5,
  "answer_weight": 0.5,
  "answer_warmup_epochs": 2
}
```

No config file is required when using the defaults above. Legacy flags from other branches (`--training-recipe`, `--loss-version`, `--no-spatial-evidence`) are not used on this branch.

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

`--batch-size` is global across GPUs (`--batch-size 4` on 2 GPUs places 2 images per GPU). FP16 AMP is enabled automatically on CUDA, and no validation runs during training. Full resumable checkpoints (including source hashes and resolved model/training configs) are saved every `--save-every` epochs alongside a separate final raw weights file.

To resume from a periodic checkpoint, pass `--resume-checkpoint outputs/checkpoint_epoch10.pt` with the original batch size, learning rate, seed, world size, and loss settings (`--num-epochs 20` specifies 20 total epochs). Raw weight files cannot be used to resume training, and checkpoints from earlier latent or spatial-evidence branches are rejected.

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

`eval.py` skips the auxiliary reader and scores masks at native ground-truth resolution (resizing logits before thresholding). Use validation metrics to select checkpoints and hyperparameters.

`diagnose.py` reports IoU across mask size groups and compares reader embedding alignment under ground-truth, predicted, and full-image support masks.

## Matched control comparison

To isolate the effect of the indirect answer gradient on mask quality, run a matched control with `"answer_weight": 0` in `--config` while keeping the architecture, reader GT training, and all other hyperparameters identical. Because the detail branch operates on 336x336 inputs, inspect the reader alignment diagnostics alongside validation mIoU to verify whether the reader learns informative regional representations.

## Encoder freezing and optimization

Omitting the flags below keeps this branch's default training configuration: Adam with learning rate `1e-5`, zero weight decay, a constant learning rate, and the branch's default encoder freezing policy. Encoder freezing is opt-in, while the detail CNN and mask decoder always remain trainable. Model state-dict keys and evaluation preprocessing are unchanged.

| Option | Values / default |
| --- | --- |
| `--freeze-backbone [BOOL]` | Freeze or unfreeze both pretrained encoders; omitting keeps the branch default |
| `--freeze-vision-encoder [BOOL]` | Freeze or unfreeze the vision encoder only |
| `--freeze-text-encoder [BOOL]` | Freeze or unfreeze the text encoder only |
| `--optimizer` | `adam` (default), `adamw` |
| `--lr` | Main learning rate; default `0.00001` |
| `--backbone-lr` | Optional separate learning rate for unfrozen encoders; defaults to `--lr` |
| `--weight-decay` | Nonnegative weight decay; default `0`. Biases, normalization parameters, and token/position embeddings are excluded from decay when parameter groups are active |
| `--lr-scheduler` | `none` (default), `cosine`, `linear`, `poly` |
| `--warmup-ratio` | Fraction of total optimizer updates used for linear warmup, in `[0, 1)`; default `0` |
| `--min-lr-ratio` | Final LR divided by initial LR for decay schedules; default `0.1` |
| `--poly-power` | Positive exponent for the `poly` scheduler; default `0.9` |
| `--config` | Path to a flat JSON file with training options; optional |

`BOOL` accepts `True` or `False`, and passing a freeze flag without an argument sets it to `True`. The negative aliases `--no-freeze-backbone`, `--no-freeze-vision-encoder`, and `--no-freeze-text-encoder` set the corresponding option to `False`. Within the same configuration source, a component-specific flag (`--freeze-vision-encoder` or `--freeze-text-encoder`) overrides `freeze_backbone`. Precedence follows **JSON config > explicit CLI flags > branch defaults**, so a `freeze_backbone` entry in `--config` overrides component flags passed on the CLI. Branch-specific recipe and loss settings can be included in the same JSON file, and unknown keys raise an error immediately. See `train.py --help` for the full list of CLI flags.

Example flags to append to a training command:

```bash
--freeze-backbone True \
--optimizer adamw \
--lr 0.0001 \
--weight-decay 0.01 \
--lr-scheduler cosine \
--warmup-ratio 0.05 \
--min-lr-ratio 0.1
```

To freeze only the vision encoder while keeping the branch's text encoder policy, pass `--freeze-vision-encoder True`. To train both encoders, pass `--freeze-backbone False`. Architectures with a fixed reference stage (such as gain-guided refinement) reject encoder unfreezing. Unused CLIP and SigLIP pooled-output parameters remain frozen even when token encoders are unfrozen.

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

When frozen, encoders are placed in `eval` mode and omitted from the optimizer. Learning rate schedulers step once per completed optimizer update (accounting for gradient accumulation) and do not advance when AMP skips a step due to gradient overflow. Checkpoints record the resolved optimization configuration, trainable parameter groups, and scheduler state. Resuming restores these settings and rejects conflicting overrides; keep `--num-epochs`, dataset length, and gradient accumulation unchanged when resuming a scheduled run so the learning rate schedule stays aligned.

Older checkpoints remain directly loadable for evaluation. Resuming a legacy checkpoint preserves its original optimizer layout and constant learning rate; changing encoder freezing, the optimizer, weight decay, or the learning rate schedule requires starting a new run (or initializing from weights where supported). Resuming also follows each branch's existing RNG and data-ordering behavior.
