# Latent-answer grounding

This branch trains a compact grounding model with CLIP image and question encoders, a latent answer predictor, and a mask decoder. Real answers act as privileged supervision during training, while student inference requires only the image and question. CLIP image inputs are resized to 336x336 pixels.

## Training

The dataset directory (`--data-root`) must contain `train_grounding.json`, `val_grounding.json`, the `train/` and `val/` image folders, and ground-truth masks under `binary_masks_png/<split>/`.

The default single-stage training configuration uses `--training-recipe full` and `--loss-version legacy`:

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

`--batch-size` is global across GPUs, so a batch size of 4 across 2 GPUs places 2 images on each GPU. Training uses CUDA and AMP automatically when available and skips in-training validation. Real answers supervise the model's own oracle path within a single stage, without requiring an external teacher checkpoint or separate teacher training step.

| `--training-recipe` | Objective |
|---|---|
| `full` | Predicted-answer mask + oracle/Q-reference masks + gain-weighted mask KD + answer alignment + attention guidance; no warmup |
| `grounding-only` | Predicted-answer mask supervision only; the latent predictor remains in the forward pass |
| `alignment` | Predicted-answer mask supervision + answer-vector alignment |
| `warmup` | Answer alignment throughout; mask supervision transitions from real `A` to predicted `A'` |

For new matched comparisons, pass `--loss-version stable` to enable the updated KL divergence and valid-answer loss reductions. `--loss-version legacy` retains the original loss formulation. Under `grounding-only`, both loss versions produce identical segmentation updates.

The `warmup` recipe defaults to 5 real-answer epochs followed by a 5-epoch linear transition. In a 20-epoch run, the predicted-mask loss weight on samples with valid answers is `0` during epochs 1-5, steps through `0.2, 0.4, 0.6, 0.8, 1.0` during epochs 6-10, and stays at `1.0` from epoch 10 onward. Samples without valid answers always use the predicted-answer grounding path. The transition schedule interpolates per-image losses rather than blending answer vectors.

Run `train.py --help` for the full CLI reference. For custom overrides, pass a flat JSON file via `--config /path/to/settings.json`:

```json
{
    "latent_weight": 0.3,
    "detach_predicted_answer": true
}
```

Supported JSON keys are `training_recipe`, `loss_version`, `dice_weight`, `latent_weight`, `detach_predicted_answer`, `freeze_text_encoder`, `warmup_epochs`, `answer_transition_epochs`, and the full-recipe weights (`oracle_weight`, `question_weight`, `kd_weight`, `attn_weight`, `gain_mix`). Warmup schedule keys are only valid when `training_recipe` is `warmup`. Any recipe or loss version specified on both the CLI and in the JSON file must match. Resolved parameters are saved in checkpoints and `history.jsonl`.

Setting `{"freeze_text_encoder": true}` in the config freezes the shared CLIP question/answer text encoder and keeps it in `eval` mode during training, while the vision encoder, text projection, latent predictor, and mask decoder remain trainable. By default `freeze_text_encoder` is `false`. Resuming a checkpoint restores the saved freezing policy before building the optimizer and rejects conflicting overrides.

Removed legacy flags (`--help-advanced`, `--teacher-checkpoint`, `--supervised-only`, and standalone per-loss weight flags) raise an immediate CLI error; use `--training-recipe grounding-only` in place of `--supervised-only`.

## Checkpoints and evaluation

`--save-every 10` saves resumable checkpoints (`checkpoint_epoch10.pt`, etc.) containing model weights, optimizer and GradScaler states, the completed epoch, and the resolved configuration. At the end of training, the script also exports raw model weights to `model_final_epoch<N>.pt`.

Pass `--resume-checkpoint` to continue training from a periodic checkpoint. The saved recipe, loss weights, detachment mode, and warmup schedule are restored automatically without needing the original JSON config file. Keep the original batch size, learning rate, and seed when resuming; `--num-epochs` sets the target total epoch count. To start a fresh run initialized from existing weights (resetting the optimizer and epoch counter), use `--init-checkpoint`. Store separate experiments in distinct output directories.

```bash
uv run python eval.py \
    --checkpoint outputs/latent_full_legacy/checkpoint_epoch20.pt \
    --dataset val \
    --data-root /path/to/vizwiz \
    --batch-size 4 \
    --num-workers 2 \
    --output-dir outputs/eval_val_ep20
```

`eval.py` computes metrics and exports predicted masks at native image resolution by default. Use the validation split for checkpoint selection. Student evaluation uses only the image and question. Single-image mask export and visualization are available through `predict_save.py` and `visualize_predictions.py`, and historical standalone teacher checkpoints are detected automatically.

## Diagnostics

Repository-root diagnostic scripts inspect how answer conditioning affects predictions. `diagnose.py` compares question-only, predicted-answer, and ground-truth-answer masks. For question/answer interventions, embedding retrieval, and conditioning traces, run:

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

Run `probe_latent_conditioning.py` as a single process (without `torchrun`) pointing to a new or empty output directory. The script leaves model weights and dataset annotations untouched and writes summary metrics, per-image logs, embeddings, and a visual gallery. Intervention metrics are scored against each image's original ground-truth mask to measure conditioning sensitivity. The pretrained text reference comparison runs on CPU and can be skipped with `--skip-pretrained-reference`.

To test sensitivity to the predicted answer vector directly, run `probe_predicted_answer.py`:

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

This validation probe caches predicted answer vectors in a first pass, then holds each image and question fixed while replaying, swapping, averaging, or zeroing the answer vector. Vector replacements preserve the answer-token slot across both cross-attention and skip conditioning, whereas the question-only view removes the slot entirely. The script outputs JSON/CSV summaries, donor mappings, and a visual gallery without writing `.pt` files or modifying checkpoints. Check the cached-replay parity metrics first when reviewing results. The mean-vector baseline averages the cached validation predictions as a diagnostic control, and distinct-answer donors are chosen by comparing normalized annotation strings. All intervention views are evaluated against the original ground-truth mask to measure sensitivity.

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
