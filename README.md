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
No separate teacher training is required for the default single-stage method.
An external teacher is optional and must have the complete latent architecture.

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

`--help` shows data/optimizer controls, recipe, and loss version.
`--help-advanced` also shows loss weights, detachment, schedule overrides, and
compatibility aliases. Existing flags remain accepted. `--supervised-only` is
an alias for grounding-only. Changing help visibility does not change training.

## Checkpoints and evaluation

`--save-every 10` writes resumable `checkpoint_epoch10.pt`, etc. Training also
writes final raw weights as `model_final_epoch<N>.pt`. Periodic checkpoints
include optimizer/scaler state, epoch, and training configuration; raw weights
contain model tensors only.

Use `--resume-checkpoint` for a periodic checkpoint. Recipe, loss weights,
detachment and schedule are restored; conflicting overrides are rejected.
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
provide single-image prediction and overlays.
