# VizWiz VQA Grounding — evidence extent

This branch predicts evidence masks from an image and question. It extends the compact CLIP grounding model with two learned tokens for evidence location and extent. Location supervision uses the mask's bounding rectangle; final-mask and area losses learn the extent, including empty masks.

Use `train.py` and `eval.py`. This branch fixes the image size at 336, token width at 64, support loss weight at 0.2, and area loss weight at 0.1. Training and evaluation use questions only. There are no architecture, text-mode, test-model, or loss-weight switches. The model adds 148,545 parameters to the compact backbone; real-data accuracy and GPU throughput remain unmeasured.

## Kaggle

Enable a GPU and Internet for pretrained CLIP downloads. Keep Kaggle's Torch/Torchvision installation and install missing dependencies:

```bash
pip install transformers tqdm pillow
```

The data directory contains `<split>_grounding.json`, image folders `train/`, `val/`, `test/`, and `binary_masks_png/<split>/`. In a Kaggle `%%bash` cell:

```bash
set -e
cd /kaggle/working/VizWiz-VQA-Grounding
DATA_ROOT=/kaggle/input/YOUR_DATASET/vizwiz
python train.py --data-root "$DATA_ROOT" --num-epochs 100 --batch-size 1 --num-workers 2 --seed 42 --output-dir /kaggle/working/outputs-extent
python eval.py --data-root "$DATA_ROOT" --checkpoint /kaggle/working/outputs-extent/checkpoint_epoch100.pt --dataset test --batch-size 1 --num-workers 2 --output-dir /kaggle/working/results-extent
```

Training selects CUDA when available. `--init-checkpoint /path/to/compact/checkpoint_epoch100.pt` optionally initializes the coarse model from a compatible corrected compact checkpoint. It starts a fresh optimizer. Matching extent checkpoints can also initialize a run.

## Optional paired questions

`--pairs /path/to/pairs.json` enables extra supervision from verified same-image question pairs. Each question must have its own real mask. Mask paths resolve relative to the manifest:

```json
[
  {
    "filename": "VizWiz_train_00000000.jpg",
    "question1": "Where is the label?",
    "question2": "Where is the whole container?",
    "mask1": "verified/label.png",
    "mask2": "verified/container.png",
    "relation": "change"
  }
]
```

This illustrates the schema, not supplied annotations. `relation` is `change` or `same`; verified `same` pairs must have identical original-resolution binary masks. Questions must be nonempty, both masks must match the image dimensions, and the image must belong to the training split. Masks are never generated from rewritten questions. Unpaired images keep ordinary supervision.

With a manifest, signed mask-difference and same-region consistency losses each have weight 0.1. All contributions are normalized by original-image count so batches with different pair counts retain consistent weighting. Pairs increase training compute and memory. Evaluation and single-image inference need neither answers nor pair files.

## Single-image prediction

```bash
python predict_save.py --checkpoint /path/to/checkpoint_epoch100.pt --image /path/to/image.jpg --question "What does it say?" --output mask.png
python visualize_predictions.py --checkpoint /path/to/checkpoint_epoch100.pt --image /path/to/image.jpg --question "What does it say?" --output overlay.png
```

`IoU.py` delegates to the same evaluator. `models/model.py` contains the extent model, the compact coarse model is in that same file, `dataset.py` the aligned data loader, and `losses.py` the objective. `docs/` and `tests/` remain ignored local development files.

## Training output and separate evaluation

Training reads only the training split: no validation dataset is loaded and no validation runs, including after the final epoch. Paired question supervision is training-only.

Training saves only periodic resumable checkpoints: `checkpoint_epoch10.pt`, `checkpoint_epoch20.pt`, etc. `--save-every` defaults to **10**. Each checkpoint contains model weights, optimizer/scaler state, completed epoch, and embedded run configuration, and uses legacy PyTorch serialization. There is no additional final raw-model export or automatic final checkpoint. If training ends between checkpoint intervals, the latest saved model is from the preceding interval; use `--save-every 1` to save every completed epoch. `--save-every 0` disables all model saving.

Existing raw weights remain loadable using this branch's default model settings. Resumable checkpoints retain their own `run_config` and `experiment_config`; no standalone configuration file is written or read. `history.jsonl` contains training losses only. Run `eval.py` separately with `--dataset val` or `--dataset test`; the commands above use the periodic checkpoint saved at their epoch count.

To resume a periodic checkpoint, use `--resume-checkpoint /path/to/checkpoint_epoch10.pt` with its existing output directory or a fresh one; optimizer/scaler state and completed epoch are restored, and `--num-epochs` is the final total epoch count. To start a new run from checkpoint weights, use `--init-checkpoint /path/to/checkpoint_epoch<N>.pt`; this starts a fresh optimizer and epoch counter. Repeat the original batch size, seed, learning rate, and experiment-specific training arguments when continuing a run. Older supported full checkpoints remain loadable.

Existing output directories are accepted for all training modes. Matching model filenames are overwritten; other model files remain. History is appended to `history.jsonl`, creating it if absent. To continue in a fresh Kaggle notebook, only the resumable checkpoint is needed to restore training state; no previous history or separate configuration file is required. The training dataset must still be available. To continue a paired run, repeat `--pairs` and provide the manifest and all masks it references.


## Code layout

- `models/model.py`: the compact coarse model and the two-token location/extent model.
- `models/image_encoder.py` and `models/text_encoder.py`: CLIP feature extraction.
- `models/mask_decoder.py`: the decoder used by this experiment.
- `train.py`: construct the model and dataset, run training, save periodic resumable checkpoints.
- `eval.py`: construct the model, load weights, evaluate and export masks.
- `utils.py`: batch transfer and a few plain weight-loading helpers.

The separate backbone, experiment, and checkpoint modules and model factory are removed. Training uses no validation and saves only resumable checkpoints every 10 epochs by default.

Training shows a batch progress bar for each epoch (rank 0 under `torchrun`), and evaluation shows a batch progress bar. Install `tqdm` alongside Transformers and Pillow in the Kaggle notebook.

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
