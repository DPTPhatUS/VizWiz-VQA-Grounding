# VizWiz VQA Grounding: Evidence Extent

This branch predicts visual evidence masks from an image and a question by extending the compact CLIP grounding model with two learned tokens representing evidence location and spatial extent. Location supervision uses the ground-truth mask's bounding box, while final-mask and area losses supervise the extent (including empty masks).

Training and evaluation run through `train.py` and `eval.py` using question-only text inputs. The image resolution is fixed at 336x336 pixels, the learned token width is 64, the support loss weight is `0.2`, and the area loss weight is `0.1`. The two-token module adds 148,545 parameters to the compact backbone.

## Kaggle

Enable a GPU and internet access (for downloading pretrained CLIP weights). Keep Kaggle's preinstalled PyTorch and Torchvision packages and install the remaining dependencies:

```bash
pip install transformers tqdm pillow
```

The dataset root (`--data-root`) must contain `<split>_grounding.json`, the image folders `train/`, `val/`, `test/`, and ground-truth masks under `binary_masks_png/<split>/`. Run training and evaluation in a Kaggle `%%bash` cell:

```bash
set -e
cd /kaggle/working/VizWiz-VQA-Grounding
DATA_ROOT=/kaggle/input/YOUR_DATASET/vizwiz
python train.py --data-root "$DATA_ROOT" --num-epochs 100 --batch-size 1 --num-workers 2 --seed 42 --output-dir /kaggle/working/outputs-extent
python eval.py --data-root "$DATA_ROOT" --checkpoint /kaggle/working/outputs-extent/checkpoint_epoch100.pt --dataset test --batch-size 1 --num-workers 2 --output-dir /kaggle/working/results-extent
```

Training uses CUDA automatically when available. Passing `--init-checkpoint /path/to/compact/checkpoint_epoch100.pt` initializes the coarse backbone from a compatible compact checkpoint (or a prior extent checkpoint) with a fresh optimizer.

## Optional paired questions

Passing `--pairs /path/to/pairs.json` adds training supervision from verified same-image question pairs, where each question has its own ground-truth mask. Mask paths in the manifest resolve relative to the manifest file:

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

Each entry sets `relation` to `"change"` or `"same"`. Entries marked `"same"` must have identical native-resolution binary masks. Both questions must be non-empty, both masks must match the image dimensions, and the image must belong to `train`. Unpaired training images continue to receive standard single-question supervision.

When `--pairs` is active, the signed mask-difference loss and same-region consistency loss are each weighted by `0.1`. Loss terms are normalized by the number of original images so batches with varying pair counts maintain consistent scaling. Evaluation and single-image inference require only an image and question.

## Single-image prediction

```bash
python predict_save.py --checkpoint /path/to/checkpoint_epoch100.pt --image /path/to/image.jpg --question "What does it say?" --output mask.png
python visualize_predictions.py --checkpoint /path/to/checkpoint_epoch100.pt --image /path/to/image.jpg --question "What does it say?" --output overlay.png
```

`IoU.py` delegates to `eval.py`. `models/model.py` defines both the compact coarse model and the two-token extent model, `dataset.py` implements the aligned dataset loader, and `losses.py` defines the training objective. `docs/` and `tests/` are git-ignored for local development.

## Training output and separate evaluation

`train.py` loads only the training split and does not run validation during or after training.

Training writes periodic resumable checkpoints (`checkpoint_epoch10.pt`, `checkpoint_epoch20.pt`, etc.), with `--save-every` defaulting to `10`. Each checkpoint stores the model weights, optimizer and GradScaler states, the completed epoch, and the embedded `run_config` and `experiment_config`. No separate `model_final_epoch*.pt`, `last.pt`, or `best.pt` file is written automatically; if training stops between checkpoint intervals, the most recent saved file is from the previous interval (use `--save-every 1` to save every epoch, or `--save-every 0` to disable saving).

Raw weight files remain loadable using the branch's default model settings. `history.jsonl` logs training losses only. Run `eval.py` with `--dataset val` or `--dataset test` to evaluate saved checkpoints.

To resume an interrupted run from a periodic checkpoint, pass `--resume-checkpoint /path/to/checkpoint_epoch10.pt`; this restores the optimizer, GradScaler, and completed epoch, and `--num-epochs` sets the target total epoch count. To initialize a new run from existing weights with a fresh optimizer and epoch counter, pass `--init-checkpoint /path/to/checkpoint_epoch<N>.pt`. Keep the original batch size, seed, learning rate, and experiment flags when continuing a run (including `--pairs` and its referenced mask files if continuing a paired-question run).

Existing output directories can be reused across runs: matching checkpoint filenames are overwritten, other files are preserved, and new log entries are appended to `history.jsonl`. Resuming in a fresh Kaggle session requires only the periodic `.pt` checkpoint and the training dataset.

## Code layout

- `models/model.py`: compact coarse model and two-token location/extent model.
- `models/image_encoder.py` and `models/text_encoder.py`: CLIP vision and text feature extractors.
- `models/mask_decoder.py`: mask decoder module.
- `train.py`: dataset loading, training loop, and periodic checkpoint saving.
- `eval.py`: checkpoint loading, native-resolution evaluation, and mask export.
- `utils.py`: batch device transfer and checkpoint loading helpers.

Both training (on rank 0 under `torchrun`) and evaluation display `tqdm` batch progress bars.

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
