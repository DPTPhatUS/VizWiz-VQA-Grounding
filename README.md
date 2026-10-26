# Compact question-only control

This branch trains a compact question-only grounding model.

The decoder bottleneck receives the cross-attention output directly without adding the original visual tokens or applying a residual scale, while keeping the multi-level decoder skip connections.

The CLIP text encoder is frozen by default (`freeze_text_encoder: true` in saved run configurations), leaving the vision encoder, text projection, cross-attention, and mask decoder trainable. Starting a fresh training run without checkpoint flags trains with the text encoder frozen from step 0.

The repository uses `train.py`, `eval.py`, `dataset.py`, and `models/` with image inputs fixed at 336x336 pixels.

## Kaggle

Clone branch `exp/grounding-controls` and enable a GPU with internet access (for downloading pretrained CLIP weights). Use Kaggle's preinstalled PyTorch and Torchvision packages and install the remaining dependencies:

```bash
pip install transformers tqdm pillow
```

The dataset root (`--data-root`) must contain `train_grounding.json`, `val_grounding.json`, `test_grounding.json`, image directories `train/`, `val/`, `test/`, and ground-truth masks under `binary_masks_png/<split>/`. Evaluating on a split requires its ground-truth masks.

Run in a `%%bash` cell, replacing the dataset path:

```bash
set -e
cd /kaggle/working/VizWiz-VQA-Grounding
DATA_ROOT=/kaggle/input/YOUR_DATASET/vizwiz
python train.py --data-root "$DATA_ROOT" --num-epochs 100 --batch-size 1 --num-workers 2 --seed 42 --output-dir /kaggle/working/outputs

python eval.py --data-root "$DATA_ROOT" --checkpoint /kaggle/working/outputs/checkpoint_epoch100.pt --dataset test --batch-size 1 --num-workers 2 --output-dir /kaggle/working/results
```

Training selects CUDA automatically and supports multi-GPU execution via `torchrun` (where `--batch-size` is the global batch size across GPUs). Evaluation accepts `--device`. Pass `--help` to either script to see its CLI options.

## Single-image prediction

```bash
python predict_save.py --checkpoint /path/to/checkpoint_epoch100.pt --image /path/to/image.jpg --question "What does it say?" --output /kaggle/working/mask.png
python visualize_predictions.py --checkpoint /path/to/checkpoint_epoch100.pt --image /path/to/image.jpg --question "What does it say?" --output /kaggle/working/overlay.png
```

`IoU.py` delegates to `eval.py`. Checkpoints may be either raw `state_dict` files or full checkpoint dictionaries whose parameter names and shapes match this branch. `docs/` and `tests/` are git-ignored for local development.

## Training output and separate evaluation

`train.py` loads only the training split and does not run validation during or after training.

Training saves periodic resumable checkpoints (`checkpoint_epoch10.pt`, `checkpoint_epoch20.pt`, etc.). `--save-every` defaults to **10**. Each checkpoint contains model weights, optimizer/scaler state, completed epoch, and embedded run configuration, serialized via standard PyTorch format. If training ends between intervals, the latest saved model is from the preceding interval; use `--save-every 1` to save every completed epoch, or `--save-every 0` to disable model saving.

Resumable checkpoints retain their own `run_config` and `experiment_config`. `history.jsonl` contains training losses only. Run `eval.py` separately with `--dataset val` or `--dataset test`.

To resume a periodic checkpoint, pass `--resume-checkpoint /path/to/checkpoint_epoch10.pt` with its existing output directory or a fresh one. Optimizer and scaler states and completed epoch are restored, and `--num-epochs` specifies the final total epoch count. To start a new run from checkpoint weights, pass `--init-checkpoint /path/to/checkpoint_epoch<N>.pt` to initialize a fresh optimizer and epoch counter. Repeat the original batch size, seed, learning rate, and experiment-specific arguments when continuing a run.

Existing output directories are accepted for all training modes. Matching model filenames are overwritten; other model files remain. History is appended to `history.jsonl`, creating it if absent. To continue in a fresh Kaggle notebook, only the resumable checkpoint is needed to restore training state. The training dataset must still be available.

## Code layout

- `models/model.py`: the compact question-only grounding model.
- `models/image_encoder.py` and `models/text_encoder.py`: CLIP feature extraction.
- `models/mask_decoder.py`: the decoder used by this experiment.
- `train.py`: construct the model and dataset, run training, save periodic resumable checkpoints.
- `eval.py`: construct the model, load weights, evaluate and export masks.
- `utils.py`: batch transfer and weight-loading helpers.

The separate backbone, experiment, and checkpoint modules and model factory are removed. Training uses no validation and saves only resumable checkpoints every 10 epochs by default.

Training displays a batch progress bar for each epoch (on rank 0 under `torchrun`), and evaluation displays a progress bar.
