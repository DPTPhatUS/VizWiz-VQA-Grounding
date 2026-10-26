# Gain-guided refinement

This experiment adds a learned refinement stage that selects local image crops and predicts mask corrections based on expected IoU gain.

The model uses a coarse-to-fine structure:
1. A frozen coarse model produces initial 336x336 mask logits.
2. An evidence router scores potential crop locations.
3. A local refiner predicts signed corrections on selected high-resolution crops.

Inputs are image + question; ground-truth answers are not used at inference time.

## Kaggle

Clone branch `exp/gain-guided-refinement` and enable a GPU with internet access (for downloading pretrained CLIP weights). Install the required dependencies:

```bash
pip install transformers tqdm pillow
```

The dataset root (`--data-root`) must contain `train_grounding.json`, `val_grounding.json`, `test_grounding.json`, image directories `train/`, `val/`, `test/`, and ground-truth masks under `binary_masks_png/<split>/`.

Run in a `%%bash` cell, replacing the dataset path:

```bash
python train.py --data-root /path/to/vizwiz --stage refiner --init-checkpoint /path/to/compact-controls/checkpoint_epoch100.pt --output-dir outputs-refiner --num-epochs 30 --batch-size 1
python train.py --data-root /path/to/vizwiz --stage router --init-checkpoint outputs-refiner/checkpoint_epoch30.pt --output-dir outputs-router --num-epochs 30 --batch-size 1
python eval.py --data-root /path/to/vizwiz --checkpoint outputs-router/checkpoint_epoch30.pt --dataset test --output-dir results-router --budget 2
```

The refiner initializes from a question-only compact controls checkpoint. The router initializes from the trained refiner stage. Either stage can also initialize from its own saved weights. Images use a 336-pixel coarse input and a 672-pixel detail canvas; crops use 336 pixels. Refiner evaluation uses fixed coverage, and router evaluation selects positive predicted gains. `--budget` chooses up to 1, 2, or 4 crops. Separate evaluation and exported masks use original image dimensions.

## Single-image prediction

```bash
python predict_save.py --checkpoint outputs-router/checkpoint_epoch30.pt --image example.jpg --question "Where is the label?" --output mask.png
python visualize_predictions.py --checkpoint outputs-router/checkpoint_epoch30.pt --image example.jpg --question "Where is the label?" --output overlay.png
```

Pass `--help` to `train.py` or `eval.py` for CLI options.

## Training output and separate evaluation

`train.py` loads only the training split and does not run validation during or after training. This applies to both the refiner and router stages.

Training saves periodic resumable checkpoints (`checkpoint_epoch10.pt`, `checkpoint_epoch20.pt`, etc.). `--save-every` defaults to **10**. Each checkpoint contains model weights, optimizer/scaler state, completed epoch, and embedded run configuration, serialized via standard PyTorch format. If training ends between intervals, the latest saved model is from the preceding interval; use `--save-every 1` to save every completed epoch, or `--save-every 0` to disable model saving.

Existing raw weights remain loadable using this branch's default model settings. Resumable checkpoints retain their own `run_config` and `experiment_config`. `history.jsonl` contains training losses only. Run `eval.py` separately with `--dataset val` or `--dataset test`.

To resume a periodic checkpoint, pass `--resume-checkpoint /path/to/checkpoint_epoch10.pt` with its existing output directory or a fresh one. Optimizer and scaler states and completed epoch are restored, and `--num-epochs` specifies the final total epoch count. To start a new run from checkpoint weights, pass `--init-checkpoint /path/to/checkpoint_epoch<N>.pt` to initialize a fresh optimizer and epoch counter. Repeat the original batch size, seed, learning rate, and experiment-specific arguments when continuing a run. Older supported full checkpoints remain loadable.

Existing output directories are accepted for all training modes. Matching model filenames are overwritten; other model files remain. History is appended to `history.jsonl`, creating it if absent. To continue in a fresh Kaggle notebook, only the resumable checkpoint is needed to restore training state. The training dataset must still be available. Repeat the saved `--stage` when resuming; the full checkpoint already contains the frozen coarse and refiner weights, so the original initialization file is unnecessary.

## Code layout

- `models/model.py`: the frozen compact coarse model, crop refiner, and gain router.
- `models/image_encoder.py` and `models/text_encoder.py`: CLIP feature extraction.
- `models/mask_decoder.py`: the decoder used by this experiment.
- `train.py`: construct the model and dataset, run training, save periodic resumable checkpoints.
- `eval.py`: construct the model, load weights, evaluate and export masks.
- `utils.py`: batch transfer and weight-loading helpers.

The separate backbone, experiment, and checkpoint modules and model factory are removed. Training uses no validation and saves only resumable checkpoints every 10 epochs by default.

For existing standalone refiner weights, pass `--stage refiner` with `eval.py`, `predict_save.py`, or `visualize_predictions.py`. Router inference is the default when no saved stage is specified.

Training displays a batch progress bar for each epoch (on rank 0 under `torchrun`), and evaluation displays a progress bar.
