# Gain-guided refinement

This branch trains a compact grounding model with a high-resolution crop refiner and an expected-IoU-gain router. Inputs and evaluation use only the question. The coarse model stays frozen; the refiner is also frozen while training the router.

Install the project dependencies, or keep your notebook's Torch/Torchvision installation and install `transformers tqdm pillow numpy`. Run from this branch with a VizWiz root containing `<split>_grounding.json`, `<split>/` images, and `binary_masks_png/<split>/` masks.

```bash
python train.py --data-root /path/to/vizwiz --stage refiner --init-checkpoint /path/to/compact-controls/model_final_epoch100.pt --output-dir outputs-refiner --num-epochs 30 --batch-size 1
python train.py --data-root /path/to/vizwiz --stage router --init-checkpoint outputs-refiner/model_final_epoch30.pt --output-dir outputs-router --num-epochs 30 --batch-size 1
python eval.py --data-root /path/to/vizwiz --checkpoint outputs-router/model_final_epoch30.pt --dataset test --output-dir results-router --budget 2
```

The refiner starts from a question-only compact controls checkpoint. The router starts from the refiner stage. Either stage can also initialize from its own saved weights. Images use a 336-pixel coarse input and a 672-pixel detail canvas; crops use 336 pixels. Refiner evaluation uses fixed coverage, and router evaluation selects positive predicted gains. `--budget` chooses up to 1, 2, or 4 crops. Separate evaluation and exported masks use original image dimensions.

```bash
python predict_save.py --checkpoint outputs-router/model_final_epoch30.pt --image example.jpg --question "Where is the label?" --output mask.png
python visualize_predictions.py --checkpoint outputs-router/model_final_epoch30.pt --image example.jpg --question "Where is the label?" --output overlay.png
```

Use `train.py --help` and `eval.py --help` for operational options. Real-data accuracy and GPU throughput require training; tiny models are used only by local offline tests.

## Training output and separate evaluation

Training reads only the training split: no validation dataset is loaded and no validation runs, including after the final epoch. This applies to both the refiner and router stages.

At completion, training saves exactly one raw `state_dict`: `model_final_epoch<N>.pt`, matching the baseline format. It contains model tensors only, with no optimizer, scaler, epoch wrapper, or RNG state. There are no automatic `last.pt` or `best.pt` saves, and `--validate-every` is removed. `--save-every` defaults to **10** and saves resumable `checkpoint_epoch10.pt`, `checkpoint_epoch20.pt`, etc., containing model, optimizer, scaler, and completed epoch. Use `--save-every 0` to disable these periodic checkpoints while still saving the final raw model.

Raw weights use this branch's default model settings. Resumable checkpoints retain their own `run_config` and `experiment_config`; no standalone configuration file is written or read. `history.jsonl` contains training losses only. Run `eval.py` separately with `--dataset val` or `--dataset test`; the commands above show the final weight filename for their epoch count.

To train further from these weights, use `--init-checkpoint /path/to/model_final_epoch<N>.pt`. This starts a new optimizer and epoch counter; `--num-epochs` is the number of additional epochs. To resume a periodic checkpoint, use `--resume-checkpoint /path/to/checkpoint_epoch10.pt` with its existing output directory or a fresh one; optimizer/scaler state and completed epoch are restored, and `--num-epochs` is the final total epoch count. Repeat the original batch size, seed, learning rate, and experiment-specific training arguments when continuing a run. Older supported full checkpoints remain loadable.

Existing output directories are accepted for all training modes. Matching model filenames are overwritten; other model files remain. History is appended to `history.jsonl`, creating it if absent. To continue in a fresh Kaggle notebook, only the resumable checkpoint is needed to restore training state; no previous history or separate configuration file is required. The training dataset must still be available. Repeat the saved `--stage` when resuming; the full checkpoint already contains the frozen coarse and refiner weights, so the original initialization file is unnecessary.


## Code layout

- `models/model.py`: the frozen compact coarse model, crop refiner, and gain router.
- `models/image_encoder.py` and `models/text_encoder.py`: CLIP feature extraction.
- `models/mask_decoder.py`: the decoder used by this experiment.
- `train.py`: construct the model and dataset, run training, save checkpoints and final weights.
- `eval.py`: construct the model, load weights, evaluate and export masks.
- `utils.py`: batch transfer and a few plain weight-loading helpers.

The separate backbone, experiment, and checkpoint modules and model factory are removed. Training still uses no validation, saves resumable checkpoints every 10 epochs by default, and writes final raw weights.

For standalone refiner weights, use `--stage refiner` with `eval.py`, `predict_save.py`, or `visualize_predictions.py`. Router inference is the default when no saved stage is available.

Training shows a batch progress bar for each epoch (rank 0 under `torchrun`), and evaluation shows a batch progress bar. Install `tqdm` alongside Transformers and Pillow in the Kaggle notebook.
