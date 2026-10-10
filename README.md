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
python eval.py --data-root "$DATA_ROOT" --checkpoint /kaggle/working/outputs-extent/model_final_epoch100.pt --dataset test --batch-size 1 --num-workers 2 --output-dir /kaggle/working/results-extent
```

Training selects CUDA when available. `--init-checkpoint /path/to/compact/model_final_epoch100.pt` optionally initializes the coarse model from a compatible corrected compact checkpoint. It starts a fresh optimizer. Matching extent checkpoints can also initialize a run.

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
python predict_save.py --checkpoint /path/to/model_final_epoch100.pt --image /path/to/image.jpg --question "What does it say?" --output mask.png
python visualize_predictions.py --checkpoint /path/to/model_final_epoch100.pt --image /path/to/image.jpg --question "What does it say?" --output overlay.png
```

`IoU.py` delegates to the same evaluator. `models/model.py` contains the extent model, the compact coarse model is in that same file, `dataset.py` the aligned data loader, and `losses.py` the objective. `docs/` and `tests/` remain ignored local development files.

## Training output and separate evaluation

Training reads only the training split: no validation dataset is loaded and no validation runs, including after the final epoch. Paired question supervision is training-only.

At completion, training saves exactly one raw `state_dict`: `model_final_epoch<N>.pt`, matching the baseline format. It contains model tensors only, with no optimizer, scaler, epoch wrapper, or RNG state. There are no automatic `last.pt` or `best.pt` saves, and `--validate-every` is removed. `--save-every` defaults to **10** and saves resumable `checkpoint_epoch10.pt`, `checkpoint_epoch20.pt`, etc., containing model, optimizer, scaler, and completed epoch. Use `--save-every 0` to disable these periodic checkpoints while still saving the final raw model.

Raw weights use this branch's default model settings. Resumable checkpoints retain their own `run_config` and `experiment_config`; no standalone configuration file is written or read. `history.jsonl` contains training losses only. Run `eval.py` separately with `--dataset val` or `--dataset test`; the commands above show the final weight filename for their epoch count.

To train further from these weights, use `--init-checkpoint /path/to/model_final_epoch<N>.pt`. This starts a new optimizer and epoch counter; `--num-epochs` is the number of additional epochs. To resume a periodic checkpoint, use `--resume-checkpoint /path/to/checkpoint_epoch10.pt` with its existing output directory or a fresh one; optimizer/scaler state and completed epoch are restored, and `--num-epochs` is the final total epoch count. Repeat the original batch size, seed, learning rate, and experiment-specific training arguments when continuing a run. Older supported full checkpoints remain loadable.

Existing output directories are accepted for all training modes. Matching model filenames are overwritten; other model files remain. History is appended to `history.jsonl`, creating it if absent. To continue in a fresh Kaggle notebook, only the resumable checkpoint is needed to restore training state; no previous history or separate configuration file is required. The training dataset must still be available. To continue a paired run, repeat `--pairs` and provide the manifest and all masks it references.


## Code layout

- `models/model.py`: the compact coarse model and the two-token location/extent model.
- `models/image_encoder.py` and `models/text_encoder.py`: CLIP feature extraction.
- `models/mask_decoder.py`: the decoder used by this experiment.
- `train.py`: construct the model and dataset, run training, save checkpoints and final weights.
- `eval.py`: construct the model, load weights, evaluate and export masks.
- `utils.py`: batch transfer and a few plain weight-loading helpers.

The separate backbone, experiment, and checkpoint modules and model factory are removed. Training still uses no validation, saves resumable checkpoints every 10 epochs by default, and writes final raw weights.

Training shows a batch progress bar for each epoch (rank 0 under `torchrun`), and evaluation shows a batch progress bar. Install `tqdm` alongside Transformers and Pillow in the Kaggle notebook.
