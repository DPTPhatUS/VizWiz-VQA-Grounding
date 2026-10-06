# VizWiz VQA Grounding — evidence extent

This branch predicts evidence masks from an image and question. It extends the compact CLIP grounding model with two learned tokens for evidence location and extent. Location supervision uses the mask's bounding rectangle; final-mask and area losses learn the extent, including empty masks.

Use `train.py` and `eval.py`. This branch fixes the image size at 336, token width at 64, support loss weight at 0.2, and area loss weight at 0.1. Training and evaluation use questions only. There are no architecture, text-mode, test-model, or loss-weight switches. The model adds 148,545 parameters to the compact backbone; real-data accuracy and GPU throughput remain unmeasured.

## Kaggle

Enable a GPU and Internet for pretrained CLIP downloads. Keep Kaggle's Torch/Torchvision installation and install missing dependencies:

```bash
pip install transformers matplotlib pillow
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

`IoU.py` delegates to the same evaluator. `models/model.py` contains the extent model, `models/backbone.py` its compact backbone, `dataset.py` the aligned data loader, and `losses.py` the objective. Detector/YOLO and alternate box-generation paths are removed. `docs/` and `tests/` remain ignored local development files.

## Training output and separate evaluation

Training reads only the training split: no validation dataset is loaded and no validation runs, including after the final epoch. This also applies to teacher training and both refinement stages.

At completion, training saves exactly one raw `state_dict`: `model_final_epoch<N>.pt`, matching the baseline format. It contains model tensors only, with no optimizer, scaler, epoch wrapper, or RNG state. There are no automatic `last.pt`, `best.pt`, or periodic checkpoint saves; `--validate-every` and `--save-every` have been removed.

Keep the existing small `config.json` beside the weight file when copying outputs to Kaggle or another directory. The evaluation, prediction, teacher, and refinement loaders use it to identify the architecture and stage. `history.jsonl` contains training losses only. Run `eval.py` separately with `--dataset val` or `--dataset test`; the commands above show the final weight filename for their epoch count.

To train further from these weights, use `--init-checkpoint /path/to/model_final_epoch<N>.pt` and a fresh output directory. This starts a new optimizer and epoch counter; `--num-epochs` is the number of additional epochs. Distillation still requires its teacher. `--resume-checkpoint` remains available only for older full checkpoints with optimizer state; their `--num-epochs` is the final total epoch count. New training does not produce full checkpoints.
