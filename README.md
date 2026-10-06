# Answer-value distillation

This branch trains a compact question-only student using a frozen joint-skip teacher. The teacher is trained with 50% answer dropout. Student distillation uses positive answer-induced teacher improvement, with fixed weight 1; the teacher is unnecessary at inference.

The experiment uses the existing `train.py`, `eval.py`, `dataset.py`, and `models/` layout. CLIP input is fixed at 336 pixels; architecture and loss settings are defined in code. No detector is involved. The offline detector experiment lives on `exp/yolo-detector`.

## Kaggle

Clone branch `exp/answer-value-distillation`. Enable a GPU and Internet for CLIP downloads. Use Kaggle's installed Torch/Torchvision and install missing dependencies:

```bash
pip install transformers matplotlib tqdm pillow
```

The dataset root contains `train_grounding.json`, `val_grounding.json`, `test_grounding.json`, image directories `train/`, `val/`, `test/`, and masks in `binary_masks_png/<split>/`. Evaluation requires masks for the selected split.

Run in a `%%bash` cell, replacing the dataset path:

```bash
set -e
cd /kaggle/working/VizWiz-VQA-Grounding
DATA_ROOT=/kaggle/input/YOUR_DATASET/vizwiz
# Train this experiment's teacher first.
python train_teacher.py --data-root "$DATA_ROOT" --num-epochs 100 --batch-size 1 --num-workers 2 --seed 42 --save-every 0 --output-dir /kaggle/working/teacher

# Train the student. Optional --init-checkpoint may point to a compact-control checkpoint.
python train.py --data-root "$DATA_ROOT" --teacher-checkpoint /kaggle/working/teacher/best.pt --num-epochs 100 --batch-size 1 --num-workers 2 --seed 42 --save-every 0 --output-dir /kaggle/working/outputs

python eval.py --data-root "$DATA_ROOT" --checkpoint /kaggle/working/outputs/best.pt --dataset test --batch-size 1 --num-workers 2 --output-dir /kaggle/working/results
```

Validation uses question-only inputs and original-resolution mean IoU. `best.pt` saves the best validation score; `last.pt` saves every epoch. `--save-every 0` disables additional numbered checkpoints. Evaluation writes masks and `metrics.json`.

For continuation, repeat the training command with `--resume-checkpoint /path/to/last.pt` and a larger `--num-epochs` (the final epoch count). Resume restores optimizer state and applies the requested `--lr` (default `1e-5`). Use the same training data and seed. You can resume a Kaggle input checkpoint into a fresh writable output directory; its `best.pt` is selected among the continued epochs. An existing output directory may only resume its own `last.pt`.

Training selects CUDA automatically and supports `torchrun`; batch size is global across GPUs. `--seed`, `--validate-every`, `--save-every`, `--num-workers`, and the usual training/data/output controls are retained. Evaluation accepts an optional `--device`. Run either script with `--help` for its short argument list.

## Single-image prediction

```bash
python predict_save.py --checkpoint /path/to/best.pt --image /path/to/image.jpg --question "What does it say?" --output /kaggle/working/mask.png
python visualize_predictions.py --checkpoint /path/to/best.pt --image /path/to/image.jpg --question "What does it say?" --output /kaggle/working/overlay.png
```

`IoU.py` delegates to `eval.py`. Corrected compact experiment checkpoint keys are preserved; historical unversioned baseline checkpoints are not interchangeable. `docs/` and `tests/` remain ignored and local. CPU smoke tests do not establish real-data accuracy or GPU performance.
