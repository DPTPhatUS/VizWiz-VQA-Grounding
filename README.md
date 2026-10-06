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
python train_teacher.py --data-root "$DATA_ROOT" --num-epochs 100 --batch-size 1 --num-workers 2 --seed 42 --output-dir /kaggle/working/teacher

# Train the student. Optional --init-checkpoint may point to a compact-control checkpoint.
python train.py --data-root "$DATA_ROOT" --teacher-checkpoint /kaggle/working/teacher/model_final_epoch100.pt --num-epochs 100 --batch-size 1 --num-workers 2 --seed 42 --output-dir /kaggle/working/outputs

python eval.py --data-root "$DATA_ROOT" --checkpoint /kaggle/working/outputs/model_final_epoch100.pt --dataset test --batch-size 1 --num-workers 2 --output-dir /kaggle/working/results
```

Training selects CUDA automatically and supports `torchrun`; batch size is global across GPUs. `--seed`, `--num-workers`, and the usual training/data/output controls are retained. Evaluation accepts an optional `--device`. Run either script with `--help` for its short argument list.

## Single-image prediction

```bash
python predict_save.py --checkpoint /path/to/model_final_epoch100.pt --image /path/to/image.jpg --question "What does it say?" --output /kaggle/working/mask.png
python visualize_predictions.py --checkpoint /path/to/model_final_epoch100.pt --image /path/to/image.jpg --question "What does it say?" --output /kaggle/working/overlay.png
```

`IoU.py` delegates to `eval.py`. Corrected compact experiment checkpoint keys are preserved; historical unversioned baseline checkpoints are not interchangeable. `docs/` and `tests/` remain ignored and local. CPU smoke tests do not establish real-data accuracy or GPU performance.


## Training output and separate evaluation

Training reads only the training split: no validation dataset is loaded and no validation runs, including after the final epoch. This also applies to teacher training and both refinement stages.

At completion, training saves exactly one raw `state_dict`: `model_final_epoch<N>.pt`, matching the baseline format. It contains model tensors only, with no optimizer, scaler, epoch wrapper, or RNG state. There are no automatic `last.pt`, `best.pt`, or periodic checkpoint saves; `--validate-every` and `--save-every` have been removed.

Keep the existing small `config.json` beside the weight file when copying outputs to Kaggle or another directory. The evaluation, prediction, teacher, and refinement loaders use it to identify the architecture and stage. `history.jsonl` contains training losses only. Run `eval.py` separately with `--dataset val` or `--dataset test`; the commands above show the final weight filename for their epoch count.

To train further from these weights, use `--init-checkpoint /path/to/model_final_epoch<N>.pt` and a fresh output directory. This starts a new optimizer and epoch counter; `--num-epochs` is the number of additional epochs. Distillation still requires its teacher. `--resume-checkpoint` remains available only for older full checkpoints with optimizer state; their `--num-epochs` is the final total epoch count. New training does not produce full checkpoints.
