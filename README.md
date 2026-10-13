# Answer-value distillation

This branch trains a compact question-only student using a frozen compact teacher with text-conditioned skip features. The teacher is trained with 50% answer dropout. Student distillation uses positive answer-induced teacher improvement, with fixed weight 1; the teacher is unnecessary at inference.

The experiment uses the existing `train.py`, `eval.py`, `dataset.py`, and `models/` layout. CLIP input is fixed at 336 pixels; architecture and loss settings are defined in code.

## Kaggle

Clone branch `exp/answer-value-distillation`. Enable a GPU and Internet for CLIP downloads. Use Kaggle's installed Torch/Torchvision and install missing dependencies:

```bash
pip install transformers tqdm pillow
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

`IoU.py` delegates to `eval.py`. Existing experiment weight keys are preserved. Raw weights and ordinary checkpoint dictionaries are accepted; model tensor names and shapes must match this branch. `docs/` and unrelated local tests remain ignored; the distillation regression test is tracked. CPU smoke tests do not establish real-data accuracy or GPU performance.


## Training output and separate evaluation

Training reads only the training split: no validation dataset is loaded and no validation runs, including after the final epoch. This applies to both teacher and student training.

At completion, training saves exactly one raw `state_dict`: `model_final_epoch<N>.pt`, matching the baseline format. It contains model tensors only, with no optimizer, scaler, epoch wrapper, or RNG state. There are no automatic `last.pt` or `best.pt` saves, and `--validate-every` is removed. `--save-every` defaults to **10** and saves resumable `checkpoint_epoch10.pt`, `checkpoint_epoch20.pt`, etc., containing model, optimizer, scaler, and completed epoch. Use `--save-every 0` to disable these periodic checkpoints while still saving the final raw model.

Raw weights use this branch's default model settings. Resumable checkpoints retain their own `run_config` and `experiment_config`; no standalone configuration file is written or read. `history.jsonl` contains training losses and teacher/gain diagnostics only. Run `eval.py` separately with `--dataset val` or `--dataset test`; the commands above show the final weight filename for their epoch count.

To train further from these weights, use `--init-checkpoint /path/to/model_final_epoch<N>.pt`. This starts a new optimizer and epoch counter; `--num-epochs` is the number of additional epochs. Distillation still requires its teacher. To resume a periodic checkpoint, use `--resume-checkpoint /path/to/checkpoint_epoch10.pt` with its existing output directory or a fresh one; optimizer/scaler state and completed epoch are restored, and `--num-epochs` is the final total epoch count. Repeat the original batch size, seed, learning rate, and experiment-specific training arguments when continuing a run. Older supported full checkpoints remain loadable.

Existing output directories are accepted for all training modes. Matching model filenames are overwritten; other model files remain. History is appended to `history.jsonl`, creating it if absent. To continue in a fresh Kaggle notebook, only the resumable checkpoint is needed to restore training state; no previous history or separate configuration file is required. The training dataset must still be available. Student continuation also requires its teacher checkpoint; teacher continuation uses `train_teacher.py`.


## Code layout

- `models/model.py`: the compact question-only student and joint question/answer teacher.
- `models/image_encoder.py` and `models/text_encoder.py`: CLIP feature extraction.
- `models/mask_decoder.py`: the decoder used by this experiment.
- `train.py`: construct the model and dataset, run training, save checkpoints and final weights.
- `eval.py`: construct the model, load weights, evaluate and export masks.
- `utils.py`: batch transfer and a few plain weight-loading helpers.

The separate backbone, experiment, and checkpoint modules and model factory are removed. Training still uses no validation, saves resumable checkpoints every 10 epochs by default, and writes final raw weights.

For standalone teacher weights, use `--teacher` with `eval.py`, `predict_save.py`, or `visualize_predictions.py`. Student inference is the default.

Training shows a batch progress bar for each epoch (rank 0 under `torchrun`), and evaluation shows a batch progress bar. Install `tqdm` alongside Transformers and Pillow in the Kaggle notebook.


## Diagnostics, controls, and existing checkpoints

The answer-value loss, model parameter names/shapes, and inference are unchanged. Existing student and compact-teacher checkpoints remain usable for evaluation and continuation without retraining. `active_fraction` is renamed to `mean_kd_weight_mass`; it measures mean weight mass, not pixel coverage. `kd_pixel_fraction` reports pixel coverage. `teacher_qa_iou`, `teacher_q_iou`, and their difference are paired training-mask diagnostics at 336 pixels, not native-resolution validation results. Existing history rows retain their old field names.

For a supervised compact control, train the identical student with `--supervised-only`, omitting `--teacher-checkpoint`. Repeat that flag when resuming the control; raw weights do not record the training objective. Default training still requires a separately trained answer-dropout compact teacher. This teacher is not the wide model from `exp/joint-text-skips`.

```bash
python train.py --data-root "$DATA_ROOT" --supervised-only --seed 42 --output-dir outputs/compact-supervised
python eval.py --data-root "$DATA_ROOT" --checkpoint outputs/compact-supervised/model_final_epoch100.pt --dataset val --output-dir results/compact-supervised
```

Use the same seeds, batch size, epochs, and native-resolution evaluator for the compact control and distilled student. Select checkpoints on validation before final test reporting. The QA baseline in `exp/baseline-fixes` is a separate full-information setting.

Run offline checks with `python -m unittest discover -s tests -p test_distillation_contracts.py`. These validate unchanged loss/gradients and raw/full checkpoint loading using tiny encoders, not real-data performance.
