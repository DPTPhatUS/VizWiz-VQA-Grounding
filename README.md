# Latent-answer distillation

This branch trains a compact question-only student with a learnable **latent answer query** and **Q/A skip gating** using single-stage self-distillation (or an optional frozen teacher via `--teacher-checkpoint`). During training, the privileged `Q + A` branch supervises both the latent answer representation (via cosine + Smooth L1 alignment) and the spatial mask predictions (via answer-value gain-weighted KL distillation); at inference, only the question and image are needed.

The experiment uses the existing `train.py`, `eval.py`, `dataset.py`, and `models/` layout. CLIP input is fixed at 336 pixels; architecture and loss settings are defined in code.

## Kaggle

Clone branch `exp/latent-answer-distillation`. Enable a GPU and Internet for CLIP downloads. Use Kaggle's installed Torch/Torchvision and install missing dependencies:

```bash
pip install transformers tqdm pillow
```

The dataset root contains `train_grounding.json`, `val_grounding.json`, `test_grounding.json`, image directories `train/`, `val/`, `test/`, and masks in `binary_masks_png/<split>/`. Evaluation requires masks for the selected split.

Run in a `%%bash` cell, replacing the dataset path:

```bash
set -e
cd /kaggle/working/VizWiz-VQA-Grounding
DATA_ROOT=/kaggle/input/YOUR_DATASET/vizwiz
# Single-stage self-distillation (no separate teacher training required; optional --teacher-checkpoint or --init-checkpoint supported).
python train.py --data-root "$DATA_ROOT" --num-epochs 100 --batch-size 4 --num-workers 4 --seed 42 --output-dir /kaggle/working/outputs

python eval.py --data-root "$DATA_ROOT" --checkpoint /kaggle/working/outputs/model_final_epoch100.pt --dataset test --batch-size 4 --num-workers 4 --output-dir /kaggle/working/results
```

Training selects CUDA automatically and supports `torchrun`; batch size is global across GPUs. `--seed`, `--num-workers`, and the usual training/data/output controls are retained. Evaluation accepts an optional `--device`. Run either script with `--help` for its short argument list.

## Single-image prediction

```bash
python predict_save.py --checkpoint /path/to/model_final_epoch100.pt --image /path/to/image.jpg --question "What does it say?" --output /kaggle/working/mask.png
python visualize_predictions.py --checkpoint /path/to/model_final_epoch100.pt --image /path/to/image.jpg --question "What does it say?" --output /kaggle/working/overlay.png
```

`IoU.py` delegates to `eval.py`. Existing experiment weight keys are preserved. Raw weights and ordinary checkpoint dictionaries are accepted; model tensor names and shapes must match this branch. `docs/` and unrelated local tests remain ignored; the latent regression test is tracked. CPU smoke tests do not establish real-data accuracy or GPU performance.


## Training output and separate evaluation

Training reads only the training split: no validation dataset is loaded and no validation runs, including after the final epoch. This applies to both teacher and student training.

At completion, training saves exactly one raw `state_dict`: `model_final_epoch<N>.pt`, matching the baseline format. It contains model tensors only, with no optimizer, scaler, epoch wrapper, or RNG state. There are no automatic `last.pt` or `best.pt` saves, and `--validate-every` is removed. `--save-every` defaults to **10** and saves resumable `checkpoint_epoch10.pt`, `checkpoint_epoch20.pt`, etc., containing model, optimizer, scaler, and completed epoch. Use `--save-every 0` to disable these periodic checkpoints while still saving the final raw model.

Raw weights use this branch's default model settings. Resumable checkpoints retain their own `run_config` and `experiment_config`; no standalone configuration file is written or read. `history.jsonl` contains training losses, teacher/gain diagnostics, and the objective version. Run `eval.py` separately with `--dataset val` or `--dataset test`; the commands above show the final weight filename for their epoch count.

To train further from these weights, use `--init-checkpoint /path/to/model_final_epoch<N>.pt`. This starts a new optimizer and epoch counter; `--num-epochs` is the number of additional epochs. Default self-distillation requires no separate teacher; repeat `--teacher-checkpoint` only for external-teacher runs. To resume a periodic checkpoint, use `--resume-checkpoint /path/to/checkpoint_epoch10.pt` with its existing output directory or a fresh one; optimizer/scaler state and completed epoch are restored, and `--num-epochs` is the final total epoch count. Repeat the original batch size, seed, learning rate, and experiment-specific training arguments when continuing a run. Older supported full checkpoints remain loadable.

Existing output directories are accepted for all training modes. Matching model filenames are overwritten; other model files remain. History is appended to `history.jsonl`, creating it if absent. To continue in a fresh Kaggle notebook, only the resumable checkpoint is needed to restore training state; no previous history or separate configuration file is required. The training dataset must still be available. Default student continuation remains single-stage and needs no external teacher. Only runs started with `--teacher-checkpoint` need that same external teacher for continuation; optional teacher continuation uses `train_teacher.py`.


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


## Corrected single-stage objective (latent-gain-v2)

Default training is still one run: the inferred-answer student, real-answer oracle, and Q-only reference share one model and one optimizer. The reference removes the answer token, uses question-only skip conditioning, and receives BCE + 0.5 Dice supervision. It no longer substitutes a question vector for an answer inside an untrained reference branch. The real-answer oracle receives its own segmentation supervision. Its KD targets and gains are detached; its segmentation loss and the Q-reference loss retain gradients.

The default total loss is `student_seg + 0.5 * oracle_seg + 0.5 * question_seg + KD + 0.5 * latent_alignment + 0.2 * attention_guidance`, where each segmentation term uses BCE + 0.5 Dice. Attention guidance pools foreground gain and foreground support to the patch grid, normalizes each independently, and mixes 75% gain with 25% foreground prior. When foreground gain is absent, it falls back to the foreground prior. Empty foreground and missing-answer samples contribute zero attention loss. Missing real answers are masked out of the oracle answer token and latent/KD alignment.

`mean_kd_weight_mass` replaces the misleading `active_fraction`; `kd_pixel_fraction` measures pixel coverage. Teacher QA/Q IoU and their difference are diagnostics on resized training masks, not validation scores. `attention_gain_image_fraction` reports how often foreground gain is available. Epoch history and resumable checkpoints identify `latent-gain-v2`; raw final weights remain model tensors only.

For the matched architecture/loss control, use `--supervised-only`: it trains the same inferred-answer student with BCE + 0.5 Dice, without oracle/Q-reference losses, latent alignment, KD, or attention supervision. Repeat this flag when continuing the control. All experiments should use matched seeds, batch sizes, epoch budgets, native-resolution evaluation, and validation checkpoint selection.

## Checkpoint compatibility and retraining

No model parameters were added, removed, or reshaped. Complete latent checkpoints from the previous version still load strictly for inference/resume, and default inference is unchanged. Existing periodic optimizer/scaler states also remain compatible. However, these checkpoints were trained with the old objective. Train from scratch to report results for `latent-gain-v2`; using `--init-checkpoint` or `--resume-checkpoint` with old weights is a warm start or mixed-objective continuation and should be labeled accordingly. Keep outputs separate from old runs.

Legacy compact checkpoints lacking the latent-answer/gating tensors are permitted only for explicit student `--init-checkpoint` initialization followed by training. Inference, resume, and frozen external-teacher loading require complete model tensors and reject such partial checkpoints. The optional external teacher must have the full latent architecture trained with answer dropout. The default self-distillation run does not need this teacher or a separate training stage.

Run offline regression checks with `python -m unittest discover -s tests -p test_latent_distillation.py`. These cover gain-sensitive attention targets, Q-reference supervision, missing-answer masking, strict teacher loading, and old raw/full checkpoint inference and optimizer compatibility with tiny encoders. They do not establish real-data accuracy or GPU execution.
