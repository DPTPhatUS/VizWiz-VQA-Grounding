# Answer-value distillation

This branch trains a compact question-only student model using distillation from a frozen compact teacher equipped with text-conditioned skip connections. The teacher is trained with 50% answer dropout, and student distillation weights pixels by the teacher's positive answer-induced improvement (with a fixed distillation weight of 1). At inference, only the student model is used.

The repository uses `train.py`, `eval.py`, `dataset.py`, and `models/` with CLIP image inputs fixed at 336x336 pixels.

## Kaggle

Clone branch `exp/answer-value-distillation` and enable a GPU and internet access (for downloading pretrained CLIP weights). Keep Kaggle's preinstalled PyTorch and Torchvision packages and install the remaining dependencies:

```bash
pip install transformers tqdm pillow
```

The dataset root (`--data-root`) must contain `train_grounding.json`, `val_grounding.json`, `test_grounding.json`, the image directories `train/`, `val/`, `test/`, and ground-truth masks under `binary_masks_png/<split>/`. Evaluating on a split requires its ground-truth masks.

Run training and evaluation in a `%%bash` cell after setting `DATA_ROOT`:

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

Training selects CUDA automatically and supports multi-GPU execution via `torchrun` (where `--batch-size` is the global batch size across GPUs). Evaluation accepts `--device`. Pass `--help` to any script to see its CLI options.

## Single-image prediction

```bash
python predict_save.py --checkpoint /path/to/model_final_epoch100.pt --image /path/to/image.jpg --question "What does it say?" --output /kaggle/working/mask.png
python visualize_predictions.py --checkpoint /path/to/model_final_epoch100.pt --image /path/to/image.jpg --question "What does it say?" --output /kaggle/working/overlay.png
```

`IoU.py` delegates to `eval.py`. Checkpoints may be either raw `state_dict` files or full checkpoint dictionaries whose parameter names and shapes match this branch. `docs/` and local test scripts are git-ignored, while `tests/test_distillation_contracts.py` is tracked.

## Training output and separate evaluation

Both `train_teacher.py` and `train.py` load only the training split and do not run validation during or after training.

Upon completion, training writes a raw `state_dict` file named `model_final_epoch<N>.pt` containing only model tensors (no optimizer state, GradScaler state, epoch metadata, or RNG state). Automatic `last.pt` and `best.pt` exports and `--validate-every` are omitted. By default, `--save-every` is `10`, writing resumable checkpoints (`checkpoint_epoch10.pt`, `checkpoint_epoch20.pt`, etc.) that bundle model weights, optimizer state, GradScaler state, and the completed epoch. Pass `--save-every 0` to skip periodic checkpoints and save only the final raw weights.

Raw weight files load with the branch's default architecture settings, while resumable checkpoints embed `run_config` and `experiment_config` directly. `history.jsonl` logs training losses and teacher gain diagnostics. Run `eval.py` with `--dataset val` or `--dataset test` to evaluate saved checkpoints.

To initialize a new training run from saved weights (with a fresh optimizer and epoch counter), pass `--init-checkpoint /path/to/model_final_epoch<N>.pt`; `--num-epochs` then specifies the number of additional epochs to run, and student distillation still requires `--teacher-checkpoint`. To resume an interrupted run from a periodic checkpoint, pass `--resume-checkpoint /path/to/checkpoint_epoch10.pt`; this restores the optimizer, GradScaler, and epoch counter, and `--num-epochs` specifies the final total epoch count. Keep the original batch size, seed, learning rate, and experiment flags when resuming.

Existing output directories can be reused across runs: matching checkpoint filenames are overwritten, other files are preserved, and new log entries are appended to `history.jsonl`. Resuming in a fresh Kaggle session requires only the periodic `.pt` checkpoint, the training dataset, and (for student runs) the teacher checkpoint.

## Code layout

- `models/model.py`: compact question-only student and joint question/answer teacher models.
- `models/image_encoder.py` and `models/text_encoder.py`: CLIP vision and text feature extractors.
- `models/mask_decoder.py`: mask decoder module.
- `train.py` and `train_teacher.py`: dataset loading, training loops, and checkpoint saving.
- `eval.py`: checkpoint loading, native-resolution evaluation, and mask export.
- `utils.py`: batch device transfer and checkpoint loading helpers.

To evaluate or visualize standalone teacher checkpoints, pass `--teacher` to `eval.py`, `predict_save.py`, or `visualize_predictions.py` (student mode is the default). Both training (on rank 0 under `torchrun`) and evaluation display `tqdm` batch progress bars.

## Diagnostics, controls, and existing checkpoints

Model parameter names, tensor shapes, inference behavior, and the answer-value distillation loss remain compatible with existing student and compact-teacher checkpoints from this branch. In `history.jsonl`, the former `active_fraction` metric is named `mean_kd_weight_mass` (measuring mean distillation weight mass), while `kd_pixel_fraction` records spatial pixel coverage. The logged `teacher_qa_iou`, `teacher_q_iou`, and their difference are computed on 336x336 training masks as in-training diagnostics.

To train a supervised question-only compact control with the same student architecture, pass `--supervised-only` and omit `--teacher-checkpoint` (and include `--supervised-only` again if resuming that control). Default distilled student training requires a compact answer-dropout teacher trained via `train_teacher.py` on this branch.

```bash
python train.py --data-root "$DATA_ROOT" --supervised-only --seed 42 --output-dir outputs/compact-supervised
python eval.py --data-root "$DATA_ROOT" --checkpoint outputs/compact-supervised/model_final_epoch100.pt --dataset val --output-dir results/compact-supervised
```

Keep the random seed, batch size, epoch count, and native-resolution evaluation settings identical when comparing the supervised compact control against the distilled student, and select checkpoints on the validation split before running test evaluation.

Run the contract test suite offline with `python -m unittest discover -s tests -p test_distillation_contracts.py` to verify loss gradients and raw/full checkpoint loading using small synthetic encoders.

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
