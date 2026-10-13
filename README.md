# VizWiz VQA Grounding — baseline fixes

CLIP grounding baseline with cross-attention and the original mask decoder. Model parameter names and shapes are unchanged by the evaluation and text-mode controls below.

Given an image, question, and answer, predict the supporting binary mask. Evaluation reports mean Intersection over Union (IoU). Inputs use the fixed 336 × 336 CLIP resolution.

## Setup

```bash
uv sync --locked
```

Place `train_grounding.json`, `val_grounding.json`, and `test_grounding.json` under `data/vizwiz/`, alongside `train/`, `val/`, `test/`, and `binary_masks_png/{train,val,test}/`. Use `--data-root` for another location. The first model run downloads the pretrained CLIP encoders if they are not cached.

## Train and evaluate

```bash
uv run train.py --data-root data/vizwiz --num-epochs 100 --batch-size 4 --output-dir outputs
uv run eval.py --checkpoint outputs/checkpoint_epoch100.pt --dataset val --data-root data/vizwiz --output-dir results/main
```

Training also supports `--lr`, `--num-workers`, `--seed`, `--resume-checkpoint`, `--validate-every`, and `--save-every`. Set `--save-every 0` to save only the final model; validation defaults to disabled. Use `torchrun` for distributed training. Evaluation requires a checkpoint and supports `--device`, `--batch-size`, and `--num-workers`; use `--dataset test` for the test split.

## Comparable evaluation and question-only control

Evaluation now defaults to interpolating logits to each original GT mask size before thresholding at 0.5. The scored mask is also the exported mask. Empty prediction and empty GT score 1, matching the distillation branches. Missing or misaligned native masks fail explicitly. CLIP inputs and training masks remain 336 by 336.

Existing QA checkpoints remain usable without retraining. Re-evaluate them under this native-resolution protocol before comparing with the distillation branches. Use `--metric-resolution resized --text-mode qa` to reproduce the legacy 336-pixel evaluation, including its empty-union score of 0. Result metadata records resolution, text mode, and empty-union convention.

Training and evaluation default to `--text-mode qa`, preserving the existing QA baseline. For a question-only control, train a separate run and evaluate it in question mode:

```bash
uv run train.py --data-root data/vizwiz --text-mode question --output-dir outputs/question-control
uv run eval.py --checkpoint outputs/question-control/model_final_epoch100.pt --dataset val --data-root data/vizwiz --text-mode question --output-dir results/question-control
```

Repeat `--text-mode question` when resuming that control or evaluating its raw weights; checkpoint tensor shapes do not encode text mode. Evaluating a QA-trained checkpoint with Q-only input is an answer-removal diagnostic, not a separately trained Q-only baseline. The compact distillation students also require a supervised compact control to isolate KD from decoder differences. Use validation for model selection and report QA and Q-only results separately.

Run the offline protocol checks with `python -m unittest discover -s tests -p test_baseline_protocol.py`. They cover native masks of different sizes, raw/full checkpoint loading, answer removal, and legacy/empty-mask scoring using synthetic data; they do not establish VizWiz accuracy or GPU performance.

## Project

Research fork of [yjh9929/VizWiz-VQA-Grounding](https://github.com/yjh9929/VizWiz-VQA-Grounding). The original baseline was developed for the [VizWiz VQA Grounding Challenge](https://vizwiz.org/tasks-and-datasets/visual-qa/) 2025 and selected as a workshop spotlight at CVPR 2025.

Licensed under [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/).

## Repository scope

This checkout contains research code for data preparation, training, evaluation, and prediction. The upstream static website and its decorative assets have been removed. The original presentation remains available in the upstream repository.

The retained upstream figures are historical illustrations, not results from this fork's experiments: [architecture](images/model.png), [grounding examples](images/task.png), and [prediction examples 1](images/m1.png), [2](images/m2.png), [3](images/m3.png).
