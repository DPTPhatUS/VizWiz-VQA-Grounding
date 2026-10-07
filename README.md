# VizWiz VQA Grounding — main

Original CLIP grounding baseline with cross-attention and the original mask decoder. This checkout runs the `main` model through `train.py` and `eval.py`.

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

## Project

Research fork of [yjh9929/VizWiz-VQA-Grounding](https://github.com/yjh9929/VizWiz-VQA-Grounding). The original baseline was developed for the [VizWiz VQA Grounding Challenge](https://vizwiz.org/tasks-and-datasets/visual-qa/) 2025 and selected as a workshop spotlight at CVPR 2025.

Licensed under [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/).

## Repository scope

This checkout contains research code for data preparation, training, evaluation, and prediction. The upstream static website and its decorative assets have been removed. The original presentation remains available in the upstream repository.

The retained upstream figures are historical illustrations, not results from this fork's experiments: [architecture](images/model.png), [grounding examples](images/task.png), and [prediction examples 1](images/m1.png), [2](images/m2.png), [3](images/m3.png).
