# VizWiz VQA Grounding — compact-decoder

Uses residual fusion, fixed joint-text skip conditioning, and a compact decoder with depthwise-separable blocks. This checkout runs the `exp/compact-decoder` model through `train.py` and `eval.py`.

Given an image, question, and answer, predict the supporting binary mask. Evaluation reports mean Intersection over Union (IoU). Inputs use the fixed 336 × 336 CLIP resolution.

## Setup

```bash
uv sync --locked
```

Place `train_grounding.json`, `val_grounding.json`, and `test_grounding.json` under `data/vizwiz/`, alongside `train/`, `val/`, `test/`, and `binary_masks_png/{train,val,test}/`. Use `--data-root` for another location. The first model run downloads the pretrained CLIP encoders if they are not cached.

## Train and evaluate

```bash
uv run train.py --data-root data/vizwiz --num-epochs 100 --batch-size 4 --output-dir outputs/compact-decoder
uv run eval.py --checkpoint outputs/compact-decoder/checkpoint_epoch100.pt --dataset val --data-root data/vizwiz --output-dir results/compact-decoder
```

Training also supports `--lr`, `--num-workers`, `--seed`, `--resume-checkpoint`, `--validate-every`, and `--save-every`. Set `--save-every 0` to save only the final model; validation defaults to disabled. Use `torchrun` for distributed training. Evaluation requires a checkpoint and supports `--device`, `--batch-size`, and `--num-workers`; use `--dataset test` for the test split.

Checkpoints retain the branch architecture identity and load strictly. Use a checkpoint from this experiment.

Compact checkpoints use `conditioning: joint`. Separate question/answer conditioning belongs to `exp/separate-qa-skips`.

## Project

Developed for the [VizWiz VQA Grounding Challenge](https://vizwiz.org/tasks-and-datasets/visual-qa/) 2025. The work was selected as a workshop spotlight at CVPR 2025.

Licensed under [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/).
