# Gain-guided refinement

This branch trains a compact grounding model with a high-resolution crop refiner and an expected-IoU-gain router. Inputs and evaluation use only the question. The coarse model stays frozen; the refiner is also frozen while training the router.

Install the project dependencies, or keep your notebook's Torch/Torchvision installation and install `transformers pillow numpy`. Run from this branch with a VizWiz root containing `<split>_grounding.json`, `<split>/` images, and `binary_masks_png/<split>/` masks.

```bash
python train.py --data-root /path/to/vizwiz --stage refiner --init-checkpoint /path/to/compact-controls/best.pt --output-dir outputs-refiner --num-epochs 30 --batch-size 1 --save-every 0
python train.py --data-root /path/to/vizwiz --stage router --init-checkpoint outputs-refiner/best.pt --output-dir outputs-router --num-epochs 30 --batch-size 1 --save-every 0
python eval.py --data-root /path/to/vizwiz --checkpoint outputs-router/best.pt --dataset test --output-dir results-router --budget 2
```

The refiner starts from a question-only compact controls checkpoint. The router starts from the refiner stage. Images use a 336-pixel coarse input and a 672-pixel detail canvas; crops use 336 pixels. Refiner evaluation uses fixed coverage, and router evaluation selects positive predicted gains. `--budget` chooses up to 1, 2, or 4 crops. Validation and exported masks use original image dimensions.

Training selects CUDA automatically when available and supports `torchrun`. `last.pt` stores continuation state and `best.pt` tracks validation mean IoU. `--save-every 0` retains only those two files. To continue, repeat the stage and data settings, replace `--init-checkpoint` with `--resume-checkpoint outputs-router/last.pt`, and increase the final `--num-epochs` value.

```bash
python predict_save.py --checkpoint outputs-router/best.pt --image example.jpg --question "Where is the label?" --output mask.png
python visualize_predictions.py --checkpoint outputs-router/best.pt --image example.jpg --question "Where is the label?" --output overlay.png
```

Use `train.py --help` and `eval.py --help` for operational options. Real-data accuracy and GPU throughput require training; tiny models are used only by local offline tests.

Resume applies `--lr` after restoring optimizer state. An existing output directory may only resume its own `last.pt`; use a fresh directory for a checkpoint copied from Kaggle input or an earlier epoch.
