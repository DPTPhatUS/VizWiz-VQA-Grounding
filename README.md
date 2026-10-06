# VizWiz VQA Grounding — exp/grounding-controls

This branch extends the CEUD grounding project, which was presented as a CVPR 2025 workshop spotlight. The model predicts visual evidence masks from an image and question.

Use **train.py** and **eval.py** directly. The experiment is integrated into the existing pipeline:

- `models/model.py`: this branch's architecture and model factory.
- `models/backbone.py`: shared CLIP grounding backbone, including frozen teacher/coarse references.
- `models/experiment.py`: compact decoder and skip conditioning.
- `dataset.py`: aligned images, masks and text views; branch-specific detail/pair data where needed.
- `losses.py`: segmentation and experiment objectives.
- `models/checkpoint.py`: checkpoint initialization and strict continuation.
- `metrics.py`: original-resolution question-only evaluation.

There is one training/evaluation pipeline. The former parallel package and alternate entry points have been removed. Existing corrected experiment checkpoint keys/configurations are preserved; historical pre-correction checkpoint formats are still rejected.

## Kaggle commands

Enable a GPU and Internet for pretrained CLIP downloads. Keep Kaggle's installed Torch/Torchvision environment and install the other imported dependencies if needed:

```bash
pip install transformers ultralytics matplotlib tqdm pillow
```

Run the following in a Kaggle `%%bash` cell after cloning this branch. Replace dataset/checkpoint placeholders. The dataset root must contain `train_grounding.json`, `val_grounding.json`, `test_grounding.json`, image directories `train/`, `val/`, `test/`, and `binary_masks_png/<split>/`.

```bash
set -e
cd /kaggle/working/VizWiz-VQA-Grounding
DATA_ROOT=/kaggle/input/YOUR_DATASET/vizwiz
python train.py --data-root "$DATA_ROOT" --architecture compact --text-mode question --num-epochs 100 --batch-size 1 --num-workers 2 --seed 42 --device cuda --save-every 1000000 --output-dir /kaggle/working/outputs-compact
python eval.py --data-root "$DATA_ROOT" --checkpoint /kaggle/working/outputs-compact/best.pt --dataset test --batch-size 1 --num-workers 2 --device cuda --output-dir /kaggle/working/results-compact

# Answer-dropout teacher for the distillation branch:
python train.py --data-root "$DATA_ROOT" --architecture joint --text-mode dropout --answer-dropout 0.5 --num-epochs 100 --batch-size 1 --num-workers 2 --seed 42 --device cuda --save-every 1000000 --output-dir /kaggle/working/outputs-teacher
python eval.py --data-root "$DATA_ROOT" --checkpoint /kaggle/working/outputs-teacher/best.pt --dataset test --batch-size 1 --num-workers 2 --device cuda --output-dir /kaggle/working/results-teacher
```

Validation is question-only and runs each epoch. `best.pt` is selected by original-resolution mean IoU; `last.pt` supports continuation. The large save interval avoids retaining many periodic checkpoints. Evaluation writes PNG masks and `metrics.json`. Use `--dataset val` for validation exports.

For continuation, repeat the same training settings, replace `--init-checkpoint` with `--resume-checkpoint /path/to/output/last.pt`, and keep the same output directory. `--num-epochs` is the final epoch count. When moving a run between Kaggle sessions, copy its saved run directory into writable storage before resuming; keep the original training configuration and data paths compatible. Teacher-dependent continuation also requires the same teacher file.

Use `--help` on either script to see the branch options. `--tiny` is only an offline integration-test model, not a research architecture. GPU speed and real-data improvements have not been validated here.

`docs/` and `tests/` are deliberately ignored and kept local. This README is the tracked run guide for clones.

## Predict or visualize one image

These scripts use the same checkpoint architecture and input preprocessing as evaluation, including detail images on the refinement branch. No ground-truth mask is needed. Replace the checkpoint path with this branch's trained output.

```bash
python predict_save.py --checkpoint /path/to/best.pt --image /path/to/image.jpg --question "What does it say?" --output /kaggle/working/mask.png --device cuda
python visualize_predictions.py --checkpoint /path/to/best.pt --image /path/to/image.jpg --question "What does it say?" --output /kaggle/working/overlay.png --device cuda
```

`IoU.py` accepts the same arguments as `eval.py` and delegates to that evaluator.
