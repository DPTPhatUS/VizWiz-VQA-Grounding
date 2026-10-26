# YOLO detector preprocessing experiment (`exp/yolo-detector`)

This branch trains a single-class YOLO detector, draws its predicted boxes onto the dataset images, and then trains the original grounding baseline on those boxed images. The grounding model architecture and checkpoint keys are unchanged. YOLO runs offline as a preprocessing step before grounding training and evaluation.

Install the dependencies in `pyproject.toml` with `uv sync` (which includes Ultralytics). Run the following commands from the repository root, with the source dataset at `data/vizwiz` (images, grounding JSON files, and binary masks):

```bash
# 1. Train detector on train annotations, using val annotations for validation.
uv run train_detector.py --data-root data/vizwiz --output-dir outputs/yolo

# 2. Generate images with predicted boxes. No masks or polygons determine boxes.
uv run generate_yolo_box.py \
  --checkpoint outputs/yolo/detector/weights/best.pt \
  --data-root data/vizwiz --output-root data/vizwiz_yolo_box \
  --splits train,val,test

# 3. Train the original grounding baseline on the generated dataset.
uv run train.py --data-root data/vizwiz_yolo_box \
  --num-epochs 100 --batch-size 4 --validate-every 1 \
  --output-dir outputs/grounding_yolo

# 4. Evaluate that grounding checkpoint on predicted-box validation images.
uv run eval.py --data-root data/vizwiz_yolo_box --dataset val \
  --checkpoint outputs/grounding_yolo/model_final_epoch100.pt \
  --output-dir results/grounding_yolo_val
```

Use `--splits train,val` if test images or JSON annotations are unavailable. Evaluating with `--dataset test` also requires test masks. The generated dataset symlinks the original annotations and masks, which remain the supervision and evaluation targets. Default rendering draws the highest-confidence detection, falling back to a full-image box when no detection is found. EXIF orientation is applied before prediction and rendering. Keep the detector checkpoint and rendering settings consistent across splits, and write to a fresh output directory when changing settings.

Grounding training and evaluation always use the baseline encoder's 336 × 336 input resolution. Detector scripts expose `--image-size` because detector training and inference resolution is configurable. `eval.py` requires a grounding checkpoint. Run any script with `--help` for additional data, training, and output flags.

Optional detector-only evaluation:

```bash
uv run eval_detector.py --data-root data/vizwiz \
  --checkpoint outputs/yolo/detector/weights/best.pt
```

`generate_random_box.py` implements the random-box control experiment, and `generate_box.py` draws boxes from ground-truth masks as an oracle diagnostic. Train a separate grounding checkpoint for each image preprocessing condition. `models/box_renderer.py` is an older tensor-rendering utility; the pipeline above uses the offline PIL renderer in `generate_yolo_box.py`.

## Project and Task

Developed for the [VizWiz-VQA-Grounding Challenge](https://vizwiz.org/tasks-and-datasets/visual-qa/) 2025 and selected as a workshop spotlight at [CVPR 2025](https://cvpr.thecvf.com/).

Given an image-question pair, the task is to predict the region in the image that supports the most common answer (answer grounding). Predictions are evaluated by mean Intersection over Union (IoU) against human-annotated binary masks.

## Project Structure

```project/
├── README.md
├── models/
    ├── init.py
    ├── image_encoder.py
    ├── text_encoder.py
    ├── concat.py
    ├── mask_decoder.py
    └── model.py 
├── data/
    ├── binary_masks_png/
    │   ├── train/
    │   └── val/
    ├── test/
    ├── train/
    ├── val/
    ├── test_grounding.json
    ├── train_grounding.json
    └── val_grounding.json
├── train.py
├── dataset.py
├── utils.py
├── visualize_predictions.py
├── IoU.py
├── metrics.py
└── config.yml
```

## Legacy Environment Setup

    apt update
    apt install -y git
    git
    git config --global user.name "<yourname>"
    git config --global user.email "<youremail>"
    git clone https://github.com/yjh9929/VizWiz-VQA-Grounding.git
    pip install torch torchvision transformers pyyaml
    pip install tqdm
    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

## Running and Evaluation

Use the four-stage workflow at the top of this file. Pointing `train.py` at the original `data/vizwiz` directory trains the baseline on unboxed images. Run `eval.py` with the grounding checkpoint and the matching image preprocessing condition.

## License

Licensed under [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/).

## Encoder freezing and optimization

Omitting the options below keeps this branch's default training setup: Adam, learning rate `1e-5`, zero weight decay, constant learning rate, and the default encoder freezing policy. Freezing is opt-in, and the detail CNN and decoder always remain trainable. Model state-dict keys and evaluation preprocessing are unchanged.

| Option | Values / default |
| --- | --- |
| `--freeze-backbone [BOOL]` | Set both pretrained encoders; omitted preserves existing policy |
| `--freeze-vision-encoder [BOOL]` | Set vision encoder only |
| `--freeze-text-encoder [BOOL]` | Set text encoder only |
| `--optimizer` | `adam` (default), `adamw` |
| `--lr` | Main learning rate; default `0.00001` |
| `--backbone-lr` | Optional separate LR for unfrozen encoders; default uses `--lr` |
| `--weight-decay` | Nonnegative; default `0`. Biases, norm vectors and token/position embeddings receive no decay when parameter groups are enabled |
| `--lr-scheduler` | `none` (default), `cosine`, `linear`, `poly` |
| `--warmup-ratio` | Fraction of planned optimizer updates, in `[0, 1)`; default `0` |
| `--min-lr-ratio` | Final LR / starting LR for decay schedules; default `0.1` |
| `--poly-power` | Positive exponent for `poly`; default `0.9` |
| `--config` | Flat JSON with training options; optional |

`BOOL` accepts `True` or `False`; passing a freeze flag without a value sets it to `True`. The negative flags `--no-freeze-backbone`, `--no-freeze-vision-encoder`, and `--no-freeze-text-encoder` set the corresponding option to `False`. Within the same source, a specific encoder flag overrides `freeze_backbone`. Precedence follows JSON config, then explicit CLI flags, then branch defaults, so a JSON config value also overrides CLI flags. Existing recipe and loss settings can share the same JSON file, and unrecognized keys raise an error immediately. See `train.py --help` for branch-specific flags.

For example, append these flags to the training command:

```bash
--freeze-backbone True \
--optimizer adamw \
--lr 0.0001 \
--weight-decay 0.01 \
--lr-scheduler cosine \
--warmup-ratio 0.05 \
--min-lr-ratio 0.1
```

To freeze only the vision encoder while keeping the default text policy, pass `--freeze-vision-encoder True`. To train both encoders, pass `--freeze-backbone False`. Gain-guided refinement models keep their reference stage fixed and reject encoder unfreezing. Unused CLIP and SigLIP pooled-output parameters stay frozen when token encoders are unfrozen.

Equivalent JSON (`--config path/to/train.json`):

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

Frozen encoders stay in `eval` mode during training and are excluded from the optimizer. Learning rate schedules step after each completed optimizer update (accounting for gradient accumulation) and do not advance when AMP skips an overflowing step. Checkpoints store the resolved optimization config, trainable parameter layout, and scheduler state; resuming restores these settings and rejects conflicting flags. When resuming a scheduled run, keep `--num-epochs`, dataset length, and gradient accumulation unchanged to preserve the learning rate schedule.

Existing checkpoints remain compatible with evaluation. Resuming a legacy checkpoint keeps its original optimizer layout and constant learning rate; changing encoder freezing, optimizer type, weight decay, or learning rate schedule requires starting a new run (or initializing from weights only where supported). Resuming follows each branch's RNG and data-order handling.
