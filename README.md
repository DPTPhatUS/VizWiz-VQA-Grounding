# YOLO detector preprocessing experiment (`exp/yolo-detector`)

This branch trains a single-class YOLO detector, draws its predicted boxes onto
the dataset images, then trains the original grounding baseline on those images.
The grounding model architecture and checkpoint keys are unchanged. YOLO runs
offline; it is not part of the grounding model's forward pass.

Install the dependencies in `pyproject.toml` with `uv sync`. This branch includes
Ultralytics. Run the following commands from the repository root, with the source
dataset at `data/vizwiz` (images, grounding JSON files, and binary masks):

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

Use `--splits train,val` if test images/JSON are unavailable. Evaluating with
`--dataset test` also requires test masks. The generated dataset symlinks the
original annotations and masks, which remain supervision/evaluation targets.
The default rendering uses the highest-confidence detection; an image with no
detections receives a full-image box. EXIF orientation is applied before prediction
and rendering. Keep the detector checkpoint and rendering settings the same across
splits, and use a fresh output directory when changing settings.

Grounding train/eval always use the baseline encoder's 336×336 input. Detector
scripts retain `--image-size` because their inference/training resolution is
configurable. `eval.py` requires a grounding checkpoint. Run any script with
`--help` for its remaining data, training, and output controls.

Optional detector-only evaluation:

```bash
uv run eval_detector.py --data-root data/vizwiz \
  --checkpoint outputs/yolo/detector/weights/best.pt
```

`generate_random_box.py` retains the random-box control experiment.
`generate_box.py` draws boxes derived from ground-truth masks: it is an oracle
diagnostic and must not be used for normal validation/test performance claims.
Train a separate grounding checkpoint for each image preprocessing condition.
`models/box_renderer.py` is a historical tensor-rendering utility; the pipeline
above uses the offline PIL renderer in `generate_yolo_box.py`.

# 🏆Workshop Spotlight at CVPR 2025
Our work has been selected as a spotlight paper at a workshop in CVPR 2025! ( https://cvpr.thecvf.com/ )
We are honored that our research was recognized and featured among the notable contributions. 

# Tasks: VizWiz-VQA-Grounding
 This project was developed for the [VizWiz-VQA-Grounding Challenge](https://vizwiz.org/tasks-and-datasets/visual-qa/) 2025. The goal is to return grounded visual evidence for answers to visual questions posed by people with visual impairments. 


## 🎶Task Objective
Given an image-question pair, the task is to predict the region in the image that supports the most common answer. This is known as **answer grounding**, and predictions are evaluated based on **mean Intersection over Union (IoU)** with human-annotated binary masks.


## 📂Project Structure
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
## ⚙️ Installation
    apt update
    apt install -y git
    git
    git config --global user.name "<yourname>"
    git config --global user.email "<youremail>"
    git clone https://github.com/yjh9929/VizWiz-VQA-Grounding.git
    pip install torch torchvision transformers pyyaml
    pip install tqdm
    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

## 🚀 Running the Code
Use the four-stage experiment commands above. Pointing `train.py` at the original
`data/vizwiz` instead trains the baseline on unboxed images.

## 🧠 Model Design

## 📊 Evaluation
Use `eval.py` with the grounding checkpoint and the same image preprocessing
condition used for training, as shown above.

## 🔗 References

## 📝 License
<a href="http://creativecommons.org/licenses/by/4.0/" rel="license"><img src="https://i.creativecommons.org/l/by/4.0/88x31.png" alt="Creative Commons License"></a>  
This work is licensed under a Creative Commons Attribution 4.0 International License.

## Encoder freezing and optimization

Omitting the options below preserves this branch's existing training behavior:
Adam, LR `1e-5`, zero weight decay, no LR schedule, and the branch's original
encoder freezing policy. Freezing is opt-in; the detail CNN and decoder remain
trainable. No model/state-dict keys or evaluation preprocessing change.

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

`BOOL` accepts `True` or `False`; a freeze flag without a value means `True`.
The aliases `--no-freeze-backbone`, `--no-freeze-vision-encoder`, and
`--no-freeze-text-encoder` mean `False`. An explicit component setting overrides
`freeze_backbone` within the same source. **JSON config > explicit CLI > branch
defaults**: a config shorthand also overrides component flags on the CLI.
Existing recipe/loss settings can share the same JSON; unsupported keys fail
early. Existing branch-specific flags remain available through `train.py --help`.

Append, for example, to this branch's existing training command:

```bash
--freeze-backbone True \
--optimizer adamw \
--lr 0.0001 \
--weight-decay 0.01 \
--lr-scheduler cosine \
--warmup-ratio 0.05 \
--min-lr-ratio 0.1
```

This is a configurable experiment example, not a validated best hyperparameter
setting. To freeze only vision while retaining the text policy, append
`--freeze-vision-encoder True`. To explicitly train both encoders, use
`--freeze-backbone False`. Models with a deliberately fixed reference stage
(gain-guided refinement) reject encoder unfreezing to preserve that algorithm.
Unused CLIP/SigLIP pooled-output parameters stay frozen when unfreezing token encoders.

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

Frozen encoders stay in eval mode during training and are excluded from the
optimizer when a freezing option is set. LR schedules advance after successful
optimizer updates, including gradient accumulation; AMP overflow skips do not
advance them. Checkpoints save the resolved optimization settings, trainable
parameter layout and schedule progress. Resume restores these settings and
rejects conflicting options. For scheduled runs, keep the original total
`--num-epochs`, loader length and accumulation so the LR trajectory is unchanged.

Old checkpoints still load for evaluation with the same architecture and
weights. Full legacy resume keeps its original optimizer layout and constant
LR behavior; changing freezing, optimizer, decay or enabling a new LR schedule
requires a new run (or an explicit weights-only initialization where supported).
Resume retains each branch's existing RNG/data-order policy; legacy branches
that do not save or reconstruct RNG state do not guarantee bitwise replay.
No freezing/optimization change retroactively changes a previously trained
checkpoint or its reported results.
