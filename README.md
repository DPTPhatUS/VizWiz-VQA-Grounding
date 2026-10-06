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
