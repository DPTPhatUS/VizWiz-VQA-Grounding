# VizWiz VQA Grounding — SigLIP 2 NaFlex baseline

Branch `exp/siglip2-naflex-baseline`, based on `exp/baseline-fixes` at `cb7f7c7`.
This experiment replaces the **vision encoder** with SigLIP 2 Base NaFlex.
It retains CLIP text, image-to-text cross-attention, the one-channel U-Net,
Adam and BCE. No F annotations, teacher or distillation.

## Backbone choices

| Vision checkpoint | Width / layers | Input variant |
|---|---|---|
| `google/siglip2-base-patch16-naflex` (default) | 768 / 12 | NaFlex |
| `google/siglip2-so400m-patch16-naflex` | 1152 / 27 | Larger NaFlex |
| `openai/clip-vit-large-patch14-336` | 1024 / 24 | Retained CLIP baseline |

Google publishes **Base and So400m NaFlex**, not Large NaFlex. SigLIP 2 Large
has a 1024-wide, 24-layer vision tower (~303M vision parameters), but its released
checkpoints are FixRes. Base has ~86M vision parameters. These counts exclude
the text tower and decoder; this branch retains the 123M CLIP text tower.
[Official checkpoint table](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md)

NaFlex still resizes, approximately preserving aspect ratio within a patch
budget. The upstream processor defaults to 256 patches; this branch defaults
to **512**. Use 1024 in a separate run to test resolution effects. There is no
336-square resize before NaFlex. Larger budgets need more compute/VRAM;
improvement on VizWiz has not been measured.
[Preprocessing documentation](https://huggingface.co/docs/transformers/model_doc/siglip2)

## Setup

```bash
uv sync --locked
```

Existing data layout: `train_grounding.json`, `val_grounding.json`,
`test_grounding.json`, `train/`, `val/`, `test/`, and
`binary_masks_png/{train,val,test}/` under `--data-root`.
The first run downloads SigLIP vision weights/processor and CLIP text
weights/tokenizer. Hugging Face may cache the full SigLIP checkpoint file,
although only its vision tower is used here.

## Train

Base, 512 patches, QA text as in the parent baseline:

```bash
uv run train.py --data-root /kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz --num-epochs 20 --batch-size 4 --validate-every 1 --save-every 5 --output-dir outputs/naflex_base_512
```

For a separate 1024-patch experiment, add `--max-num-patches 1024` and use a new
output directory. For the larger NaFlex tower:

```bash
uv run train.py --data-root data/vizwiz --vision-model google/siglip2-so400m-patch16-naflex --max-num-patches 512 --num-epochs 20 --batch-size 4 --output-dir outputs/naflex_so400m_512
```

Start with Base. GPU peak memory has not been measured; lower batch size if
necessary and report the changed effective batch size. No gradient accumulation
flag is implemented. Both encoders start pretrained; fusion/decoder layers are
normally initialized. CLIP checkpoints cannot resume a NaFlex model.

`--text-mode question` trains a separate Q-only control. Use the same text mode
on both backbones for comparison. QA and Q-only are separate settings. NaFlex
also changes image/GT spatial sampling; gains cannot be attributed solely to
pretrained visual features without a resolution control.

Retained CLIP path:

```bash
uv run train.py --data-root data/vizwiz --vision-model openai/clip-vit-large-patch14-336 --num-epochs 20 --output-dir outputs/clip_control
```

Existing `--lr`, `--num-workers`, `--seed`, `--resume-checkpoint`,
`--validate-every` and `--save-every` remain. Training validation reports BCE,
not mIoU; run eval.py for mIoU. `--save-every 0` saves only the final model.

## Evaluate and resume

```bash
uv run eval.py --checkpoint outputs/naflex_base_512/checkpoint_epoch20.pt --dataset val --data-root /kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz --output-dir outputs/eval_naflex_base_512
uv run train.py --data-root data/vizwiz --resume-checkpoint outputs/naflex_base_512/checkpoint_epoch20.pt --num-epochs 30 --output-dir outputs/naflex_base_512
```

All NaFlex checkpoints, including final files, record architecture,
preprocessing, patch budget and text mode. Eval/resume restore these
automatically without redownloading encoder weights; tokenizer files must
be cached or available. Resume rejects changing backbone/budget/text mode.
Periodic checkpoints contain optimizer/scaler state. Final NaFlex files
contain weights+metadata only, so their optimizer starts fresh on resume.

Old CLIP raw/full checkpoints still load as CLIP. Baseline-fixes itself is an
unchanged separate branch. Start a new NaFlex experiment from pretrained
SigLIP; there is no direct CLIP checkpoint conversion.

Eval defaults to native GT size: resize logits, then threshold, score and
export the same mask. `--metric-resolution resized` uses the legacy **336 GT
reference**, including for NaFlex. JSON records backbone, patch budget, text
mode and resolution. Eval restores checkpoint text mode unless overridden;
legacy files default to QA. Removing A from a QA-trained checkpoint is a
diagnostic, not a Q-only-trained baseline.

Single-image prediction:

```python
from models import GroundingModel
model = GroundingModel.from_checkpoint('outputs/naflex_base_512/checkpoint_epoch20.pt')
result = model.predict('image.jpg', question='What does this label say?')
result['pred_pil'].save('mask.png')  # NaFlex: original image dimensions
```

Legacy example scripts select NaFlex preprocessing for NaFlex checkpoints;
update their hardcoded paths/questions first. Prefer eval.py for benchmark
metrics and metadata.

## Shape and loss handling

- Processor emits patch pixels, `pixel_attention_mask` and per-image
  `spatial_shapes=(patch_height,patch_width)`. There is no first CLS token to drop.
- Intermediate layers 4/7/9 and the final state supply decoder features. These
  remain one spatial grid, not a true spatial pyramid.
- Group equal H/W grids for decoding. Padding is removed before all CNN
  bottlenecks/skips, and never participates in BCE.
- Use SigLIP processor normalization. GT uses nearest resize to the same
  processed image size; native image/mask alignment is mandatory.
- Average BCE per image, then across the batch, retaining equal image weights.
- Decoder channels stay 1024/512/256/128. Input/skip channels and text projection
  adapt to the vision width (768 or 1152).
- NaFlex DDP uses **local BatchNorm**: ranks may have different numbers of
  spatial groups, so per-group SyncBN collectives could hang. CLIP retains
  SyncBN. Grouping changes BN statistics relative to a full fixed-grid batch;
  CUDA/DDP execution has not been tested here.

## Verification

```bash
uv run python -m unittest discover -s tests -v
```

11 CPU/offline tests pass with the real NaFlex processor and tiny randomly
initialized vision/text models. They cover rectangular grids, mask alignment,
padding invariance, gradients, equal-image loss, extraction from a full SigLIP
checkpoint, checkpoint roundtrip, native prediction/export, train/eval/resume
and legacy CLIP compatibility.

No full pretrained Base/So400m training, CUDA memory benchmark, DDP execution
or VizWiz accuracy result has been established. This is an experimental
baseline implementation, not evidence that NaFlex improves mIoU.

## Project

Research fork of [yjh9929/VizWiz-VQA-Grounding](https://github.com/yjh9929/VizWiz-VQA-Grounding).
Licensed under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/).
Figures in images/ are historical illustrations, not results of this experiment.
