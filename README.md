# VizWiz VQA Grounding — fixed SigLIP 2 Base NaFlex

Branch `exp/siglip2-naflex-baseline`, derived from `exp/baseline-fixes`.
Vision is fixed to **`google/siglip2-base-patch16-naflex`** (768 channels,
12 transformer layers). The separate `exp/siglip2-so400m-naflex-baseline`
branch fixes vision to So400m (1152 channels, 27 layers).
There is no vision-backbone switch in either branch.

`models/image_encoder.py` owns the vision encoder and patch grids;
`models/text_encoder.py` owns text; `models/model.py` owns cross-attention,
decoder and prediction. Both branches retain the same three feature skips
(hidden states 4/7/9), final-state fusion, decoder channels 1024/512/256/128,
Adam and per-image BCE. These skips have the same spatial grid, not a spatial
pyramid. No teacher, distillation or additional annotation is involved.

## Text and image inputs

**CLIP text is the default**, using `openai/clip-vit-large-patch14-336`.
`--text-encoder siglip2` instead selects the text tower from this branch's
fixed SigLIP2 checkpoint. Its tokenizer uses fixed-length 64-token padding;
CLIP uses dynamic padding with a 77-token limit. Projection into the fusion
layer uses the actual text/vision widths. The unused SigLIP pooled text head
is frozen; token features are used for grounding.

`--text-mode qa` merges question and answer into one sequence:
`Q: <question> A: <answer>`. `--text-mode question` trains a separate Q-only
control. Changing text encoder or removing A is a separate experiment.

Images go directly from raw RGB PIL into the SigLIP2 processor. It resizes
approximately preserving aspect ratio under `--max-num-patches`, patchifies,
rescales by `1/255`, then normalizes with RGB mean/std `[0.5,0.5,0.5]`:
`(pixel/255 - 0.5)/0.5`. There is no preceding 336-square resize or CLIP
normalization. Saved processor metadata is validated on restoration.
GT masks remain binary labels and use nearest resize to the processed grid.
Padding is removed before CNN decoding and loss computation.
[Official Base processor](https://huggingface.co/google/siglip2-base-patch16-naflex/blob/main/preprocessor_config.json)

NaFlex still resizes images. The default patch budget here is 512, compared
with the upstream processor's 256. Test 1024 separately with a new output
directory. Resolution and pretrained vision features both change relative
to CLIP; a gain alone does not identify which caused it.

## Setup and training

```bash
uv sync --locked
```

Data layout under `--data-root`: `train_grounding.json`, `val_grounding.json`,
`test_grounding.json`, `train/`, `val/`, `test/`, and
`binary_masks_png/{train,val,test}/`. First use downloads encoder weights,
processor and tokenizers; Hugging Face can cache a full checkpoint file even
when only one tower is loaded.

Suggested starting configuration for two GPUs:

```bash
uv run torchrun --nproc_per_node=2 train.py \
    --data-root /kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz \
    --num-epochs 20 \
    --batch-size 2 \
    --gradient-accumulation-steps 2 \
    --lr 0.00001 \
    --num-workers 2 \
    --max-num-patches 512 \
    --text-encoder clip \
    --text-mode qa \
    --gradient-checkpointing \
    --amp \
    --validate-every 1 \
    --save-every 10 \
    --seed 42 \
    --output-dir outputs/naflex_base_clip_512
```

`--batch-size` is the **global micro-batch**, so this uses one image per GPU.
The effective batch is `2 * 2 = 4` images per optimizer update, except the
last partial window. Do not multiply by GPU count again. Worker count is
per process. AMP uses FP16 with gradient scaling on CUDA. Gradient
checkpointing recomputes encoder activations to reduce memory.

Use the same micro-batch/accumulation on both experiment branches for a
matched comparison. NaFlex decoding groups images by patch H/W and uses
local BatchNorm: different ranks can have different group counts, making
per-group SyncBN unsafe. Accumulation does not reproduce the BatchNorm
statistics of the original larger CLIP micro-batch.

This is a conservative starting suggestion, **not a measured VRAM fit**.
Full GPU runs have not been performed here. So400m is substantially heavier;
switching its text to SigLIP2 also adds a large text tower and optimizer state.
Reducing micro-batch alone may be insufficient for that configuration on
16 GB GPUs. Keep CLIP text first to assess vision changes.

All training options are shown above except `--resume-checkpoint`, which is
omitted for a fresh run. Disable the booleans with `--no-amp` or
`--no-gradient-checkpointing`. `--save-every 0` disables periodic checkpoints;
the final model is still saved. `--validate-every 0` disables epoch validation.
Validation during training reports BCE, not mIoU; the distributed validation
sampler can duplicate its last sample. Use eval.py for the full validation
set's mIoU without sampler duplication.

## Checkpoints, evaluation and resume

```bash
uv run eval.py --checkpoint outputs/naflex_base_clip_512/checkpoint_epoch20.pt --dataset val --data-root /kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz --output-dir outputs/eval_naflex_base_clip_512
uv run train.py --data-root data/vizwiz --resume-checkpoint outputs/naflex_base_clip_512/checkpoint_epoch20.pt --num-epochs 30 --output-dir outputs/naflex_base_clip_512
```

Checkpoints save architecture, encoder configurations, preprocessing, patch
budget, text encoder/mode and training arguments. Eval/resume restore these;
tokenizer files must be cached or accessible. Resume rejects changing vision,
text encoder/mode or patch budget. Micro-batch, accumulation, AMP and gradient
checkpointing restore from saved arguments unless explicitly overridden.
`--num-epochs` on resume is the total desired epoch count.

Periodic checkpoints contain optimizer/scaler state; final model files contain
weights and metadata only. Resuming a final file starts a new optimizer.
Existing Base NaFlex checkpoints with CLIP text retain their parameter
keys/shapes; missing text-encoder metadata defaults to CLIP. Use the matching
experiment branch to load its checkpoint. Original CLIP-vision checkpoints
belong to `exp/baseline-fixes`; this experiment does not convert them.

Eval defaults to native GT resolution: resize logits, threshold, score and
export the same mask. `--metric-resolution resized` instead uses the legacy
336-square GT reference. Compare scores using the same resolution and text
mode. Removing A at eval from a QA-trained model is a diagnostic, not a
Q-only-trained baseline. Metadata records the selected protocol.

```python
from models import GroundingModel
model = GroundingModel.from_checkpoint('outputs/naflex_base_clip_512/checkpoint_epoch20.pt')
result = model.predict('image.jpg', question='What does this label say?', answer='example')
result['pred_pil'].save('mask.png')  # original image dimensions
```

Update the example scripts' hardcoded paths/questions before using them.
Prefer eval.py for dataset metrics and exported prediction masks.

## Verification

```bash
uv run python -m unittest discover -s tests -v
```

12 CPU tests cover rectangular grids, GT alignment, normalization, padding,
gradients, equal-image loss, pretrained tower extraction, checkpoint/native
prediction, train/eval/resume, text switching and incomplete accumulation.
They include two-process Gloo DDP with different per-rank grid groups and
gradient checkpointing. Actual pretrained tokenizer loading was also checked.
Model tests use tiny random towers; no full pretrained training, CUDA/NCCL
execution, GPU memory benchmark or VizWiz accuracy result is established.

## Project

Research fork of [yjh9929/VizWiz-VQA-Grounding](https://github.com/yjh9929/VizWiz-VQA-Grounding).
Licensed under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/).
Figures in images/ are historical illustrations, not experiment results.
