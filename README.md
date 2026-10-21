# Question-guided native-image zoom

Branch: `exp/question-guided-zoom`, based on the original
`exp/question-guided-detail-pyramid` at `3eaa010`. This experiment adds a small
crop reader to the global pyramid. It does not include token/region ranking or
an answer predictor. Train fresh from pretrained CLIP; parent experiment
checkpoints are deliberately rejected rather than partially loaded.

The hypothesis is that reading selected regions before full-image resizing
can preserve evidence that the global 768-pixel view loses. This implementation
does not establish a VizWiz accuracy improvement.

## Architecture

```text
Upright native RGB + question Q
  |
  +-- resize 336 --> trainable CLIP vision --+-- 3 skips + Q-FiLM
  |                                        +-- image/text cross-attention
  |                                                      |
  +-- resize 768 --> small CNN/detail pyramid --> global decoder
  |                                          48 -> 96 -> 192 -> 384
  |                                                |           |
  |                                        semantic F96    global refiner
  |                                                |           |
  |                     projected Q ----------> locator96  global logits768
  |                                                |           |
  |                          relevance + uncertainty --> two crop boxes
  |                                                        |
  +-- cut two native RGB crops, resize each to 256 <---------+
                          |
                shared small U-Net crop reader
                RGB: 16@256 -> 32@128 -> 64@64
                          + cropped global F96 -> 64@64
                          + full Q token cross-attention at 64
                          |
                32@128 + skip -> 16@256 + skip
                          |
                  correction256 + crop quality
                          |
    global logits + weighted crop corrections in original coordinates
```

CLIP is `openai/clip-vit-large-patch14-336`. Text is frozen and stays in eval
mode. Vision is fine-tuned except its unused pooled-CLS post-layernorm. The
parent global branch is retained: three skips `1024 -> 256/GN/GELU -> Q-FiLM
-> 128`, four standard-convolution upsampling blocks, four question-guided
detail fusions, and a signed-logit refiner at 768.

The locator receives `(B,32,96,96)` semantic features and broadcast projected
Q with 16 channels. It predicts mask relevance independently of the global
mask head. One crop at each scale is selected on a deterministic 16x16 center
grid. Crop sides are 0.25 and 0.5 times the native shorter side, shifted into
the image and snapped outward to native pixel edges. They remain approximately
square (up to one pixel rounding). Each scale searches all centers, using:

```text
utility = sigmoid(locator) * (1 + 0.25 * 4*p*(1-p))
p = sigmoid(global logits resized to locator resolution)
score = mean utility within the candidate window
second-scale score *= 1 - 0.5 * IoU(candidate, first crop)
```

Window means use integral images rather than sparse point sampling. The
overlap penalty discourages redundant crops but does not prohibit overlap.
The discrete box selection is detached. Its supervision comes from the
locator loss, not from differentiating through argmax. An incorrectly low
locator response can still miss the target; crop recall is measured at eval.

Native RGB stays uint8 through loading/collation/device transfer. Only sliced
crops are converted to `[0,1]`, resized with bicubic antialiasing, clamped to
`[0,1]` and normalized with CLIP mean/std. Feature and mask crops share the
same boxes. Feature sampling is bilinear; binary GT sampling is nearest.
All geometry uses normalized xyxy pixel edges and `align_corners=False`.

Local spatial features query all non-padding CLIP question tokens. They also
receive the matching global semantic crop. The U-Net predicts a signed logit
correction and one quality logit per crop. The correction head starts at zero,
so the initial final mask equals the global mask. Quality starts at sigmoid(-2).

At any output grid, including original resolution:

```text
w_k(x) = sigmoid(quality_k) * tapered_crop_window_k(x)
z_final(x) = resize(z_global)(x) + sum_k(w_k(x)*delta_k(x)) / max(1, sum_k(w_k(x)))
```

The window is flat in the interior and tapers over the outer 10% on each side,
reaching zero at/outside the crop edges. A high-quality crop can replace the
global prediction in the interior; overlapping crops do not amplify corrections.
At eval, crop corrections are placed directly
on the native grid, avoiding an intermediate crop-to-768-to-native resize.
The global logits are still interpolated from 768. No RGB restoration or OCR
model is added; zoom cannot recover information absent in the original photo.

Additional parameters: **250,723**. Total excluding both CLIP backbones:
**6,899,824**. These are implementation counts, not GPU memory estimates.

## Training and GT boundaries

The final mask always uses exactly two model-selected crops in training and
inference. No GT box or mask affects their selection or forward prediction.
There is no teacher-forced final mask or train/inference crop curriculum.

For training only, the same reader also processes one teaching crop centered
on a foreground pixel nearest the GT centroid, at the smaller crop scale.
An empty GT uses the image center and an empty target. This third crop is
supervised locally and is excluded from final mask composition. GroupNorm and
per-crop attention keep its activations independent of the two prediction
crops. No extra annotation is needed; neither train nor inference reads A.

Default objective:

```text
L = L_final768 + 0.3*L384 + 0.2*L192 + 0.1*L96
    + 0.2*L_locator + 0.5*L_crop + 0.05*L_quality
```

Each segmentation loss is mean pixel BCE + 0.5 times mean per-image soft Dice
loss. Main masks retain the parent's nearest-resized 768 targets. Auxiliary
targets use area interpolation. Locator targets use area interpolation
directly from native GT, and crop targets are sampled directly from native GT.
`L_crop` averages all three fixed-count crops per image, then images. Quality
BCE targets the detached hard foreground IoU of each local prediction against
its local GT, using threshold zero and empty-union IoU=1. Quality predicts
local correctness, not a guaranteed improvement over the global mask.

These coefficients and crop sizes are initial experimental settings. Optional
`--config` JSON exposes loss settings without adding architecture switches:

```json
{
  "overlap_loss": "dice",
  "overlap_weight": 0.5,
  "fp_weight": 0.7,
  "aux96_weight": 0.1,
  "aux192_weight": 0.2,
  "aux384_weight": 0.3,
  "locator_weight": 0.2,
  "crop_weight": 0.5,
  "quality_weight": 0.05
}
```

`fp_weight` applies only to optional Tversky loss, not default Dice. Setting
zoom losses to zero does not remove the crop architecture or create a parent
baseline. Use the parent branch for that separately trained control.

## Kaggle run

Enable Internet and two T4 GPUs, then install dependencies before torchrun:

```bash
git clone --single-branch --branch exp/question-guided-zoom https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
cd VizWiz-VQA-Grounding
uv sync --frozen
export DATA_ROOT=/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz
export OMP_NUM_THREADS=1
```

First check real-backbone AMP/DDP execution:

```bash
uv run --frozen torchrun --standalone --nproc_per_node=2 preflight.py \
  --data-root "$DATA_ROOT" --batch-size 2 --num-workers 0 --steps 3 \
  --gradient-checkpointing
```

Preflight requires finite gradients on all trainable parameters. It records
initial AMP scale backoff and retries the same batch, up to eight reductions.
Missing gradients, nonfinite forward losses or persistent overflow fail. It
writes no checkpoints. Pass the same `--config` to preflight and train if using
custom loss weights.

```bash
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py \
  --data-root "$DATA_ROOT" --output-dir outputs \
  --num-epochs 20 --batch-size 2 --gradient-accumulation-steps 2 \
  --gradient-checkpointing --num-workers 2 --lr 0.00001 --seed 42 --save-every 10
```

Global microbatch 2 means one image per GPU; accumulation 2 gives effective
batch 4. CUDA FP16 AMP is automatic. Adam, LR1e-5, gradient clip norm1, no
scheduler or epoch validation. Full checkpoints at epochs10/20 support resume
with the same batch/accumulation/LR/seed/world size; raw final weights do not.
All checkpoints stay directly in `outputs/`. Fresh runs reject existing run
history/checkpoints. New native-image transfers and crop processing increase
runtime; CPU checks do not establish T4 capacity or a full-session duration.

## Evaluation and diagnostic

Run all four combinations `(val,test) x (epoch10,epoch20)` with distinct output
folders. Local test scoring requires test annotations and GT masks.

```bash
uv run --frozen python eval.py \
  --checkpoint outputs/checkpoint_epoch20.pt --dataset val \
  --data-root "$DATA_ROOT" --output-dir outputs/eval_val_ep20 \
  --batch-size 1 --num-workers 2 --device cuda
uv run --frozen python diagnose.py \
  --predictions outputs/eval_val_ep20 --data-root "$DATA_ROOT" \
  --output-dir outputs/diagnostic_ep20
```

Evaluation is FP32, one process, original-resolution mean per-image foreground
IoU, threshold logit0, empty-union IoU1. Batch size4 is allowed if it fits.
Outputs contain native-size binary PNGs and `metrics.json` with both final
`mean_iou` and `global_mean_iou` from the same checkpoint. `zoom_diagnostics`
records each image's global/final IoU, native GT coverage of the two crops,
normalized boxes, and predicted crop quality. GT is read only after prediction
for these diagnostic measurements, never for inference selection.

The global view is an intervention on a jointly trained model, not an
independently trained parent baseline. Compare against the parent run as well.
Inspect small-region precision/recall and proposal coverage, not overall IoU
alone. Select checkpoint/settings using validation rather than test scores.

Offline checks cover native geometry, antialiasing, GT independence, Q-token
masking, finite gradients, accumulation, two-rank Gloo with reduced real CLIP,
FP16/BF16, tiny overfit, strict raw/full reload, exact epoch resume, and agreement
between eval PNGs, native diagnostics and single-image prediction. Production
heads also pass CPU forward/backward with synthetic CLIP features. Actual
pretrained CLIP on CUDA, T4 VRAM/runtime and VizWiz accuracy remain unverified.
