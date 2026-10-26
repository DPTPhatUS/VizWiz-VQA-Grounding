# Question-guided native-image zoom

Branch: `exp/question-guided-zoom`, based on `exp/question-guided-detail-pyramid` at `3eaa010`. This experiment adds a native-resolution crop locator and shared U-Net crop reader to the global detail pyramid so fine local regions can be read directly from the full-resolution image before downsampling. Runs start fresh from pretrained CLIP weights, and parent checkpoints are rejected to avoid partial initialization.

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

CLIP is `openai/clip-vit-large-patch14-336`. The text encoder is frozen and stays in `eval` mode, while the vision encoder is fine-tuned except for its unused pooled-CLS `post_layernorm`. The global pyramid branch is unchanged: three skip projections (`1024 -> 256/GN/GELU -> Q-FiLM -> 128`), four standard-convolution upsampling blocks, four question-guided detail fusions, and a signed-logit refiner at 768 resolution.

At 96x96 resolution, the crop locator takes `(B, 32, 96, 96)` decoder semantic features and 16-channel broadcast question features to predict a spatial relevance map independently of the global mask head. It selects one crop at each of two scales (`0.25` and `0.5` of the native shorter image side) over a deterministic 16x16 center grid. Candidate boxes are clamped inside image bounds and snapped outward to integer native pixel edges (staying square up to 1-pixel rounding). Candidate windows are scored by:

```text
utility = sigmoid(locator) * (1 + 0.25 * 4*p*(1-p))
p = sigmoid(global logits resized to locator resolution)
score = mean utility within the candidate window
second-scale score *= 1 - 0.5 * IoU(candidate, first crop)
```

Window means are computed exactly via integral images, and the overlap penalty on the second scale discourages redundant boxes. Discrete box selection is detached and supervised through the locator loss.

Native RGB images remain `uint8` through dataset loading, collation, and host-to-device transfer. Only the selected native crops are converted to `[0, 1]` floats, resized to 256x256 with bicubic antialiasing, clamped to `[0, 1]`, and normalized with CLIP mean/std. Feature and mask crops share identical normalized `xyxy` pixel-edge coordinates with `align_corners=False` (using bilinear interpolation for features and nearest-neighbor sampling for binary ground-truth masks).

Inside the shared U-Net crop reader, 64x64 local features fuse with the corresponding crop of global `F96` semantic features and attend to all non-padding CLIP question tokens. For each crop, the U-Net outputs a 256x256 signed logit correction and a scalar quality logit. The correction head is zero-initialized (so initial predictions match the global mask), and the quality bias is initialized to `sigmoid(-2)`.

At any target resolution, including native image dimensions, crop corrections are blended via:

```text
w_k(x) = sigmoid(quality_k) * tapered_crop_window_k(x)
z_final(x) = resize(z_global)(x) + sum_k(w_k(x)*delta_k(x)) / max(1, sum_k(w_k(x)))
```

The spatial window equals 1 across the crop interior and tapers linearly to 0 over the outer 10% border on each side, preventing boundary seams or amplification where the two crops overlap. During evaluation, crop corrections are projected directly onto the native image grid while global logits are interpolated from 768.

The zoom components add **250,723 parameters**, bringing total non-backbone parameters to **6,899,824**.

## Training and GT boundaries

Both training and inference form the final mask from the two model-selected crops using only the image and question. Ground-truth masks never enter box selection or forward mask prediction.

During training only, the shared crop reader also runs on a third teaching crop centered on the foreground pixel closest to the ground-truth centroid at the `0.25` crop scale (or the image center with an empty target when ground truth is empty). This third crop receives local crop supervision but is excluded from the final composited mask, and GroupNorm plus per-crop attention keep its activations isolated from the two model-selected crops.

Default training objective:

```text
L = L_final768 + 0.3*L384 + 0.2*L192 + 0.1*L96
    + 0.2*L_locator + 0.5*L_crop + 0.05*L_quality
```

Each segmentation term is mean pixel BCE plus `0.5` times mean per-image soft Dice loss. Global 768 masks use nearest-resized 768 targets, auxiliary pyramid heads use area-interpolated targets, locator supervision uses area interpolation directly from the native ground-truth mask, and crop targets are sampled directly from native ground truth. `L_crop` averages over all three crops per image and then across images. `L_quality` applies BCE against the detached binary foreground IoU of each crop prediction (thresholded at logit 0, with empty-union IoU equal to 1.0).

Optional `--config` JSON (showing defaults):

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

`fp_weight` is used only when `overlap_loss="tversky"`. To compare against a no-zoom baseline, use the parent branch `exp/question-guided-detail-pyramid` rather than zeroing the zoom loss weights.

## Kaggle run

Enable Internet and two T4 GPUs, then clone the repository and install the locked dependencies before running `torchrun`:

```bash
git clone --single-branch --branch exp/question-guided-zoom https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
cd VizWiz-VQA-Grounding
uv sync --frozen
export DATA_ROOT=/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz
export OMP_NUM_THREADS=1
```

Run the multi-GPU preflight check first:

```bash
uv run --frozen torchrun --standalone --nproc_per_node=2 preflight.py \
  --data-root "$DATA_ROOT" --batch-size 2 --num-workers 0 --steps 3 \
  --gradient-checkpointing
```

Preflight verifies finite gradients across all trainable parameters under AMP and DDP without writing checkpoints. Initial AMP overflow steps trigger `GradScaler` backoff and retry the same batch (up to 8 reductions), while missing gradients, nonfinite forward losses, or persistent overflow fail immediately. Pass the same `--config` file to both `preflight.py` and `train.py` when customizing loss weights.

Run the 20-epoch training job:

```bash
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py \
  --data-root "$DATA_ROOT" --output-dir outputs \
  --num-epochs 20 --batch-size 2 --gradient-accumulation-steps 2 \
  --gradient-checkpointing --num-workers 2 --lr 0.00001 --seed 42 --save-every 10
```

A global microbatch of 2 places 1 image on each GPU, and `--gradient-accumulation-steps 2` gives an effective batch size of 4. Training uses CUDA FP16 AMP, Adam with LR `1e-5`, gradient clipping at norm 1.0, and no LR scheduler or per-epoch validation. Full resumable checkpoints are saved at epochs 10 and 20 directly in `outputs/` alongside final raw weights; resuming requires matching batch size, accumulation steps, learning rate, seed, and world size. Fresh runs abort if `outputs/` already contains training history or checkpoints.

## Evaluation and diagnostic

Evaluate epochs 10 and 20 on `val` (and `test` when ground-truth annotations and masks are present) using distinct output directories:

```bash
uv run --frozen python eval.py \
  --checkpoint outputs/checkpoint_epoch20.pt --dataset val \
  --data-root "$DATA_ROOT" --output-dir outputs/eval_val_ep20 \
  --batch-size 1 --num-workers 2 --device cuda
uv run --frozen python diagnose.py \
  --predictions outputs/eval_val_ep20 --data-root "$DATA_ROOT" \
  --output-dir outputs/diagnostic_ep20
```

Evaluation runs in a single process in FP32, scoring native-resolution masks thresholded at logit 0 (with empty-union IoU set to 1.0; `--batch-size 4` can also be used if memory allows). `eval.py` writes native binary PNG masks and `metrics.json`, which records both the final `mean_iou` and the pre-zoom coarse `global_mean_iou` from the same forward pass. `zoom_diagnostics` logs per-image coarse and final IoU, native ground-truth recall inside the two selected crops, normalized crop boxes, and predicted crop quality scores.

Select checkpoints and hyperparameters on the validation split only, and compare both small-region precision/recall and crop proposal coverage against the standalone parent pyramid run.

## Encoder freezing and optimization

Omitting the options below keeps the branch's default training configuration: Adam with LR `1e-5`, zero weight decay, constant LR, and the default encoder freezing policy. Encoder freezing is opt-in, while the detail CNN and decoder always remain trainable. Model state-dict keys and evaluation preprocessing are unchanged.

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

`BOOL` accepts `True` or `False`; passing a freeze flag without a value sets it to `True`. The negative aliases `--no-freeze-backbone`, `--no-freeze-vision-encoder`, and `--no-freeze-text-encoder` set the corresponding flag to `False`. Within the same configuration source, a specific encoder flag overrides `freeze_backbone`. Precedence follows **JSON config > explicit CLI > branch defaults**, so a JSON shorthand also overrides component flags passed on the CLI. Recipe and loss settings can share the same JSON file, and unrecognized keys raise an error immediately. Run `train.py --help` to see all branch-specific flags.

Example CLI flags to freeze the backbone and use AdamW with a cosine schedule:

```bash
--freeze-backbone True \
--optimizer adamw \
--lr 0.0001 \
--weight-decay 0.01 \
--lr-scheduler cosine \
--warmup-ratio 0.05 \
--min-lr-ratio 0.1
```

To freeze only the vision encoder while keeping the default text policy, pass `--freeze-vision-encoder True`. To train both encoders explicitly, pass `--freeze-backbone False`. Branches that rely on a fixed reference stage (such as gain-guided refinement) reject encoder unfreezing. Unused CLIP/SigLIP pooled-output parameters remain frozen when token encoders are unfrozen.

Equivalent JSON configuration (`--config path/to/train.json`):

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

Frozen encoders stay in `eval` mode during training and are excluded from the optimizer. Learning-rate schedules step once per completed optimizer update (accounting for gradient accumulation) and do not advance when AMP skips an update due to gradient overflow. Checkpoints record the resolved optimization config, trainable parameter layout, and scheduler state; resuming restores those settings and rejects conflicting overrides. When resuming a scheduled run, keep `--num-epochs`, dataloader length, and gradient accumulation unchanged so the LR schedule trajectory stays consistent.

Existing checkpoints remain compatible with evaluation and legacy constant-LR resume. Changing encoder freezing, optimizer type, weight decay, or LR scheduling requires starting a new run (or using weights-only initialization where supported).

## Foreground global IoU

`eval.py` writes both dataset-level IoU metrics to `metrics.json` under `summary`, computed from the final native-resolution binary masks:

- `mean_iou`: arithmetic mean of per-image foreground IoUs (weighting each image equally).
- `global_iou`: total foreground intersection pixels divided by total foreground union pixels across the evaluated split.
- `total_intersection` and `total_union`: exact integer pixel counts across the split.

Values are reported as fractions in `[0, 1]` (multiply by 100 for percentages). Logits are resized to each image's original resolution before thresholding at zero, and an empty union evaluates to an IoU of 1.0.

`diagnose.py` also records `global_iou` under `overall` and `by_gt_area` in `summary.json` directly from the exported PNG masks and native ground truth on CPU:

```bash
uv run diagnose.py --predictions outputs/eval_val_ep20 \
    --data-root /path/to/vizwiz --output-dir outputs/diagnostic_global_iou_ep20
```

On this zoom branch, `global_mean_iou` in `metrics.json` refers to the per-image mean IoU of the coarse global branch before applying local zoom corrections, whereas `global_iou` is the dataset-wide intersection-over-union of the final composited prediction.
