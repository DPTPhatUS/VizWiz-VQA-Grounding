# Native crop and point refinement

Branch: `exp/ocr-native-refinement`.

These two experiments refine masks instead of choosing a fixed OCR polygon.
They reuse the frozen B+5 parent and existing OCR cache, with no answer input.

| Run | `--architecture` | Added trainable parameters | Head + AdamW checkpoint |
|---|---|---:|---:|
| A | native_crop | 183,514 | about 2.13 MiB |
| B | native_point | 111,931 | about 1.30 MiB |

The parent still has 433,217,037 parameters, stored separately. These are new
heads trained independently, not continuations of the OCR pointer/scorer runs.
CPU checks establish implementation contracts, not accuracy or T4 throughput.

## A: native context crops + global cleanup

```text
image + Q -> frozen B+5 -> base logits / semantic / detail / Q tokens
                     |
OCR polygons/text/crops -> Q-token attention + OCR-set context
                     |
raw-image context crops [B,N,3,192,192], N <= 24
  context box = 2x OCR box, at least 32 native pixels per dimension
                     |
small U-Net-like encoder/decoder + semantic crops + Q/OCR conditioning
                     |
absolute crop-mask logits [B,N,1,192,192]
                     |
  +-- crop probability field at 96x96
  |       + semantic/detail/Q/base -> signed GLOBAL correction [B,1,96,96]
  |
  +-- inverse crop rendering at ORIGINAL image pixel centres
                     |
inside context crops: average overlapping local logits
outside context crops: original base logits + global correction
                     |
native mask (H_original, W_original)
```

Every retained OCR region is processed; GT never selects a crop. This avoids
requiring a trained top-candidate selector before local segmentation can learn.
Crop RGB comes from the upright original image, with continuous sampling into
192x192. This still resizes individual crops; it does not claim lossless imagery.
The global parent continues to provide whole-image semantics.

Local masks are absolute predictions. They can replace a confidently wrong parent
inside their support. Averaging overlaps is deterministic, without a learned
quality gate. The global correction has whole-image support and can remove inherited
foreground OUTSIDE OCR/crop support. There is no parent-preservation distillation.
Wrong local masks can still damage a good parent, so large-region harm must be measured.

The final output is not restricted to OCR polygons. Missing OCR disables the local
path; the trained global correction remains active. `missing` therefore does not
have to equal `base` in this branch.

## B: native coordinate point head

```text
frozen parent + Q/OCR context -> global signed correction at 96x96
                         |
query at a NATIVE pixel centre:
  semantic/detail/Q/OCR context features
  + native RGB 3x3 neighbourhood (integer gathers)
  + normalized xy + globally corrected logit
                         |
small shared MLP -> signed point correction
                         |
final native pixel logit
```

Training samples 8,192 points from a known proposal mixture: 50% uniform image,
25% OCR-context support and 25% uncertainty. Empty OCR support or underflowed
uncertainty uses a uniform component. Confidently wrong areas remain reachable
through the uniform component. A CDF sampler avoids the multinomial category-count
limit on very large images.

There are also 1,024 uniformly sampled foreground and 1,024 background points
when those classes exist. GT is used ONLY to choose these TRAINING samples and
provide labels; it is never an inference input or point feature. Exact native
integer indices retrieve the labels and RGB neighbours, without mask interpolation.

Inference evaluates every native pixel in chunks of 65,536. It is a coordinate
point head, not a reproduction of PointRend's adaptive subdivision. Trainable
parameter count is small, but inference work scales with native image area.
This branch does not claim sparse inference or a measured speed advantage.

## Geometry, objectives and numerical precision

One implementation defines the native output for training, evaluation and prediction.
Normalized support is half-open: `x1 <= (x+.5)/W < x2`, and similarly for y.
In exact arithmetic the integer limits are `ceil(x1*W-.5)` / `ceil(x2*W-.5)`.
The implementation searches float64 pixel centres directly, preserving exact-centre
ties when multiply-back rounding would otherwise move a boundary by one pixel. There is no
`+0.999999`, rounded GT resize, or validation-dependent mask alignment.
Geometry is formed in float64. Native local-mask logit interpolation also uses
float64 before returning float32 logits; feature/RGB interpolation uses float32.
Sampling always uses `align_corners=False` and explicit border handling. Local rendering uses
bounded chunks and activation recomputation for backward.

Let `S(z,y)` be full-native mean pixel BCE plus soft Dice:

```text
S = mean(BCEWithLogits(z,y))
    + 1 - (2*sum(sigmoid(z)*y) + 1e-6)
              / (sum(sigmoid(z)) + sum(y) + 1e-6)
```

GT is binary on its ORIGINAL grid. Nonlinearities are float32; reductions and Dice
ratios are float64. The small smoothing constant handles empty masks; it is not
an extra foreground pixel. DDP graph-anchor sums are float32 too, preventing FP16 overflow before a zero
multiplication. Training never thresholds predicted logits. Integer
native intersection/union and float64 division define evaluation IoU.

```text
A: L = S(final_native, GT)
       + 0.25 * [S(global_native, GT) + mean_ROIs S(local_native, GT_inside_ROI)]
```

ROI loss uses every native pixel centre inside the same support used by rendering.
It is averaged within each image, then across original images. Duplicate/overlapping
crops do not increase an image's outer loss weight.

For B, let q(p) be the actual normalized pixel sampling probability, M=8,192 and
A the original image pixel count:

```text
L_pixel = sum_sampled BCE(z_p,y_p) / (M*A*q(p))
L_balanced = mean over present classes [mean uniformly sampled class BCE]
B: L = L_pixel + 0.25*L_balanced + 0.5*S(global_native, GT)
```

`L_pixel` is an unbiased Monte Carlo estimate of mean native pixel BCE, up to
floating-point arithmetic. The uniform proposal bounds each importance factor
`1/(A*q(p))` by approximately 2. `L_balanced` deliberately changes the objective
to strengthen small foreground learning; it is not passed off as ordinary pixel
BCE. No sampled-point Dice ratio is used. The point loss has sampling variance;
it is not an exact full-image final-output loss. Full-image Dice supervises the
global branch. Losses average by ORIGINAL image, including partial accumulation
windows and ranks/images with no OCR.

IEEE floating-point arithmetic still has finite precision, and CUDA interpolation
backward can be nondeterministic. The checks target coordinate/target consistency,
finite gradients, estimator correctness and bounded numerical error, not a promise
of zero rounding error or bitwise CPU/CUDA equality.

## Kaggle: two separate GPU sessions, 20 epochs each

Use Internet + 2xT4. Attach the existing VizWiz, B+5 checkpoint and OCR cache inputs.
The standard pilot below needs no new CPU cache or parent retraining. It does NOT
claim the parent is unseen on training images.

### Cell 1: setup

```python
!pip install -q uv
!git clone --single-branch --branch exp/ocr-native-refinement https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen

import os
from pathlib import Path

os.environ["RUN_VARIANT"] = "A"  # Fresh second session: "B"
os.environ["ARCHITECTURE"] = "native_crop" if os.environ["RUN_VARIANT"] == "A" else "native_point"
os.environ["DATA_ROOT"] = "/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz"
os.environ["PARENT_CKPT"] = "/kaggle/input/datasets/khoahoang0512/vizwiz-size-weighted-pyramid-cp/mild_focus/checkpoint_epoch5.pt"
os.environ["OCR_CACHE"] = "/kaggle/input/datasets/khoahoang0512/vizwiz-ocr-cache/ocr_cache"
os.environ["RUN_ROOT"] = "outputs"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
assert os.environ["RUN_VARIANT"] in ("A", "B")
assert Path(os.environ["PARENT_CKPT"]).is_file()
assert (Path(os.environ["OCR_CACHE"]) / "manifest.json").is_file()
```

Use the exact B+5 **mild_focus/checkpoint_epoch5.pt**, SHA256
`d0058ab5d7ae4c9470b98f0728c15f20df4c32e68cbfa1a9d7bdf1960ac8953e`.
Do not use an OCR adapter as the parent. Both parent encoders and all parent
layers stay frozen; only the new head trains.

### Cell 2: train

```bash
%%bash
set -euo pipefail
uv run torchrun --standalone --nproc_per_node=2 train_ocr.py \
    --parent-checkpoint "$PARENT_CKPT" \
    --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
    --output-dir "$RUN_ROOT" \
    --architecture "$ARCHITECTURE" \
    --evidence-mode full --answer-weight 0 \
    --num-epochs 20 \
    --batch-size 2 --gradient-accumulation-steps 2 \
    --num-workers 2 \
    --lr 0.0001 --optimizer adamw --weight-decay 0.01 \
    --max-candidates 24 \
    --ocr-dropout 0.2 --text-dropout 0.2 \
    --save-every 5 --seed 42 --amp
```

Batch 2 is GLOBAL, 1 per GPU; accumulation2 gives effective global batch4. This
bounds crop activations while preserving the effective batch for A/B. The recipes
have different stochastic operations, so equal seeds do not imply identical
point/dropout draws. A/B are architectural experiments, not a one-factor ablation.
No validation runs inside training. Exactly four checkpoints are written at
5/10/15/20, excluding the parent. Together they require only a few tens of MiB,
including optimizer state and small protocol metadata.

### Cell 3: validation at each checkpoint

```bash
%%bash
set -euo pipefail
for EPOCH in 5 10 15 20; do
    uv run eval_ocr.py \
        --parent-checkpoint "$PARENT_CKPT" \
        --checkpoint "$RUN_ROOT/checkpoint_epoch$EPOCH.pt" \
        --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
        --dataset val --view full --batch-size 2 --num-workers 2 \
        --output-dir "$RUN_ROOT/eval_val_ep$EPOCH"
done
```

### Cell 4: select by validation, evaluate test and ablations

```python
import json
import subprocess

root = Path(os.environ["RUN_ROOT"])
best = max((5, 10, 15, 20), key=lambda epoch: (
    json.loads((root / f"eval_val_ep{epoch}/metrics.json").read_text())["summary"]["mean_iou"],
    -epoch,
))
(root / "selection.json").write_text(json.dumps({
    "variant": os.environ["RUN_VARIANT"], "architecture": os.environ["ARCHITECTURE"],
    "selected_epoch": best, "selection_metric": "validation_mean_iou",
    "tie_break": "earlier_epoch",
}, indent=2))
for split, view in [
    ("test", "full"), ("val", "global"), ("val", "base"),
    ("val", "missing"), ("val", "geometry"), ("val", "shuffled_text"),
]:
    subprocess.run([
        "uv", "run", "eval_ocr.py", "--parent-checkpoint", os.environ["PARENT_CKPT"],
        "--checkpoint", str(root / f"checkpoint_epoch{best}.pt"),
        "--data-root", os.environ["DATA_ROOT"], "--cache-root", os.environ["OCR_CACHE"],
        "--dataset", split, "--view", view, "--batch-size", "2", "--num-workers", "2",
        "--output-dir", str(root / f"eval_{split}_{view}_ep{best}"),
    ], check=True)
for view, folder in [("full", f"eval_val_ep{best}"), ("global", f"eval_val_global_ep{best}")]:
    subprocess.run([
        "uv", "run", "diagnose.py", "--data-root", os.environ["DATA_ROOT"],
        "--predictions", str(root / folder),
        "--output-dir", str(root / f"diagnostic_{view}"),
    ], check=True)
print(f"Selected epoch {best}, architecture {os.environ['ARCHITECTURE']}")
```

`global` disables the local overwrite/point correction at OUTPUT while retaining
the learned global branch and its OCR inputs. `base` returns the untouched parent.
`missing` disables OCR while retaining the refiner; it is NOT an exact parent
fallback. `geometry` removes recognized-string embeddings; RGB/context remain.
`shuffled_text` changes recognized strings across images. The old `evidence` view
is not defined for these heads. Test never chooses epochs or hyperparameters.

Use `eval_ocr.py`, not `eval.py`. Metrics include native mean/global IoU, size
groups, parent scores and native masks. General `diagnose.py` is appropriate here;
quality-regression/pointer diagnostics are not. Compare small/medium gains with
large-region harm and per-image changes, not only overall mean.

### Cell 5: ZIP results without checkpoints

```python
import zipfile

archive = Path(f"ocr_native_{os.environ['RUN_VARIANT']}_ep20_seed42.zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() not in (".pt", ".tmp", ".zip"):
            bundle.write(path, path.relative_to(root))
print(archive.resolve())
```

Download `ocr_native_A_ep20_seed42.zip` / `ocr_native_B_ep20_seed42.zip` and keep
small `.pt` files separately. Outputs stay directly under `outputs/`.

## Optional: learn from parent-unseen TRAIN images

This is a separate, more expensive protocol. The existing B+5 cannot establish
that property; it already learned on the full train split. Do not relabel it as
out-of-fold or unseen-parent training.

1. Generate deterministic disjoint image lists:

```bash
uv run prepare_parent_split.py --data-root "$DATA_ROOT" --output-dir splits --seed 42
```

2. In an EXTRA session, train a NEW parent from scratch on `splits/parent_fit.json`:

```bash
uv run torchrun --standalone --nproc_per_node=2 train.py \
    --data-root "$DATA_ROOT" --train-image-list splits/parent_fit.json \
    --output-dir outputs --num-epochs 20 --batch-size 4 --num-workers 2 \
    --lr 0.00001 --seed 42 --save-every 20
```

Keep the full `checkpoint_epoch20.pt` (with run metadata) and both list files.
The raw `model_final_epoch20.pt` lacks lineage and cannot satisfy the strict flag.
Saving every20 limits this optional session to one full checkpoint plus the legacy
raw final-weight file. Attach the full checkpoint and lists to the refiner
session. Do NOT resume B+5, a full-train parent or an unknown-lineage checkpoint.
Parent training records its image membership and whether resumed lineage is known.

3. For each native-refiner run, replace `PARENT_CKPT` with that NEW parent and add:

```text
--train-image-list /kaggle/input/YOUR_SPLIT/refiner_fit.json
--require-unseen-parent
```

The flag rejects unknown parent lineage or any image-ID overlap before writing
training outputs. It verifies membership relative to the canonical dataset IDs,
not semantic near-duplicate detection across different filenames or foundation-model
pretraining contamination. Split files
contain image names; no GT is changed and no images are copied.

85% of train fits the parent; the remaining15% fits the refiner. These are not
validation/test images. Compare against THAT parent's base predictions, since it
has different training data from B+5. This is one held-out-parent split, not K-fold
cross-fitting. Its extra parent training cost must be counted. No GPU execution
of this optional protocol is claimed by the implementation tests.

## Flags, resume and checks

Existing optimization flags remain. Optional flat config uses **config > CLI >
default**, with paths on CLI. New controls are `--architecture native_crop|native_point`,
`--train-image-list` and `--require-unseen-parent` / `--no-require-unseen-parent`.
Standard pilot runs need neither split flag. Loss constants/sampling/geometry,
training membership, cache and parent exposure are recorded in `run_config.json`.

Resume the SAME architecture with `--resume-checkpoint .../checkpoint_epoch10.pt`
and total `--num-epochs 20`, in a new empty output directory. Keep optimizer,
cache/GT, architecture, split membership, seed and effective batch settings unchanged.
Attach earlier checkpoints or adjust the evaluation epoch list when resuming.
Legacy residual/routing/selection/scope/pointer checkpoints retain their original
classes and versions. Old parent runs keep their default data order when no image
list is supplied.

Checks cover half-open fractional boxes, multichunk inverse rendering/gradients,
overlap order, extreme/empty-mask losses, importance correction, tiny foreground
sampling, native output shape, missing-OCR ranks, strict checkpoints, two-process
training/accumulation/exact resume, deployment views and GT-free prediction.
Production-channel CPU BF16 backward is checked. This does not establish GPU speed,
VRAM limits, statistical gains or transfer to another dataset/backbone.

The architecture borrows established local refinement and point-based segmentation
principles; see [CascadePSP](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_CascadePSP_Toward_Class-Agnostic_and_Very_High-Resolution_Segmentation_via_Global_and_CVPR_2020_paper.pdf)
and [PointRend](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html).
These are task-specific adaptations, not reproductions or claims that local/point
refinement is new. Another backbone needs a compatible frozen feature wrapper;
interface flexibility is not measured transfer. Blur, ambiguous questions and
missing OCR evidence remain real limits.
