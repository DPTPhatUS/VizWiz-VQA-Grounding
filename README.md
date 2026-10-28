# Joint native crop grounding

Branch: `exp/joint-native-crop`.

Two fresh 20-epoch runs test whether native OCR-context crops help when their
loss can adapt the grounding features from the start. This follows the frozen
native-refinement pilot, whose measured gains were negligible. Improvement is a
hypothesis; this implementation does not establish an accuracy gain.

| Session | `--variant` | Inference |
|---|---|---|
| A | `pyramid` | Question-guided detail pyramid control |
| B | `joint-crop` | Same pyramid + native context crop decoder + global cleanup |

Both initialize CLIP encoders from pretrained weights and initialize grounding
layers afresh. Both train vision, freeze text, and use question-only input.
There is **no B+5 checkpoint, teacher, answer input, or frozen trained parent**.
The extra crop module has 183,514 parameters with the current CLIP/pyramid sizes;
the frozen CLIP text encoder is reused for Q and OCR strings, not duplicated.
The existing `train.py`, `train_ocr.py` and their checkpoint loaders stay unchanged.

## Architecture and controlled comparison

```text
Original RGB image + question
  |
  +-- 336x336 view -> pretrained CLIP vision (TRAINABLE) -> 4 feature maps
  +-- Q -> pretrained CLIP text (FROZEN) -> full question token sequence
  +-- 768x768 view -> trainable detail CNN pyramid
                      |
           question-guided detail decoder / refiner
                      |
             raw pyramid logits [B,1,768,768]
                      |
           interpolate logits to native image size
                      |
        A: native pyramid mask

B additionally:
  Cached frozen OCR -> polygons + recognized strings + confidence
    -> context boxes: 2x OCR bounds, minimum 32 native pixels per axis
    -> original RGB sampled as [B,N,3,192,192], N <= 24
    -> small U-Net-like crop decoder + trainable pyramid semantic crops
       + question-token/OCR-set attention
    -> absolute crop logits [B,N,1,192,192]
       |
       +-- crop field + pyramid detail/semantic/Q -> global signed correction
       +-- inverse crop rendering at original image pixel centres
                      |
     inside context boxes: average overlapping local logits
     outside context boxes: pyramid logits + global correction
                      |
               B: native final mask

Final, pyramid, crop and auxiliary losses all backpropagate into the pyramid.
```

The global correction may change pixels anywhere; final crop masks are not
restricted to OCR polygons. There is no learned selector or parent-preservation
loss. Missing OCR removes local overwrite, while the learned global correction
remains. Crops still resize RGB to 192x192; this is not lossless image processing.

To isolate architecture from extra local supervision, **A and B both receive the
same native GT supervision within the same OCR context boxes**. In A the local
prediction is a slice of the native pyramid mask; in B it is the native rendering
of the crop decoder. Consequently A is a new matched control, not an exact
reproduction of the historical pyramid training recipe. Compare B against A;
compare historical B+5 separately and count its full 25-epoch training cost.

For a native logit mask `z` and binary GT `y`, `S(z,y)` is mean pixel BCE plus
soft Dice loss, with smoothing `1e-6`. The common objective is:

```text
S(final_native, GT)
+ 0.25 * S(raw_pyramid_native, GT)
+ 0.25 * mean_valid_crops S(local_native, GT_inside_context_box)
+ 0.10 * S(aux96_native, GT)
+ 0.20 * S(aux192_native, GT)
+ 0.30 * S(aux384_native, GT)
```

A's final and raw pyramid predictions coincide, so their two terms explicitly
sum to `1.25 * S(pyramid_native, GT)`. Crop losses average within each original
image, then all terms average over original images. Images with no retained OCR
have zero crop loss. All three auxiliary heads remain supervised. The `.25`
weights are declared pilot settings, not coefficients established by the data.

No GT mask is resized for these losses. Geometry uses float64 pixel centres and
half-open supports; native crop-logit interpolation uses float64 before returning
float32. BCE/sigmoid run in float32 and loss reductions in float64. This reuses
the verified native geometry implementation. Floating-point error and CUDA
interpolation nondeterminism still exist; bitwise GPU reproducibility is not claimed.

Pyramid initialization, data order, encoder policy, dropout draws, batch and
optimizer settings match across A/B. Crop-specific operations can still change
numerical execution; equal seeds are not a claim of identical GPU trajectories.
OCR cache generation is fixed, question/answer independent, and outside the
trainable model. GT is used only for training losses or evaluation metrics.

## Kaggle: two separate GPU sessions

Use Internet and 2xT4. Attach VizWiz and the **existing OCR cache**. No new CPU
cache run or trained parent input is needed. Session A and B each train 20 epochs.
Only change `RUN_VARIANT` in cell 1 for the second fresh session.

### Cell 1: setup

```python
!pip install -q uv
!git clone --single-branch --branch exp/joint-native-crop https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen

import os
from pathlib import Path

os.environ["RUN_VARIANT"] = "pyramid"  # Session B: "joint-crop"
os.environ["DATA_ROOT"] = "/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz"
os.environ["OCR_CACHE"] = "/kaggle/input/datasets/khoahoang0512/vizwiz-ocr-cache/ocr_cache"
os.environ["RUN_ROOT"] = "outputs"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
assert os.environ["RUN_VARIANT"] in ("pyramid", "joint-crop")
assert (Path(os.environ["OCR_CACHE"]) / "manifest.json").is_file()
```

Adjust only the cache path if the attached input has a different folder name.
Each fresh session writes directly into `outputs/`.

### Cell 2: train

```bash
%%bash
set -euo pipefail
uv run torchrun --standalone --nproc_per_node=2 train_joint.py \
    --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
    --output-dir "$RUN_ROOT" --variant "$RUN_VARIANT" \
    --num-epochs 20 \
    --batch-size 2 --gradient-accumulation-steps 2 \
    --num-workers 2 \
    --lr 0.00001 --optimizer adam --weight-decay 0 \
    --lr-scheduler none --warmup-ratio 0 \
    --freeze-vision-encoder False --freeze-text-encoder True \
    --max-candidates 24 --ocr-dropout 0.2 --text-dropout 0.2 \
    --gradient-checkpointing --amp \
    --save-every 5 --seed 42
```

`--batch-size` is the GLOBAL microbatch: 2 means 1/GPU. Accumulation2 gives
an effective global batch4. Both variants use the same setting. Native crop
activations are checkpointed along with the pyramid/vision backbone. Larger
microbatches are optional only after checking actual GPU memory; no T4 throughput
or full-model GPU memory measurement has been made locally.

There is no validation inside the trainer. Evaluate the four saved checkpoints
below, then select the checkpoint by validation mean IoU, with earlier epoch
breaking ties. Never select using test scores.

### Checkpoints and disk budget

The trainer writes:

- `checkpoint_epoch5.pt`, `checkpoint_epoch10.pt`, `checkpoint_epoch15.pt`,
  `checkpoint_epoch20.pt`: inference weights + architecture/protocol metadata.
- **One** `latest_resume.pt`: model + Adam + scaler + training-control state and
  history, atomically replaced at each save event.

Fully frozen pretrained encoders are omitted from these files and recreated
from Hugging Face on load. An exact tensor fingerprint is checked before use;
missing/incompatible pretrained weights fail explicitly. Keep the HF cache or
Internet access. Trainable vision and all trained grounding tensors are saved.
Tiny contract models save everything and never omit random encoder tensors.

With frozen text and trainable vision, expect roughly **1.24 GB per snapshot**
and **3.72 GB for the resume file**, before small metadata/serialization overhead.
Four snapshots + resume occupy about 8.7 GB; the old/new resume coexist during
atomic replacement, bringing transient checkpoint storage to about **12.4 GB**.
The exact tensor-byte estimate is printed and saved in `run_config.json`.
Leave room for result masks, HF/uv caches and temporary files. If space is tight,
use `--save-every 10` **in both runs** and evaluate only 10/20. Unfreezing text
increases both VRAM and checkpoint storage; the estimates above no longer apply.
Startup checks the projected checkpoint peak plus a 1 GiB result reserve.
The save routine also checks free space, retains the old resume on failure, and never
silently deletes checkpoints. Disk I/O synchronization uses a separate Gloo group.

### Cell 3: validation at each checkpoint

```bash
%%bash
set -euo pipefail
for EPOCH in 5 10 15 20; do
    uv run eval_joint.py \
        --checkpoint "$RUN_ROOT/checkpoint_epoch$EPOCH.pt" \
        --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
        --dataset val --view full --batch-size 1 --num-workers 2 \
        --output-dir "$RUN_ROOT/eval_val_ep$EPOCH"
done
```

Evaluation is float32. Batch1 bounds native-output/crop memory; batch2/4 can be
used if GPU memory allows. It does not change the requested metric definition.
Mean IoU, global IoU, size groups, per-image scores and native PNG masks are saved.
`base_*` in `metrics.json` refers to the raw pyramid **of this same checkpoint**,
which has been jointly trained in B. It is not the independently trained run A.

### Cell 4: select, evaluate test and diagnose

```python
import json
import subprocess

root = Path(os.environ["RUN_ROOT"])
epochs = (5, 10, 15, 20)
best = max(epochs, key=lambda epoch: (
    json.loads((root / f"eval_val_ep{epoch}/metrics.json").read_text())["summary"]["mean_iou"],
    -epoch,
))
(root / "selection.json").write_text(json.dumps({
    "variant": os.environ["RUN_VARIANT"], "selected_epoch": best,
    "selection_metric": "validation_mean_iou", "tie_break": "earlier_epoch",
}, indent=2))

def evaluate(split, view, epoch):
    subprocess.run([
        "uv", "run", "eval_joint.py", "--checkpoint", str(root / f"checkpoint_epoch{epoch}.pt"),
        "--data-root", os.environ["DATA_ROOT"], "--cache-root", os.environ["OCR_CACHE"],
        "--dataset", split, "--view", view, "--batch-size", "1", "--num-workers", "2",
        "--output-dir", str(root / f"eval_{split}_{view}_ep{epoch}"),
    ], check=True)

for epoch in sorted({best, 20}):
    evaluate("test", "full", epoch)  # Compare validation-selected and fixed epoch20 separately.
if os.environ["RUN_VARIANT"] == "joint-crop":
    for view in ("pyramid", "global", "missing", "geometry", "shuffled_text"):
        evaluate("val", view, best)
for view, folder in [("full", f"eval_val_ep{best}")] + (
    [("pyramid", f"eval_val_pyramid_ep{best}"), ("global", f"eval_val_global_ep{best}")]
    if os.environ["RUN_VARIANT"] == "joint-crop" else []
):
    subprocess.run([
        "uv", "run", "diagnose.py", "--data-root", os.environ["DATA_ROOT"],
        "--predictions", str(root / folder), "--output-dir", str(root / f"diagnostic_{view}"),
    ], check=True)
print(f"Selected epoch {best}: {os.environ['RUN_VARIANT']}")
```

Views do not retrain or use GT as input:

| View | Meaning |
|---|---|
| `full` | Final jointly trained output |
| `pyramid` | Raw pyramid branch of the same checkpoint |
| `global` | Keep crop evidence/global correction, disable final local overwrite |
| `missing` | Disable all OCR candidates; global correction remains |
| `geometry` | Zero recognized-string embeddings; Q, image, boxes and RGB remain |
| `shuffled_text` | Permute recognized strings across images, preserving boxes/RGB |

For A all views return the same pyramid prediction; the guide skips redundant
interventions. For B, compare full/global/pyramid and missing/shuffled-text to
separate crop replacement, global cleanup and recognized-string contributions.
Compare A/B small (<1% area), medium (1-10%) and large (>=10%) performance, along
with per-image wins/losses and global IoU. OCR only covers some failure cases;
this experiment does not promise to fix tiny non-text objects or blurred evidence.

### Cell 5: ZIP results, excluding checkpoints

```python
import zipfile

archive = Path(f"joint_native_{os.environ['RUN_VARIANT']}_ep20_seed42.zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() not in (".pt", ".tmp", ".zip"):
            bundle.write(path, path.relative_to(root))
print(archive.resolve())
```

Download the ZIP for analysis. Keep desired `.pt` files separately as Kaggle
outputs/inputs. `latest_resume.pt` supports training recovery; weights-only
snapshots support evaluation/deployment and deliberately reject exact resume.

## Resume, configuration and deployment

For epoch-boundary recovery, use the same training command with
`--resume-checkpoint /kaggle/input/YOUR_RUN/latest_resume.pt` and a **new, empty**
`--output-dir`. Keep variant, seed, batch/accumulation, cache, objective and encoder
policy unchanged. `--num-epochs 20` is the total target, not 20 additional epochs.
Data order and RNG are reseeded by epoch/rank, and optimizer/scaler/control state
are restored. No mid-epoch resume is claimed. With an active LR schedule, keep the
original total step budget; constant-LR runs can extend their total epochs.

### Kaggle: resume an interrupted run

Download `latest_resume.pt` separately from the previous session (the analysis
ZIP excludes `.pt` files). Upload it as a Kaggle Dataset and attach it to the new
GPU notebook. Run **Cell 1** first, keeping the original `RUN_VARIANT` (`joint-crop`
for Session B). Replace Cell 2 with these cells; do not also run fresh training.

```python
os.environ["RESUME_CHECKPOINT"] = "/kaggle/input/datasets/YOUR_USERNAME/YOUR_DATASET/latest_resume.pt"
assert Path(os.environ["RESUME_CHECKPOINT"]).is_file()
assert not Path(os.environ["RUN_ROOT"]).exists() or not any(Path(os.environ["RUN_ROOT"]).iterdir())
```

Replace the checkpoint path with the actual attached input path. Keep `RUN_ROOT`
as `outputs`, empty in this new session.

```bash
%%bash
set -euo pipefail
uv run torchrun --standalone --nproc_per_node=2 train_joint.py \
    --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
    --output-dir "$RUN_ROOT" --variant "$RUN_VARIANT" \
    --resume-checkpoint "$RESUME_CHECKPOINT" \
    --num-epochs 20 \
    --batch-size 2 --gradient-accumulation-steps 2 \
    --num-workers 2 \
    --lr 0.00001 --optimizer adam --weight-decay 0 \
    --lr-scheduler none --warmup-ratio 0 \
    --freeze-vision-encoder False --freeze-text-encoder True \
    --max-candidates 24 --ocr-dropout 0.2 --text-dropout 0.2 \
    --gradient-checkpointing --amp \
    --save-every 5 --seed 42
```

A file saved after epoch 10 continues at epoch 11 and stops at epoch 20; it does
not train 20 additional epochs. Work since the last completed epoch is repeated.
Preserve any custom settings from the original run. Earlier snapshots are not
copied into the new `outputs/`: keep them separately and evaluate only checkpoint
files actually present. The new session writes an updated `latest_resume.pt` and
its scheduled snapshots.

Optional `--config path.json` follows **config > explicit CLI > defaults**.
Paths remain on CLI. Available optimizer/schedule/freezing options are shared
with the project: `--optimizer adam|adamw`, `--lr`, `--weight-decay`,
`--backbone-lr`, `--lr-scheduler none|cosine|linear|poly`, `--warmup-ratio`,
`--min-lr-ratio`, `--poly-power`, `--freeze-backbone True|False`,
`--freeze-vision-encoder True|False`, `--freeze-text-encoder True|False`.
For example, `{"roi_weight": 0.25, "pyramid_weight": 0.25}` changes objective
settings. Auxiliary keys are `aux96_weight`, `aux192_weight`, `aux384_weight`.
The guide needs no JSON file. Do not tune these settings differently across A/B
if interpreting their difference as an architecture comparison.

GT-free deployment uses an OCR record under `CACHE/single/IMAGE_STEM.json`, with
the same manifest/provider protocol as training, or `--no-ocr`:

```bash
uv run predict_joint.py --checkpoint outputs/checkpoint_epoch20.pt \
    --image /path/image.jpg --question "What does the label say?" \
    --cache-root /path/ocr_cache --output prediction.png
# Replace --cache-root ... with --no-ocr for the missing-evidence path.
```

Use `eval_joint.py` / `predict_joint.py` for new checkpoints. Old pyramid/OCR
checkpoints remain supported by their original entrypoints on their branches.
This is a new experiment and requires fresh training.

## Verification boundary

Offline checks cover native geometry/loss contracts, crop-to-pyramid gradients,
empty OCR, strict checkpoint loading, two-rank CPU/Gloo accumulation and exact
epoch-boundary resume, all inference views, native mean/global IoU, diagnostics
and GT-free prediction. Mixed-precision checks exercise CPU fp16/bfloat16 paths.
They do not establish real GPU accuracy, speed, VRAM usage or transfer performance.


## Optimized execution and existing checkpoints

The packed-native runtime keeps the same learned tensors, architecture metadata,
loss weights, precision policy and checkpoint format. It packs only candidates
that remain valid after OCR dropout into the CNN, then scatters masks back to
the original slots. Padding/dropped slots produce zeros and do not enter the
objective. Entirely empty OCR batches skip the crop CNN with a zero-gradient DDP
anchor; the global correction still runs. GroupNorm normalizes each sample, so
removing padded samples does not change valid-sample normalization.

Dataset workers generate RGB context crops only for real candidates. Native
composition accumulates occupied windows instead of allocating a padded full-image
tensor per candidate. Training reuses those exact native crop-logit windows for
ROI loss, and avoids a redundant global-logit interpolation. The raw-pyramid
inference view bypasses the unused crop head and RGB crop preparation.

Frozen OCR text descriptors are cached in a bounded CPU LRU of16,384 strings per
process (about48 MiB for768-dimensional FP32 vectors). The cache is nonpersistent,
rebuilds after resume, separates autocast/device/mode settings and invalidates when
encoder parameters are loaded. Trainable text bypasses caching. This is internal
runtime caching; the existing OCR dataset cache remains unchanged.

Keep the same train command and all settings. To continue an interrupted old
joint-crop run, add `--resume-checkpoint /kaggle/input/YOUR_RUN/latest_resume.pt`
and use a fresh empty output directory. Keep `--num-epochs 20`: it denotes the
total target, not20 more epochs. A weights-only snapshot cannot recover Adam state.
No retraining from epoch0 is required for compatibility. Existing inference
checkpoints remain loadable. Runtime changes are recorded in run_config.json;
history now includes epoch_seconds, excluding checkpoint I/O. Do not combine
old/new timing segments as though execution were unchanged.

Output/loss/gradient equivalence was checked against the preceding implementation
on CPU fixtures, including empty OCR and overlapping native windows. Small
floating-point differences from packed convolution batches or accumulation order
can occur; identical GPU trajectories are not promised. A CPU fixture benchmark
with batch2,8 slots/image and only1 valid crop total measured about1.78s versus
0.14s per forward+backward. This sparse synthetic workload does not predict T4
speedup, real candidate density, or Kaggle session duration. Measure the new
GPU epoch_seconds before estimating remaining training time.
