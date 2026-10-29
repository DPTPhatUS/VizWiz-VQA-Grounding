# Evidence Transport: selection and evidence use

Branch: `exp/evidence-transport`.

Two independent **20-epoch adapter** experiments share one frozen-pyramid feature
cache. This is a new training protocol, not a continuation of joint-crop.
The parent is the **matched pyramid control from `exp/joint-native-crop`, epoch
20**. Do not substitute a joint-crop checkpoint, B+5, or an older pyramid
checkpoint: the extractor checks the checkpoint family and variant.

The previous joint-crop's selected validation checkpoint barely changed when OCR
was removed or its strings shuffled. These experiments remove the semantic-mask
bypass from the evidence branch and explicitly supervise which evidence matters.
They do not guarantee improvement or resolve an intrinsically ambiguous question.

## Experiments

| Recipe | What is trained | Question tested |
|---|---|---|
| `transport` | word selection + local masks + evidence segmentation + scope gate | Can a separately grounded evidence mask improve the frozen parent? |
| `intervention` | identical architecture and initialization, plus leave-one-evidence-out contribution loss | Does supervising actual evidence contribution improve selection/use? |

The adapter adds **169,891 parameters** at CLIP dimensions (about 0.68 MB of FP32 weights).
Both parent encoders **and the entire parent decoder are frozen**. OCR remains
external/frozen and uses the existing PP-OCR cache. No answer is needed for
training or inference. GT masks provide training labels only. The scope target
never depends on parent accuracy, avoiding the old train-seen parent-quality
calibration shortcut. This does not eliminate all possible overfitting.

### Architecture

```text
Image + Q -- frozen parent encoder --> region visual descriptors, scene, full Q tokens
OCR strings -- same frozen text encoder --> word descriptors
OCR boxes --> 2x context boxes (minimum 32 native pixels) --> RGB crops 64 x 64
                         |
word + region visual + geometry -- Q attention -- set attention --> 64-dim tokens
                         |                                      |
                    selection logits                     scope gate g
                         |
RGB crop --> 16/24-channel CNN + token conditioning --> 64 x 64 local probabilities
                         |
    inverse warp into image + competition with explicit background
                         |
                  evidence-only probability E
                         |
Final P = (1-g) * sigmoid(native parent logits) + g * E
```

The evidence branch cannot read parent logits, decoder semantic features, or GT.
It uses pre-fusion region features, full Q tokens, recognized text, geometry and
RGB. Empty OCR gives `g=0` and exactly the parent's prediction. Multiple words
can contribute simultaneously; there is no hard top-one OCR selection.
Outside all context boxes, `E=0`, so an evidence route can suppress excess
foreground outside OCR, unlike a correction restricted to OCR support.

The head accepts arbitrary text/visual dimensions from the cache manifest and
has no CLIP-specific hidden width. A different backbone/data adapter must produce
the same feature schema and a **new cache**; cached features are not portable
between checkpoints. OCR is not appropriate for every non-text grounding task.

### Targets and losses

Native GT polygon purity is `intersection / polygon area`. Selection learns this
soft target, balancing present positive/negative candidate groups within each
image. The scope target is 1 only when the union of polygons with purity >=0.5
covers >=80% of GT and has precision >=50%; otherwise it is 0. Scope BCE remains
unweighted to avoid silently changing its probability calibration.

The evidence-only segmentation loss is applied on scope-positive images. All
valid local masks receive BCE supervision. The parent/final mask's training
accuracy is **not** used to choose the gate target. No validation/test labels are
stored in the feature records or consumed by the head.

```text
L_transport = L_selection + L_scope + L_local + L_evidence
L_intervention = L_transport + 0.5 * L_leave_one_out
```

`L_evidence` is BCE + soft Dice. Global training targets use area-resampled
192 x 192 masks; local targets use supersampled pixel-centre sampling followed
by averaging to 64 x 64. These are explicit training surrogates, **not the old
full-native-resolution objective**. Native polygon purity and scope use original
GT resolution. All final masks/mean IoU/global IoU use original image resolution.

For intervention, analytically remove each candidate from the probability
competition. On unique GT mass of a positive polygon, its foreground contribution
is encouraged towards 1; on a negative polygon's background pixels it is
encouraged towards 0. Overlapping positive polygons are not falsely required to
be independently indispensable. Prefix/suffix log-sum-exp computes all removals
in O(B*N*H*W), with no extra CNN/attention forwards and no unstable `1-weight`
subtraction. This tests spatial evidence use; it does not guarantee semantic
understanding of recognized strings. The shuffled-text diagnostic tests that.

## Kaggle: one GPU session for cache + both runs

Use a GPU notebook, internet enabled, and attach:

1. VizWiz grounding data.
2. The existing OCR cache (no new OCR generation session).
3. The **matched pyramid control epoch-20 checkpoint** downloaded from the recent
   `joint_native_pyramid` run. Its `latest_resume.pt` is also accepted by the
   extractor, but a weights-only snapshot is smaller.

One GPU is sufficient: **do not use torchrun**. Both recipes run sequentially
with fresh model/optimizer/scaler and identical seed/order. Each result has a
20-epoch parent history plus 20 adapter epochs; it is not a fresh 20-epoch
end-to-end model. This is 40 lightweight
head epochs, not 40 full-backbone epochs. Actual Kaggle duration is unmeasured;
the history records seconds per epoch. Keep the effective batch unchanged when
comparing the two recipes.

### 1. Setup

```python
!pip install -q uv
!git clone --single-branch --branch exp/evidence-transport https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen

import os
from pathlib import Path

os.environ['DATA_ROOT'] = '/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz'
os.environ['OCR_CACHE'] = '/kaggle/input/datasets/khoahoang0512/vizwiz-ocr-cache/ocr_cache'
os.environ['PYRAMID_CKPT'] = '/kaggle/input/YOUR_PYRAMID_DATASET/checkpoint_epoch20.pt'
os.environ['RUN_ROOT'] = 'outputs'
os.environ['FEATURE_CACHE'] = 'outputs/feature_cache'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
Path(os.environ['RUN_ROOT']).mkdir(exist_ok=True)
assert Path(os.environ['PYRAMID_CKPT']).is_file()
assert (Path(os.environ['OCR_CACHE']) / 'manifest.json').is_file()
```

Change `PYRAMID_CKPT` to the actual Kaggle Input path. Do not copy it to working
disk. The two run folders will be `outputs/transport` and `outputs/intervention`.

### 2. Build the shared feature cache once

```bash
%%bash
set -euo pipefail
uv run cache_evidence_transport.py \
    --checkpoint "$PYRAMID_CKPT" \
    --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
    --output-dir "$FEATURE_CACHE" \
    --max-candidates 24 --batch-size 2 --num-workers 2
```

The cache is GT-free on val/test; only train has a separate target payload.
Training extraction skips the high-resolution parent decoder entirely. Region
and full-token descriptors are stored FP16; context RGB is uint8; losses run at
FP32 with FP64 Dice reductions. Area targets stay floating point so subpixel
GT mass is not rounded away to uint8; contribution normalization also preserves
fractional pixel mass. The same rounded cached inputs are used by both
recipes. Parent evaluation logits retain FP32 at their original grid size so
cache compression does not change the reference mask. Sparse polygon targets
avoid storing 24 dense 192 x 192 planes per training image.

For the full dataset, FP32 parent grids on val/test alone take about **8.27 GB**
(decimal); other features/targets add storage. Startup checks a projected cache
budget plus 2 GiB reserve, and every atomic write checks free space. The estimate
is not an absolute bound on image/candidate-dependent cache sizes. If it fails,
use a separate GPU cache session, ZIP its feature folder, download/upload it as
Kaggle Input, and set `FEATURE_CACHE` to that input folder in the training session.
Do not delete or duplicate existing user checkpoints to make space.

A completed feature cache can be reused for additional head runs. Check
`manifest.json` for `complete: true`, the parent SHA256 and the count
`training_scope_positives`. Cache generation requires a new/empty destination;
an interrupted extraction must be rerun into a new/empty cache directory.

### 3. Train, validate, test and run interventions

```bash
%%bash
set -euo pipefail
uv run run_evidence_transport.py \
    --feature-cache "$FEATURE_CACHE" --data-root "$DATA_ROOT" \
    --output-dir "$RUN_ROOT" --recipe both \
    --num-epochs 20 --batch-size 8 --num-workers 2 \
    --lr 0.0003 --weight-decay 0.01 --seed 42 --amp
```

For each recipe the command automatically:

- trains a freshly initialized adapter for 20 epochs;
- saves small weights-only snapshots at 5/10/15/20 and one replacing
  `latest_resume.pt` **every epoch**;
- evaluates all four on validation, selecting highest mean IoU (earlier epoch
  breaks ties);
- evaluates test at the validation-selected epoch and epoch 20 separately;
- evaluates selected-checkpoint val with missing OCR, zero word descriptors and
  globally shuffled recognized-word descriptors, independently of batch size;
- saves native PNG masks, per-image metrics, size groups, global IoU, gate/word
  scores, evidence-only IoU, rescue/harm counts and a GT oracle envelope.

The oracle envelope chooses between the **predicted** parent and evidence masks
using GT, only to diagnose unused potential. It is never a deployable result or
a checkpoint-selection metric. Compare oracle potential, actual gains, selection
scores, missing/shuffled sensitivity and large-region losses together.

Keep batch 8 for the matched comparison; reduce it for **both** runs if needed.
The code uses packed valid crops, shared CNN output for counterfactuals, batched
inverse warps, persistent loader workers, pinned transfers, AMP and foreach
AdamW on GPU. It avoids backbone backprop, per-candidate full-native losses and
repeated full-model training. No compilation/recompilation overhead or OCR model
is added to these head runs. None of these choices proves a measured GPU speedup.

### 4. ZIP analysis results, excluding weights and the shared cache

```python
import zipfile

root = Path(os.environ['RUN_ROOT'])
archive = Path('evidence_transport_two_runs_ep20_seed42.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for recipe in ('transport', 'intervention'):
        for path in sorted((root / recipe).rglob('*')):
            if path.is_file() and path.suffix.lower() not in ('.pt', '.tmp', '.zip'):
                bundle.write(path, path.relative_to(root))
    bundle.write(Path(os.environ['FEATURE_CACHE']) / 'manifest.json', 'feature_manifest.json')
print(archive.resolve())
```

Download this ZIP for analysis. Stop here for the ordinary training session;
the optional cache export below is not needed to analyze either run. Download desired small adapter checkpoints
separately. For a reusable cache, ZIP **the feature cache only**, separately:

```python
import shutil

cache = Path(os.environ['FEATURE_CACHE']).resolve()
archive = Path('evidence_transport_feature_cache.zip').resolve()
if cache == archive.parent or cache in archive.parents:
    raise ValueError('Write the ZIP outside the feature cache')
if archive.exists():
    raise FileExistsError(archive)
files = [p for p in cache.rglob('*') if p.is_file()]
# Conservative incompressible-data estimate, headers and 2 GiB notebook reserve.
required = int(sum(p.stat().st_size for p in files) * 1.01) + len(files) * 4096 + 2 * 1024**3
free = shutil.disk_usage(archive.parent).free
if free < required:
    print(f'SKIP cache ZIP: need {required / 1024**3:.2f} GiB free; have {free / 1024**3:.2f} GiB. Analysis ZIP is sufficient.')
else:
    print(shutil.make_archive(str(archive.with_suffix('')), 'zip', str(cache)))
```

**Only run the optional cache ZIP in a separate cache-export session with enough
free disk for a second copy**; it is deliberately excluded from the ordinary
single-session workflow. Keep the source parent checkpoint for future inference.
Adapters are small: four snapshots plus one AdamW resume per recipe are only
about 5 MB of tensor data (serialization adds overhead). The large shared
grids/features are the main disk cost here. A previous notebook output can also
be attached directly as Kaggle Input for reuse, without copying the cache.

## Resume or evaluate existing adapter checkpoints

Resume one recipe in a new session with a completed feature cache attached as
Input, the same code/seed/batch/optimizer settings, and a new output directory:

```bash
uv run run_evidence_transport.py \
    --feature-cache /kaggle/input/YOUR_FEATURE_CACHE \
    --data-root "$DATA_ROOT" --output-dir outputs --recipe intervention \
    --num-epochs 20 --batch-size 8 --num-workers 2 \
    --lr 0.0003 --weight-decay 0.01 --seed 42 --amp \
    --resume-checkpoint /kaggle/input/YOUR_RUN/latest_resume.pt \
    --previous-checkpoint-dir /kaggle/input/YOUR_RUN
```

`num-epochs` is the **total** target. Old snapshots are read directly from Input,
new ones from Output; all four participate in val selection. Upload the earlier
snapshots alongside the resume file. Source/cache/training-control mismatch is
rejected instead of silently resetting the optimizer. Recovery starts at the
next epoch. Weights-only snapshots intentionally cannot exact-resume AdamW.

To evaluate existing snapshots without training, use a new output directory:

```bash
uv run run_evidence_transport.py \
    --feature-cache "$FEATURE_CACHE" --data-root "$DATA_ROOT" \
    --output-dir outputs_eval --recipe transport --batch-size 8 \
    --evaluate-checkpoints /kaggle/input/YOUR_RUN/checkpoint_epoch5.pt \
        /kaggle/input/YOUR_RUN/checkpoint_epoch10.pt \
        /kaggle/input/YOUR_RUN/checkpoint_epoch15.pt \
        /kaggle/input/YOUR_RUN/checkpoint_epoch20.pt
```

## GT-free online prediction

The cache is a training acceleration, not an inference requirement. Keep the
same parent checkpoint, a selected adapter snapshot and the OCR provider/cache
protocol. Put the image's OCR record at `OCR_CACHE/single/IMAGE_STEM.json`.

```bash
uv run predict_evidence_transport.py \
    --checkpoint outputs/transport/checkpoint_epoch10.pt \
    --parent-checkpoint "$PYRAMID_CKPT" \
    --image /path/image.jpg --question "What does the label say?" \
    --cache-root "$OCR_CACHE" --output prediction.png
```

Use the epoch actually chosen in `selection.json`. Replace `--cache-root ...`
with `--no-ocr` for the exact parent fallback. The online path shares the same
feature conversion/storage rounding as extraction and verifies the parent hash
and OCR protocol. It consumes image, Q and OCR only; no GT masks or answers.

For comparison, use the native cached parent metrics (`base_mean_iou` and
`base_metrics`) and the two adapters' validation-selected results. Only the two
adapter recipes form a matched objective ablation; comparing against older
end-to-end runs also changes freezing, training budget and objective.

## Validation boundary

CPU fixtures test finite FP16 forward/backward, empty OCR fallback, GT/answer
independence, native versus batched composition, stable leave-one-out against
explicit removal, checkpoint recovery, original-resolution metrics and the full
cache/train/eval pipeline. These checks do not establish Kaggle speed, real-data
accuracy or cross-backbone/dataset transfer. Selection, scope and local-mask
errors can still coexist; the diagnostics distinguish them before another
full-backbone experiment is justified.

The existing `train_joint.py`, `eval_joint.py`, old checkpoints and their behavior
are unchanged. Their guide remains on [exp/joint-native-crop](https://github.com/DPTPhatUS/VizWiz-VQA-Grounding/tree/exp/joint-native-crop).
