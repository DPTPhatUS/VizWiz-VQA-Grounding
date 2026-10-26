# OCR evidence routing

Branch: `exp/ocr-evidence-routing`.

Two feasibility experiments replace the previous support-only residual adapter.
The same frozen B+5 pyramid and frozen mobile OCR are retained. Each experiment
trains its own small head from scratch; neither uses the old OCR adapter weights.

## What changes

The previous adapter could only change pixels inside OCR boxes. It could find a
correct logo but could not remove the parent's much larger false foreground.
These experiments predict an independent evidence mask and route between it and
the full-image mask.

| Session | --architecture | Trainable parameters | Question tested |
|---|---|---:|---|
| A | proposal | 288,483 | Does independent evidence-mask routing solve the residual restriction? |
| B | pyramid | 343,085 | Does OCR fusion during multi-scale evidence decoding help further? |

Counts exclude the 433M-parameter frozen parent and the frozen OCR pipeline.
Both runs use recognized text, geometry and native appearance. The only planned
difference BETWEEN A AND B is the evidence decoder. Compared to the old residual,
A changes a head/loss/routing bundle; it is not a single-variable causal ablation. Default width, seed, initialization of shared
head modules, optimizer, data and targets are matched. These are hypotheses;
accuracy, transfer and Kaggle runtime have not been established.

## Architecture

```text
upright image + Q
    |
    +-- frozen B+5, ONE forward -------------------> base probability P_base
    |        +-- Q tokens [B,L,768]
    |        +-- Q-conditioned semantic features [B,16,384,384]
    |        +-- detail features [B,16,768,768]
    |        +-- detail levels 96 / 192 / 384 (B only)
    |
    +-- cached frozen mobile OCR
          polygons, strings, confidence; max24 candidates, independent of GT/Q
                       |
          native RGB crops [B,N,3,48,128] + soft polygon priors
          same frozen CLIP text encoder for short OCR strings
                       |
          candidate bank [B,N,128]
          attends to ALL Q tokens and to the candidate set
                       |
               independent inclusion scores (sigmoid, not softmax)
                       |
          independent local masks [B,N,1,48,128]
          initialized from OCR polygons, NOT the parent's mask
                       |
              multi-region soft union
                       |
           +-----------+--------------------------------+
           | A: proposal                                | B: pyramid
           | paste independent local masks              | token field + proposal prior
           | over the image                             | + reused image detail
           |                                            | fuse at96 ->192 ->384
           |                                            | full-resolution768 refinement
           +--------------------+-----------------------+
                                |
                         evidence probability P_evidence
                                |
      Q + candidate set + pooled visual detail -> separate router g
                                |
            P_final = (1-g)*P_base + g*P_evidence
```

A can erase inherited foreground ANYWHERE by routing to an evidence mask that is
zero outside its selected regions. Unlike A, B can also extend its evidence mask
outside candidate support through its dense decoder. Candidate inclusion is
multi-label: multiple true words do not compete for a single softmax mass.
The union is computed stably in optical-density space.

The parent mask is not used as input to the independent local mask head.
Candidate appearance, detail, Q and reused Q-conditioned parent semantic features
supply context. A small learned projection aligns semantic/detail channels, so
routing sees visual meaning rather than only crop texture and recognized text. OCR features in B are
projected to only8 channels before spatial fusion; no new vision/language
backbone is added.

The router starts with bias -3 (about4.7% evidence weight), a conservative mixture
rather than an exact identity for nonempty OCR. With missing/dropped OCR, final
logits exactly reproduce the parent even after training. A wrong nonempty OCR
route can still damage large targets; fallback is not an accuracy guarantee.

The adapter constructors accept text/detail channel dimensions. The current
wrapper exposes CLIP-pyramid features; another backbone needs a matching feature
wrapper and training/evaluation. OCR cache input remains provider neutral.
Do not call a portable interface evidence of transfer accuracy.

## Targets and loss

No new annotations or GT answers are required.

- Local masks: native GT crops, with area downsampling. All valid candidates,
  including negatives, receive local supervision independently of the parent.
- Inclusion: soft GT foreground purity within each OCR polygon crop. Scores are
  independent rather than normalized across candidates.
- Router: a documented training heuristic. Select candidates with purity>=0.5;
  if their padded support boxes cover>=80% of native GT foreground, the evidence
  route target is1; otherwise0. Empty GT or dropped/missing OCR has target0.
  This labels recoverable evidence geometry, not a manually annotated semantic
  category and not whether the candidate beats the overfit parent on train.
- Evidence branch: full-image GT segmentation loss on route-positive images.
  This trains mask prediction even when the parent already fits that train image.
- Final mixture: ordinary full-image segmentation loss on every image.

All segmentation terms use per-image BCE+0.5Dice. Defaults:

```text
L = L_final +0.1 L_inclusion +0.1 L_local +0.2 L_route +0.5 L_evidence
```

Terms average over original images, with invalid candidates masked. Evidence loss
is zero on route-negative images; it is not separately re-normalized per rank's
positive count, preserving DDP/accumulation semantics. Target constants are stored
as target_protocol in run_config/checkpoints. They are fixed for the first
feasibility runs, not tuned against test.

Training-only target fields stay outside the model's inference path. Parent,
OCR and CLIP text weights are frozen. Only the new adapter is optimized.

## Kaggle: two GPU sessions, 20 epochs each

Reuse the OCR dataset you already created:
`/kaggle/input/datasets/khoahoang0512/vizwiz-ocr-cache/ocr_cache`.

No new CPU caching session, OCR training or OCR download is needed for these
runs. The existing manifest and train/val/test JSON files are compatible.
For new images later, the same PP-OCRv5 mobile detector/recognizer is used frozen;
its compute/weights must be counted in end-to-end deployment cost.

Create TWO fresh sessions, each with Internet and 2xT4. Attach:
1. VizWiz.
2. The exact B+5 checkpoint dataset.
3. The existing OCR cache dataset.

Use the same cells in each session. Set ARCHITECTURE to proposal for A, pyramid
for B. B starts independently from B+5 and a fresh head; never resume A into B.

### Cell 1: setup and paths

```python
!pip install -q uv
!git clone --single-branch --branch exp/ocr-evidence-routing https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen

import os
from pathlib import Path

os.environ["ARCHITECTURE"] = "proposal"  # Session A: proposal; Session B: pyramid
os.environ["DATA_ROOT"] = "/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz"
os.environ["PARENT_CKPT"] = "/kaggle/input/datasets/khoahoang0512/vizwiz-size-weighted-pyramid-cp/mild_focus/checkpoint_epoch5.pt"
os.environ["OCR_CACHE"] = "/kaggle/input/datasets/khoahoang0512/vizwiz-ocr-cache/ocr_cache"
os.environ["RUN_ROOT"] = "outputs"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

assert os.environ["ARCHITECTURE"] in ("proposal", "pyramid")
assert Path(os.environ["PARENT_CKPT"]).is_file()
assert (Path(os.environ["OCR_CACHE"]) / "manifest.json").is_file()
for split in ("train", "val", "test"):
    assert (Path(os.environ["DATA_ROOT"]) / f"{split}_grounding.json").is_file()
    assert (Path(os.environ["OCR_CACHE"]) / split).is_dir()
```

Change paths only if the Kaggle mounts differ. Do not use old OCR adapter weights,
the original pyramid20, natural/A, crop or interpolation checkpoints as the parent.

### Cell 2: train this architecture for20 epochs

```bash
%%bash
set -euo pipefail
uv run torchrun --standalone --nproc_per_node=2 train_ocr.py \
    --parent-checkpoint "$PARENT_CKPT" \
    --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
    --output-dir "$RUN_ROOT" --architecture "$ARCHITECTURE" \
    --evidence-mode full --num-epochs 20 \
    --batch-size 4 --gradient-accumulation-steps 1 --num-workers 2 \
    --lr 0.0001 --optimizer adamw --weight-decay 0.01 \
    --max-candidates 24 --ocr-dropout 0.2 --text-dropout 0.2 \
    --selection-weight 0.1 --local-weight 0.1 --route-weight 0.2 --evidence-weight 0.5 \
    --save-every 5 --seed 42 --amp
```

All architecture/loss/optimization options can be overridden with flat JSON:
**config > explicit CLI > default**. Paths stay on CLI. Normal runs need no config.
Defaults: constant LR, frozen entire parent, no per-epoch validation, AMP on CUDA.
These are20 NEW head epochs, not40 epochs in a single session.

Keep all four checkpoints at5/10/15/20. They contain only adapter, optimizer,
scaler and provenance; the large parent is stored separately and checked by
SHA256. Output stays directly in outputs/. Checkpoints use atomic writes and
rank-zero I/O errors propagate through a CPU group.

### Cell 3: evaluate and diagnose validation at10 and20

```bash
%%bash
set -euo pipefail
for EPOCH in 10 20; do
    uv run eval_ocr.py \
        --parent-checkpoint "$PARENT_CKPT" \
        --checkpoint "$RUN_ROOT/checkpoint_epoch$EPOCH.pt" \
        --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
        --dataset val --view full --batch-size 4 --num-workers 2 \
        --output-dir "$RUN_ROOT/eval_val_ep$EPOCH"
    uv run diagnose.py --data-root "$DATA_ROOT" \
        --predictions "$RUN_ROOT/eval_val_ep$EPOCH" \
        --output-dir "$RUN_ROOT/diagnostic_val_ep$EPOCH"
done
```

Metrics include native mean/global IoU, size groups, paired parent scores,
standalone evidence IoU and a GT-dependent branch-choice oracle. Per-image records
include route/inclusion probabilities and boxes, so evidence failure and routing
failure can be distinguished without rerunning training.

### Cell 4: select on validation, test and inspect routing

```python
import json
import subprocess
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
architecture = os.environ["ARCHITECTURE"]
best = max((10, 20), key=lambda epoch: (
    json.loads((root / f"eval_val_ep{epoch}/metrics.json").read_text())["summary"]["mean_iou"],
    -epoch,
))
for split, view in [
    ("test", "full"),
    ("val", "base"),
    ("val", "missing"),
    ("val", "geometry"),
    ("val", "shuffled_text"),
    ("val", "evidence"),
]:
    subprocess.run([
        "uv", "run", "eval_ocr.py",
        "--parent-checkpoint", os.environ["PARENT_CKPT"],
        "--checkpoint", str(root / f"checkpoint_epoch{best}.pt"),
        "--data-root", os.environ["DATA_ROOT"],
        "--cache-root", os.environ["OCR_CACHE"],
        "--dataset", split, "--view", view,
        "--batch-size", "4", "--num-workers", "2",
        "--output-dir", str(root / f"eval_{split}_{view}_ep{best}"),
    ], check=True)

subprocess.run([
    "uv", "run", "diagnose_ocr_routing.py",
    "--predictions", str(root / f"eval_val_ep{best}"),
    "--data-root", os.environ["DATA_ROOT"],
    "--cache-root", os.environ["OCR_CACHE"],
    "--output-dir", str(root / "routing_diagnostic"),
], check=True)

(root / "selection.json").write_text(json.dumps({
    "architecture": architecture, "selected_epoch": best,
    "selection_metric": "validation_mean_iou", "tie_break": "earlier_epoch",
}, indent=2))
print(f"Selected {architecture} epoch {best}")
```

Only validation selects the checkpoint. Test is not used for choosing architecture,
epoch, route threshold or coefficients. Base/missing views should match exactly.
Geometry view removes recognized content but retains OCR polygons and crop RGB.
Shuffled text uses a deterministic cross-image permutation. Evidence view forces
the independent branch; it is an ablation, not normal inference.

The route oracle uses GT to choose between the two branch masks. It is NOT a
deployable score or a bound on every possible soft mixture. The route-coverage
target is a training heuristic, not proof of semantic object/evidence categories.

### Cell 5: ZIP this session's results

```python
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(os.environ["RUN_ROOT"])
architecture = os.environ["ARCHITECTURE"]
archive_path = root / f"ocr_routing_{architecture}_ep20_seed42.zip"
with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
    for item in sorted(root.rglob("*")):
        if item.is_file() and item.suffix not in {".pt", ".tmp", ".zip"}:
            archive.write(item, item.relative_to(root))
print(archive_path)
```

Download both result ZIPs for analysis:
- A: ocr_routing_proposal_ep20_seed42.zip
- B: ocr_routing_pyramid_ep20_seed42.zip

Checkpoints remain outside the ZIP; download the four small files separately.
Do not copy the input OCR cache or parent into outputs.

## Checkpoint compatibility and recovery

Old residual OCR checkpoints remain loadable with eval_ocr.py/predict_ocr.py.
Raw B+5 pyramid train/eval behavior and weights remain compatible. New routing
checkpoints have a separate version and exact architecture metadata; no partial
or silent loading into the old adapter is allowed.

To resume, use the same architecture/settings/cache/parent and a NEW output
directory, adding --resume-checkpoint /path/checkpoint_epoch10.pt. Keep
--num-epochs 20 as the TOTAL head epoch count. Earlier checkpoints are not copied
to the new directory; copy the small epoch10 adapter AFTER training if reusing
the evaluation cells unchanged. Never resume between proposal and pyramid.

predict_ocr.py handles arbitrary images with no GT files. Use the same existing
cache_ocr.py --image command in a separate Paddle environment to cache a new image,
then predict with the corresponding parent+adapter. --no-ocr gives exact visual
fallback. Original OCR cache generation and provider schema are unchanged.

## How to judge feasibility

Look for small AND medium gains, inspect large damage, and compare each run to
its paired B+5 rather than comparing unrelated parent/epoch runs.

- Evidence improves but the final output does not: router/calibration is the next bottleneck.
- Correct polygons exist but evidence masks remain poor: inspect independent mask learning/inclusion.
- B helps over A: multi-scale feature fusion contributes beyond replacement.
- Missing OCR changes the parent: implementation error.
- Only test improves by a tiny amount: no strong success claim; inspect validation and seed variance.

Monitor route_target_rate and evidence/local losses, not train_loss alone.
No second large reader, OCR fine-tuning, answer vocabulary or new annotation is
introduced. Transfer requires held-out data/backbone evaluation; perturbation
views and generic interfaces do not establish transfer.

The frozen provider is PP-OCRv5 mobile detection+recognition (same cached versions).
Runtime reporting must include OCR as well as the full frozen parent and new
head. Cached grounding timing alone excludes OCR and is labeled in metrics.
Components use established cross-attention, soft union and mixture mechanisms;
the contribution sought is resolving evidence-granularity errors in this task.

References: [PaddleOCR API](https://www.paddleocr.ai/v3.3.1/en/version3.x/pipeline_usage/OCR.html),
[PP-OCR](https://arxiv.org/abs/2009.09941).
