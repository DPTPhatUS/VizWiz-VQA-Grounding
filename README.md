# OCR evidence selection

Branch: `exp/ocr-evidence-selection`.

This experiment learns to choose a grounding mask from the frozen B+5 prediction
and separate OCR polygon hypotheses. It does not merge every OCR word into one
mask before judging its quality. No extra vision/language backbone is introduced.

| Run | `--architecture` | `--answer-weight` | Question |
|---|---|---:|---|
| A | `hypotheses` | 0.0 | Can native OCR hypotheses and quality ranking improve grounding? |
| B | `hypotheses` | 0.2 | Does supported answer-to-OCR supervision help the same selector? |

A and B have the same **259,490 trainable parameters**, candidate bank, initialization,
seed, parent, optimizer and schedule. Only the auxiliary answer loss weight differs.
The count excludes the 433,217,037-parameter frozen parent and mobile OCR pipeline.
Both heads start fresh from the same B+5 parent; B never resumes A. These are
feasibility experiments. Accuracy, GPU throughput and transfer remain unmeasured.

## Method

```text
image + question
    |
    +-- frozen B+5 pyramid, one forward
    |       question tokens: [B,L,768]
    |       Q-conditioned semantics: [B,16,384,384]
    |       detail features: [B,16,768,768]
    |       base mask logits: [B,1,768,768]
    |
    +-- existing frozen mobile OCR cache
            max N=24 polygons + strings + confidence
            native crops: [B,N,3,48,128]
                    |
            compact appearance / text / geometry encoder
            attends to Q tokens and other candidates
            candidate features: [B,N,128]
                    |
            fixed mask hypothesis bank (GT-free)
              index 0: base prediction
              each individual OCR polygon
              bounded nearby groups of up to three words
              union of all words, as one optional hypothesis
                    |
            <=2N+2 slots, with padding/duplicates masked
            masks pooled at96x96 ONLY for quality features
            shared scorer: actual mask + features + Q + relation to base
                    |
            estimated mask quality in [0,1]
                    |
            replace base only if predicted gain >0.02
                    |
            selected raw polygon union at NATIVE image resolution,
            or unchanged base logits resized by the original evaluator
```

Groups use native pixel distances, comparable word heights and line overlap.
Membership depends only on OCR geometry, never on Q, answer or GT. A selected
hypothesis can contain multiple words. There is no forced top1 OCR token and no
obligatory union of every word. Missing OCR reproduces the parent exactly.
The 0.02 margin is a fixed conservative pilot setting, not a calibrated guarantee
against harmful switches. It is saved in the checkpoint.

Native polygons are retained as raw options rather than passed through a learned
refiner. Selected OCR masks are rasterized directly at the original size; they
are not resized down to768 and back up. The base uses its original bilinear-logit
resize and zero threshold. Both train quality targets and evaluation use exactly
these respective output rules. This rendering change is part of the experiment;
it is not an isolated loss-only comparison with the previous routing branch.

The model returns `base_logits`, quality scores and `selected_index`, not a dense
mixed output. Use `eval_ocr.py` / `predict_ocr.py`, which implement the same native
selection contract. Plain `eval.py` is for parent checkpoints and cannot load
these adapter checkpoints. `diagnose.py` reads the exported native PNGs as usual.

## Training objective

For every fixed OCR hypothesis, compute native foreground IoU with training GT.
The live base prediction is resized to native GT resolution for its IoU target.
To avoid treating the parent as an always-good training category, half of the base
masks receive train-only shifts or dilation/erosion (2--8% image-size kernels,
computed with separable pooling). Their native IoUs are recomputed after the
change. Images/features stay unchanged; inference never applies this augmentation.
A/B share this fixed protocol, recorded in checkpoint metadata.
Targets are detached; no GT or answer participates in forward inference.

```text
L = L_quality + L_ranking + answer_weight * L_answer
```

- `L_quality`: BCE with continuous IoU targets. Overlapping and disjoint hypotheses
  are balanced within each image, with invalid slots ignored.
- `L_ranking`: within-image pairwise softplus, weighted by the true quality gap;
  compare hypotheses when their IoUs differ by more than0.05. Normalize within
  each image. OCR-vs-OCR comparisons remain active even when the base is best.
- `L_answer` (B only): a pointer-style cross-entropy over candidate scores. A
  positive candidate requires normalized full-answer phrase containment in the
  recognized string AND single-polygon native IoU>=0.5. If there are several
  supported matches, use a uniform soft target over them. With no supported match,
  this auxiliary loss is zero; a missing match is not a negative answer label.

The auxiliary head shares candidate features with quality scoring. The inference
path never receives answer text or the matching labels. A/B have identical modules;
A sets the auxiliary weight to zero. There is no separate teacher or answer model.
Quality targets use masks on all training examples, not only answer-matched ones.

Each loss is averaged over original images, preserving DDP and gradient
accumulation semantics even when one rank has no OCR or no answer supervision.
The entire parent and OCR stay frozen; only the selector is optimized. There is
no final segmentation loss through the hard choice, because hypothesis masks are
fixed in this experiment. A learned mask refiner is deliberately deferred.

The principles build on explicit OCR candidate prediction in
[M4C](https://arxiv.org/abs/1911.06258) and mask quality prediction in
[Mask Scoring R-CNN](https://arxiv.org/abs/1903.00241). Their results on other tasks
are not evidence of gains here. Backbone compatibility is an interface goal;
only the current CLIP pyramid wrapper is implemented and tested locally.

## Kaggle: two independent GPU sessions, 20 epochs each

Reuse the existing OCR cache. No new CPU caching session or OCR training is needed.
In each fresh notebook, enable Internet and 2xT4; attach VizWiz, the exact B+5
checkpoint dataset and the OCR cache dataset. Use the same cells below, changing
only `RUN_VARIANT` between A and B. Output stays directly under `outputs/`.

### Cell 1: setup

```python
!pip install -q uv
!git clone --single-branch --branch exp/ocr-evidence-selection https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen

import os
from pathlib import Path

os.environ["RUN_VARIANT"] = "A"  # Fresh second session: "B"
os.environ["ANSWER_WEIGHT"] = "0.0" if os.environ["RUN_VARIANT"] == "A" else "0.2"
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

The parent is **mild_focus checkpoint_epoch5.pt (B+5)**, not an old OCR adapter,
raw pyramid epoch20, natural/A, crop or interpolated model. Parent SHA256 is checked
when loading a selector checkpoint. Retain the exact parent as a separate input.

### Cell 2: train

```bash
%%bash
set -euo pipefail
uv run torchrun --standalone --nproc_per_node=2 train_ocr.py \
    --parent-checkpoint "$PARENT_CKPT" \
    --data-root "$DATA_ROOT" \
    --cache-root "$OCR_CACHE" \
    --output-dir "$RUN_ROOT" \
    --architecture hypotheses \
    --evidence-mode full \
    --answer-weight "$ANSWER_WEIGHT" \
    --min-gain 0.02 \
    --num-epochs 20 \
    --batch-size 4 \
    --gradient-accumulation-steps 1 \
    --num-workers 2 \
    --lr 0.0001 \
    --optimizer adamw \
    --weight-decay 0.01 \
    --max-candidates 24 \
    --ocr-dropout 0.2 \
    --text-dropout 0.2 \
    --save-every 5 \
    --seed 42 \
    --amp
```

Batch size is GLOBAL across both GPUs:4 means2 per GPU. `--batch-size 2
--gradient-accumulation-steps 2` preserves effective global batch4 if needed.
LR is constant. There is no validation inside training and no need for a
`--validate-every` flag in this script. These are20 new selector epochs on the
fixed B+5 parent, not end-to-end training of the parent.

Save all four selector checkpoints at5/10/15/20. Each contains selector, optimizer,
scaler and provenance only; the large parent is excluded. No checkpoint is deleted.
Regular checkpoints support exact continuation with the same training protocol.

### Cell 3: validation and diagnostics for every saved checkpoint

```bash
%%bash
set -euo pipefail
for EPOCH in 5 10 15 20; do
    uv run eval_ocr.py \
        --parent-checkpoint "$PARENT_CKPT" \
        --checkpoint "$RUN_ROOT/checkpoint_epoch$EPOCH.pt" \
        --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
        --dataset val --view full --batch-size 4 --num-workers 2 \
        --output-dir "$RUN_ROOT/eval_val_ep$EPOCH"
    uv run diagnose_evidence_selection.py \
        --predictions "$RUN_ROOT/eval_val_ep$EPOCH" \
        --output-dir "$RUN_ROOT/selection_diagnostic_ep$EPOCH"
done
```

Metrics contain native per-image mean/global IoU, size groups, paired base scores,
selected membership and predicted/actual quality per hypothesis. GT-dependent
hypothesis oracle and regret are diagnostics only. Selected masks are exported
as native PNGs. No GT is used to choose a hypothesis during inference.

### Cell 4: choose best validation checkpoint, test and robustness views

```python
import json
import subprocess

root = Path(os.environ["RUN_ROOT"])
best = max((5, 10, 15, 20), key=lambda epoch: (
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
    "uv", "run", "diagnose.py", "--data-root", os.environ["DATA_ROOT"],
    "--predictions", str(root / f"eval_val_ep{best}"),
    "--output-dir", str(root / "mask_diagnostic"),
], check=True)
(root / "selection.json").write_text(json.dumps({
    "variant": os.environ["RUN_VARIANT"], "selected_epoch": best,
    "selection_metric": "validation_mean_iou", "tie_break": "earlier_epoch",
}, indent=2))
print(f"Selected epoch {best} for run {os.environ['RUN_VARIANT']}")
```

Select ONLY by validation mIoU, with the earlier epoch breaking exact ties.
Do not select the architecture or margin using test. A/B compare the same protocol;
inspect small/medium gains together with large-region damage and overall mIoU.

`base` and `missing` must match. `geometry` removes recognized-string embeddings
but keeps polygons/crop RGB. `shuffled_text` deterministically permutes strings
across images. `evidence` forces the highest-scored valid OCR hypothesis, ignoring
the base/margin; images with no OCR still fall back to the parent. It is an
ablation, not deployment. It no longer means one dense noisy-OR evidence branch.

### Cell 5: ZIP results without checkpoints

```python
import zipfile

archive = Path(f"ocr_selection_{os.environ['RUN_VARIANT']}_ep20_seed42.zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() not in (".pt", ".tmp", ".zip"):
            bundle.write(path, path.relative_to(root))
print(archive.resolve())
```

Download this ZIP and the four small `.pt` files separately. The ZIP includes
history/config, checkpoint selection, metrics, diagnostics and prediction PNGs.
It excludes all `.pt`, `.tmp` and nested ZIP files.

## Resume and configuration

To resume the SAME run in a fresh session, use a new empty output directory:

```bash
uv run torchrun --standalone --nproc_per_node=2 train_ocr.py \
    --parent-checkpoint "$PARENT_CKPT" --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
    --output-dir outputs --resume-checkpoint /kaggle/input/YOUR_SELECTOR_DATASET/checkpoint_epoch10.pt \
    --architecture hypotheses --evidence-mode full --answer-weight "$ANSWER_WEIGHT" --min-gain 0.02 \
    --num-epochs 20 --batch-size 4 --gradient-accumulation-steps 1 --num-workers 2 \
    --lr 0.0001 --optimizer adamw --weight-decay 0.01 --max-candidates 24 \
    --ocr-dropout 0.2 --text-dropout 0.2 --save-every 5 --seed 42 --amp
```

`--num-epochs 20` means total target epoch20, not20 extra epochs. Keep variant,
parent, OCR cache, native training annotations/masks, margin, optimizer,
batch/accumulation and seed unchanged. Native supervision has its own SHA256 digest. Resume
A as A or B as B; adapters from routing/residual architectures cannot initialize
this selector. Copy earlier checkpoints/evaluation folders into outputs only AFTER
training if reusing the four-checkpoint evaluation cells; otherwise adjust that
cell to the checkpoints present. Never copy old files into a training output
folder before resume: nonempty output directories are rejected.

`--config path.json` accepts flat JSON overrides for architecture/training options:
**config > explicit CLI > default**. Paths stay on CLI. Example for run B:

```json
{"architecture": "hypotheses", "answer_weight": 0.2, "min_gain": 0.02}
```

The explicit train command above needs no JSON file. Relevant options:

| Option | Default | Meaning |
|---|---:|---|
| `--architecture` | proposal | Explicitly set `hypotheses` for this experiment |
| `--answer-weight` | 0.0 | Auxiliary train-only candidate retrieval loss |
| `--min-gain` | 0.02 | Fixed predicted-quality improvement required to switch |
| `--max-candidates` | 24 | Candidate cap before deterministic grouping |
| `--batch-size` | 4 | Global microbatch across ranks |
| `--gradient-accumulation-steps` | 1 | Effective batch multiplier |
| `--lr`, `--optimizer`, `--weight-decay` | 1e-4, adamw, 0.01 | Selector optimization |
| `--ocr-dropout`, `--text-dropout` | 0.2, 0.2 | Training-only OCR/recognized-text dropout |
| `--num-epochs`, `--save-every`, `--seed` | 20, 5, 42 | Total epochs, checkpoint interval, seed |
| `--amp` / `--no-amp` | enabled on CUDA | Mixed precision |
| `--num-workers` | 2 | Loader workers per rank |

Legacy residual/proposal/pyramid architectures and their versioned checkpoints
remain loadable. Their `selection/local/route/evidence-weight` options retain the
old behavior; those losses are not used by `hypotheses`. This experiment has a
separate checkpoint version and strict architecture/protocol validation.

## Validation and limits

Local checks cover native target/render equivalence, GT-free inference, answer
labels, finite gradients with empty OCR ranks, adapter checkpoint round trips,
two-process Gloo training/resume, all eval views and legacy compatibility.
Production channel/spatial shapes have CPU BF16 forward/backward checks. These
checks do not establish CUDA/T4 speed, memory headroom or accuracy gains.

The frozen hypothesis bank cannot find missing OCR regions or arbitrary non-text
objects. Local word grouping is bounded and may miss long/multiline answers.
Raw masks can include OCR detector errors. The fixed switch margin can still make
wrong selections. Evaluate the actual final masks rather than quoting oracle
headroom as model performance. A/B isolate answer supervision only; comparisons
with old routing change candidate representation, losses and output rendering.
