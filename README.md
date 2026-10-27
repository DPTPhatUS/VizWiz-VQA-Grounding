# OCR sufficiency pointer

Branch: `exp/ocr-sufficiency-pointer`.

The previous scope scorer still predicted about 80% quality for small-region parent
masks whose true mean IoU was about 20%. Removing explicit reference cues did not
solve the deployment failure. This experiment changes the learning problem:
**select an adequate OCR mask, or defer**. It never learns the parent's quality.

| Run | `--pointer-recipe` | Hypothesis |
|---|---|---|
| A | categorical | Direct phrase selection + a supervised defer class avoids parent-quality overestimation |
| B | structured | Native word-purity supervision and a token-set score help reject irrelevant words in an otherwise plausible group |

Both start fresh, use the same initialization, image split, parent, cache and
optimizer. B adds a training/score bundle, not an isolated loss ablation. A also
changes representation and training data relative to older selectors; comparisons
with those runs do not isolate one causal change. Accuracy gains remain unmeasured.

## Architecture

```text
image + question
    |
    +-- frozen B+5 parent, one forward
    |      Q tokens [B,L,768]
    |      semantic [B,16,384,384], detail [B,16,768,768]
    |      original base logits [B,1,768,768] ----------- fallback only
    |
    +-- existing frozen mobile OCR cache
           native polygons, crops, recognized strings, confidence
                    |
           crop CNN + local image features + geometry + frozen text encoding
                    |
           Q-token attention -> OCR-set context -> word features [B,N,128]
                    |
           unchanged native mask bank: singles / local groups / all words
                    |
           own-mask visual features + Q-token attention
                    |
           masked attention to each phrase's MEMBER WORDS
                    |
           shared phrase pointer + learned defer query
                    |
           B only: word-purity logits -> token-set score -> add to OCR scores
                    |
           categorical softmax over [defer, OCR mask 1, ..., OCR mask K]
                    |
           OCR total mass > defer? best OCR mask : original base
```

The head works at 96x96, while OCR crops retain native-region appearance at
48x128 before the small crop encoder. Native output rasterization is unchanged.
There is no trainable detector, duplicated text encoder, answer generator or extra
large backbone. The parent and its vision/text encoders stay frozen and in eval
mode. Adapter checkpoints exclude the parent.

The learned defer query sees Q, global image features and OCR context. It never
sees base-mask geometry, confidence, overlap or a parent IoU target. Changing only
base logits cannot change ANY pointer score. Parent features are reused, so this
is not independence from the parent's feature distribution or a transfer result.

Each phrase attends to its member-word features rather than averaging every word.
All Q tokens remain available in attention. Learned attention can still ignore
useful content; the intervention views below test this rather than assuming it.

## Targets and objectives

Let `u_i` be exact native IoU between OCR hypothesis i and the training GT mask.
The base is NOT one of these training alternatives.

- If at least one valid OCR mask has `u_i >= 0.5`, supervise masks within 0.05 of
  the best IoU, also requiring `u_i >= 0.5`. Their target masses are proportional
  to their IoUs; near ties are not arbitrarily assigned to one bank position.
- Otherwise, the target is defer. Missing/dropped OCR always targets defer.
- A shared OCR-only retrieval loss also trains the best available phrase when
  its IoU is at least 0.1, even if the correct deployment decision is defer.
- Targets are recomputed against the words actually available after OCR dropout.
  No parent prediction, GT answer, validation mask or test label defines training.

```text
A: L = categorical CE + 0.25 * OCR retrieval CE
B: L = categorical CE + 0.25 * OCR retrieval CE + 0.25 * word-purity BCE
```

The word target in B is native `intersection(OCR polygon, GT) / polygon area`.
This is purity, not whole-mask IoU: a relevant word in a multiword answer can have
high purity while low individual IoU. Positive/negative word groups are balanced
within each image, then losses average over original images for correct DDP and
accumulation behavior.

For B, the score correction of OCR hypothesis i is
`sum_j member[i,j] * word_logit[j] / sqrt(number_of_valid_words)`.
Within OCR hypotheses this is a scaled Bernoulli set log likelihood up to a common
constant. Including an irrelevant word lowers the correction; including several
relevant words can help a true phrase. The defer score does not receive this term.
The token head starts at zero, giving A/B identical initial scores and parameters.

No margin or post-hoc threshold is tuned on validation/test. Normal inference
uses OCR when its TOTAL probability mass exceeds defer probability (fixed 0.5),
then selects the highest-scored OCR mask. This avoids a group of near-tied valid
phrases splitting their probability and incorrectly losing to defer.
Probabilities are NOT estimated IoUs.
`actual_quality[0]` in exported metrics is the fallback's measured IoU for auditing;
it never becomes the defer label. A sufficient OCR mask can still be worse than
a good parent mask. This method therefore does not guarantee preservation of
large-region scores; harmful switches are a required outcome measure.

### Why a scorer holdout, and what it does not establish

A fixed hash of image name + seed splits TRAIN into 85% fit / 15% scorer holdout.
The same images are used by A/B, recorded explicitly in `run_config.json` and
checked on resume/evaluation. The scorer never fits its holdout. No additional
annotation or CPU caching session is needed.

The parent has already seen these images. This is **not an unseen-parent split or
out-of-fold experiment**. The new targets do not depend on parent correctness, so
we can check the scorer's own generalization without fitting a parent-quality
regressor. Shared parent features can still bias it. An actual out-of-fold parent
comparison would require retraining parents; this guide does not pretend to do it.

Choose among epochs 5/10/15/20 by the LOWEST scorer-holdout categorical CE, with
earlier epoch breaking exact ties. Validation and test then report final native
mIoU/global IoU and small/medium/large behavior. Do not choose a different checkpoint
or threshold after inspecting test. This pilot's selection rule differs from the
older validation-mIoU rule and is recorded in `selection.json`.

## Kaggle: two fresh GPU sessions, 20 epochs each

Enable Internet and 2xT4. Attach the same VizWiz dataset, B+5 parent and OCR cache.
Run these cells once with A, then in a fresh session with B. Only the small head
trains for 20 epochs; neither run resumes the previous selector or the other run.

### Cell 1: setup

```python
!pip install -q uv
!git clone --single-branch --branch exp/ocr-sufficiency-pointer https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen

import os
from pathlib import Path

os.environ["RUN_VARIANT"] = "A"  # Fresh second session: "B"
os.environ["POINTER_RECIPE"] = "categorical" if os.environ["RUN_VARIANT"] == "A" else "structured"
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

Use the exact **mild_focus/checkpoint_epoch5.pt (B+5)**, SHA256
`d0058ab5d7ae4c9470b98f0728c15f20df4c32e68cbfa1a9d7bdf1960ac8953e`.
The parent is pyramid20 + mild-focus5, not an OCR adapter. Its training cost still
counts when comparing complete systems. No new cache upload is required.

### Cell 2: train

```bash
%%bash
set -euo pipefail
uv run torchrun --standalone --nproc_per_node=2 train_ocr.py \
    --parent-checkpoint "$PARENT_CKPT" \
    --data-root "$DATA_ROOT" \
    --cache-root "$OCR_CACHE" \
    --output-dir "$RUN_ROOT" \
    --architecture pointer \
    --pointer-recipe "$POINTER_RECIPE" \
    --evidence-mode full \
    --answer-weight 0 \
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

Batch 4 is GLOBAL, 2 per GPU. If necessary, batch 2 + accumulation 2 keeps the same
effective global batch. No validation runs inside training. Checkpoints at
5/10/15/20 contain only the adapter + optimizer/scaler, not the large frozen parent.
Both heads have **347,970 parameters**. Head + AdamW state is about 4 MiB per
checkpoint before small protocol metadata; all four require only a few tens of
MiB. The frozen parent still has 433,217,037 parameters stored separately.

### Cell 3: evaluate the scorer holdout at each checkpoint

```bash
%%bash
set -euo pipefail
for EPOCH in 5 10 15 20; do
    uv run eval_ocr.py \
        --parent-checkpoint "$PARENT_CKPT" \
        --checkpoint "$RUN_ROOT/checkpoint_epoch$EPOCH.pt" \
        --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
        --dataset train-holdout --view full --batch-size 4 --num-workers 2 \
        --output-dir "$RUN_ROOT/eval_holdout_ep$EPOCH"
    uv run diagnose_ocr_pointer.py \
        --predictions "$RUN_ROOT/eval_holdout_ep$EPOCH" \
        --output-dir "$RUN_ROOT/pointer_holdout_ep$EPOCH"
done
```

`train-holdout` is available only for pointer checkpoints. It uses the saved split
seed and checks the held-out image list. `pointer_choice_ce` uses OCR/GT sufficiency
labels and ignores the parent IoU. Holdout grounding IoU is exported for context;
it is not a claim about unseen-parent performance.

### Cell 4: select, evaluate validation/test and run interventions

```python
import json
import subprocess

root = Path(os.environ["RUN_ROOT"])
best = min((5, 10, 15, 20), key=lambda epoch: (
    json.loads((root / f"eval_holdout_ep{epoch}/metrics.json").read_text())["summary"]["pointer_choice_ce"],
    epoch,
))
(root / "selection.json").write_text(json.dumps({
    "variant": os.environ["RUN_VARIANT"],
    "pointer_recipe": os.environ["POINTER_RECIPE"],
    "selected_epoch": best,
    "selection_metric": "train_scorer_holdout_choice_ce", "direction": "min",
    "tie_break": "earlier_epoch", "parent_unseen_holdout": False,
}, indent=2))
for split, view in [
    ("val", "full"),
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
for split in ("val", "test"):
    subprocess.run([
        "uv", "run", "diagnose_ocr_pointer.py",
        "--predictions", str(root / f"eval_{split}_full_ep{best}"),
        "--output-dir", str(root / f"pointer_{split}"),
    ], check=True)
subprocess.run([
    "uv", "run", "diagnose.py", "--data-root", os.environ["DATA_ROOT"],
    "--predictions", str(root / f"eval_val_full_ep{best}"),
    "--output-dir", str(root / "mask_diagnostic"),
], check=True)
print(f"Selected epoch {best}, recipe {os.environ['POINTER_RECIPE']}")
```

Use `eval_ocr.py`, not the parent's `eval.py`. Metrics include original-resolution
mean/global IoU, size groups, candidate probabilities, true IoUs and native PNGs.
The pointer diagnostic reports sufficiency recall, retrieval success, switches,
helpful/harmful changes and missed gains. Oracle bounds use GT and are diagnostic
only. A high sufficiency probability is not a safety guarantee or predicted IoU.

`base`/`missing` must match exactly. `geometry` removes string embeddings, keeping
RGB/polygons. `shuffled_text` changes recognized strings across images. `evidence`
forces the best OCR candidate except when OCR is absent; it is not deployment.
Use `diagnose_ocr_pointer.py` for this branch, not the quality-regression diagnostic.

### Cell 5: download results, excluding checkpoints

```python
import zipfile

archive = Path(f"ocr_pointer_{os.environ['RUN_VARIANT']}_ep20_seed42.zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() not in (".pt", ".tmp", ".zip"):
            bundle.write(path, path.relative_to(root))
print(archive.resolve())
```

Download `ocr_pointer_A_ep20_seed42.zip` / `ocr_pointer_B_ep20_seed42.zip`. Keep the
four small `.pt` files separately. All outputs stay directly under `outputs/`.
The ZIP includes configs, split membership, history, selection, metrics, masks and
diagnostics; `.pt`, `.tmp` and ZIP files are excluded.

## Flags, configuration and resume

The only new training flag is `--pointer-recipe categorical|structured`, default
`categorical`. Explicitly set `--architecture pointer` (the legacy default remains
`proposal`). Standard optimizer/LR/batch/AMP/save flags are unchanged. Optional flat
JSON uses **config > CLI > default**, with paths on CLI. Normal training needs no
JSON. `min-gain`, `selector-recipe` and legacy loss weights do not control pointer
training; `answer-weight` must remain zero.

Resume the SAME recipe in a new empty output directory with
`--resume-checkpoint /kaggle/input/YOUR_DATASET/checkpoint_epoch10.pt` and total
`--num-epochs 20`. Keep parent/cache/annotations/seed/split/recipe/optimizer/batch
settings unchanged. For Cell 3, include only epochs present, or attach earlier
small checkpoints. Old OCR scope/selection/routing/residual checkpoints continue
to load as their original architectures; they are not converted to pointer heads.

## Scientific boundaries and attribution

The explicit no-match category follows the established idea used in
[DETR](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf).
This is an adaptation to cached OCR mask sufficiency, not DETR, Hungarian matching,
or a claim that no-match classification is new. Native GT overlap provides all
new labels; GT answers are not needed, including at inference.

The backbone interfaces accept different text/feature widths, normalized boxes
and native polygons. Another backbone/dataset still needs an appropriate wrapper
and real experiments. The OCR provider remains external/frozen; cached timing
excludes running OCR. This pilot adds no OCR model to the trainable backbone.

Important limitations: categorical scores can still overfit, rare sufficient cases
can be missed, OCR polygons/grouping can omit the correct answer scope, and defer
can choose a bad parent or OCR can replace a better one. The 0.5 label threshold
is fixed before running, not optimized on validation. A/B test the hypothesis;
they do not establish gains until measured. Do not increase width/epochs solely
because the training loss decreases.
