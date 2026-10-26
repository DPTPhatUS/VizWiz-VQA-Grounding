# OCR evidence pyramid

Branch: `exp/ocr-evidence-pyramid`.

Keep the exact B +5 / mild_focus epoch5 pyramid frozen. Train a small OCR evidence
adapter to refine regions supported by native-image text proposals. This tests
whether explicit evidence localization improves small and medium targets without
sacrificing whole-object grounding. Accuracy and transfer are hypotheses, not
established results.

## Architecture

```text
upright RGB + question
    |
    +-- frozen B+5 pyramid ---------------------------------> base logits B,1,768,768
    |       +-- frozen question tokens B,L,768
    |       +-- frozen high-resolution detail B,16,768,768
    |
    +-- frozen mobile OCR, cached once
            +-- polygons + recognized strings + confidence
            +-- top 24 candidates, chosen WITHOUT Q/answer/GT
                    |
                    +-- native RGB crops B,N,3,48,128 -> shared small CNN
                    +-- aligned pyramid detail + base-logit crops
                    +-- geometry / confidence
                    +-- same frozen CLIP text encoder for OCR strings (full run)
                            |
                        candidate features B,N,128
                            |
                  candidate queries attend to ALL question tokens
                            |
                  candidate self-attention + explicit NULL candidate
                            |
                  probabilities B,N+1 (NULL means visual fallback)
                            |
                  local signed correction B,N,1,48,128
                            |
                  inverse crop mapping within candidate support
                            |
final logits = base logits + sum(candidate_probability * local_correction)
```

The production adapter has **293,938 trainable parameters**. This excludes the
frozen parent and frozen OCR; report all three components when claiming efficiency.
No second language backbone is instantiated. Only short OCR strings are mean
pooled; the selector receives the complete question token sequence.

The final correction layer starts at zero: initialization reproduces B+5 exactly.
Missing OCR gives exact fallback even after training. Nonempty but wrong OCR
does not have an accuracy guarantee: the model must learn NULL. Corrections can
add or remove foreground, but only within 1.25x candidate boxes on the 768 grid.
Native resizing can affect boundary pixels; bounding support does not guarantee
large-object preservation. OCR polygons are NOT used directly as final masks.

The adapter accepts arbitrary text/detail channel dimensions; the current wrapper
connects the existing CLIP pyramid. Another backbone needs the corresponding
feature wrapper and training; existing weights are not automatically transferable.
Cache schema is OCR-provider neutral, with upright native xy coordinates and
decoded-RGB hashes. No image IDs, answer vocabulary or GT-size gates enter the
adapter. `predict_ocr.py` supports arbitrary images with no GT files.

## Two matched runs

| Run | Candidate geometry + native appearance | Recognized OCR text |
|---|---|---|
| geometry | Yes | No |
| full | Yes | Yes, same frozen CLIP encoder |

Both start independently from the same B+5 and adapter seed; both parents stay
frozen. This is adapter training, NOT continuing B+5's optimizer.
OCR dropout=0.2 removes all candidates for a training image; text dropout=0.2
removes recognized content while retaining geometry/appearance.
Defaults: AdamW, LR 1e-4 constant, decay 0.01, global batch 4, accumulation 1,
FP16 AMP on CUDA, 20 epochs, save every 5, no per-epoch validation.
All listed training controls can be overridden by flat JSON: **config > CLI > default**.
Paths stay on CLI. See `uv run train_ocr.py --help`.

Loss: final per-image BCE + 0.5 Dice, plus 0.1 selection and 0.1 local loss.
Selection uses soft native-GT box Dice across relevant candidates; NULL is
supervised when all candidate Dice values are below 0.1. Local native-crop
targets use area downsampling; unrelated candidates do not receive local
segmentation supervision. This avoids learning a gate solely from whether an
adapter beats an overfit parent's training predictions. All GT relevance/patch
targets are dataset-side training fields and are never model inputs.
This gate threshold is a training relevance definition, not a tuned inference
threshold. Empty-GT images use NULL.

## Kaggle workflow: one CPU session, then GPU sessions A and B

Use THREE fresh sessions in this order. Do not train both experiments in one session.

| Session | Accelerator | Inputs to attach | Work | Download |
|---|---|---|---|---|
| CPU | None | VizWiz | Cache OCR for train/val/test; audit coverage | `ocr_cache_and_coverage.zip` |
| A | 2xT4 | VizWiz + B+5 + uploaded OCR cache | Train `geometry` for 20 epochs; evaluate/diagnose | `ocr_geometry_ep20_seed42.zip` and adapter checkpoints |
| B | 2xT4 | Same inputs as A | Train `full` for 20 epochs; evaluate/diagnose | `ocr_full_ep20_seed42.zip` and adapter checkpoints |

A and B start independently from the SAME B+5 parent and seed42. B does NOT resume
A. The complete parent (CLIP encoders, CNN pyramid and decoder) stays frozen.
Only the 293,938-parameter adapter is trained. Frozen parent forward computation
is still required each epoch; runtime on 2xT4 has not been measured.

## CPU session — create the reusable OCR cache

Enable Internet and set Accelerator to None. Attach only VizWiz. No B+5 checkpoint
is needed here. Do NOT run `uv sync` in this CPU session: OCR preprocessing and
coverage do not need PyTorch or CLIP.

### CPU cell 1 — Clone and paths

```python
!pip install -q uv
!git clone --single-branch --branch exp/ocr-evidence-pyramid https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding

import os
os.environ["DATA_ROOT"] = "/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz"
os.environ["RUN_ROOT"] = "outputs"
os.environ["OCR_CACHE"] = "outputs/ocr_cache"
os.environ["OMP_NUM_THREADS"] = "2"
```

### CPU cell 2 — Install OCR and cache all three splits

The separate `/tmp/ocr-env` is only a local Python environment, not an artifact
to upload. The mobile OCR weights download automatically. These pinned backend
versions were smoke-tested. Document dewarping/orientation modules are disabled;
max detection side=1600 bounds compute and coordinates refer to upright native
RGB images. All recognition confidences are retained for the downstream selector.

```bash
%%bash
set -euo pipefail
uv venv /tmp/ocr-env --python 3.12
uv pip install --python /tmp/ocr-env/bin/python \
    "paddleocr==3.3.2" "paddlepaddle==3.2.2"
/tmp/ocr-env/bin/python cache_ocr.py \
    --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
    --splits train val test --device cpu --max-side 1600
/tmp/ocr-env/bin/python audit_ocr.py \
    --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
    --split val --max-candidates 24 --output-dir outputs/ocr_coverage
```

This processes each image once, without using Q/answer/GT as OCR input.
The audit reads validation masks to measure proposal coverage; it does not train
a model. Read `outputs/ocr_coverage/coverage.json` before starting A/B. Missing
target coverage is a reason to reconsider OCR proposals before spending GPU time.
Coverage/oracle metrics are diagnostics, not model accuracy or transfer evidence.

If interrupted within the same session, rerun cache_ocr with the same cache
directory: existing image records are verified and skipped. For an audit rerun,
choose a fresh output directory because the audit does not overwrite results.

### CPU cell 3 — Verify completeness and ZIP for download

```python
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(os.environ["RUN_ROOT"])
cache = Path(os.environ["OCR_CACHE"])
assert (cache / "manifest.json").is_file(), "OCR manifest is missing"
for split in ("train", "val", "test"):
    names = json.loads((Path(os.environ["DATA_ROOT"]) / f"{split}_grounding.json").read_text())
    missing = [
        name for name in names
        if not (cache / split / Path(name).with_suffix(".json")).is_file()
    ]
    assert not missing, f"Incomplete {split} cache: {len(missing)} missing records"
assert (root / "ocr_coverage/coverage.json").is_file(), "Coverage audit is missing"

archive_path = root / "ocr_cache_and_coverage.zip"
with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
    for folder in (cache, root / "ocr_coverage"):
        for item in sorted(folder.rglob("*")):
            if item.is_file() and item.suffix not in {".pt", ".tmp", ".zip"}:
                archive.write(item, item.relative_to(root))
print(archive_path)
```

Download `outputs/ocr_cache_and_coverage.zip`, extract it, and upload the
`ocr_cache/` folder as a new Kaggle dataset. You may include `ocr_coverage/`
for reference. Keep this folder structure:

```text
ocr_cache/
    manifest.json
    train/*.json
    val/*.json
    test/*.json
ocr_coverage/
    coverage.json
```

The cache contains polygons, recognized strings, confidence, image hashes and
provenance. Do not upload images, GT masks, Python environments or OCR weights
again. In both GPU notebooks, OCR_CACHE must point to the folder containing
`manifest.json` and the three split directories. Use its actual mounted path
from the Kaggle sidebar.

## GPU sessions A and B — run the following cells once per session

Create TWO fresh notebooks/sessions with Internet and 2xT4.

Attach the same three inputs in each:
1. VizWiz.
2. The existing B+5 checkpoint dataset containing `mild_focus/checkpoint_epoch5.pt`.
3. The OCR cache dataset uploaded after the CPU session.

| Setting | Session A | Session B |
|---|---|---|
| MODE in cell 1 | `geometry` | `full` |
| Adapter epochs | 20 | 20 |
| Parent | B+5 | Same B+5 |
| OCR cache | Uploaded CPU cache | Same cache |
| Trainable weights | Fresh adapter | Independent fresh adapter |
| Result ZIP | `ocr_geometry_ep20_seed42.zip` | `ocr_full_ep20_seed42.zip` |

The remaining cells are identical. Each notebook trains ONLY its selected MODE.
No Paddle installation or OCR regeneration is needed on the GPU.

### GPU cell 1 — Setup (set MODE for A or B)

```python
!pip install -q uv
!git clone --single-branch --branch exp/ocr-evidence-pyramid https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen

import os
from pathlib import Path

os.environ["MODE"] = "geometry"  # Session A: geometry; Session B: full
os.environ["DATA_ROOT"] = "/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz"
os.environ["PARENT_CKPT"] = "/kaggle/input/datasets/khoahoang0512/vizwiz-size-weighted-pyramid-cp/mild_focus/checkpoint_epoch5.pt"
os.environ["OCR_CACHE"] = "/kaggle/input/datasets/YOUR_USERNAME/YOUR_OCR_CACHE_DATASET/ocr_cache"
os.environ["RUN_ROOT"] = "outputs"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

assert os.environ["MODE"] in ("geometry", "full")
assert Path(os.environ["PARENT_CKPT"]).is_file(), "Check the B+5 checkpoint path"
assert (Path(os.environ["OCR_CACHE"]) / "manifest.json").is_file(), "Set OCR_CACHE to the uploaded folder containing manifest.json"
for split in ("train", "val", "test"):
    assert (Path(os.environ["DATA_ROOT"]) / f"{split}_grounding.json").is_file()
    assert (Path(os.environ["OCR_CACHE"]) / split).is_dir(), f"Missing OCR split: {split}"
```

Edit OCR_CACHE to the real mounted path in BOTH notebooks. Change MODE to
`full` in session B. Do not use original parent20, A/natural, crop or interpolation
checkpoints as the parent. Output stays directly in `outputs/`; session A and B
are separate filesystems and do not need nested run folders.

### GPU cell 2 — Train this session's adapter for 20 epochs

```bash
%%bash
set -euo pipefail
uv run torchrun --standalone --nproc_per_node=2 train_ocr.py \
    --parent-checkpoint "$PARENT_CKPT" \
    --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
    --output-dir "$RUN_ROOT" --evidence-mode "$MODE" \
    --num-epochs 20 --batch-size 4 --gradient-accumulation-steps 1 \
    --num-workers 2 --lr 0.0001 --optimizer adamw --weight-decay 0.01 \
    --max-candidates 24 --ocr-dropout 0.2 --text-dropout 0.2 \
    --selection-weight 0.1 --local-weight 0.1 \
    --save-every 5 --seed 42 --amp
```

There is no per-epoch validation. Save epochs5/10/15/20 and retain all four files.
Each contains only adapter/optimizer/scaler/provenance, not another copy of B+5.
Keep the original B+5 separately: both parent and adapter are required for loading.
The loader checks the exact parent's SHA256. No automatic checkpoint deletion
or duplicate final model is performed.

### GPU cell 3 — Validation and diagnostics at epochs10 and20

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
    uv run diagnose.py \
        --data-root "$DATA_ROOT" \
        --predictions "$RUN_ROOT/eval_val_ep$EPOCH" \
        --output-dir "$RUN_ROOT/diagnostic_val_ep$EPOCH"
done
```

`--view full` here means normal inference for the saved architecture: a geometry
checkpoint still ignores recognized content. Evaluation reports native-resolution
mean/global IoU, original-GT size groups and the frozen parent's paired score.

### GPU cell 4 — Select on validation, test, and robustness diagnostics

This selects the better validation mean IoU of epochs10/20; ties prefer epoch10.
Test scores are never used for selection. Base/missing-OCR checks run in both
sessions; removing/shuffling recognized text additionally runs in session B.
Shuffling is fixed across images and independent of evaluation batch size.

```python
import json
import subprocess
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
mode = os.environ["MODE"]
best = max((10, 20), key=lambda epoch: (
    json.loads((root / f"eval_val_ep{epoch}/metrics.json").read_text())["summary"]["mean_iou"],
    -epoch,
))
views = [("test", "full"), ("val", "base"), ("val", "missing")]
if mode == "full":
    views += [("val", "geometry"), ("val", "shuffled_text")]

for split, view in views:
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
    "uv", "run", "audit_ocr.py",
    "--data-root", os.environ["DATA_ROOT"],
    "--cache-root", os.environ["OCR_CACHE"],
    "--split", "val", "--max-candidates", "24",
    "--predictions", str(root / f"eval_val_base_ep{best}"),
    "--output-dir", str(root / "ocr_oracle_headroom"),
], check=True)

(root / "selection.json").write_text(json.dumps({
    "mode": mode,
    "selected_epoch": best,
    "selection_metric": "validation_mean_iou",
    "tie_break": "earlier_epoch",
}, indent=2))
print(f"Session {mode}: selected adapter epoch {best}")
```

Candidate-set and perfect-local-correction oracles use GT: they are diagnostic
upper bounds, not deployable model scores.

### GPU cell 5 — ZIP this session's results

```python
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(os.environ["RUN_ROOT"])
mode = os.environ["MODE"]
archive_path = root / f"ocr_{mode}_ep20_seed42.zip"
with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
    for item in sorted(root.rglob("*")):
        if item.is_file() and item.suffix not in {".pt", ".tmp", ".zip"}:
            archive.write(item, item.relative_to(root))
print(archive_path)
```

Download this result ZIP from each session for analysis. It contains masks,
metrics, diagnostics and configs; it does NOT contain .pt files, the input OCR
cache or the parent checkpoint. Download the four adapter checkpoints separately
if you want to keep/resume them. Both sessions retain their checkpoints.

## Resume and inference notes

If training is interrupted after a saved epoch, resume the SAME mode with the
same parent/cache/settings, a NEW output directory and total `--num-epochs 20`.
Add `--resume-checkpoint /path/checkpoint_epoch10.pt` to GPU cell 2 and use, for
example, `--output-dir outputs_resume`. Then point RUN_ROOT at the resumed
directory before evaluation. B must never resume A. The epoch counter refers to
adapter epochs, not the parent's history.

A resumed segment does not recreate earlier checkpoint files. To reuse GPU cells
3-5 unchanged, copy the original small adapter checkpoint_epoch10.pt into the
new RUN_ROOT AFTER resumed training finishes. If epoch10 is unavailable, evaluate
the available checkpoints and adjust the selection list accordingly.

For an arbitrary image, create its OCR record in an environment with the same
pinned backend, then predict without any GT files:

```bash
/tmp/ocr-env/bin/python cache_ocr.py --image /path/photo.jpg \
    --cache-root outputs/new_image_ocr --device cpu --max-side 1600
uv run predict_ocr.py --parent-checkpoint "$PARENT_CKPT" \
    --checkpoint outputs/checkpoint_epoch20.pt \
    --image /path/photo.jpg --question "What does the label say?" \
    --cache-root outputs/new_image_ocr --output outputs/photo_mask.png
```

The CPU-only session does not install PyTorch; the GPU sessions do not install
Paddle. This arbitrary-image example therefore needs both environments available
on the same machine, or an OCR record transferred between them. It is not an
extra required Kaggle training cell. To bypass OCR entirely, replace
`--cache-root ...` in predict_ocr with `--no-ocr`.

## Interpretation and transfer protocol

Compare each run to its frozen parent's paired native predictions. Require useful
small AND medium gains, inspect large-region damage, and report overall mean IoU
alongside global IoU. Missing-OCR must reproduce the parent. Shuffled/removed text
tests whether recognized content is used; it does not prove semantic reasoning.
Do not infer transfer from these perturbations: evaluate a held-out dataset or
backbone with its own preprocessing adapter and disclose whether any fine-tuning
was performed. Full OCR runtime plus grounding runtime is the deployment cost;
cached grounding timing alone excludes OCR and is explicitly labeled in metrics.

Known limits: unreadable text, logos missed by the detector, targets outside
candidate support, top-N truncation, line-level rather than word-level boxes, and
CLIP's text length limit. Non-text regions rely on the parent. Different OCR
protocols must be evaluated as a separate experiment, not silently substituted.

Implementation is based on standard OCR, cross-attention and residual refinement.
Relevant references: [PaddleOCR API](https://www.paddleocr.ai/v3.3.1/en/version3.x/pipeline_usage/OCR.html),
[PP-OCR](https://arxiv.org/abs/2009.09941),
[M4C / OCR-aware multimodal reasoning](https://openaccess.thecvf.com/content_CVPR_2020/html/Hu_Iterative_Answer_Prediction_With_Pointer-Augmented_Multimodal_Transformers_for_TextVQA_CVPR_2020_paper.html).
These references motivate components, not a claim of already improved VizWiz scores.

Legacy pyramid utilities remain available. Original train.py/eval.py and old
checkpoints retain their behavior; use train_ocr.py/eval_ocr.py for this experiment.
