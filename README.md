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

## Kaggle notebook: setup, cache, train and evaluate

Default: use ONE notebook/session with Internet and 2xT4. Attach only:
1. The existing VizWiz dataset.
2. The existing B+5 checkpoint dataset containing
   `mild_focus/checkpoint_epoch5.pt`.

You do NOT need to upload a new OCR dataset, new annotations, or OCR model
weights. The OCR models download automatically; cache JSON files are generated
locally in `outputs/ocr_cache` and reused by both runs.

OCR preprocessing runs on CPU in the cells below. Keeping it in the GPU session
is simpler but uses session time while GPUs are idle. If you prefer a separate
CPU session, the optional handoff instructions appear below. Two 20-epoch runs
plus preprocessing have not been timed on Kaggle; do not assume they will fit
the session limit. A prepared cache can avoid repeating preprocessing.

### Cell 1 — Setup

```python
!pip install -q uv
!git clone --single-branch --branch exp/ocr-evidence-pyramid https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen

import os
os.environ["DATA_ROOT"] = "/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz"
os.environ["RUN_ROOT"] = "outputs"
os.environ["OCR_CACHE"] = "outputs/ocr_cache"
os.environ["PARENT_CKPT"] = "/kaggle/input/datasets/khoahoang0512/vizwiz-size-weighted-pyramid-cp/mild_focus/checkpoint_epoch5.pt"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
```

### Cell 2 — Install OCR and inspect validation coverage

Training/evaluation still use `uv run`, as in the previous branches. OCR uses a
separate environment at `/tmp/ocr-env` to avoid mixing Paddle and PyTorch
dependencies. This is local setup only; do not upload that environment.

These tested backend versions use
mobile detection/recognition, disable document dewarping/orientation modules and
retain all recognition confidences. Max side=1600 caps detection compute; the
backend returns coordinates in the original upright image.

```bash
%%bash
set -euo pipefail
uv venv /tmp/ocr-env --python 3.12
uv pip install --python /tmp/ocr-env/bin/python \
    "paddleocr==3.3.2" "paddlepaddle==3.2.2"
/tmp/ocr-env/bin/python cache_ocr.py \
    --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
    --splits val --device cpu --max-side 1600
/tmp/ocr-env/bin/python audit_ocr.py \
    --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
    --split val --max-candidates 24 --output-dir outputs/ocr_coverage
```

Read `outputs/ocr_coverage/coverage.json` first. It reports full-validation and
size-group coverage, top-24 versus all-proposal best polygon IoU, support area and
GT coverage. These metrics do NOT measure OCR string accuracy or prove transfer.
If proposals miss the target regions, improve the detector/input policy before
spending GPU time on the adapter. This is a human research decision; no automatic
validation-tuned training gate is installed.

### Cell 3 — Cache the remaining images

When coverage warrants training, generate train/test cache files in the SAME
local `outputs/ocr_cache` directory:

```bash
%%bash
set -euo pipefail
/tmp/ocr-env/bin/python cache_ocr.py \
    --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
    --splits train test --device cpu --max-side 1600
```

Continue directly to Cell 4. No upload, download, new Kaggle dataset or path
change is needed between these cells. Missing cache files stop training rather
than silently masquerading as images with no text. OCR is run only on RGB;
question/answer/GT are never passed to the backend.

### Cell 4 — Train both adapters

Use the same notebook and environment variables from Cell 1. Keep
`OCR_CACHE=outputs/ocr_cache`; do not clone again or point it at a new dataset.
Both runs start independently from B+5. Do not use original parent20, natural/A,
crop checkpoints or interpolation endpoints as the parent.

```bash
%%bash
set -euo pipefail
for MODE in geometry full; do
    uv run torchrun --standalone --nproc_per_node=2 train_ocr.py \
        --parent-checkpoint "$PARENT_CKPT" \
        --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
        --output-dir "$RUN_ROOT/$MODE" --evidence-mode "$MODE" \
        --num-epochs 20 --batch-size 4 --gradient-accumulation-steps 1 \
        --num-workers 2 --lr 0.0001 --optimizer adamw --weight-decay 0.01 \
        --max-candidates 24 --ocr-dropout 0.2 --text-dropout 0.2 \
        --selection-weight 0.1 --local-weight 0.1 \
        --save-every 5 --seed 42 --amp
done
```

All epoch5/10/15/20 checkpoints are retained. Each contains only adapter weights,
optimizer, scaler and provenance; no duplicate parent. They should be a few MB
each, not GB. Keep the original B+5 separately: both artifacts are required for
loading. SHA256 mismatch is an error. Rank-zero disk errors are propagated through
a CPU group rather than leaving other ranks waiting in NCCL.

### Cell 5 — Evaluate and diagnose

Evaluate validation at epochs10 and20, with native masks, mean/global IoU, size
groups and the unchanged parent's paired score. These are NOT test-selected runs.

```bash
%%bash
set -euo pipefail
for MODE in geometry full; do
    for EPOCH in 10 20; do
        uv run eval_ocr.py \
            --parent-checkpoint "$PARENT_CKPT" \
            --checkpoint "$RUN_ROOT/$MODE/checkpoint_epoch$EPOCH.pt" \
            --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
            --dataset val --view full --batch-size 4 --num-workers 2 \
            --output-dir "$RUN_ROOT/$MODE/eval_val_ep$EPOCH"
        uv run diagnose.py \
            --data-root "$DATA_ROOT" \
            --predictions "$RUN_ROOT/$MODE/eval_val_ep$EPOCH" \
            --output-dir "$RUN_ROOT/$MODE/diagnostic_val_ep$EPOCH"
    done
done
```

Select the epoch with validation mean IoU (tie -> earlier epoch), then run test
for that checkpoint in each experiment. Also evaluate full's robustness views at
its selected epoch. Shuffled text uses a fixed cross-image permutation independent
of evaluation batch size; boxes/appearance stay aligned.

```python
import json
from pathlib import Path
import subprocess

root = Path(os.environ["RUN_ROOT"])
selection = {}
for mode in ("geometry", "full"):
    best = max((10, 20), key=lambda e: (
        json.loads((root / mode / f"eval_val_ep{e}/metrics.json").read_text())["summary"]["mean_iou"], -e))
    selection[mode] = best
    views = [("test", "full")] + (
        [("val", v) for v in ("base", "missing", "geometry", "shuffled_text")]
        if mode == "full" else [])
    for split, view in views:
        subprocess.run([
            "uv", "run", "eval_ocr.py",
            "--parent-checkpoint", os.environ["PARENT_CKPT"],
            "--checkpoint", str(root / mode / f"checkpoint_epoch{best}.pt"),
            "--data-root", os.environ["DATA_ROOT"],
            "--cache-root", os.environ["OCR_CACHE"],
            "--dataset", split, "--view", view,
            "--batch-size", "4", "--num-workers", "2",
            "--output-dir", str(root / mode / f"eval_{split}_{view}_ep{best}")
        ], check=True)
(root / "selection.json").write_text(json.dumps(selection, indent=2))
```

Optional coverage/oracle headroom relative to a saved B+5 prediction folder:

```bash
%%bash
set -euo pipefail
uv run audit_ocr.py --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
    --split val --max-candidates 24 \
    --predictions outputs/full/eval_val_base_ep20 \
    --output-dir outputs/ocr_oracle_headroom
```

Change ep20 to the selected epoch if necessary. Candidate-set and perfect local
correction oracles use GT and are diagnostic upper bounds, never model scores.

### Cell 6 — ZIP results for download

The result ZIP contains metrics, masks, diagnostics and configs; checkpoints stay
outside it for separate Kaggle download. All adapter checkpoints are retained.

```python
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(os.environ["RUN_ROOT"])
archive_path = root / "ocr_evidence_two_runs_ep20_seed42.zip"
with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix not in {".pt", ".tmp", ".zip"}:
            archive.write(path, path.relative_to(root))
print(archive_path)
```

### Optional: prepare OCR in a separate CPU session

This is optional, to avoid spending GPU-session time on OCR. Generate the same
cache using Cells 1-3 (the OCR cells themselves do not need a GPU), then save it:

```python
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(os.environ["RUN_ROOT"])
with ZipFile(root / "ocr_cache_and_coverage.zip", "w", ZIP_DEFLATED) as archive:
    for folder in ("ocr_cache", "ocr_coverage"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix not in {".pt", ".tmp", ".zip"}:
                archive.write(path, path.relative_to(root))
```

Only in this separate-session workflow, download this ZIP and upload its extracted
`ocr_cache/` directory as a Kaggle dataset. It contains OCR polygons, recognized
strings, confidence and provenance JSONs; it does not need images, GT masks,
checkpoints, environments or OCR model weights.

In the new GPU notebook, run Cell 1 and then replace just this variable:

```python
os.environ["OCR_CACHE"] = "/kaggle/input/datasets/YOUR_USERNAME/YOUR_OCR_DATASET/ocr_cache"
```

Use the actual path from the Kaggle sidebar. Attach this OCR cache dataset in
addition to VizWiz and B+5, skip Cells 2-3, then run Cells 4-6. Caching is resumable
within a session: existing image records are verified and skipped.

### Resume and arbitrary-image inference

Resume uses a NEW output directory, the same parent/cache/settings, and a larger
total epoch count. Both mode and optimizer settings must match. For example, add
`--resume-checkpoint outputs/full/checkpoint_epoch10.pt` to the full command,
set output to `outputs/full_resume`, and keep `--num-epochs 20`.
The epoch counter refers to adapter epochs, not the parent's training history.
No automatic checkpoint deletion or extra final duplicate is performed.

For an arbitrary image, run cache_ocr in the separate OCR environment:

```bash
/tmp/ocr-env/bin/python cache_ocr.py --image /path/photo.jpg \
    --cache-root outputs/new_image_ocr --device cpu --max-side 1600
uv run predict_ocr.py --parent-checkpoint "$PARENT_CKPT" \
    --checkpoint outputs/full/checkpoint_epoch20.pt \
    --image /path/photo.jpg --question "What does the label say?" \
    --cache-root outputs/new_image_ocr --output outputs/photo_mask.png
```

To bypass OCR entirely for deployment fallback, replace `--cache-root ...` in
the prediction command with `--no-ocr`. The parent checkpoint is still required.

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
