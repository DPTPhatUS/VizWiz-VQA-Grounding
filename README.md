# OCR evidence pyramid

Branch: `exp/ocr-evidence-pyramid`.

This branch freezes the `B+5` (`mild_focus` epoch 5) pyramid and trains a lightweight OCR evidence adapter to refine predictions within native-image text regions, testing whether explicit text-proposal localization improves small and medium targets while preserving whole-object grounding.

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

The adapter adds **293,938 trainable parameters** on top of the frozen `B+5` pyramid and frozen mobile OCR pipeline. Rather than instantiating a second language model, the `full` run encodes short recognized OCR strings with the existing frozen CLIP text encoder (mean-pooled per candidate), while candidate queries attend to the full sequence of question tokens.

The final correction layer is zero-initialized so the model reproduces `B+5` at initialization, and images with zero OCR detections fall back to `B+5` even after training. When OCR detections are present but irrelevant to the question, the selector can route probability to the explicit `NULL` candidate. Local corrections can add or remove foreground within a `1.25x` padded bounding box around each candidate on the `768x768` grid (OCR polygons themselves are not used directly as output masks).

The adapter constructor accepts arbitrary text and detail channel dimensions and is wired here to the 336/768 CLIP pyramid. The OCR cache schema stores upright native `(x, y)` polygon coordinates, recognized strings, confidences, and decoded-RGB SHA256 hashes without image IDs, answer vocabularies, or ground-truth size gates. `predict_ocr.py` runs inference on arbitrary images without ground-truth annotations.

## Two matched runs

| Run | Candidate geometry + native appearance | Recognized OCR text |
|---|---|---|
| `geometry` | Yes | No |
| `full` | Yes | Yes (via frozen CLIP text encoder) |

Both runs start independently from the same frozen `B+5` parent and adapter seed (`42`). During training, `ocr_dropout = 0.2` drops all OCR candidates for an image, and `text_dropout = 0.2` masks out recognized strings while keeping candidate geometry and visual appearance.

Default training hyperparameters: AdamW, constant learning rate `1e-4`, weight decay `0.01`, global batch size `4`, gradient accumulation `1`, FP16 AMP on CUDA, 20 epochs, checkpoint saving every 5 epochs, and no per-epoch validation. Training controls can be overridden via a flat JSON config (**JSON config > CLI flag > default**; file paths remain on the CLI). See `uv run train_ocr.py --help` for all options.

The training loss combines final per-image `BCE + 0.5 * Dice` with `0.1 * selection_loss` and `0.1 * local_loss`. Candidate selection supervision uses soft native-GT box Dice across relevant candidates, supervising `NULL` whenever all candidate Dice overlaps fall below `0.1` (including on empty-GT images). Local crop supervision uses area-downsampled ground-truth masks on relevant candidates only, so candidate gating is supervised by ground-truth overlap rather than whether the adapter improves on the parent's training logits. All ground-truth relevance and crop targets are computed in the dataset loader during training only.

## Kaggle workflow: one CPU session, then GPU sessions A and B

Run three fresh Kaggle sessions in order:

| Session | Accelerator | Inputs to attach | Work | Download |
|---|---|---|---|---|
| CPU | None | VizWiz | Cache OCR for train/val/test; audit coverage | `ocr_cache_and_coverage.zip` |
| A | 2xT4 | VizWiz + `B+5` + uploaded OCR cache | Train `geometry` for 20 epochs; evaluate/diagnose | `ocr_geometry_ep20_seed42.zip` and adapter checkpoints |
| B | 2xT4 | Same inputs as A | Train `full` for 20 epochs; evaluate/diagnose | `ocr_full_ep20_seed42.zip` and adapter checkpoints |

Sessions A and B each train only the 293,938-parameter adapter while running frozen parent forward passes each epoch.

## CPU session: create the reusable OCR cache

Enable Internet, set Accelerator to **None**, and attach only the VizWiz dataset. Skip `uv sync` in the CPU session because OCR caching and coverage auditing do not use PyTorch or CLIP.

### CPU cell 1: Clone and paths

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

### CPU cell 2: Install OCR and cache all three splits

The `/tmp/ocr-env` virtual environment is a local scratch environment for running PaddleOCR on CPU. Mobile OCR weights download automatically, document orientation/dewarping modules are disabled, and `--max-side 1600` bounds detection resolution while recording coordinates in upright native RGB image space.

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

`cache_ocr.py` processes each image once without reading questions, answers, or ground-truth masks. `audit_ocr.py` then compares the cached validation proposals against validation ground-truth masks and writes `outputs/ocr_coverage/coverage.json` so you can inspect proposal coverage before running the GPU sessions. If `cache_ocr.py` is interrupted, rerunning it with the same `--cache-root` verifies and skips completed image records.

### CPU cell 3: Verify completeness and ZIP for download

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

Download `outputs/ocr_cache_and_coverage.zip`, extract it, and upload the `ocr_cache/` directory as a Kaggle dataset (optionally alongside `ocr_coverage/`):

```text
ocr_cache/
    manifest.json
    train/*.json
    val/*.json
    test/*.json
ocr_coverage/
    coverage.json
```

In both GPU sessions, point `OCR_CACHE` to the mounted directory containing `manifest.json` and the `train/`, `val/`, and `test/` folders.

## GPU sessions A and B: run the following cells once per session

Create two separate Kaggle sessions with Internet and **2x T4** GPUs enabled, attaching the same three inputs in each:

1. VizWiz dataset.
2. The `B+5` checkpoint dataset containing `mild_focus/checkpoint_epoch5.pt`.
3. The uploaded `ocr_cache` dataset from the CPU session.

| Setting | Session A | Session B |
|---|---|---|
| `MODE` in cell 1 | `geometry` | `full` |
| Adapter epochs | 20 | 20 |
| Parent | `B+5` | Same `B+5` |
| OCR cache | Uploaded CPU cache | Same cache |
| Trainable weights | Fresh adapter | Independent fresh adapter |
| Result ZIP | `ocr_geometry_ep20_seed42.zip` | `ocr_full_ep20_seed42.zip` |

All remaining cells are identical across Sessions A and B, and no PaddleOCR installation is needed on the GPU machines.

### GPU cell 1: Setup (set MODE for A or B)

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

Update `OCR_CACHE` to the mounted dataset path in both notebooks, and change `MODE` from `"geometry"` to `"full"` in Session B.

### GPU cell 2: Train this session's adapter for 20 epochs

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

This saves compact adapter checkpoints at epochs 5, 10, 15, and 20 (`checkpoint_epoch5.pt` through `checkpoint_epoch20.pt`), storing only the adapter weights, optimizer state, AMP scaler, and provenance (including the parent checkpoint's SHA256 hash) rather than duplicating the frozen `B+5` parent weights.

### GPU cell 3: Validation and diagnostics at epochs 10 and 20

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

Passing `--view full` runs normal inference for the checkpoint's trained architecture (a `geometry` checkpoint still ignores recognized text). Evaluation computes native-resolution mean and global IoU, size-group breakdowns, and paired deltas against the frozen `B+5` parent.

### GPU cell 4: Select on validation, test, and ablation diagnostics

This cell picks the higher validation mean IoU between epochs 10 and 20 (breaking ties in favor of epoch 10), evaluates that checkpoint on the test split, and runs validation ablations (`base` parent-only and `missing` empty-OCR in both sessions, plus `geometry`-only and `shuffled_text` in Session B) along with an oracle headroom audit.

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

Note that `ocr_oracle_headroom` uses validation ground truth to compute upper-bound diagnostic metrics for the candidate set.

### GPU cell 5: ZIP this session's results

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

Download `ocr_<mode>_ep20_seed42.zip` from each GPU session, and download the four compact adapter `.pt` files separately if you want to keep or resume them.

## Resume and inference notes

If a GPU session is interrupted after saving an intermediate checkpoint (such as `checkpoint_epoch10.pt`), resume that same mode in a new output directory (e.g., `--output-dir outputs_resume`) with `--resume-checkpoint /path/to/checkpoint_epoch10.pt` and `--num-epochs 20`. After training finishes, copy the earlier `checkpoint_epoch10.pt` into `outputs_resume` and set `RUN_ROOT = "outputs_resume"` so GPU cells 3 through 5 can evaluate both epoch 10 and epoch 20 unchanged.

To run standalone prediction on a new image without ground-truth files, cache its OCR proposals with the pinned PaddleOCR environment and run `predict_ocr.py`:

```bash
/tmp/ocr-env/bin/python cache_ocr.py --image /path/photo.jpg \
    --cache-root outputs/new_image_ocr --device cpu --max-side 1600
uv run predict_ocr.py --parent-checkpoint "$PARENT_CKPT" \
    --checkpoint outputs/checkpoint_epoch20.pt \
    --image /path/photo.jpg --question "What does the label say?" \
    --cache-root outputs/new_image_ocr --output outputs/photo_mask.png
```

To bypass OCR at inference time and fall back to the frozen parent, replace `--cache-root ...` with `--no-ocr`.

## Interpretation and references

Compare each adapter run against its frozen `B+5` parent on small, medium, and large ground-truth regions as well as overall mean and global IoU. Verify that `--view missing` matches `--view base` (`B+5`), and compare `full` against `geometry` and `shuffled_text` to measure the contribution of recognized text strings. Note that cached evaluation timing measures only the pyramid + adapter pass; end-to-end deployment latency also includes running mobile OCR on the input image.

Known limitations of the OCR pipeline include severely blurred or unreadable text, logos missed by the text detector, targets outside the `1.25x` candidate box support, top-24 candidate truncation, line-level rather than word-level polygon boxes, and CLIP's token length limit.

References: [PaddleOCR API](https://www.paddleocr.ai/v3.3.1/en/version3.x/pipeline_usage/OCR.html), [PP-OCR](https://arxiv.org/abs/2009.09941), and [M4C / OCR-aware multimodal reasoning](https://openaccess.thecvf.com/content_CVPR_2020/html/Hu_Iterative_Answer_Prediction_With_Pointer-Augmented_Multimodal_Transformers_for_TextVQA_CVPR_2020_paper.html). Standard pyramid training and evaluation (`train.py` and `eval.py`) remain unchanged on this branch.
