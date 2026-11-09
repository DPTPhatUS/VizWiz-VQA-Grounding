# Kaggle: export B0 validation masks, seeds 41/42/43

This is a new inference-only notebook. It uses the existing evaluator to export epoch-20 validation PNGs, checks all 1,131 images per seed against original validation results, and creates a downloadable ZIP. It does not train a model or read test metric contents. Keep the original training notebooks and checkpoints.

## Before opening the notebook

Create a **new Kaggle Notebook**, select a Python notebook, enable **Internet**, and select a GPU accelerator. One GPU or two T4s are supported. The runner detects CUDA devices and uses at most two, with one evaluator process per GPU. No CUDA device is an error; there is no silent CPU-inference fallback.

Use **Add Input** to attach:

1. The VizWiz dataset `phtphantun/vizwiz-vqa-grounding` used for B0. Only `val_grounding.json`, `val/` images and `binary_masks_png/val/` masks are required by this workflow.
2. Each saved B0 notebook output containing the actual `ocrg_b0_s41/checkpoint_epoch20.pt`, `ocrg_b0_s42/checkpoint_epoch20.pt`, and `ocrg_b0_s43/checkpoint_epoch20.pt`. Notebook outputs can be attached directly; no need to copy these multi-GB files into `/kaggle/working`. The small results ZIPs alone **do not contain checkpoints**.
3. Original result ZIPs `ocrg_b0_s41.zip`, `ocrg_b0_s42.zip`, `ocrg_b0_s43.zip`, or their extracted `ocrg_b0_sSEED/eval_val_ep20/metrics.json` files. Upload these as a **private Kaggle Dataset** if they are not already attached notebook outputs. They stay outside Git. The runner reads only the explicit validation member from a ZIP.

Use only your own retained checkpoints: the original PyTorch checkpoint format uses pickle. Internet downloads CLIP config/tokenizer/pretrained files used to reconstruct the existing model before checkpoint loading. No Hugging Face token is required for the public CLIP model.

## Cell 1: obtain the public code

**GitHub route, after these local changes have been published to `ocrg/main`:**

```bash
%%bash
set -euo pipefail
test ! -e /kaggle/working/VizWiz-VQA-Grounding
git clone --depth 1 --single-branch --branch ocrg/main \
  https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git \
  /kaggle/working/VizWiz-VQA-Grounding
cd /kaggle/working/VizWiz-VQA-Grounding
git log --oneline -1
git ls-files --error-unmatch scripts/export_b0_validation_masks.py docs/kaggle_b0_mask_export.md
```

**Until publication:** a GitHub clone cannot contain unpushed changes. On the local model checkout, create a source ZIP using `git archive --format=zip --output=/tmp/b0-mask-export-source.zip HEAD`. This includes committed, tracked public files only; it excludes `personal/`, data, checkpoints, and untracked files. Upload that ZIP as a private Kaggle Dataset named `b0-mask-export-source`, attach it, and run the following cell **instead of the clone cell**. Kaggle may expose the ZIP or unpack it automatically:

```python
from pathlib import Path
import shutil, zipfile

target = Path('/kaggle/working/VizWiz-VQA-Grounding')
assert not target.exists(), f'Use a new notebook/session: {target} exists'
source = Path('/kaggle/input/b0-mask-export-source')
assert source.exists(), 'Set source to the actual attached source Dataset directory'
archives = list(source.rglob('b0-mask-export-source.zip'))
if archives:
    assert len(archives) == 1
    with zipfile.ZipFile(archives[0]) as z:
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
        z.extractall(target)
else:
    candidates = list(source.rglob('scripts/export_b0_validation_masks.py'))
    assert len(candidates) == 1, candidates
    shutil.copytree(candidates[0].parents[1], target)
assert (target / 'scripts/b0_source_manifest.json').is_file()
```

Source ZIPs have no Git history; provenance records a null workflow commit plus the exact workflow script hash and original B0 source manifest. Keep the locally printed source commit and source ZIP hash with your notebook notes. Both code routes verify the same original B0 file hashes; neither requires current Git HEAD to remain the old B0 commit.

## Cell 2: environment and original-code verification

```python
%cd /kaggle/working/VizWiz-VQA-Grounding
%env UV_PROJECT_ENVIRONMENT=/root/ocrg-export-venv
%env OMP_NUM_THREADS=1
```

```bash
%%bash
set -euo pipefail
command -v uv >/dev/null || python -m pip install uv
uv sync --frozen --quiet
# Kaggle's sitecustomize needs wrapt in every Python environment.
uv pip install --python /root/ocrg-export-venv wrapt
uv run --frozen python -c "from huggingface_hub import snapshot_download; print(snapshot_download('openai/clip-vit-large-patch14-336', allow_patterns=['*.json','*.txt','pytorch_model.bin']))"
uv run --frozen python -c "from scripts.export_b0_validation_masks import verify_source; import torch; print(verify_source()['baseline_commit']); print('CUDA devices:', torch.cuda.device_count()); assert torch.cuda.is_available()"
uv run --frozen python -m unittest discover -s tests -p test_b0_export.py -v
```

The environment is outside the 20 GB working disk. `uv sync --frozen` keeps the committed lockfile; `wrapt` is an environment-only addition and is recorded with installed package versions. The manifest hashes the original B0 Python files, `pyproject.toml`, `uv.lock`, and Python version file. Adding deployment docs/scripts changes Git HEAD but does not change any of these checked files. Never edit the hashes to bypass a mismatch.

## Cell 3: resolve actual Kaggle Input paths

Kaggle Input paths vary. This cell discovers candidates without opening test results. It fails on missing or ambiguous candidates, rather than choosing a checkpoint silently.

```python
import json, os
from pathlib import Path

inputs = Path('/kaggle/input')
datasets = list(inputs.rglob('val_grounding.json'))
assert len(datasets) == 1, f'Select the intended dataset root explicitly: {datasets}'
data = datasets[0].parent
assert (data / 'val').is_dir() and (data / 'binary_masks_png/val').is_dir()
records = json.loads((data / 'val_grounding.json').read_text())
assert len(records) == 1131
checkpoints = list(inputs.rglob('checkpoint_epoch20.pt'))
print('All attached epoch-20 candidates:', *map(str, checkpoints), sep='\n')
mapping = {}
for seed in (41, 42, 43):
    run = f'ocrg_b0_s{seed}'
    weights = [p for p in checkpoints if run in p.parts]
    references = list(inputs.rglob(f'{run}.zip'))
    if not references:
        references = [p for p in inputs.rglob('metrics.json')
                      if run in p.parts and p.parent.name == 'eval_val_ep20']
    assert len(weights) == 1, f'{run}: select exact checkpoint path: {weights}'
    assert len(references) == 1, f'{run}: select exact original validation reference: {references}'
    mapping[str(seed)] = {'checkpoint': str(weights[0]), 'reference': str(references[0])}
manifest = Path('/kaggle/working/b0_export_inputs.json')
assert not manifest.exists(), 'Manifest exists; inspect it or use a new session'
manifest.write_text(json.dumps(mapping, indent=2))
os.environ['DATA_ROOT'] = str(data)
os.environ['EXPORT_ROOT'] = '/kaggle/working/b0_validation_masks'
print(manifest.read_text())
print('DATA_ROOT:', data)
```

If your saved outputs renamed the run directories, use the printed candidates and write the map explicitly instead: `mapping = {'41': {'checkpoint': '/kaggle/input/ACTUAL/.../checkpoint_epoch20.pt', 'reference': '/kaggle/input/ACTUAL/.../ocrg_b0_s41.zip'}, ...}` for all three seeds, followed by the manifest/environment assignments above. These are actual Kaggle paths, never WSL paths. Checkpoint metadata must identify the correct seed, epoch, clean original source commit, recipe and model config; a misleading directory name cannot pass that check.

## Cell 4: CPU preflight, then validation-only export

```bash
%%bash
set -euo pipefail
uv run --frozen python scripts/export_b0_validation_masks.py \
  --inputs /kaggle/working/b0_export_inputs.json --data-root "$DATA_ROOT" \
  --output-root "$EXPORT_ROOT" --check-only
```

CPU preflight loads each trusted checkpoint serially, checks all three references and original source, and checks epoch/seed/config metadata before any inference. It needs host RAM for one full B0 checkpoint (including optimizer state). If memory is insufficient, stop and preserve the original checkpoint; do not replace it with a casually stripped state dict.

```bash
%%bash
set -euo pipefail
uv run --frozen python scripts/export_b0_validation_masks.py \
  --inputs /kaggle/working/b0_export_inputs.json --data-root "$DATA_ROOT" \
  --output-root "$EXPORT_ROOT" --gpus auto --batch-size 2 --num-workers 2
```

`--batch-size 2` is **per GPU**, unlike B0's global training batch. Evaluation remains FP32, native-resolution bilinear logits, threshold zero, and canonical per-image IoU. One GPU runs plain `eval.py`; two run `torch.distributed.run` with the existing non-padding evaluation sampler. `--gpus 1` forces one GPU if two are available. No training process is launched.

Each seed's output is immediately verified on CPU: exact image IDs, exactly 1,131 binary PNGs, native EXIF-upright source/GT/prediction dimensions, recomputed canonical IoU versus exported metrics (absolute error ≤1e-6), versus original per-image IoU (≤1e-4), and aggregate mean versus original (≤1e-5). Tolerances are in IoU units [0,1]; do not enlarge them to hide a mismatch. An ID, geometry, metadata or score mismatch fails the job and retains evidence for investigation.

All output goes under `/kaggle/working/b0_validation_masks`, with `s41_ep20_val`, `s42_ep20_val`, `s43_ep20_val`, per-seed logs, provenance, and a `COMPLETE.json` marker written only after all seeds pass. The adjacent ZIP is created only on success. The runner refuses an existing output directory **or ZIP**, including partial earlier outputs. For OOM, inspect the log and rerun with `--batch-size 1` and a **new** `EXPORT_ROOT`; do not delete or overwrite the failed export. There is no automatic partial resume.

## Cell 5: independently recheck and download

```bash
%%bash
set -euo pipefail
uv run --frozen python scripts/export_b0_validation_masks.py \
  --inputs /kaggle/working/b0_export_inputs.json --data-root "$DATA_ROOT" \
  --output-root "$EXPORT_ROOT" --verify-only
test -f "$EXPORT_ROOT/COMPLETE.json"
sha256sum "$EXPORT_ROOT.zip"
```

```python
from IPython.display import FileLink, display
from pathlib import Path
export = Path(os.environ['EXPORT_ROOT'])
assert json.loads((export / 'COMPLETE.json').read_text())['predictions_per_seed'] == 1131
display(FileLink(str(export.with_suffix('.zip'))))
```

Use **Save Version → Save & Run All** to persist the notebook's working outputs, then download the ZIP from the notebook Output panel if the inline link is unavailable. Alternatively download the ZIP from the current session before it ends; only the saved version's outputs are durable. A rerun must start in a clean session or use new output paths. Retain the original checkpoints in their saved notebook outputs separately; this export ZIP deliberately contains no weights, test results, images or GT masks.

The ZIP contains masks and `metrics.json` in the exact directory layout consumed by the local CPU diagnostics, original **validation-only** metrics, per-image dimensions/question/GT+prediction hashes, checkpoint SHA256s, source hashes, workflow commit/hash, environment package versions, GPU names, commands, input paths and validation annotation checksum. Original ZIP files that also contain test metrics are not copied into it.

After download, extract it into the personal research workspace and point `src/diagnostics.py analyze --eval-dir .../s42_ep20_val` (and the other seeds) at the original local VizWiz data and OCR cache. The CPU diagnostic script remains private; Kaggle does not need to import anything under `personal/`.

## Verification status

Deployment guards and native mask verification were tested with 1,131 synthetic CPU fixtures; original B0 source hashes were checked. Source-change, missing-mask, native-shape, checkpoint-identity, no-GPU/two-GPU request and sealed-test protections are covered. GPU inference and the live Kaggle setup were not executed during preparation. Actual GPU memory, runtime and reproduction tolerances must pass on the retained real checkpoints before outputs are accepted.
