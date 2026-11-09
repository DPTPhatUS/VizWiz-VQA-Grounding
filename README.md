# VizWiz-VQA-Grounding: CLIP ViT-B/16 baseline (`ocrg/exp-backbone-clip-vitb16`)

Question-only answer grounding on [VizWiz-VQA-Grounding](https://vizwiz.org/tasks-and-datasets/answer-grounding-for-vqa/): given a photo and a question, predict the mask of the image region that supports the answer. The model never sees the answer. This branch changes the **image encoder only** from the `ocrg/main` B0 to CLIP ViT-B/16. It keeps the frozen ViT-L question encoder, learned text projection, three skips, FusionDecoder, loss and native-resolution evaluation. No OCR or B1/H1/H2 modules are used.

## B0 with ViT-B/16

```
images 336x336 ─ CLIP ViT-B/16 (fine-tuned) ─ skips: blocks 4, 7, 9 ; last V   (21x21, 768 ch)
question ─ original CLIP ViT-L text encoder (frozen) ─ tokens T ─ Linear 768→768
F = V + α · MHA(query=V, key=T, value=T)                      α init 0.01; 8 heads
Decoder (FusionDecoder):
  x = Conv1x1(F) → 256; x ← x + Bottleneck(x + Conv1x1(s_j)), j = 3, 2, 1
  4 × [bilinear ×2 → (Conv3x3 → GroupNorm → GELU) × 2]: 256→128→64→32→16
    grid 21→42→84→168→336; stages 42/84/168 have question attention (γ init 0)
  Conv1x1 → 1 logit map (336x336)
Loss: BCE + 0.5 · per-image soft Dice against the 336px training mask
```

Image weights: `openai/clip-vit-base-patch16`, revision `57c216476eefef5ab752ec549e440a49ae4ae5f3`. The native224 image encoder uses explicit positional interpolation from14x14 to21x21 at input336, retaining its original pretrained position table and CLS position. Question weights/tokenizer: `openai/clip-vit-large-patch14-336`, revision `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`.

Compared with primary B0, vision depth changes24→12, width1024→768 and patch grid24x24→21x21. Decoder widths and question width768 stay fixed. Skip indices4/7/9 stay fixed, but their relative depths differ. The decoder emits336 rather than384; both paths resize logits to native dimensions for evaluation. This is a hybrid with the original question encoder, not the native150M ViT-B/16 dual encoder. Actual total/trainable parameter counts are recorded at run start. Across-backbone scores do not isolate capacity alone.

**Pass `--vision-backbone clip-vit-b16`.** The default remains historical ViT-L B0 for backward compatibility. B/16 pins the two revisions above; optional `--vision-revision`/`--text-revision` require immutable40-character commit hashes. Checkpoints save the architecture/revisions and reconstruct interpolation automatically. Old B0 configs without the new fields retain their original path; mismatched-backbone resumes fail. Initialize this baseline from pretrained encoders and a new grounding head, never from the trained ViT-L grounding checkpoint.

## Data

Point `--data-root` at a folder containing `{train,val}_grounding.json`, `{train,val}/` and `binary_masks_png/{train,val}/`. Full-frame images are EXIF-rotated, resized to336x336 (bicubic, antialiased, without preserving aspect ratio), and CLIP-normalised. Masks use nearest resizing. Question-only prompts and split membership remain unchanged. The optional `--text-mode qa` feeds the gold answer and is an ablation, not this baseline. Missing or misaligned masks fail.

## Train

```bash
torchrun --nproc_per_node 2 train.py --vision-backbone clip-vit-b16 \
  --data-root DATA --output-dir OUT --num-epochs 20 --batch-size 8 \
  --dice-weight 0.5 --freeze-text-encoder --no-freeze-vision-encoder \
  --optimizer adamw --weight-decay 0.05 --lr 1e-4 --backbone-lr 1e-5 --lr-scheduler cosine --warmup-ratio 0.05 \
  --grad-clip-norm 1.0 --validate-every 2 --save-every 10 \
  --expect vision_backbone=clip-vit-b16 --expect freeze_text_encoder=true --expect optimizer=adamw
```

- This retains the original B0 recipe:20 epochs, AdamW, no decay on norms/biases/embeddings/α/γ, head LR1e-4, vision LR1e-5, cosine with5% warmup and clipping1.0. It is a fixed-budget baseline, not an optimal-recipe claim. Unspecified optimizer defaults remain historical Adam/constant.
- Global batch8 means4 per GPU on two GPUs; epoch validation also uses4 per GPU. Training uses FP16 autocast; validation uses FP32 without gradients. One GPU uses global batch8 on that GPU; memory feasibility is not established locally.
- Use a fresh output directory per seed. Keep configs, history, the complete validation curve and full-state checkpoints10/20. Select only between saved10/20 on validation, not an unsaved epoch. Checkpoints include optimizer/scaler/scheduler state and are written atomically.

## Evaluate

```bash
uv run --frozen python eval.py --checkpoint OUT/checkpoint_epoch20.pt --split val \
  --data-root DATA --output-dir OUT/eval_val_ep20 --batch-size 4 --save-masks
```

Use `torchrun --nproc_per_node 2 eval.py …` to shard validation across GPUs. Evaluation batch size is per GPU. Every image is scored exactly once: bilinear resize of logits to native dimensions, threshold0, per-image IoU (two empty masks score1), then mean over images. Training validation and standalone evaluation use the same function. Existing evaluation metrics are never overwritten. No test evaluation is needed.

The historical B0 source manifest/exporter remains unchanged and **intentionally rejects this branch**. Export original ViT-L B0 masks from `ocrg/main` using [its guide](https://github.com/DPTPhatUS/VizWiz-VQA-Grounding/blob/ocrg/main/docs/kaggle_b0_mask_export.md); do not weaken its source guard.

## Tests

```bash
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 python -m unittest discover -s tests -p 'test_*.py' -v
```

Tiny random encoders and the cached original tokenizer exercise B0 compatibility, B/16 positional interpolation/gradients/skips, checkpoint reconstruction, native masks, `predict()` and two-process CPU training. These are software checks, not pretrained accuracy or GPU-runtime measurements.

## Origin

Research fork of [yjh9929/VizWiz-VQA-Grounding](https://github.com/yjh9929/VizWiz-VQA-Grounding), via `exp/baseline-fixes` and `ocrg/main`. Licensed under [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/).

## Kaggle: ViT-B/16 B0, seed 42 (first run)

Enable Internet and **two T4 GPUs**. Attach `phtphantun/vizwiz-vqa-grounding`. No B0 checkpoint or OCR Input is required. Run cells in order; every shell cell stops on failure. This branch must be pushed before Kaggle can clone it.

| Cell | What |
|---|---|
| 1 | Clone this branch, install, cache pinned CLIP snapshots, check data/GPUs |
| 2 | Offline CPU tests |
| 3 | Optional memory/speed check |
| 4 | Train the20-epoch baseline |
| 5 | Export validation masks at10/20 and verify results |
| Z | ZIP results without weights; retain checkpoints separately |

### Cell 1: setup

```python
!git clone --depth 1 --single-branch --branch ocrg/exp-backbone-clip-vitb16 https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!git log --oneline -1
%env UV_PROJECT_ENVIRONMENT=/root/ocrg-vitb16-venv
%env OMP_NUM_THREADS=1
%env MKL_NUM_THREADS=1
%env MPLBACKEND=Agg
```

The environment stays outside the20GB working disk. Install Kaggle's required `wrapt` in that environment without changing the lockfile:

```bash
%%bash
set -euo pipefail
uv sync --frozen --quiet
uv pip install -q --python /root/ocrg-vitb16-venv wrapt hf_transfer
uv run --frozen python - <<'PY'
from huggingface_hub import snapshot_download
from models.model import CLIP_NAME, VITB16_NAME, VITB16_REVISION, TEXT_REVISION
for name, revision in ((VITB16_NAME, VITB16_REVISION), (CLIP_NAME, TEXT_REVISION)):
    print(snapshot_download(name, revision=revision, allow_patterns=['*.json', '*.txt', 'pytorch_model.bin']))
PY
```

Both models are public. An `HF_TOKEN` environment variable is optional if you already use a Kaggle secret. Both snapshots are needed because the image and question encoders differ.

```python
import hashlib, json, os, subprocess
from pathlib import Path
assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'ocrg/exp-backbone-clip-vitb16'
assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(), 'Use a clean checkout'
os.environ['DATA_ROOT'] = '/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz'
os.environ['RUN_NAME'] = 'ocrg_b0_vitb16_336_s42'
os.environ['OUT'] = f"/kaggle/working/outputs/{os.environ['RUN_NAME']}"
os.environ['BATCH'] = '8'
os.environ['SEED'] = '42'
data = Path(os.environ['DATA_ROOT'])
for split, n in (('train', 6494), ('val', 1131)):
    records = json.loads((data / f'{split}_grounding.json').read_text())
    assert len(records) == n, (split, len(records))
    assert all((data / split / name).is_file() and
               (data / 'binary_masks_png' / split / (Path(name).stem + '.png')).is_file() for name in records)
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                     check=True, capture_output=True, text=True)
print(gpu.stdout)
assert len(gpu.stdout.strip().splitlines()) == 2, 'Enable two GPUs'
assert not Path(os.environ['OUT']).exists(), 'Choose a fresh RUN_NAME'
```

If the Input mount differs, change only `DATA_ROOT` to the folder containing `train_grounding.json`. No WSL or ignored `personal/` file is required.

### Cell 2: offline tests

```bash
%%bash
set -euo pipefail
uv run --frozen python -c "from transformers import CLIPTokenizer; CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14-336')"
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -v
```

The last line must be `OK`. Do not continue after a failing cell.

### Cell 3: optional memory/speed check

This runs30 updates through the real training path and writes `probe.json`, without checkpoints or validation. It is optional and does not establish accuracy. Keep batch8 for matched comparisons; report an OOM before changing the recipe.

```bash
%%bash
set -euo pipefail
test ! -e /kaggle/working/outputs/probe_vitb16_bs8
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py --vision-backbone clip-vit-b16 \
  --data-root "$DATA_ROOT" --output-dir /kaggle/working/outputs/probe_vitb16_bs8 --batch-size "$BATCH" --num-workers 2 --max-steps 30 \
  --dice-weight 0.5 --freeze-text-encoder --no-freeze-vision-encoder \
  --optimizer adamw --weight-decay 0.05 --lr 1e-4 --backbone-lr 1e-5 --lr-scheduler cosine --warmup-ratio 0.05 \
  --grad-clip-norm 1.0 --seed "$SEED" --expect vision_backbone=clip-vit-b16
```

### Cell 4: train B0

```bash
%%bash
set -euo pipefail
test ! -e "$OUT"
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py --vision-backbone clip-vit-b16 \
  --data-root "$DATA_ROOT" --output-dir "$OUT" --num-epochs 20 --batch-size "$BATCH" --num-workers 2 \
  --dice-weight 0.5 --freeze-text-encoder --no-freeze-vision-encoder \
  --optimizer adamw --weight-decay 0.05 --lr 1e-4 --backbone-lr 1e-5 --lr-scheduler cosine --warmup-ratio 0.05 \
  --grad-clip-norm 1.0 --validate-every 2 --save-every 10 --seed "$SEED" \
  --expect vision_backbone=clip-vit-b16 --expect text_mode=question --expect dice_weight=0.5 \
  --expect freeze_text_encoder=true --expect freeze_vision_encoder=false --expect optimizer=adamw \
  --expect lr_scheduler=cosine --expect decoder=fusion --expect attention_residual=true \
  --expect stage_attention=true --expect grad_clip_norm=1.0 --expect world_size=2
```

Progress: loss/timing each epoch, validation every2 epochs, checkpoints10/20. For seeds41/43, choose a new `RUN_NAME`/`OUT` and matching `SEED`; keep the recipe fixed. If interrupted, persist the checkpoint and attach it to a new session. Use a fresh output directory and add `--resume-checkpoint /kaggle/input/<attached-output>/.../checkpoint_epoch10.pt` to Cell4 with identical recipe flags. Optimizer/scaler/scheduler resume checks remain in force.

### Cell 5: validation masks

Validation chooses between saved10/20; no test split is accessed.

```bash
%%bash
set -euo pipefail
for ep in 10 20; do
  uv run --frozen torchrun --standalone --nproc_per_node=2 eval.py \
    --checkpoint "$OUT/checkpoint_epoch${ep}.pt" --split val --data-root "$DATA_ROOT" \
    --output-dir "$OUT/eval_val_ep${ep}" --batch-size 4 --num-workers 2 --save-masks
done
```

```python
from PIL import Image, ImageOps
out = Path(os.environ['OUT'])
records = json.loads((data / 'val_grounding.json').read_text())
curve = {r['epoch']: r['val_miou'] for r in map(json.loads, (out / 'val_curve.jsonl').read_text().splitlines())}
for ep in (10, 20):
    folder = out / f'eval_val_ep{ep}'
    metrics = json.loads((folder / 'metrics.json').read_text())
    assert metrics['summary']['split'] == 'val'
    assert metrics['summary']['model_config']['vision_backbone'] == 'clip-vit-b16'
    assert len(metrics['per_sample']) == 1131 and set(metrics['per_sample']) == set(records)
    assert abs(metrics['summary']['mean_iou'] - curve[ep]) < 1e-6
    assert len(list((folder / 'masks').glob('*.png'))) == 1131
    for name in records:
        with Image.open(data / 'val' / name) as image, Image.open(folder / 'masks' / (Path(name).stem + '.png')) as mask:
            assert mask.size == ImageOps.exif_transpose(image).size, name
    print(ep, metrics['summary']['mean_iou'], '1131 native masks verified')
```

### Cell Z: ZIP results (no checkpoints)

```python
from zipfile import ZIP_DEFLATED, ZipFile
out = Path(os.environ['OUT'])
def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()
provenance = out / 'artifact_provenance.json'
assert not provenance.exists(), 'Provenance already exists'
info = {
    'git': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'git_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True),
    'annotations': {s: sha256(data / f'{s}_grounding.json') for s in ('train', 'val')},
    'checkpoints': {p.name: sha256(p) for p in sorted(out.glob('checkpoint_epoch*.pt'))},
}
assert set(info['checkpoints']) == {'checkpoint_epoch10.pt', 'checkpoint_epoch20.pt'}
provenance.write_text(json.dumps(info, indent=2))
archive = Path('/kaggle/working') / f"{os.environ['RUN_NAME']}.zip"
members = {}
with ZipFile(archive, 'x', compression=ZIP_DEFLATED) as bundle:
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.suffix.lower() not in {'.pt', '.tmp', '.zip'}:
            name = str(path.relative_to(out.parent))
            members[name] = sha256(path)
            bundle.write(path, name)
with ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    for name, expected in members.items():
        assert hashlib.sha256(bundle.read(name)).hexdigest() == expected, name
print('Verified ZIP:', archive)
print('ZIP SHA256:', sha256(archive))
with Path(str(archive) + '.sha256').open('x') as checksum:
    checksum.write(sha256(archive) + '  ' + archive.name + '\n')
```

Persist the notebook outputs before ending the session. Use **Save Version → Save & Run All** for a fresh unattended execution; it starts the experiment again, so do not use it to rerun a completed interactive training merely for saving. Download `ocrg_b0_vitb16_336_s42.zip` and keep the two `.pt` files separately in Kaggle outputs. The ZIP contains configs, curves, per-image validation metrics, native masks and checkpoint hashes, excluding heavyweight weights. No pretrained accuracy or GPU-runtime result is claimed until the actual run completes.
