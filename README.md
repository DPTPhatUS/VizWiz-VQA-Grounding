# VizWiz-VQA-Grounding: SigLIP2 Base/16-384 baseline (`ocrg/exp-backbone-siglip2-b16`)

Question-only answer grounding on [VizWiz-VQA-Grounding](https://vizwiz.org/tasks-and-datasets/answer-grounding-for-vqa/): given a photo and a question, predict the mask of the image region that supports the answer. The model never sees the answer. This branch replaces the `ocrg/main` backbone with the **paired Google SigLIP2 Base/16-384 vision and text encoders** at their native `384x384` resolution. Text stays frozen; vision is fine-tuned. Learned text projection, three skips, FusionDecoder structure, loss and native-resolution evaluation are retained. No OCR or B1/H1/H2 modules are used.

## B0 with SigLIP2 Base/16-384

```
images 384x384 ─ SigLIP2 Base/16-384 (fine-tuned) ─ skips: blocks 4, 7, 9 ; last V   (24x24, 768 ch)
question ─ paired SigLIP2 Base/16-384 text encoder (frozen; 64 tokens) ─ tokens T ─ Linear 768→768
F = V + α · MHA(query=V, key=T, value=T)                                              α init 0.01; 8 heads
Decoder (FusionDecoder):
  x = Conv1x1(F) → 256; x ← x + Bottleneck(x + Conv1x1(s_j)), j = 3, 2, 1
  4 × [bilinear ×2 → (Conv3x3 → GroupNorm → GELU) × 2]: 256→128→64→32→16
    grid 24→48→96→192→384; stages 48/96/192 have question attention (γ init 0)
  Conv1x1 → 1 logit map (384x384)
Loss: BCE + 0.5 · per-image soft Dice against the 384px training mask
```

Both encoders and the tokenizer use `google/siglip2-base-patch16-384`, revision `932518b266bc21ea2a4291b915a912d43aeb3a48`. `models/image_encoder.py` loads `SiglipVisionModel` (replacing the unused attention pooling head with `nn.Identity()`) and `models/text_encoder.py` separately loads `SiglipTextModel` and its Gemma-based tokenizer (`AutoTokenizer`), from this same checkpoint/revision. No combined `SiglipModel` is loaded. Because `384 / 16 = 24`, the native `384x384` image encoder produces a `24x24` (`576` tokens, no `CLS` token) patch grid without positional interpolation.

Compared with primary ViT-L/14-336 B0, vision depth changes `24→12`, vision width changes `1024→768`, and input resolution changes `336x336→384x384` with SigLIP normalization (`mean=(0.5, 0.5, 0.5)`, `std=(0.5, 0.5, 0.5)`), while keeping the same `24x24` patch grid and `48/96/192/384` decoder stages. Text width remains `768` (`256,000`-token multilingual Gemma vocabulary, padded to `64` tokens); the learned global text projection is `768→768` and each decoder stage's text projection accepts `768` channels. Decoder widths, head counts and fusion rules stay fixed. Skip indices `4/7/9` stay fixed, and the decoder emits `384x384` logits matching the `384x384` training mask; evaluation resizes logits to native image dimensions. Actual grounding-model total/trainable parameter counts (`373,549,813` total / `91,244,533` trainable) are recorded at run start. Across-backbone scores compare complete pretrained encoder pairs, not image capacity alone.

**Pass `--vision-backbone siglip2-b16-384`.** Despite the retained CLI name, this selects the entire matching encoder pair and native `384x384` preprocessing. The default remains historical ViT-L B0 for backward compatibility. SigLIP2 pins both encoders to the revision above; explicit `--vision-revision`/`--text-revision` must be identical immutable 40-character commits. Checkpoints save both identities/revisions and reconstruct `384x384` preprocessing automatically. Historical B0 configs without the new fields retain their original path; mismatched-backbone resumes fail. Initialize from pretrained encoders and a new grounding head, never from a trained ViT-L, ViT-B/16 or RN50 grounding checkpoint.

## Data

Point `--data-root` at a folder containing `{train,val}_grounding.json`, `{train,val}/` and `binary_masks_png/{train,val}/`. For `--vision-backbone siglip2-b16-384`, full-frame images are EXIF-rotated, resized to `384x384` (bicubic, antialiased, without preserving aspect ratio), and SigLIP-normalised (`mean=(0.5, 0.5, 0.5)`, `std=(0.5, 0.5, 0.5)`). Masks use nearest resizing to `384x384`. Question-only prompts and split membership remain unchanged. The optional `--text-mode qa` feeds the gold answer and is an ablation, not this baseline. Missing or misaligned masks fail.

## Train

```bash
torchrun --nproc_per_node 2 train.py --vision-backbone siglip2-b16-384 \
  --data-root DATA --output-dir OUT --num-epochs 20 --batch-size 8 \
  --dice-weight 0.5 --freeze-text-encoder --no-freeze-vision-encoder \
  --optimizer adamw --weight-decay 0.05 --lr 1e-4 --backbone-lr 1e-5 --lr-scheduler cosine --warmup-ratio 0.05 \
  --grad-clip-norm 1.0 --validate-every 2 --save-every 10 \
  --expect vision_backbone=siglip2-b16-384 --expect freeze_text_encoder=true --expect optimizer=adamw
```

- This retains the original B0 recipe: 20 epochs, AdamW, no decay on norms/biases/embeddings/α/γ, head LR 1e-4, vision LR 1e-5, cosine with 5% warmup and clipping 1.0. It is a fixed-budget baseline, not an optimal-recipe claim. Unspecified optimizer defaults remain historical Adam/constant.
- Global batch 8 means 4 per GPU on two GPUs; epoch validation also uses 4 per GPU. Training uses FP16 autocast; validation uses FP32 without gradients.
- Use a fresh output directory per seed. Keep configs, history, the complete validation curve and full-state checkpoints 10/20. Select only between saved 10/20 on validation, not an unsaved epoch. Checkpoints include optimizer/scaler/scheduler state and are written atomically.

## Evaluate

```bash
uv run --frozen python eval.py --checkpoint OUT/checkpoint_epoch20.pt --split val \
  --data-root DATA --output-dir OUT/eval_val_ep20 --batch-size 4 --save-masks
```

Use `torchrun --nproc_per_node 2 eval.py …` to shard validation across GPUs. Evaluation batch size is per GPU. Every image is scored exactly once: bilinear resize of logits to native dimensions, threshold 0, per-image IoU (two empty masks score 1), then mean over images. Training validation and standalone evaluation use the same function. Existing evaluation metrics are never overwritten. No test evaluation is needed.

The historical B0 source manifest/exporter remains unchanged and **intentionally rejects this branch**. Export original ViT-L B0 masks from `ocrg/main` using [its guide](https://github.com/DPTPhatUS/VizWiz-VQA-Grounding/blob/ocrg/main/docs/kaggle_b0_mask_export.md); do not weaken its source guard.

## Tests

```bash
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 python -m unittest discover -s tests -p 'test_*.py' -v
```

Tiny random encoders and the cached historical ViT-L tokenizer exercise B0 compatibility, native `384x384` `24x24` feature extraction, SigLIP normalization, projection/gradients, checkpoint reconstruction, native masks, `predict()` and two-process CPU training. Cache the historical tokenizer in Cell 2 before offline tests. These are software checks, not pretrained accuracy or GPU-runtime measurements.

## Origin

Research fork of [yjh9929/VizWiz-VQA-Grounding](https://github.com/yjh9929/VizWiz-VQA-Grounding), via `exp/baseline-fixes` and `ocrg/main`. Licensed under [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/).

## Kaggle: SigLIP2 Base/16-384 B0, seed 42 (first run)

Enable Internet and **two T4 GPUs**. Attach `phtphantun/vizwiz-vqa-grounding`. No B0 checkpoint or OCR Input is required. Run cells in order; every shell cell stops on failure. This branch must be pushed before Kaggle can clone it.

| Cell | What |
|---|---|
| 1 | Clone this branch, install, cache pinned SigLIP2 snapshot, check data/GPUs |
| 2 | Offline CPU tests |
| 3 | Optional memory/speed check |
| 4 | Train the 20-epoch baseline |
| 5 | Export validation masks at 10/20 and verify results |
| Z | ZIP results without weights; retain checkpoints separately |

### Cell 1: setup

```python
!git clone --depth 1 --single-branch --branch ocrg/exp-backbone-siglip2-b16 https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!git log --oneline -1
%env UV_PROJECT_ENVIRONMENT=/root/ocrg-siglip2-venv
%env OMP_NUM_THREADS=1
%env MKL_NUM_THREADS=1
%env MPLBACKEND=Agg
```

The environment stays outside the 20GB working disk. Install Kaggle's required `wrapt` in that environment without changing the lockfile:

```bash
%%bash
set -euo pipefail
uv sync --frozen --quiet
uv pip install -q --python /root/ocrg-siglip2-venv wrapt hf_transfer
uv run --frozen python - <<'PY'
from huggingface_hub import snapshot_download
from models.model import SIGLIP2_NAME, SIGLIP2_REVISION
print(snapshot_download(SIGLIP2_NAME, revision=SIGLIP2_REVISION, allow_patterns=['*.json', '*.model', 'model.safetensors']))
PY
```

The model is public; an `HF_TOKEN` environment variable is optional if you already use a Kaggle secret. One `google/siglip2-base-patch16-384` snapshot supplies both separately loaded vision and text encoders plus the tokenizer. CPU compatibility tests also cache the historical ViT-L tokenizer below; its model weights are not needed for SigLIP2 training.

```python
import hashlib, json, os, subprocess
from pathlib import Path
assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'ocrg/exp-backbone-siglip2-b16'
assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(), 'Use a clean checkout'
os.environ['DATA_ROOT'] = '/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz'
os.environ['RUN_NAME'] = 'ocrg_b0_siglip2_b16_384_s42'
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

This runs 30 updates through the real training path and writes `probe.json`, without checkpoints or validation. It is optional and does not establish accuracy. Keep batch 8 for matched comparisons; report an OOM before changing the recipe.

```bash
%%bash
set -euo pipefail
test ! -e /kaggle/working/outputs/probe_siglip2_bs8
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py --vision-backbone siglip2-b16-384 \
  --data-root "$DATA_ROOT" --output-dir /kaggle/working/outputs/probe_siglip2_bs8 --batch-size "$BATCH" --num-workers 2 --max-steps 30 \
  --dice-weight 0.5 --freeze-text-encoder --no-freeze-vision-encoder \
  --optimizer adamw --weight-decay 0.05 --lr 1e-4 --backbone-lr 1e-5 --lr-scheduler cosine --warmup-ratio 0.05 \
  --grad-clip-norm 1.0 --seed "$SEED" --expect vision_backbone=siglip2-b16-384
```

### Cell 4: train B0

```bash
%%bash
set -euo pipefail
test ! -e "$OUT"
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py --vision-backbone siglip2-b16-384 \
  --data-root "$DATA_ROOT" --output-dir "$OUT" --num-epochs 20 --batch-size "$BATCH" --num-workers 2 \
  --dice-weight 0.5 --freeze-text-encoder --no-freeze-vision-encoder \
  --optimizer adamw --weight-decay 0.05 --lr 1e-4 --backbone-lr 1e-5 --lr-scheduler cosine --warmup-ratio 0.05 \
  --grad-clip-norm 1.0 --validate-every 2 --save-every 10 --seed "$SEED" \
  --expect vision_backbone=siglip2-b16-384 --expect text_mode=question --expect dice_weight=0.5 \
  --expect freeze_text_encoder=true --expect freeze_vision_encoder=false --expect optimizer=adamw \
  --expect lr_scheduler=cosine --expect decoder=fusion --expect attention_residual=true \
  --expect stage_attention=true --expect grad_clip_norm=1.0 --expect world_size=2
```

Progress: loss/timing each epoch, validation every 2 epochs, checkpoints 10/20. For seeds 41/43, choose a new `RUN_NAME`/`OUT` and matching `SEED`; keep the recipe fixed. If interrupted, persist the checkpoint and attach it to a new session. Use a fresh output directory and add `--resume-checkpoint /kaggle/input/<attached-output>/.../checkpoint_epoch10.pt` to Cell 4 with identical recipe flags. Optimizer/scaler/scheduler resume checks remain in force.

### Cell 5: validation masks

Validation chooses between saved 10/20; no test split is accessed.

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
    assert metrics['summary']['model_config']['vision_backbone'] == 'siglip2-b16-384'
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

Persist the notebook outputs before ending the session. Use **Save Version → Save & Run All** for a fresh unattended execution; it starts the experiment again, so do not use it to rerun a completed interactive training merely for saving. Download `ocrg_b0_siglip2_b16_384_s42.zip` and keep the two `.pt` files separately in Kaggle outputs. The ZIP contains configs, curves, per-image validation metrics, native masks and checkpoint hashes, excluding heavyweight weights. No pretrained accuracy or GPU-runtime result is claimed until the actual run completes.
