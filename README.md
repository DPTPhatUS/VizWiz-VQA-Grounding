# VizWiz-VQA-Grounding: OpenCLIP ConvNeXt-Base baseline (`ocrg/exp-backbone-convnext-base`)

Question-only answer grounding on [VizWiz-VQA-Grounding](https://vizwiz.org/tasks-and-datasets/answer-grounding-for-vqa/): given a photo and a question, predict the mask of the image region that supports the answer. The model never sees the answer. This branch replaces the `ocrg/main` backbone with the **paired OpenCLIP ConvNeXt-Base vision and text encoders**. Text stays frozen; vision is fine-tuned. Learned text projection, three hierarchical skips, FusionDecoder structure, loss and native-resolution evaluation are retained. No OCR or B1/H1/H2 modules are used.

## B0 with OpenCLIP ConvNeXt-Base

```
images 336x336 ─ OpenCLIP ConvNeXt-Base (fine-tuned)
  stem (84x84, 128 ch) → s1=stages[0] (84x84, 128 ch) → s2=stages[1] (42x42, 256 ch)
                       → s3=stages[2] (21x21, 512 ch) → V=stages[3] (10x10, 1024 ch)
question ─ paired OpenCLIP ConvNeXt-Base text encoder (frozen) ─ tokens T ─ Linear 512→1024
F = V + α · MHA(query=V, key=T, value=T)                      α init 0.01; 8 heads
Decoder (FusionDecoder, hierarchical skips):
  x = Conv1x1(F) → 256 (10x10) → bilinear to 21x21; x ← x + Bottleneck_256(x + Conv1x1_512→256(s3))
  stage 0: bilinear to 42x42 → ConvBlock 256→128 → fuse s2 (256→128) → question attention (γ init 0)
  stage 1: bilinear to 84x84 → ConvBlock 128→64  → fuse s1 (128→64)  → question attention (γ init 0)
  stage 2: bilinear ×2 (168x168) → ConvBlock 64→32                   → question attention (γ init 0)
  stage 3: bilinear ×2 (336x336) → ConvBlock 32→16
  Conv1x1 → 1 logit map (336x336)
Loss: BCE + 0.5 · per-image soft Dice against the 336px training mask
```

Both encoders load from the official OpenCLIP `convnext_base` (`laion400m_s13b_b51k`) release packaged on Hugging Face Hub (`laion/CLIP-convnext_base-laion400M-s13B-b51K`, `open_clip_pytorch_model.bin`), pinned to its immutable 40-character commit `4c478c07a1511dafb7331443cd20bf7867a0f1b1`. The BPE tokenizer is identical to the standard 49,408-token CLIP vocabulary and is loaded from `openai/clip-vit-base-patch16` at revision `57c216476eefef5ab752ec549e440a49ae4ae5f3`. `models/image_encoder.py` extracts the `timm` `ConvNeXt` visual trunk (`trunk.stem` and `trunk.stages[0..3]`, replacing the unused `trunk.norm_pre`, `trunk.head` and `visual.head` pooling/projection layers with `nn.Identity()`), and `models/text_encoder.py` converts the paired 12-layer GELU text transformer into `CLIPTextModel`.

Compared with primary ViT-L/14-336 B0, the vision backbone changes from a uniform `24x24` (`1024` ch) patch grid to a multiscale CNN pyramid (`84x84` `128` ch, `42x42` `256` ch, `21x21` `512` ch, `10x10` `1024` ch). Instead of fusing all three skips at the bottleneck resolution, `FusionDecoder` upsamples the `10x10` bottleneck to `21x21` to fuse `s3`, then fuses `s2` at `42x42` (`128` ch) and `s1` at `84x84` (`64` ch) before question attention at `42x42`, `84x84` and `168x168`. Text width changes `768→512`; the learned cross-attention text projection becomes `512→1024` and each decoder stage's text projection accepts `512` channels. Total parameters are `156.8M` (`156,772,469`), with `93.6M` (`93,606,517`) trainable (`87.6M` ConvNeXt-Base visual trunk + `4.7M` global cross-attention + `1.3M` hierarchical `FusionDecoder`) and `63.2M` (`63,165,952`) frozen in the text encoder. The decoder emits `336x336` logits rather than `384x384`; both paths resize logits to native dimensions for evaluation. Across-backbone scores compare complete pretrained encoder pairs, not image capacity alone.

**Pass `--vision-backbone convnext-base`.** Despite the retained CLI name, this selects the entire matching encoder pair. The default remains historical ViT-L B0 for backward compatibility. ConvNeXt-Base pins both encoders to the commit hash above; explicit `--vision-revision`/`--text-revision` must be identical immutable 40-character commits. Checkpoints save both identities/revisions and reconstruct hierarchical skip fusion automatically. Historical B0 configs without the new fields retain their original path; mismatched-backbone resumes fail. Initialize from pretrained encoders and a new grounding head, never from a trained ViT-L, ViT-B/16, RN50 or SigLIP2 grounding checkpoint.

## Data

Point `--data-root` at a folder containing `{train,val}_grounding.json`, `{train,val}/` and `binary_masks_png/{train,val}/`. Full-frame images are EXIF-rotated, resized to 336x336 (bicubic, antialiased, without preserving aspect ratio), and CLIP-normalised (`mean=(0.48145466, 0.4578275, 0.40821073)`, `std=(0.26862954, 0.26130258, 0.27577711)`). Masks use nearest resizing. Question-only prompts and split membership remain unchanged. The optional `--text-mode qa` feeds the gold answer and is an ablation, not this baseline. Missing or misaligned masks fail.

## Train

```bash
torchrun --nproc_per_node 2 train.py --vision-backbone convnext-base \
  --data-root DATA --output-dir OUT --num-epochs 20 --batch-size 8 \
  --dice-weight 0.5 --freeze-text-encoder --no-freeze-vision-encoder \
  --optimizer adamw --weight-decay 0.05 --lr 1e-4 --backbone-lr 1e-5 --lr-scheduler cosine --warmup-ratio 0.05 \
  --grad-clip-norm 1.0 --validate-every 2 --save-every 10 \
  --expect vision_backbone=convnext-base --expect freeze_text_encoder=true --expect optimizer=adamw
```

- This retains the original B0 recipe: 20 epochs, AdamW, no decay on norms/biases/embeddings/α/γ, head LR 1e-4, vision LR 1e-5, cosine with 5% warmup and clipping 1.0. ConvNeXt-Base uses `LayerNorm2d`/`LayerNorm` and `GroupNorm` throughout, which are batch-size independent across GPUs.
- Global batch 8 means 4 per GPU on two GPUs; epoch validation also uses 4 per GPU. Training uses FP16 autocast; validation uses FP32 without gradients.
- Use a fresh output directory per seed. Keep configs, history, the complete validation curve and full-state checkpoints 10/20. Select only between saved 10/20 on validation, not an unsaved epoch. Checkpoints include optimizer/scaler/scheduler state and are written atomically.

## Evaluate

```bash
uv run --no-sync python eval.py --checkpoint OUT/checkpoint_epoch20.pt --split val \
  --data-root DATA --output-dir OUT/eval_val_ep20 --batch-size 4 --save-masks
```

Use `torchrun --nproc_per_node 2 eval.py …` to shard validation across GPUs. Evaluation batch size is per GPU. Every image is scored exactly once: bilinear resize of logits to native dimensions, threshold 0, per-image IoU (two empty masks score 1), then mean over images. Training validation and standalone evaluation use the same function. Existing evaluation metrics are never overwritten. No test evaluation is needed.

The historical B0 source manifest/exporter remains unchanged and **intentionally rejects this branch**. Export original ViT-L B0 masks from `ocrg/main` using [its guide](https://github.com/DPTPhatUS/VizWiz-VQA-Grounding/blob/ocrg/main/docs/kaggle_b0_mask_export.md); do not weaken its source guard.

## Tests

```bash
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 python -m unittest discover -s tests -p 'test_*.py' -v
```

Tiny random encoders and the cached BPE and historical ViT-L tokenizers exercise B0 compatibility, ConvNeXt-Base pyramid shapes, hierarchical skip fusion, `open_clip` text conversion, gradients, checkpoint reconstruction, native masks, `predict()` and two-process CPU training. Cache the paired snapshot in Cell 1 and historical tokenizer in Cell 2 before offline tests. These are software checks, not pretrained accuracy or GPU-runtime measurements.

## Origin

Research fork of [yjh9929/VizWiz-VQA-Grounding](https://github.com/yjh9929/VizWiz-VQA-Grounding), via `exp/baseline-fixes` and `ocrg/main`. Licensed under [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/).

## Kaggle: OpenCLIP ConvNeXt-Base B0, seed 42 (first run)

Enable Internet and **two T4 GPUs**. Attach `phtphantun/vizwiz-vqa-grounding`. No B0 checkpoint or OCR Input is required. Run cells in order; every shell cell stops on failure. This branch must be pushed before Kaggle can clone it.

| Cell | What |
|---|---|
| 1 | Clone this branch, install, cache pinned OpenCLIP ConvNeXt-Base weights and tokenizer, check data/GPUs |
| 2 | Offline CPU tests |
| 3 | Optional memory/speed check |
| 4 | Train the 20-epoch baseline |
| 5 | Export validation masks at 10/20 and verify results |
| Z | ZIP results without weights; retain checkpoints separately |

### Cell 1: setup

```python
!git clone --depth 1 --single-branch --branch ocrg/exp-backbone-convnext-base https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!git log --oneline -1
%env UV_PROJECT_ENVIRONMENT=/root/ocrg-convnext-venv
%env OMP_NUM_THREADS=1
%env MKL_NUM_THREADS=1
%env MPLBACKEND=Agg
```

The environment stays outside the 20GB working disk. Install Kaggle's required `wrapt` plus `open-clip-torch` and `timm` in that environment without changing the lockfile, and use `uv run --no-sync` in subsequent cells so `uv` keeps those extra packages installed:

```bash
%%bash
set -euo pipefail
uv sync --frozen --quiet
uv pip install -q --python /root/ocrg-convnext-venv wrapt hf_transfer open-clip-torch timm
uv run --no-sync python - <<'PY'
from huggingface_hub import snapshot_download
from models.model import CONVNEXT_NAME, CONVNEXT_REVISION, CONVNEXT_TOKENIZER_NAME, CONVNEXT_TOKENIZER_REVISION
print(snapshot_download(CONVNEXT_NAME, revision=CONVNEXT_REVISION, allow_patterns=['*.json', 'open_clip_pytorch_model.bin']))
print(snapshot_download(CONVNEXT_TOKENIZER_NAME, revision=CONVNEXT_TOKENIZER_REVISION, allow_patterns=['*.json', '*.txt']))
PY
```

The model and tokenizer are public; an `HF_TOKEN` environment variable is optional if you already use a Kaggle secret. One `laion/CLIP-convnext_base-laion400M-s13B-b51K` snapshot (`open_clip_pytorch_model.bin`) supplies both separately loaded vision and text encoders. CPU compatibility tests also cache the historical ViT-L tokenizer below; its model weights are not needed for ConvNeXt-Base training.

```python
import hashlib, json, os, subprocess
from pathlib import Path
assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'ocrg/exp-backbone-convnext-base'
assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(), 'Use a clean checkout'
os.environ['DATA_ROOT'] = '/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz'
os.environ['RUN_NAME'] = 'ocrg_b0_convnext_336_s42'
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
uv run --no-sync python -c "from transformers import CLIPTokenizer; CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14-336')"
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 uv run --no-sync python -m unittest discover -s tests -p 'test_*.py' -v
```

The last line must be `OK`. Do not continue after a failing cell.

### Cell 3: optional memory/speed check

This runs 30 updates through the real training path and writes `probe.json`, without checkpoints or validation. It is optional and does not establish accuracy. Keep batch 8 for matched comparisons; report an OOM before changing the recipe.

```bash
%%bash
set -euo pipefail
test ! -e /kaggle/working/outputs/probe_convnext_bs8
uv run --no-sync torchrun --standalone --nproc_per_node=2 train.py --vision-backbone convnext-base \
  --data-root "$DATA_ROOT" --output-dir /kaggle/working/outputs/probe_convnext_bs8 --batch-size "$BATCH" --num-workers 2 --max-steps 30 \
  --dice-weight 0.5 --freeze-text-encoder --no-freeze-vision-encoder \
  --optimizer adamw --weight-decay 0.05 --lr 1e-4 --backbone-lr 1e-5 --lr-scheduler cosine --warmup-ratio 0.05 \
  --grad-clip-norm 1.0 --seed "$SEED" --expect vision_backbone=convnext-base
```

### Cell 4: train B0

```bash
%%bash
set -euo pipefail
test ! -e "$OUT"
uv run --no-sync torchrun --standalone --nproc_per_node=2 train.py --vision-backbone convnext-base \
  --data-root "$DATA_ROOT" --output-dir "$OUT" --num-epochs 20 --batch-size "$BATCH" --num-workers 2 \
  --dice-weight 0.5 --freeze-text-encoder --no-freeze-vision-encoder \
  --optimizer adamw --weight-decay 0.05 --lr 1e-4 --backbone-lr 1e-5 --lr-scheduler cosine --warmup-ratio 0.05 \
  --grad-clip-norm 1.0 --validate-every 2 --save-every 10 --seed "$SEED" \
  --expect vision_backbone=convnext-base --expect text_mode=question --expect dice_weight=0.5 \
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
  uv run --no-sync torchrun --standalone --nproc_per_node=2 eval.py \
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
    assert metrics['summary']['model_config']['vision_backbone'] == 'convnext-base'
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

Persist the notebook outputs before ending the session. Use **Save Version → Save & Run All** for a fresh unattended execution; it starts the experiment again, so do not use it to rerun a completed interactive training merely for saving. Download `ocrg_b0_convnext_336_s42.zip` and keep the two `.pt` files separately in Kaggle outputs. The ZIP contains configs, curves, per-image validation metrics, native masks and checkpoint hashes, excluding heavyweight weights. No pretrained accuracy or GPU-runtime result is claimed until the actual run completes.
