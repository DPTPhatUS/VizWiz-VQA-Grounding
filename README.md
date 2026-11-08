# VizWiz-VQA-Grounding: OCR evidence grounding (`ocrg/main`)

Question-only answer grounding on [VizWiz-VQA-Grounding](https://vizwiz.org/tasks-and-datasets/answer-grounding-for-vqa/): given a photo and a question, predict the mask of the image region that supports the answer. The model never sees the answer. This branch holds the no-OCR baseline **B0**; OCR models are built on top of it. Research decisions are logged in the project's decision log (D016, D017).

## B0

```
images 336x336 ─ CLIP ViT-L/14-336 (fine-tuned) ─ skips: blocks 4, 7, 9 ; last block V      (all 24x24, 1024 ch)
question ─ CLIP text encoder (frozen) ─ tokens T ─ Linear 768→1024
F = V + α · MHA(query=V, key=T, value=T)                      α learnable, init 0.01
Decoder (FusionDecoder):
  x = Conv1x1(F) → 256;  x ← x + Bottleneck(x + Conv1x1(s_j))  for j = 3, 2, 1   (24x24; last GN starts at 0)
  4 × [bilinear ×2 → (Conv3x3 → GroupNorm → GELU) × 2]: 256→128→64→32→16, 24→384 px
    stages 48², 96², 192² then: x ← x + γ_s · MHA(GN(x), T, T)    (T projected per stage; 4/2/1 heads of 32; γ_s init 0)
  Conv1x1 → 1 logit map (384x384)
Loss: BCE + 0.5 · per-image soft Dice, on logits resized to the 336 px training mask
```

Trainable: 310.6M (vision encoder 303.5M, attention 5.0M, decoder 2.11M of which stage attention 0.26M); text encoder 123.1M frozen. `--no-stage-attention` builds the decoder without the stage attentions (identical to B0 at initialisation, since every γ_s starts at 0). The upstream architecture remains available with `--decoder unet --no-attention-residual --no-stage-attention`.

## Data

Point `--data-root` at a folder with `{train,val,test}_grounding.json`, `{train,val,test}/` images and `binary_masks_png/{train,val,test}/`. Images are EXIF-rotated, resized to 336×336 without keeping the aspect ratio (bicubic, antialiased) and CLIP-normalised; training masks use nearest resizing (no mask loses its foreground at 336 px). By default only the question is read (the task input). `--text-mode qa` also feeds the most common answer (`Q: … A: …`) for ablations only: it is recorded in the recipe, `eval.py` follows the checkpoint's mode unless `--text-mode` overrides it (e.g. a qa-trained model scored with the question only), results report `answer_in_input`, and qa mode fails on records without an answer (the test split). A missing or misaligned mask is an error.

## Train

```bash
torchrun --nproc_per_node 2 train.py --data-root DATA --output-dir OUT --num-epochs 20 --batch-size 8 \
  --dice-weight 0.5 --freeze-text-encoder --no-freeze-vision-encoder \
  --optimizer adamw --weight-decay 0.05 --lr 1e-4 --backbone-lr 1e-5 --lr-scheduler cosine --warmup-ratio 0.05 \
  --grad-clip-norm 1.0 --validate-every 2 --save-every 10 \
  --expect dice_weight=0.5 --expect freeze_text_encoder=true --expect stage_attention=true --expect optimizer=adamw
```

- This is the B0 recipe (decision D017): AdamW (no decay on norms, biases, embeddings, α, γ), learning rate 1e-4 for new layers and 1e-5 for the vision encoder, cosine schedule with 5% warmup, gradient clipping 1.0. Without these flags the defaults are Adam at 1e-5, constant. `--batch-size` is global (split across GPUs); training uses FP16 autocast.
- `--expect KEY=VALUE` stops before training if the resolved recipe differs. Resuming (`--resume-checkpoint`) refuses a checkpoint with a different recipe or architecture.
- Outputs: `run_config_from_epoch<E>.json`, `history.jsonl` (loss, learning rate, seconds, seconds waiting for data), `val_curve.jsonl` (validation mIoU) and `checkpoint_epoch<E>.pt` (every `--save-every` epochs and always at the last epoch; written atomically). Checkpoints store the architecture and the recipe.

## Evaluate

```bash
uv run eval.py --checkpoint OUT/checkpoint_epoch20.pt --split val --data-root DATA --output-dir OUT/eval_val_ep20
torchrun --nproc_per_node 2 eval.py ...   # same result, images split across GPUs
```

Metric: logits resized bilinearly to each image's native resolution, thresholded at 0, per-image IoU (two empty masks score 1), averaged over images. Validation during training uses the same code, so the logged validation mIoU equals `eval.py` on the saved checkpoint. Writes `metrics.json` (`summary`, `per_sample`); `--save-masks` also saves the predicted masks. Existing results are never overwritten. Choose checkpoints on validation only.

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Offline (tiny random encoders, cached CLIP tokenizer), including two-process CPU runs of training and evaluation.

## Origin

Research fork of [yjh9929/VizWiz-VQA-Grounding](https://github.com/yjh9929/VizWiz-VQA-Grounding) (CVPR 2025 VizWiz workshop spotlight), via `exp/baseline-fixes`. Licensed under [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/).

## Kaggle: B0, seed 42 (first run)

Enable Internet and **two T4 GPUs**. Attach the VizWiz dataset (`vizwiz-vqa-grounding`). Run the cells in order. Every cell stops loudly when something is wrong; send its output instead of working around it. The recipe is decision D017 in the project log.

| Cell | What | Time (estimate) |
|---|---|---|
| 1 | Setup: clone `ocrg/main`, install, fast HF download, cache CLIP, check GPUs and data | ~5 min |
| 2 | Offline tests | ~2 min |
| 3 | Memory probe: 4 images per GPU fit? Speed? | ~5 min |
| 4 | Train B0 (20 epochs) | from Cell 3's estimate |
| 5 | Evaluate epochs 10 and 20 on val and test | ~20 min |
| Z | ZIP the results (no checkpoints) | ~1 min |

### Cell 1: setup

```python
!git clone --depth 1 --single-branch --branch ocrg/main https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!git log --oneline -1
%env UV_PROJECT_ENVIRONMENT=/root/ocrg-venv
!uv sync --frozen --quiet
```

`UV_PROJECT_ENVIRONMENT` puts the virtual environment (several GB with CUDA torch) outside `/kaggle/working`, keeping the 20 GB working disk for checkpoints; every later `uv run` uses it. It is rebuilt in each new session (rerun Cell 1, e.g. after a resume).

Faster Hugging Face downloads (optional; needs a Kaggle secret `HF_TOKEN`). `uv pip install` adds the packages to the environment without editing `pyproject.toml` or `uv.lock`, so the checkout stays at the recorded commit (`uv add` would change both files):

```python
!uv pip install -q wrapt hf_transfer
%env HF_HUB_ENABLE_HF_TRANSFER=1
%env HF_XET_HIGH_PERFORMANCE=1

import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

Download CLIP once (~1.7 GB). Every later cell starts two processes that load the model; with the cache filled they read it from disk instead of the network:

```python
!uv run --frozen python -c "from huggingface_hub import snapshot_download; print(snapshot_download('openai/clip-vit-large-patch14-336', allow_patterns=['*.json', '*.txt', 'pytorch_model.bin']))"
```

```python
import json, os, subprocess
from pathlib import Path

os.environ['DATA_ROOT'] = '/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz'
os.environ['RUN_NAME'] = 'ocrg_b0_s42'
os.environ['OUT'] = f"outputs/{os.environ['RUN_NAME']}"
os.environ['OMP_NUM_THREADS'] = '1'

data = Path(os.environ['DATA_ROOT'])
for split, n in (('train', 6494), ('val', 1131), ('test', 2373)):
    records = json.loads((data / f'{split}_grounding.json').read_text())
    masks = len(list((data / 'binary_masks_png' / split).glob('*.png')))
    assert len(records) == n and masks == n, (split, len(records), masks)
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout)
assert not Path(os.environ['OUT']).exists(), 'Output folder exists: choose a new RUN_NAME'
```

### Cell 2: offline tests

Downloads the CLIP tokenizer once, then runs the 24 tests (tiny random models, including a two-process run).

```bash
%%bash
uv run --frozen python -c "from transformers import CLIPTokenizer; CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14-336')"
HF_HUB_OFFLINE=1 uv run --frozen python -m unittest discover -s tests -p "test_*.py" 2>&1 | tail -5
```

The last line must be `OK`.

### Cell 3: memory probe

Runs the real B0 training path (both GPUs, real data, mixed precision, full optimiser) for 30 steps with 4 images per GPU, then reports peak memory and speed. No checkpoint is written.

```bash
%%bash
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py \
  --data-root "$DATA_ROOT" --output-dir outputs/probe_bs8 --batch-size 8 --num-workers 4 --max-steps 30 \
  --dice-weight 0.5 --freeze-text-encoder --no-freeze-vision-encoder \
  --optimizer adamw --weight-decay 0.05 --lr 1e-4 --backbone-lr 1e-5 --lr-scheduler cosine --warmup-ratio 0.05 \
  --grad-clip-norm 1.0 2>&1 | grep -E "Probe:|Error|error" 
```

```python
probe = json.loads(Path('outputs/probe_bs8/probe.json').read_text())
peak, total = max(probe['peak_memory_gib_per_gpu']), probe['gpu_memory_gib']
steps_per_epoch = 6494 // 8 + (6494 % 8 > 0)
hours = probe['seconds_per_step'] * steps_per_epoch * 20 / 3600
print(f'peak {peak:.1f} of {total:.1f} GiB per GPU; {probe["seconds_per_step"]:.2f} s/step; '
      f'data wait {probe["data_wait_seconds"]:.0f} s of {probe["seconds"]:.0f} s; training ~{hours:.1f} h + validation')
os.environ['BATCH'] = '8' if peak <= total - 1.0 else '4'
print('Use batch size', os.environ['BATCH'])
```

- **Out-of-memory error, or `Use batch size 4`:** rerun the probe with `--batch-size 4 --output-dir outputs/probe_bs4`, then set `os.environ['BATCH'] = '4'`.
- **Training estimate above ~10.5 h:** the run will not fit in one 12-hour session. Run Cell 4 anyway; it saves a checkpoint at epoch 10, and the resume note below continues from there.
- **Send the printed line in any case:** the speed and data-wait numbers go into the decision log.

### Cell 4: train B0

```bash
%%bash
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py \
  --data-root "$DATA_ROOT" --output-dir "$OUT" --num-epochs 20 --batch-size "$BATCH" --num-workers 4 \
  --dice-weight 0.5 --freeze-text-encoder --no-freeze-vision-encoder \
  --optimizer adamw --weight-decay 0.05 --lr 1e-4 --backbone-lr 1e-5 --lr-scheduler cosine --warmup-ratio 0.05 \
  --grad-clip-norm 1.0 --validate-every 2 --save-every 10 --seed 42 \
  --expect text_mode=question --expect dice_weight=0.5 --expect freeze_text_encoder=true \
  --expect freeze_vision_encoder=false --expect optimizer=adamw --expect lr_scheduler=cosine \
  --expect decoder=fusion --expect attention_residual=true --expect stage_attention=true \
  --expect grad_clip_norm=1.0 --expect world_size=2 2>&1 | grep -v "it/s\]"
```

Progress: one line per epoch (loss, seconds, seconds waiting for data) and validation mIoU every 2 epochs. Checkpoints (about 4 GB each) are written at epochs 10 and 20.

**If the session ends before epoch 20:**
1. Keep `checkpoint_epoch10.pt`: save the notebook version with its output, and attach that output to a new session.
2. Rerun Cells 1–2.
3. Run Cell 4 again with `--resume-checkpoint /kaggle/input/<attached output>/.../checkpoint_epoch10.pt` added. Everything else stays identical; the run refuses to resume if anything in the recipe differs.

### Cell 5: evaluate

Validation chooses between epochs 10 and 20; test is reported for both but chooses nothing.

```bash
%%bash
for ep in 10 20; do
  for split in val test; do
    uv run --frozen torchrun --standalone --nproc_per_node=2 eval.py --checkpoint "$OUT/checkpoint_epoch${ep}.pt" \
      --split "$split" --data-root "$DATA_ROOT" --output-dir "$OUT/eval_${split}_ep${ep}" --batch-size 8 --num-workers 4 \
      2>&1 | grep "mean IoU"
  done
done
```

### Cell Z: ZIP results (no checkpoints)

```python
from zipfile import ZIP_DEFLATED, ZipFile

archive = Path('/kaggle/working') / f"{os.environ['RUN_NAME']}.zip"
with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as bundle:
    for root in (Path(os.environ['OUT']), Path('outputs/probe_bs8'), Path('outputs/probe_bs4')):
        for path in sorted(root.rglob('*')) if root.exists() else []:
            if path.is_file() and path.suffix.lower() not in {'.pt', '.tmp', '.zip'}:
                bundle.write(path, path.relative_to('outputs'))
print('Download:', archive)
```

Send `ocrg_b0_s42.zip`. It holds the run config, history, validation curve, `probe.json` and the four `metrics.json` files, and is analysed read-only.
