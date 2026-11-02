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
  Conv1x1 → 1 logit map (384x384)
Loss: BCE + 0.5 · per-image soft Dice, on logits resized to the 336 px training mask
```

Trainable: 310.3M (vision encoder 303.5M, attention 5.0M, decoder 1.85M); text encoder 123.1M frozen. The upstream architecture remains available with `--decoder unet --no-attention-residual`.

## Data

Point `--data-root` at a folder with `{train,val,test}_grounding.json`, `{train,val,test}/` images and `binary_masks_png/{train,val,test}/`. Images are EXIF-rotated, resized to 336×336 without keeping the aspect ratio (bicubic, antialiased) and CLIP-normalised; training masks use nearest resizing (no mask loses its foreground at 336 px). By default only the question is read (the task input). `--text-mode qa` also feeds the most common answer (`Q: … A: …`) for ablations only: it is recorded in the recipe, `eval.py` follows the checkpoint's mode unless `--text-mode` overrides it (e.g. a qa-trained model scored with the question only), results report `answer_in_input`, and qa mode fails on records without an answer (the test split). A missing or misaligned mask is an error.

## Train

```bash
torchrun --nproc_per_node 2 train.py --data-root DATA --output-dir OUT --num-epochs 20 --batch-size 4 \
  --dice-weight 0.5 --freeze-text-encoder --no-freeze-vision-encoder --validate-every 2 --save-every 10 \
  --expect dice_weight=0.5 --expect freeze_text_encoder=true --expect decoder=fusion
```

- `--batch-size` is global (split across GPUs); training uses FP16 autocast and Adam (lr 1e-5) by default.
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
