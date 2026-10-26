# Mask-conditioned answer supervision

Branch: `exp/mask-conditioned-answer`, created from `exp/answer-token-supervision` at `cb805f8`. Runs train fresh for 20 epochs from pretrained CLIP weights without requiring a teacher checkpoint or external VLM reader.

## Architecture

The main question-only grounding pyramid is unchanged (`CLIP I336 + detail CNN I768 + Q -> I768 mask logits`). A training-only mask-conditioned answer reader adds **739,587 parameters**. Standard evaluation removes the reader before moving the model to the GPU, so inference requires only the image and question.

A 2-layer, width-128 Transformer reader takes 32-channel decoder features at 192 resolution, area-pooled onto a fixed full-image 32x32 grid so ground-truth and predicted mask reads share identical spatial coordinates. Mask coverage enters the reader as a normalized log-attention prior. Answer byte teacher forcing uses a 64-byte limit, skipping missing, unanswerable, or overlength answers. Under AMP, the reader runs in float32 using PyTorch's `MATH` attention backend so variable-length answer queries and continuous log-attention priors support second-backward/mask-gradient computation cleanly on CUDA.

```text
Image + Q -> Pyramid -> Final logits -> segmentation + auxiliary mask losses
                |            |
                |            +-> soft mask -> fixed reader -> mask-answer hinge
                |                               ^
                +-> detached decoder feature ---+
                |
                +-> feature + GT mask -> trainable reader -> answer CE
                +-> feature + full mask -> no-gradient reader reference
```

For each image during training:

- `C_gt`: Answer cross-entropy conditioned on the ground-truth mask, updating both the reader and shared pyramid features.
- `C_full`: Detached answer cross-entropy conditioned on a uniform full-image mask.
- `C_pred`: Answer cross-entropy conditioned on the predicted soft mask. Reader parameters and decoder features are detached via `torch.func.functional_call` (without duplicating the module or mutating DDP `requires_grad` flags), so gradients from `C_pred` flow strictly into the predicted mask logits.
- `reliable = valid_GT_answer_and_region and (C_full - C_gt > reader_margin)`.
- `L_mask = reliable * relu(C_pred - stopgrad(C_gt) - mask_answer_tolerance)`.
- Total loss combines the pyramid segmentation and auxiliary mask losses with `scheduled_answer_weight * mean(C_gt)` and `scheduled_mask_answer_weight * mean(L_mask)`.

Batch reductions divide by the total number of images in the batch rather than the count of valid or reliable samples, preserving exact gradient scaling across DDP ranks and partial accumulation windows. The detached reliability gate activates mask guidance only on samples where the ground-truth mask yields a lower answer loss than the full image by at least `reader_margin`, and the hinge loss stops pushing once `C_pred` comes within `mask_answer_tolerance` of `C_gt`.

## Kaggle setup

Enable Internet and two GPUs, and use a clean `outputs` directory.

Python cell:

```python
!git clone --single-branch --branch exp/mask-conditioned-answer https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen
```

Python cell:

```python
import json
import os
from pathlib import Path

os.environ["DATA_ROOT"] = "/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz"
os.environ["RUN_ROOT"] = "outputs"
os.environ["RUN_NAME"] = "mask_conditioned_answer_ep20_seed42"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
root = Path(os.environ["RUN_ROOT"])
(root / "configs").mkdir(parents=True, exist_ok=True)
config = {
    "overlap_loss": "dice", "overlap_weight": 0.5, "fp_weight": 0.7,
    "aux96_weight": 0.1, "aux192_weight": 0.2, "aux384_weight": 0.3,
    "answer_weight": 0.01, "answer_warmup_epochs": 0, "answer_ramp_epochs": 1,
    "mask_answer_weight": 0.01,
    "mask_answer_warmup_epochs": 5, "mask_answer_ramp_epochs": 5,
    "reader_margin": 0.05, "mask_answer_tolerance": 0.05,
}
(root / "configs/answer.json").write_text(json.dumps(config, indent=2))
```

Precedence follows **JSON config > explicit CLI flags > defaults**. Loss weights are configured via JSON, while optimizer and encoder-freezing options can be set via either CLI flags or JSON (`fp_weight` applies only when `overlap_loss="tversky"`).

## Preflight

Both ranks must pass preflight. Preflight exercises the full loss graph (including active mask-answer guidance) for 3 steps and discards parameter updates without writing checkpoints.

```bash
%%bash
set -euo pipefail
uv run --frozen torchrun --standalone --nproc_per_node=2 preflight.py \
    --data-root "$DATA_ROOT" \
    --batch-size 2 --num-workers 0 --steps 3 \
    --gradient-checkpointing \
    --freeze-vision-encoder False --freeze-text-encoder True \
    --config "$RUN_ROOT/configs/answer.json"
```

## Train 20 epochs

A global microbatch of 2 places 1 image per GPU, and `--gradient-accumulation-steps 2` gives an effective batch size of 4 (if memory permits, verify `--batch-size 4` in preflight and run with `--batch-size 4 --gradient-accumulation-steps 1`). The vision encoder is fine-tuned while the text encoder stays frozen.

```bash
%%bash
set -euo pipefail
uv run --frozen torchrun --standalone --nproc_per_node=2 train.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$RUN_ROOT" \
    --num-epochs 20 \
    --batch-size 2 --gradient-accumulation-steps 2 \
    --num-workers 2 \
    --lr 0.00001 --optimizer adam --weight-decay 0 \
    --lr-scheduler none --warmup-ratio 0 \
    --freeze-vision-encoder False --freeze-text-encoder True \
    --gradient-checkpointing \
    --seed 42 --save-every 10 \
    --config "$RUN_ROOT/configs/answer.json"
```

Epochs 1 to 5 train segmentation and the ground-truth answer reader (`answer_weight=0.01`, `mask_answer_weight=0.0`). Epochs 6 to 10 linearly ramp `mask_answer_weight` from `0.002` to `0.01`, and epochs 11 to 20 train with the full `0.01` weight on both terms.

## Four evaluations

Run validation and test evaluations at epochs 10 and 20 into separate directories. `eval.py` reports native-resolution per-image `mean_iou` and dataset-wide `global_iou` (scoring the test split requires local test ground-truth masks). Select checkpoints on the validation split only.

```bash
%%bash
set -euo pipefail
for epoch in 10 20; do
    for split in val test; do
        uv run --frozen eval.py \
            --checkpoint "$RUN_ROOT/checkpoint_epoch${epoch}.pt" \
            --dataset "$split" --data-root "$DATA_ROOT" \
            --batch-size 4 --num-workers 2 \
            --output-dir "$RUN_ROOT/eval_${split}_ep${epoch}"
    done
done
```

## Diagnostics in the same session

`diagnose.py` evaluates the exported validation masks on CPU. `diagnose_answers.py` runs the grounding forward pass with image + question, then compares teacher-forced byte cross-entropy and accuracy under ground-truth masks, predicted masks, full-image masks, and zeroed visual features.

```bash
%%bash
set -euo pipefail
for epoch in 10 20; do
    uv run --frozen diagnose.py \
        --predictions "$RUN_ROOT/eval_val_ep${epoch}" \
        --data-root "$DATA_ROOT" \
        --output-dir "$RUN_ROOT/diagnostic_ep${epoch}"
    uv run --frozen diagnose_answers.py \
        --checkpoint "$RUN_ROOT/checkpoint_epoch${epoch}.pt" \
        --data-root "$DATA_ROOT" \
        --batch-size 4 --num-workers 2 \
        --output-dir "$RUN_ROOT/answer_diagnostic_ep${epoch}"
done
```

`history.jsonl` logs `answer_loss`, byte accuracy, `answer_valid_fraction`, `mask_answer_loss`, `reader_reliable_fraction`, `mask_answer_active_fraction`, `reader_visual_gain`, `predicted_answer_loss`, and the scheduled weights at each epoch.

## ZIP without checkpoints

Python cell:

```python
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(os.environ["RUN_ROOT"])
archive = root / (os.environ["RUN_NAME"] + ".zip")
with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() not in {".pt", ".zip", ".tmp"}:
            bundle.write(path, path.relative_to(root))
print(archive)
```

Download the `.pt` checkpoint files separately from Kaggle Outputs.

## Controls and checkpoint compatibility

- **Default run**: Uses both `answer_weight=0.01` and `mask_answer_weight=0.01` as configured above.
- **Reader-only control (`mask_answer_weight=0`)**: Trains the fixed-grid GT reader and shared features without pushing predicted mask logits through the hinge loss.
- **No-answer control (`answer_weight=0`, `mask_answer_weight=0`)**: Disables both answer losses (note that `mask_answer_weight` must not be enabled when `answer_weight=0`, since an untrained reader cannot provide meaningful mask guidance).
- **Resuming**: Pass `--resume-checkpoint` with a full checkpoint from this branch, keeping the same architecture, configuration, and annotation hashes while setting `--num-epochs` to the target total epoch count.
