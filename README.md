# Mask-conditioned answer supervision

Branch: `exp/mask-conditioned-answer`, from token supervision commit `cb805f8`. Start a fresh 20-epoch experiment from pretrained CLIP. No teacher checkpoint, external reader or old experiment checkpoint is required.

## Architecture

The main question-only pyramid is unchanged: CLIP I336 + detail CNN I768 + Q -> I768 mask logits. The training-only reader adds **739,587 parameters**. Standard eval removes it before GPU transfer; inference requires only I+Q.

A two-layer width-128 Transformer reads 32-channel decoder features at scale 192, area-pooled to a fixed full-image 32x32 grid. GT and predicted reads use identical positions, without GT-dependent cropping. Mask coverage becomes a normalized log attention prior. Byte teacher forcing uses a 64-byte limit; missing, unanswerable and overlength answers are skipped, not truncated. Reader math stays float32 under AMP and uses the reference MATH attention backend, avoiding fused CUDA backward alignment failures with variable-length answer queries and differentiable mask priors. Backbone attention is unchanged.

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

For each image:

- `C_gt`: GT-region answer CE trains the reader and shared features.
- `C_full`: detached full-image answer CE.
- `C_pred`: predicted-region answer CE. Reader weights and input feature are detached, but mask gradients stay live. This uses `torch.func.functional_call`, with no extra model copy or mutation of DDP parameter flags.
- `reliable = valid_GT_answer_and_region AND (C_full - C_gt > reader_margin)`.
- `L_mask = reliable * relu(C_pred - stopgrad(C_gt) - mask_answer_tolerance)`.
- Total loss is original segmentation/auxiliary losses + scheduled `answer_weight * mean(C_gt)` + scheduled `mask_answer_weight * mean(L_mask)`.

Means count original images, not just valid/reliable examples, preserving DDP and partial accumulation weighting. The reliability decision is detached. It measures region informativeness, not guaranteed answer correctness. The hinge stops pushing when the predicted-region read reaches the GT reference within tolerance.

Normalization removes absolute mask mass as a reader cue, but cannot guarantee that expansion will never help. Segmentation still determines correct extent. Fixed-grid pooling preserves tiny GT coverage but cannot recover lost image detail. The decoder features are already Q-conditioned and may carry context beyond the selected region; this method is not proof of causal localization. Benefits remain experimental.

## Kaggle setup

Enable Internet and two GPUs. Use a fresh output directory. Local tests are CPU-only; GPU memory/runtime are not measured. All outputs remain directly under `outputs`.

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

Config > explicit flags > defaults. Loss controls belong in JSON; optimizer/freeze controls also support flags. `fp_weight` has no effect with Dice.

## Preflight

Both ranks must pass. Full-strength guidance is exercised even though normal training warms it up. Updates are discarded; no checkpoint is written. This checks execution/gradients, not accuracy.

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

Global microbatch 2 is one image/GPU; accumulation 2 gives effective batch 4. If memory permits, preflight with batch 4, then use batch 4 / accumulation 1. Vision fine-tunes; text is frozen. No internal per-epoch validation runs.

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

Epochs 1-5 train segmentation and the GT reader. Epochs 6-10 ramp mask guidance from 0.002 to 0.01; epochs 11-20 use full configured weights. The reader keeps learning from GT; only its use on predicted masks has detached weights.

## Four evaluations

Validation/test at epoch 10/20 go into four separate folders. Batch 4 is for eval; reduce if necessary. Reports include native mean IoU and separate global foreground IoU. Local test metrics require test GT. Select checkpoints/settings on validation, not test.

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

Mask analysis is CPU-only. Answer diagnostic reuses I+Q grounding features, then uses GT validation masks/answers for GT-region, predicted-region, full-image and zero-visual reads. Byte accuracy is teacher-forced, not free-generation VQA accuracy.

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

Inspect `history.jsonl`: answer loss/accuracy/valid fraction, `mask_answer_loss`, `reader_reliable_fraction`, `mask_answer_active_fraction`, `reader_visual_gain`, `predicted_answer_loss` and effective weights. Fractions count all images. A low reliable/active fraction means limited guidance, not automatically a bug. Check size-group IoU, precision/recall and mask area; answer loss alone is not evidence of success.

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

Download `.pt` checkpoints separately and keep the selected one.

## Controls and checkpoint compatibility

- First run defaults above.
- To isolate mask guidance, set only `mask_answer_weight=0` and train fresh. The fixed-grid GT reader still learns; this differs from the earlier cropped token-reader branch.
- To disable all answer supervision, set both `answer_weight=0` and `mask_answer_weight=0`. A fresh untrained reader must not guide masks with its own training disabled.
- Resume only this branch's full checkpoint with `--resume-checkpoint` and original settings. `--num-epochs` is the total desired epochs. Architecture/settings/annotation hashes must match.
- Old checkpoints remain valid on their original branches, but cannot resume this experiment. This branch needs fresh training.

CPU checks validate execution, gradient routing, checkpoint and evaluation behavior, not CUDA memory fit, runtime or improved accuracy. Full train + eval + diagnostics can still exceed a session's runtime budget.
