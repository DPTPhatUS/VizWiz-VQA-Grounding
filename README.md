# OCR scope ranking

Branch: `exp/ocr-scope-ranking`.

The previous selector nearly always retained B+5 and overrated its mask on small
targets. Its candidate scores tracked agreement with the parent more than agreement
with GT. This experiment removes that explicit reference shortcut and tests a
training recipe focused on wrong mask scope and harmful top choices.

| Run | `--architecture` | `--selector-recipe` | Change |
|---|---|---|---|
| A | independent | ranking | Shared scorer without reference/type/confidence shortcuts; previous quality+pairwise loss |
| B | independent | scope | Same scorer; scope counterexamples and decision-regret losses |

Both have **281,793 trainable parameters** and identical initialization. The frozen
parent has 433,217,037 parameters and is stored separately. A/B use the same parent,
data, seed, optimizer, global batch and matched dropout draws. B changes a training
bundle (augmentation AND ranking objective); it is not an isolated loss ablation.
Compared with the old selector, A changes the representation/context bundle; it
is not a one-feature causal ablation. These changes address observed failures;
accuracy gains are not established yet.

## Architecture and invariants

```text
image + question
    |
    +-- frozen B+5, one forward
    |       Q tokens [B,L,768]
    |       semantic features [B,16,384,384]
    |       detail features [B,16,768,768]
    |       base logits [B,1,768,768]
    |
    +-- existing frozen mobile OCR cache
            polygons / text / confidence / native RGB crops
                    |
            shared word encoder + Q attention + OCR context
                    |
            unchanged, GT-free mask bank (max N=24 OCR regions)
              base mask; individual polygons; nearby word groups; all-word union
                    |
            EACH mask uses the SAME representation:
              own region + surrounding ring + scene features
              OCR content covered by that mask + ALL Q tokens
              own geometry; no source ID or agreement with another mask
                    |
            shared quality head -> compare estimated qualities
                    |
            use OCR only if predicted gain over base >0.02
                    |
            native selected polygon union OR original base output
```

A small CNN combines reused semantic/detail features with RGB at 96x96. Native
OCR appearance remains available through the crop encoder. Quality descriptors
use binary-mask geometry, supersampled at 192 and pooled to 96. Parent sigmoid
confidence is not an input cue. The final base still uses the original logits
and native interpolation; only its descriptor is binarized.

The scorer has no base token, source ID, reference IoU, intersection fraction or
candidate/base disagreement feature. It still reuses parent features, which can
carry the parent's learned biases. Reference-free means no DIRECT comparison
features; it does not mean independence from the frozen feature extractor.

Tested contracts:
- Changing only base-mask geometry cannot change OCR hypothesis scores.
- Increasing base-logit magnitude without changing the mask cannot change scores.
- Identical mask descriptors receive identical scores regardless of bank slot.
- Reordering hypotheses only reorders their scores.

The candidate bank and native output protocol are unchanged from the previous
selection experiment. With N=24 there are at most 50 slots; invalid/duplicate padding
is ignored. A hypothesis may contain several words. OCR polygons are rasterized
at native resolution after selection; there is no 768px output detour for them.
The base uses its original bilinear-logit resize and zero threshold. Missing OCR
returns the exact base output. The 0.02 margin is fixed, not a calibrated safety
guarantee. Wrong switches can still damage large targets.

## Training recipes

Both use native GT mask IoU as detached quality targets, with train-only mask
perturbations applied before recomputing the base-slot target. They never use GT
or answers to choose a hypothesis at inference.

**A: ranking control**

```text
L_A = quality BCE + gap-weighted pairwise ranking
```

This matches the previous selector's loss mathematically (answer weight 0).
Half the base-slot masks receive shifts or morphology, as a training-quality
augmentation. The other half retain the original parent mask.

**B: scope-focused training**

```text
L_B = quality BCE + decision regret +2 * OCR retrieval regret
```

Half the base slots retain the original prediction. The other half, when OCR
exists, are replaced with a randomly sampled tight OCR region, its expanded
context rectangle (2--4x), or a union with another OCR region. Sampling uses only
OCR geometry/availability, not GT. An added region is not automatically labelled
wrong: the resulting native IoU determines its quality. These counterexamples
model confusion between a local answer word and broader context.

Decision regret focuses on the mask ACTUALLY chosen by the deployment policy,
including the margin. If it loses more than 0.05 IoU against a better available
hypothesis, a gap-weighted softplus pushes the better score above the wrong one.
Adding many easy low-quality masks does not dilute this term. The OCR-only term
trains retrieval even when the original parent is best; it is active only when
there is OCR evidence with native IoU >= 0.1. Ties and small quality differences are
not forced into an arbitrary winner.

Quality BCE balances overlapping/disjoint hypotheses inside each image. All
terms then average over ORIGINAL images, preserving DDP/accumulation when one
rank has no OCR. Recipe-specific augmentation uses the same RNG budget so A/B
OCR/text-dropout patterns remain matched. Constants and recipe are recorded in
checkpoints. Both runs disable the old answer auxiliary loss: it did not improve
held-out mask selection in the previous experiment.

The parent, its CLIP text/vision encoders and OCR remain frozen. Only the scorer
trains. No teacher checkpoint, new OCR cache, additional annotations or backbone
is needed. This is not out-of-fold calibration: parent training-distribution bias
can persist despite removing explicit shortcuts. It must be assessed from results.

## Kaggle: two fresh GPU sessions, 20 epochs each

Enable Internet and 2xT4. Attach VizWiz, the exact B+5 parent and the existing OCR
cache. Use these cells in each session, changing only `RUN_VARIANT` to A or B.
B starts independently; do not resume any A/B selector from the previous branch.

### Cell 1: setup

```python
!pip install -q uv
!git clone --single-branch --branch exp/ocr-scope-ranking https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen

import os
from pathlib import Path

os.environ["RUN_VARIANT"] = "A"  # Fresh second session: "B"
os.environ["SELECTOR_RECIPE"] = "ranking" if os.environ["RUN_VARIANT"] == "A" else "scope"
os.environ["DATA_ROOT"] = "/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz"
os.environ["PARENT_CKPT"] = "/kaggle/input/datasets/khoahoang0512/vizwiz-size-weighted-pyramid-cp/mild_focus/checkpoint_epoch5.pt"
os.environ["OCR_CACHE"] = "/kaggle/input/datasets/khoahoang0512/vizwiz-ocr-cache/ocr_cache"
os.environ["RUN_ROOT"] = "outputs"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

assert os.environ["RUN_VARIANT"] in ("A", "B")
assert Path(os.environ["PARENT_CKPT"]).is_file()
assert (Path(os.environ["OCR_CACHE"]) / "manifest.json").is_file()
```

The parent is **mild_focus/checkpoint_epoch5.pt (B+5)**. Do not substitute the raw
pyramid epoch20, natural/A, interpolation/crop model or any old OCR adapter.
Parent SHA256, OCR protocol and native training-supervision digest are checked.

### Cell 2: train

```bash
%%bash
set -euo pipefail
uv run torchrun --standalone --nproc_per_node=2 train_ocr.py \
    --parent-checkpoint "$PARENT_CKPT" \
    --data-root "$DATA_ROOT" \
    --cache-root "$OCR_CACHE" \
    --output-dir "$RUN_ROOT" \
    --architecture independent \
    --selector-recipe "$SELECTOR_RECIPE" \
    --evidence-mode full \
    --answer-weight 0 \
    --min-gain 0.02 \
    --num-epochs 20 \
    --batch-size 4 \
    --gradient-accumulation-steps 1 \
    --num-workers 2 \
    --lr 0.0001 \
    --optimizer adamw \
    --weight-decay 0.01 \
    --max-candidates 24 \
    --ocr-dropout 0.2 \
    --text-dropout 0.2 \
    --save-every 5 \
    --seed 42 \
    --amp
```

Batch 4 is GLOBAL:2 per GPU. Batch 2 with accumulation 2 preserves effective global
batch 4 if needed. LR is constant; the parent is frozen. No validation runs inside
training. `--save-every 5` saves 5/10/15/20. Each production scorer checkpoint with
AdamW state is approximately 3.25 MiB; all four require about 13 MiB, excluding results
and dependency caches. The large parent is NOT saved in these files.

### Cell 3: evaluate every saved checkpoint and diagnose selection

```bash
%%bash
set -euo pipefail
for EPOCH in 5 10 15 20; do
    uv run eval_ocr.py \
        --parent-checkpoint "$PARENT_CKPT" \
        --checkpoint "$RUN_ROOT/checkpoint_epoch$EPOCH.pt" \
        --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
        --dataset val --view full --batch-size 4 --num-workers 2 \
        --output-dir "$RUN_ROOT/eval_val_ep$EPOCH"
    uv run diagnose_evidence_selection.py \
        --predictions "$RUN_ROOT/eval_val_ep$EPOCH" \
        --output-dir "$RUN_ROOT/selection_diagnostic_ep$EPOCH"
done
```

Evaluation includes native mean/global IoU, size groups, all predicted/actual
hypothesis qualities and native PNGs. Diagnostics report base-quality bias,
quality MAE, OCR ranking mistakes, missed retrieved gains and harmful switches.
GT-based oracle/regret values are diagnostic only. No GT enters the selection
policy. Use `eval_ocr.py`, not the parent's `eval.py`, for scorer checkpoints.

### Cell 4: select by validation, evaluate test and robustness

```python
import json
import subprocess

root = Path(os.environ["RUN_ROOT"])
best = max((5, 10, 15, 20), key=lambda epoch: (
    json.loads((root / f"eval_val_ep{epoch}/metrics.json").read_text())["summary"]["mean_iou"],
    -epoch,
))
for split, view in [
    ("test", "full"),
    ("val", "base"),
    ("val", "missing"),
    ("val", "geometry"),
    ("val", "shuffled_text"),
    ("val", "evidence"),
]:
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
    "uv", "run", "diagnose.py", "--data-root", os.environ["DATA_ROOT"],
    "--predictions", str(root / f"eval_val_ep{best}"),
    "--output-dir", str(root / "mask_diagnostic"),
], check=True)
(root / "selection.json").write_text(json.dumps({
    "variant": os.environ["RUN_VARIANT"],
    "selector_recipe": os.environ["SELECTOR_RECIPE"],
    "selected_epoch": best,
    "selection_metric": "validation_mean_iou", "tie_break": "earlier_epoch",
}, indent=2))
print(f"Selected epoch {best}, recipe {os.environ['SELECTOR_RECIPE']}")
```

Only validation chooses the checkpoint; test never chooses the margin or recipe.
`base`/`missing` must match. `geometry` removes recognized-string embeddings but
keeps crop RGB and polygons; `shuffled_text` permutes strings across images.
`evidence` forces the highest-scored OCR hypothesis, with base fallback only when
OCR is absent. It is an ablation, not normal inference. Unchanged final scores
under an intervention do not prove internal OCR scores are unchanged.

### Cell 5: download results without checkpoint files

```python
import zipfile

archive = Path(f"ocr_scope_{os.environ['RUN_VARIANT']}_ep20_seed42.zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() not in (".pt", ".tmp", ".zip"):
            bundle.write(path, path.relative_to(root))
print(archive.resolve())
```

Download `ocr_scope_A_ep20_seed42.zip` / `ocr_scope_B_ep20_seed42.zip` and keep the
four small `.pt` checkpoints separately. The ZIP contains histories, configs,
metrics, native masks, selection and diagnostics; all `.pt`, `.tmp` and ZIP files
are excluded. Output stays directly under `outputs/`.

## Resume and configuration

For the SAME recipe, add `--resume-checkpoint /kaggle/input/YOUR_DATASET/checkpoint_epoch10.pt`
to Cell 2 in a fresh session, with a new empty output directory. Keep total
`--num-epochs 20`, the exact parent, cache, training annotations, recipe and training
settings. Do not copy previous files into the output directory before training.
After training, bring earlier checkpoints/evaluation folders back only if needed
for Cell 3, or adjust its epoch list to the checkpoints present.

`--config path.json` supports flat overrides: **config > explicit CLI > default**;
paths remain on CLI. Normal commands need no JSON. Relevant defaults:

| Option | Default | Purpose |
|---|---|---|
| `--architecture` | proposal | Explicitly set independent for these runs |
| `--selector-recipe` | ranking | ranking control or scope training bundle |
| `--min-gain` | 0.02 | Fixed predicted-quality gain required to replace base |
| `--max-candidates` | 24 | OCR cap, before deterministic grouping |
| `--batch-size`, `--gradient-accumulation-steps` |4,1| Global microbatch and accumulation |
| `--lr`, `--optimizer`, `--weight-decay` |1e-4,adamw,0.01| Scorer optimization |
| `--ocr-dropout`, `--text-dropout` |0.2,0.2| Train-only dropout |
| `--num-epochs`, `--save-every`, `--seed` |20,5,42| Total epochs, checkpoint interval, seed |
| `--amp` / `--no-amp`, `--num-workers` |on,2| CUDA precision, workers per rank |

Independent scorers do not use answer_weight or the legacy selection/local/route/
evidence loss weights. A nonzero answer_weight is rejected for this architecture.
The old residual/proposal/pyramid/hypotheses models, defaults and checkpoint
versions remain supported. New scorers use a separate version; recipe changes
cannot be silently resumed into an existing run.

## Verification and limits

Local checks cover source-independent scoring, identical-mask/permutation
contracts, confidence invariance, regret-loss gradients, native output, no GT or
answer leakage, empty-OCR ranks, strict checkpoint round trips, DDP training,
accumulation, resume, all evaluation views and old-checkpoint compatibility.
Production-shape CPU BF16 forward/backward and checkpoint storage are checked.
These do not establish CUDA/T4 throughput, VRAM headroom or segmentation gains.

The adapter accepts different feature/text dimensions; another backbone still
needs an appropriate wrapper and actual experiments. Raw OCR cannot recover
regions missing from its proposals. Grouping can miss long/multiline answers.
Shared scoring can still learn visual or area priors; scope counterexamples are
synthetic and may not match real failures. Evaluate small and medium gains together
with large-region damage, calibration and final mIoU before expanding the model.
