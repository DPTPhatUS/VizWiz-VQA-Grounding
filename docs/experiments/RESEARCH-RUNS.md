# Evidence location and extent experiment

Branch: `exp/evidence-extent`, derived from shared corrected controls commit `48ef79e`.
Run from this worktree, using the repository environment and an absolute data path. Legacy entry points remain unchanged.

The compact question-only grounder is wrapped by two learned tokens. They cross-attend to padding-masked question features, then visual features. Their dynamic visual maps represent broad location support and an extent correction. Final logits equal compact logits plus the upsampled location map and extent residual. The location target is the bounding rectangle of the annotated mask; an empty mask has an empty support target. Final-mask BCE (plus optional Dice), support BCE, and squared normalized-area error train the model. The area term includes empty masks.

At production CLIP-L dimensions (text 768, visual 1024), width 64 adds **148,545 parameters**, counted using actual standalone PyTorch head modules without downloading encoders. Tiny width 16 adds 2,833. The extent correction is spatially low-resolution; the compact decoder retains the finer mask path. No evidence shows accuracy improvements yet.

```bash
python train_research.py --data-root /path/to/data/vizwiz --text-mode question --num-epochs 100 --seed 42 --output-dir outputs-extent-s42
python train_research.py --data-root /path/to/data/vizwiz --init-checkpoint /path/to/controls/best.pt --num-epochs 100 --seed 42 --output-dir outputs-extent-init-s42
python eval_research.py --data-root /path/to/data/vizwiz --checkpoint outputs-extent-s42/best.pt --dataset test --output-dir results-extent-s42
```

Only compact architecture and question text are accepted. Validation and evaluation use no answers or pair artifacts. With an explicit `question_text` field, direct model calls ignore alternative `text`/`answer_text` views. Bare `text` is treated as the caller-provided question.

Defaults: `--support-weight 0.2 --area-weight 0.1 --dice-weight 0 --evidence-width 64` (tiny defaults to width 16). Ablations should use separate directories: `--support-weight 0`, `--area-weight 0`, both zero, and `--evidence-width 32|128`. Production widths must be positive multiples of four. Compare against the compact control branch with identical preprocessing, seed, initialization policy, and segmentation loss. Width changes require a fresh run or matching weights; strict resume rejects them. Repeat seeds 42, 43, 44. Parameter counts are not latency measurements.

## Optional verified paired questions

Use paired losses only when independently verified same-image question annotations exist, with **each question's own mask**. The manifest is a JSON list; mask paths resolve relative to the manifest:

```json
[
  {
    "filename": "VizWiz_train_00000000.jpg",
    "question1": "Where is the label?",
    "question2": "Where is the whole container?",
    "mask1": "verified/label.png",
    "mask2": "verified/container.png",
    "relation": "change"
  },
  {
    "filename": "VizWiz_train_00000001.jpg",
    "question1": "Where is the label?",
    "question2": "Which region contains the label?",
    "mask1": "verified/label2.png",
    "mask2": "verified/label2_paraphrase.png",
    "relation": "same"
  }
]
```

These are schema illustrations, not supplied annotations. Never infer new masks from a rewritten question. `same` declares a verified equivalent-region pair, and original-resolution binary masks must match exactly; mismatches are rejected before resizing. Both mask dimensions must match the image. Filenames must belong to the training split. Manifest questions and paths must be valid. The loader samples one declared pair per available image each epoch; unpaired images retain ordinary single-question supervision. The two explicit paired masks replace the ordinary mask for that pair. Synthetic test fixtures are not valid research annotation sources.

```bash
python train_research.py --data-root /path/to/data/vizwiz --pairs /path/to/verified/pairs.json --pair-delta-weight 0.5 --pair-consistency-weight 0.5 --num-epochs 100 --output-dir outputs-extent-paired-s42
```

Either positive pair weight requires `--pairs`. Defaults for both are zero. Providing a manifest also trains both question/mask views even with both pair penalties zero, so that setting is the additional-supervised-data control. Signed delta loss is MSE between `sigmoid(logits2)-sigmoid(logits1)` and `mask2-mask1`, retaining addition/removal signs. Same-region consistency is MSE between predictions only for declared `same` pairs. Each available question gets segmentation, support and area supervision; a single concatenated forward processes all views. All loss terms sum contributions over views or eligible pairs and divide by the number of original images. Thus equal-sized distributed shards retain the same weighting when their pair counts differ; unpaired training keeps its ordinary loss scale. Paired examples therefore increase training memory and compute. No paired data is required for the default experiment or deployment.

## Checkpoints and verification

```bash
python train_research.py --data-root /path/to/data/vizwiz --output-dir outputs-extent-s42 --resume-checkpoint outputs-extent-s42/last.pt --num-epochs 150
```

Resume requires the original training arguments, including all objective weights and pair manifest, except supported runtime and epoch overrides. Resume must retain the checkpoint output directory to preserve best-checkpoint history. `--num-epochs` is the final epoch. `--init-checkpoint` accepts matching extent weights or an exact compatible corrected compact control checkpoint; it starts a new optimizer/run. Original-resolution mean per-image IoU selects `best.pt`; all provenance and checkpoint semantics of shared controls apply. Only load trusted checkpoints.

Exact offline verification command executed in this worktree:

```bash
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/vizwiz-mpl /home/dptphat/research/VizWiz-VQA-Grounding/.venv/bin/python -m unittest discover -s tests -v
```

Result: 31 tests collected, 30 passed, one intentional inherited controls-CLI skip. New coverage checks both token gradients, answer immunity, question conditioning, padding invariance, empty masks/area, envelope geometry, signed changes, verified same masks at original resolution, width and checkpoint incompatibility, controls initialization, mixed unpaired/change/same fixtures, tiny CPU CLI training, in-process resume and original-size evaluation. The pair files are deleted before evaluation to verify training-artifact independence.

No real-data training, encoder downloads, CUDA execution, multi-rank DDP execution, GPU memory/throughput measurements, or accuracy/novelty claims are included. Tiny offline models are test fixtures. A regression verifies combined-batch loss and all components equal the mean of two equal-sized original-image shards with unequal pair counts, for both change and same-region pairs. Shared provenance now hashes both manifest content and referenced mask contents, rejecting changed annotations on resume.

Strict continuation requires the checkpoint to reside in the same `--output-dir`; use weight initialization for a new run/directory. Provenance hashes both JSON annotations and their binary mask contents (including paired masks). CPU checkpoints can resume on CUDA with a fresh AMP scaler; CUDA execution itself is unverified here.

Strict resume accepts `last.pt` only, avoiding stale best-checkpoint selection when rewinding within a run. Start a new initialized run to reuse older/best weights.
