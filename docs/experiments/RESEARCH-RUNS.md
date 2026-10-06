# Gain-guided high-resolution refinement

Branch: `exp/gain-guided-refinement`, based on `exp/grounding-controls`. Historical `train.py` and `eval.py` are unchanged. Run these commands inside this worktree and supply absolute data/checkpoint paths when sharing data with another branch. Train the corrected compact question-only coarse checkpoint on the controls branch first. Historical checkpoints or answer-trained controls are not accepted as a frozen coarse reference.

```bash
python train_research.py --data-root /path/to/data/vizwiz --stage refiner --init-checkpoint /path/to/controls/best.pt --detail-size 672 --crop-size 336 --num-epochs 30 --seed 42 --output-dir outputs-refiner-s42
python train_research.py --data-root /path/to/data/vizwiz --stage router --init-checkpoint outputs-refiner-s42/best.pt --detail-size 672 --crop-size 336 --num-epochs 30 --seed 42 --output-dir outputs-router-s42
python eval_research.py --data-root /path/to/data/vizwiz --checkpoint outputs-router-s42/best.pt --dataset test --policy gain --budget 2 --output-dir results-gain-b2-s42
```

Both stages use question-only text. The coarse reference stays frozen and in evaluation mode. Stage `refiner` trains one uniformly sampled candidate per image with crop BCE plus optional `--dice-weight 1`; its default validation policy is `fixed`. Stage `router` freezes the trained refiner, evaluates every candidate independently against the same coarse prediction, and regresses each image's actual change in mask IoU with MSE. Router labels use the aligned detail-resolution ground-truth mask, not predicted uncertainty. Validation and deployment access only RGB and question features; there is no oracle policy.

The candidate pool contains 13 windows: the four cells of a 2×2 grid and nine cells of a 3×3 grid. Each scale covers the whole image, including right/bottom boundaries for nondivisible dimensions. Raw `detail_image` and `detail_mask` are resized directly from the source RGB/mask with shared geometry. A shallow three-convolution refiner consumes normalized crop RGB, cropped coarse logits, and pooled frozen visual/question context. It produces a residual on a 336×336 crop canvas. Residuals are scattered to the 672×672 detail canvas; overlaps are averaged and all untouched pixels preserve the coarse logits. The full mask is then resized to original dimensions for scoring.

The router reads coarse visual/question context, crop coordinates, mean foreground probability, uncertainty, and foreground fraction. It is scored before any detail crops are refined. Selected crops are executed sequentially, so actual refiner work is bounded by the chosen budget. The production head dimensions (1024 visual + 768 text, hidden width 64) add **190,977 refiner parameters + 115,265 router parameters = 306,242**. The frozen compact coarse model is additional. These counts were measured by instantiating the standalone heads without downloading CLIP.

Evaluate the same trained router checkpoint for each policy and budget:

```bash
python eval_research.py --data-root /path/to/data/vizwiz --checkpoint outputs-router-s42/best.pt --policy uncertainty --budget 1 --output-dir results-uncertainty-b1-s42
python eval_research.py --data-root /path/to/data/vizwiz --checkpoint outputs-router-s42/best.pt --policy random --budget 4 --output-dir results-random-b4-s42
python eval_research.py --data-root /path/to/data/vizwiz --checkpoint outputs-router-s42/best.pt --policy fixed --budget 2 --output-dir results-fixed-b2-s42
python eval_research.py --data-root /path/to/data/vizwiz --checkpoint outputs-router-s42/best.pt --policy relevance --budget 2 --output-dir results-relevance-b2-s42
python eval_research.py --data-root /path/to/data/vizwiz --checkpoint outputs-router-s42/best.pt --policy gain --budget 2 --no-skip-nonpositive --output-dir results-gain-forced-b2-s42
```

Use budgets `1`, `2`, and `4` and repeat training seeds 42, 43, and 44. `fixed` follows row-major order, larger scale first. `uncertainty` averages `4p(1-p)` inside each candidate. `relevance` uses cosine similarity between pooled coarse visual features and the coarse model's projected pooled question features; it is a heuristic control, not an extra learned ranking head. `random` uses a SHA256-derived filename/run-seed generator, invariant to batch composition and process sharding. An image-content fallback is used only for direct model calls without filenames.

Every policy uses greedy diversity suppression with `--diversity-iou 0.3`; use `1` to disable suppression. Only `gain` skips predicted nonpositive gains, enabled by default; `--no-skip-nonpositive` forces selection up to the budget. Diversity may leave fewer candidates than the cap. `metrics.json` records policy, budget, suppression threshold, skip behavior and routing seed. Evaluation restores strict checkpoint weights before applying these inference-only overrides. Gain evaluation rejects a refiner-stage checkpoint because its router is untrained.

```bash
python train_research.py --data-root /path/to/data/vizwiz --stage router --detail-size 672 --crop-size 336 --resume-checkpoint outputs-router-s42/last.pt --num-epochs 50 --seed 42 --output-dir outputs-router-s42
torchrun --standalone --nproc_per_node=2 train_research.py --data-root /path/to/data/vizwiz --stage router --init-checkpoint outputs-refiner-s42/best.pt --batch-size 4 --output-dir outputs-router-ddp
```

Use `--init-checkpoint` for a fresh stage/optimizer. Full model weights, including the frozen coarse model and refiner, survive the stage transition. Use `--resume-checkpoint` only within the same stage and training configuration; changing policy, budget, world size, loss weights or annotation contents is not a resume. Stage and inference settings are run metadata, while tensor architecture has its own strict identity. Trusted checkpoint files are required. Checkpoint-driven evaluation does not need the original initialization checkpoint file.

Offline verification from this worktree:

```bash
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/vizwiz-mpl /home/dptphat/research/VizWiz-VQA-Grounding/.venv/bin/python -m unittest discover -s tests -v
```

The suite covers crop geometry/overlap averaging, signed IoU gains, diversity/negative-gain selection, detail-pixel dependence, annotation-free inference, frozen-module modes/gradients, deterministic random routing, exact stage initialization, resume rejection, and all five original-resolution evaluation policies. Synthetic integration uses the actual shared parser/training/evaluation functions, temporary two-image train/validation splits, and real tiny PyTorch encoders. It trains each stage for one epoch and resumes router training for a second epoch. `--tiny --image-size 28 --detail-size 56 --crop-size 16 --num-workers 0 --device cpu` is exclusively an offline integration model.

No real-data training, pretrained CLIP download, CUDA benchmarking or accuracy experiment has been run. Independent single-crop gain targets do not model multi-crop interactions; diversity suppression is a heuristic. Router labels are measured on the detail canvas, while validation/model selection uses original-size IoU. The shallow RGB refiner and coarse-grid context pooling are deliberate cost limits; the sequential crop implementation prioritizes explicit budget behavior over maximum GPU utilization.

Strict continuation requires the checkpoint to reside in the same `--output-dir`; use weight initialization for a new run/directory. Provenance hashes both JSON annotations and their binary mask contents (including paired masks). CPU checkpoints can resume on CUDA with a fresh AMP scaler; CUDA execution itself is unverified here.

Strict resume accepts `last.pt` only, avoiding stale best-checkpoint selection when rewinding within a run. Start a new initialized run to reuse older/best weights.
