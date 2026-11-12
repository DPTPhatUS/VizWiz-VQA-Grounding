# Context–Membership grounding experiments

This branch trains and evaluates the paper's question-only Context–Membership (CM) models, relation controls, the exact FTC baseline, and the FTC capacity control. Every campaign trains fresh models and saves research results. The instructor's essential/high-priority additions are symmetric three-role membership, a trained refinement-depth study, and efficiency measurements.

For Vast.ai, upload [vast.ipynb](vast.ipynb) to `/workspace` on your single-GPU NVIDIA CUDA instance. Enter your Hugging Face token, edit the settings, and **Run All**. Setup requires a new `/workspace/cm-study` directory. If anything fails, delete that directory and start again. See [the Vast guide](docs/vast.md).

## Experiment matrix

| Configuration | Seeds | Decoder parameters |
|---|---|---:|
| Symmetric-affinity, separate | 42, 123, 456 | 1,120,550 |
| Two-role CM, separate | 42, 123, 456 | 1,120,568 |
| FTC | 42, 123, 456 | 690,753 |
| CM-Shared | 42, 123, 456 | 942,953 |
| CM-Separate | 42, 123, 456 | 1,120,585 |
| Context-only, separate | 42, 123, 456 | Registered/trainable counts reported separately |
| Membership-only, separate | 42, 123, 456 | Registered/trainable counts reported separately |
| CM-Separate without auxiliary loss | 42, 123, 456 | 1,120,585 |
| FTC capacity control | 42, 123, 456 | 1,120,845 |

The original [manifest](configs/high_priority.json) contains 27 runs. Its affinity and two-role controls come first, followed by the other 21 runs. Each uses 30 epochs, global batch 4, Adam at 1e-5, zero weight decay, gradient clipping at 1.0, a frozen text encoder, and a trainable image encoder/decoder. Text contains questions only. Validation runs after every epoch at probability threshold 0.5 on the original image grid. Highest validation mIoU selects `best.pt`; ties retain the earlier epoch. Test results do not select the checkpoint.

CM controls are `--model cm`, `--connection-mode shared|separate`, `--local-refinement membership|symmetric-affinity|two-role|symmetric-membership`, `--membership-steps`, `--disable-context`, `--disable-membership`, and `--edge-loss-weight`. Default refinement is the original three-role membership with two updates. The auxiliary weight defaults to 0.1 with local refinement, otherwise zero. Context-only bypasses membership updates/supervision; membership-only bypasses context aggregation. A zero auxiliary weight retains the complete forward computation. Disabled modules preserve common initialization but are frozen and excluded from optimization. FTC options are `--model ftc` and `--model ftc-capacity`; CM-specific flags are rejected for them. A refinement control cannot be combined with `--disable-membership`.

### Local refinement controls

Both controls retain CM-Separate's global context, question conditioning, unary features, readout, 48×48 local grid, eight neighbors, and two update steps. Common tensors and the random state after model construction match the full model for the same seed. The output head is reduced after the usual initialization; no unused parameter rows are retained. Checkpoints identify the refinement explicitly and load with strict tensor checks.

**Symmetric affinity (`cm_affinity_s42`, `cm_affinity_s123`, `cm_affinity_s456`).** Average the scalar scores for both orientations of each valid edge before sigmoid, so `A_ij = A_ji`. From receiver/source occupancies `y_i,y_j`, the same-class target is `y_i*y_j + (1-y_i)*(1-y_j)`. Auxiliary binary cross-entropy is balanced separately over foreground agreements, background agreements, and disagreements: their soft masses are `y_i*y_j`, `(1-y_i)*(1-y_j)`, and `y_i*(1-y_j)+(1-y_i)*y_j`. Each group's loss is divided by its mass; supported groups are averaged per image, then over images. Invalid border edges contribute nothing.

With unit self edges, `P = D^-1 (I+A)` is row-normalized and the signed unary logits are refined by `z[0]=u; z[t+1]=(1-alpha)*u + alpha*P*z[t]`. The learned scalar `alpha=sigmoid(theta)` starts at 0.5. Affinities are symmetric before normalization; `P` need not be symmetric. This preserves background evidence and leaves isolated cells unchanged. It compares refinement formulations, rather than isolating directionality alone.

**Two-role CM (`cm_two_role_s42`, `cm_two_role_s123`, `cm_two_role_s456`).** Use an include/exclude softmax and the original signed, degree-averaged update with the same two learned strengths. The auxiliary target masses are `y_i*y_j` and `(1-y_i)*y_j`. They sum to source occupancy `y_j`: equivalently, supervise conditional targets `(y_i,1-y_i)` weighted by `y_j`. Normalize each supported role's loss by its target mass, then average over roles and images. Pure background sources have no auxiliary target; every valid edge still participates in the forward update and segmentation loss. An empty mask gives zero auxiliary loss. There is no ground-truth gating during inference. This removes explicit neutral modeling and supervision; opposing messages can still cancel.

Both use segmentation BCE plus `0.1 * relation_loss`, the same training budget and validation checkpoint rule. The affinity loss is logged as `affinity`; two-role loss is logged as `membership`. Existing three-role checkpoints and behavior remain unchanged.

## Instructor high-priority experiments

Select [configs/instructor_high_priority.json](configs/instructor_high_priority.json) for the latest review. It contains **12 new runs and three CM-Separate reference runs**, ordered by priority. Every configuration uses seeds 42, 123, and 456 and the same 30-epoch protocol above. The cohort name remains `matched-30ep-v1` so reference runs with the same settings can be summarized together; its default Hub prefix is `instructor-high-priority`.

| Run ID prefix (append `_s42`, `_s123`, or `_s456`) | Experiment |
|---|---|
| `cm_symmetric_membership` | Symmetric three-role predictions and targets; two signed updates |
| `cm_steps0` | Directed membership, zero updates, relation supervision retained |
| `cm_steps1` | Directed membership, one update |
| `cm_steps3` | Directed membership, three updates |
| `cm_separate` | Reference: directed membership, two updates |

**Symmetric three-role membership.** Average reciprocal three-component logits before softmax. Keep the signed update, update strengths, degree averaging, context branch, unary/readout path, and all 1,120,585 decoder parameters. Initial tensors and RNG state match CM-Separate. Supervise reciprocal averages of the original target distributions:

```text
include = y_i * y_j
exclude = ((1-y_i)*y_j + (1-y_j)*y_i) / 2
neutral = 1 - (y_i+y_j) / 2
```

Targets sum to one on valid edges. A mixed foreground/background pair receives `(0, 0.5, 0.5)` in both directions. This avoids demanding incompatible exclusion/neutral predictions from a symmetric head. Reciprocal averaging preserves the total mass of each role on the valid graph, so the class-balanced loss is also equivalent to constraining predictions to be symmetric under the original directional objective. Invalid border edges remain excluded. Checkpoints record `local_refinement=symmetric-membership`.

**Refinement depth.** `--membership-steps 0|1|2|3` changes the number of updates during both training and inference; it is recorded in the checkpoint. Zero steps feeds unary logits to the same learned readout. Local relation prediction and its auxiliary loss remain active, while the two unused update-rate parameters are frozen. This is different from the existing context-only ablation, which also removes relation supervision. The other depths have identical parameter counts and initialization. The existing CM-Separate runs supply the depth-two reference; select their IDs to train new references if desired. Nothing is resumed or automatically skipped.

Run the next experiment, or queue several IDs:

```bash
uv run --frozen python -m experiments.launch --manifest configs/instructor_high_priority.json --data-root /path/to/vizwiz --processes 1 --runs cm_symmetric_membership_s42 --hf-repo OWNER/REPOSITORY
```

Train the three symmetric seeds first, then depths 0, 1, and 3. Match the process count used by your reference runs. Omitting `--runs` trains **all 15 entries from scratch**, including the references. The original 27-run preset is unchanged. Use the review manifest when combining results:

```bash
uv run --frozen python -m experiments.summarize --manifest configs/instructor_high_priority.json --root outputs --output-dir outputs/review_summary
```

The summary reports validation/test mIoU and Boundary IoU, sample standard deviations, and paired test differences against CM-Separate. It uses whatever seeds are complete. Earlier result folders can be placed under the same parent for analysis; missing Boundary IoU is explicitly reported and never treated as zero. The launcher itself always starts fresh.

**Efficiency.** These commands benchmark FTC, the capacity control, and all four CM depths at batch 1 and input 336×336, with 10 warm-up and 50 measured passes. They need no newly trained checkpoints: the fixed-shape networks use pretrained encoders and initialized prediction layers.

```bash
uv run --frozen python -m diagnostics.benchmark --models ftc ftc-capacity cm-separate --membership-steps 0 1 2 3 --device cuda --output outputs/review_efficiency_fp32.json
uv run --frozen python -m diagnostics.benchmark --models ftc ftc-capacity cm-separate --membership-steps 0 1 2 3 --device cuda --precision fp16 --output outputs/review_efficiency_fp16.json
```

Both commands measure synchronized mean/median latency, throughput, peak allocated/reserved GPU memory, and registered/trainable parameters. Full forward includes text encoding/tokenization; decoder timing caches the encoders and bottleneck cross-attention but includes skip fusion, refinement, and upsampling. File loading and native resizing are outside timing. Each JSON records hardware, precision, software, and the timing boundaries. Keep FP32 and FP16 results separate.

FLOPs use PyTorch's convolution/matrix/attention counter, with a multiply-add counted as two operations. Counting uses equivalent unfused attention without changing the timing backend. These are **counted FLOPs**, excluding elementwise operations, graph reductions, normalization, and interpolation; the exclusions are saved in the report. The count alone therefore does not capture the cost of extra membership updates; use measured latency for that comparison. Add `cm-symmetric-membership` to `--models` to measure that control too. Benchmark JSONs are local artifacts; upload them explicitly with `hf upload OWNER/REPOSITORY outputs/review_efficiency_fp32.json instructor-high-priority/review_efficiency_fp32.json` if wanted.

The lower-priority loss-weight sweep, size-stratified boundary analysis, intermediate visualizations, and VGOQ comparison are not part of this implementation.

## Linux commands

Run commands from the source root. Python 3.12 and locked dependencies can be installed with:

```bash
python -m pip install uv
uv python install 3.12
uv sync --frozen --python 3.12
```

The data root contains `train_grounding.json`, `val_grounding.json`, `test_grounding.json`, the `train/`, `val/`, `test/` image directories, and `binary_masks_png/<split>/`.

Authenticate with `HF_TOKEN` in the environment or `uv run --frozen hf auth login`. The public destination defaults to `Last-Bullet/vizwiz-cm-aciids-2027` and is created automatically.

```bash
uv run --frozen python -m experiments.launch --data-root /path/to/vizwiz --processes 1
```

Omit `--runs` for the whole study, or select models:

```bash
uv run --frozen python -m experiments.launch --data-root /path/to/vizwiz --processes 1 --runs ftc_s42 cm_shared_s42 cm_separate_s42
```

For the original affinity/two-role study, select IDs in this order (the latest instructor experiments use the separate preset above):

```text
cm_affinity_s42
cm_affinity_s123
cm_affinity_s456
cm_two_role_s42
cm_two_role_s123
cm_two_role_s456
```

```bash
uv run --frozen python -m experiments.launch --data-root /path/to/vizwiz --processes 1 --runs cm_affinity_s42 --hf-repo OWNER/REPOSITORY
```

Use your current Hugging Face destination and the same process count as the CM runs being compared. Several IDs after `--runs` queue them sequentially. Each invocation is fresh; omitting `--runs` launches all 27, including previously completed configurations.

Each invocation creates `outputs/<timestamp>/`, trains every selected model from scratch, evaluates `best.pt` on validation and test, and uploads each completed model before starting the next. The final study summary is also uploaded. Remote paths are `matched-30ep-v1/<timestamp>/<run-id>/` and `matched-30ep-v1/<timestamp>/summary/`. A new timestamp keeps earlier results separate. `--hf-repo` and `--hf-prefix` change the destination; `--no-hub` runs locally. `--dry-run` prints commands without creating outputs or contacting Hugging Face. `--processes 2` uses torchrun with microbatch 2 per GPU; validation covers each image once.

A direct training command also starts fresh:

```bash
uv run --frozen python train.py --model cm --connection-mode separate --data-root /path/to/vizwiz --num-epochs 30 --batch-size 4 --lr 0.00001 --seed 42 --output-dir outputs/my_cm
```

Add `--local-refinement symmetric-affinity`, `two-role`, or `symmetric-membership` for a direct control run. Use `--membership-steps` to change the refinement depth. The launcher also evaluates and uploads automatically when selecting the corresponding manifest entry.

Summarize runs from separate campaigns by giving their common parent (or pass one timestamp directory):

```bash
uv run --frozen python -m experiments.summarize --root outputs --output-dir outputs/combined_summary
```

The summary includes paired CM-Separate-minus-control differences for matching seeds. Put the earlier CM-Separate result folders under the same parent to include these comparisons; weights and logits are unnecessary for summarization. A repeated run ID is rejected instead of selecting a result by its test score. Use a root containing one result per run, and a new summary output directory each time.

## Saved research artifacts

```text
outputs/<timestamp>/
├── <run-id>/
│   ├── best.pt
│   ├── run.json
│   ├── history.jsonl
│   ├── val/
│   │   ├── metrics.json
│   │   └── logits.npz
│   └── test/                       # Same result files as val/
└── summary/
    ├── summary.json
    └── runs.csv
```

- `best.pt`: selected weights, architecture/settings, selected epoch, validation mIoU, and source/software information. It has no optimizer state.
- `run.json`: resolved settings, model/component choice, seed, parameter counts, source commit, software versions, and selected epoch/score.
- `history.jsonl`: per-epoch total/segmentation/relation losses and validation mIoU. The unweighted relation loss is named `membership` or `affinity` according to the control.
- `metrics.json`: native-resolution overall results and one record per image containing pixel counts, IoU, precision, recall, area ratio, relative area error, boundary intersection/union, and Boundary IoU. Boundary bands use 3×3 zero-padded erosion for `max(1, round(0.02 * native image diagonal))` iterations, including image borders. The summary is a mean of per-image scores. It references the run record and identifies the selected experiment/epoch in plain text. Boundary scores are computed in final validation/test evaluation; checkpoint selection still uses only validation mIoU.
- `logits.npz`: one losslessly compressed archive per split. Each image has its own float32 array at model output resolution. The `metadata` array holds a JSON string mapping filenames to array keys and original `[height, width]`, plus input size, split, selected model/epoch, dtype, and interpolation settings. Read it with `np.load(..., allow_pickle=False)`; individual image arrays load on demand. Export streams one image at a time without stacking the entire split in memory. Resize logits with bilinear interpolation and `align_corners=False` before applying a threshold.
- Summary JSON/CSV: individual test results and validation mIoU/Boundary IoU, per-model means and sample standard deviations, missing runs, and paired CM-Separate-minus-FTC/control seed differences. Fractions remain in 0–1 units; severe failures are counts. A configuration with only one completed seed has no estimated standard deviation.

Completed-run uploads contain seven files: the checkpoint, settings/history, and one metrics JSON and logit NPZ for each of validation and test. The summary folder uploads after all selected models finish (191 files for one campaign of all 27 runs and its two summary files). There are no continuation checkpoints, backup/status manifests, custom artifact versions, application hashes, or automatic study-wide ZIPs.

PNG masks are optional (`SAVE_MASKS=True` in the notebook or `--save-masks` in the launcher/evaluator) and remain local. They can also be exported from saved logits without model inference or dataset access:

```bash
uv run --frozen python predictions.py --predictions outputs/TIMESTAMP/cm_separate_s42/val --output-dir outputs/masks/cm_val --threshold 0.5
```

Manual archive export remains available:

```bash
uv run --frozen python -m experiments.export --root outputs/TIMESTAMP --output /tmp/cm_results.zip
```

To upload an individually trained model after its evaluation:

```bash
uv run --frozen python -m experiments.hub --run-dir outputs/my_cm --path-in-repo manual-study/my_cm
```

## Evaluation and diagnostics

Evaluate any saved model, export one prediction, or make an overlay:

```bash
uv run --frozen python eval.py --checkpoint outputs/my_cm/best.pt --data-root /path/to/vizwiz --dataset val --output-dir outputs/my_cm/val
uv run --frozen python predict_save.py --checkpoint outputs/my_cm/best.pt --image photo.jpg --question 'What does it say?' --output prediction.png
uv run --frozen python visualize_predictions.py --checkpoint outputs/my_cm/best.pt --image photo.jpg --question 'What does it say?' --output overlay.png
```

`IoU.py` remains an evaluation alias. The loader applies EXIF orientation; `bake_exif.py` remains available for explicit dataset preparation. `preflight.py` is an optional standalone data/model check and is not part of the automatic launcher or notebook.

Threshold calibration can use saved logits on CPU:

```bash
uv run --frozen python -m diagnostics.thresholds validation --predictions outputs/TIMESTAMP/cm_separate_s42/val --data-root /path/to/vizwiz --device cpu --output-dir outputs/calibration/val
uv run --frozen python -m diagnostics.thresholds test --predictions outputs/TIMESTAMP/cm_separate_s42/test --data-root /path/to/vizwiz --device cpu --selection outputs/calibration/val/selection.json --output-dir outputs/calibration/test
```

Alternatively, replace `--predictions` with `--checkpoint /path/to/best.pt` for live inference. Validation sweeps 0.10–0.90 in steps of 0.05 and exports metrics, pixel counts, and PDF curves. Selection maximizes validation mIoU; ties prefer proximity to 0.5, then the lower threshold. Test uses that selection and 0.5. Ground-truth masks are still required. Area ratios exclude empty ground-truth masks and report their support.

Further diagnostics are explicit commands; the notebook does not launch them automatically:

```bash
uv run --frozen python -m diagnostics.questions --checkpoint /path/to/best.pt --data-root /path/to/vizwiz --donors outputs/donors_val.json --output-dir outputs/questions
uv run --frozen python -m diagnostics.roles --checkpoint /path/to/cm_best.pt --data-root /path/to/vizwiz --output-dir outputs/roles
uv run --frozen python -m diagnostics.benchmark --device cuda --output outputs/benchmark_fp32.json
```

Question diagnostics intervene on the actual text, including repeated, shuffled, generic, and empty questions; identical shuffled texts are excluded. Preserve the shared donor mapping for comparable conditions. Evaluation, prediction, threshold calibration, and question diagnostics load both new controls. Relation-role diagnostics currently require three-role CM and reject the new controls rather than labeling their outputs as include/exclude/neutral. Benchmarking supports FTC, CM-Shared, CM-Separate, capacity control, the replicated U-Net, and optional `--models cm-affinity cm-two-role`, reporting full/decoder latency, memory, throughput, and parameters. FP16 is a separate measurement via `--precision fp16`. Decoder timing caches encoder/fusion outputs; full timing includes text encoding and excludes input-file loading/native resizing.

## Tests and repository layout

```bash
uv run --frozen python -m unittest discover -s tests -v
```

Offline tests use tiny encoders and synthetic masks. They cover reference CM/FTC behavior, ablations, gradients, capacity counts, native metrics, validation selection, lossless logits, calibration, fresh campaigns, research-only uploads, and notebook setup/interruption. CUDA measurements and the full study require the target hardware.

Root scripts/shared files remain directly in the repository, with `models/`, `experiments/`, `diagnostics/`, `configs/`, `docs/`, and `tests/` as folders. See [structure](docs/structure.md), [Kaggle](docs/kaggle.md), [Vast](docs/vast.md), and [verification](docs/verification.md).
