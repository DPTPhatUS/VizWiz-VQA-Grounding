# Context–Membership grounding experiments

This branch trains and evaluates the paper's question-only Context–Membership (CM) models, the exact FTC baseline, and the FTC capacity control. Every campaign trains fresh models and saves research results.

For Vast.ai, upload [vast.ipynb](vast.ipynb) to `/workspace` on your single-GPU NVIDIA CUDA instance. Enter your Hugging Face token, edit the settings, and **Run All**. Setup requires a new `/workspace/cm-study` directory. If anything fails, delete that directory and start again. See [the Vast guide](docs/vast.md).

## Experiment matrix

| Configuration | Seeds | Decoder parameters |
|---|---|---:|
| FTC | 42, 123, 456 | 690,753 |
| CM-Shared | 42, 123, 456 | 942,953 |
| CM-Separate | 42, 123, 456 | 1,120,585 |
| Context-only, separate | 42 | Registered/trainable counts reported separately |
| Membership-only, separate | 42 | Registered/trainable counts reported separately |
| CM-Separate without auxiliary loss | 42 | 1,120,585 |
| FTC capacity control | 42 | 1,120,845 |

The [manifest](configs/high_priority.json) contains all thirteen runs. Each uses 30 epochs, global batch 4, Adam at 1e-5, zero weight decay, gradient clipping at 1.0, a frozen text encoder, and a trainable image encoder/decoder. Text contains questions only. Validation runs after every epoch at probability threshold 0.5 on the original image grid. Highest validation mIoU selects `best.pt`; ties retain the earlier epoch. Test results do not select the checkpoint.

CM controls are `--model cm`, `--connection-mode shared|separate`, `--disable-context`, `--disable-membership`, and `--edge-loss-weight`. The auxiliary weight defaults to 0.1 with membership, otherwise zero. Context-only bypasses membership updates/supervision; membership-only bypasses context aggregation. A zero auxiliary weight retains the complete forward computation. Disabled modules preserve common initialization but are frozen and excluded from optimization. FTC options are `--model ftc` and `--model ftc-capacity`; CM-specific flags are rejected for them.

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

Each invocation creates `outputs/<timestamp>/`, trains every selected model from scratch, evaluates `best.pt` on validation and test, and uploads each completed model before starting the next. The final study summary is also uploaded. Remote paths are `matched-30ep-v1/<timestamp>/<run-id>/` and `matched-30ep-v1/<timestamp>/summary/`. A new timestamp keeps earlier results separate. `--hf-repo` and `--hf-prefix` change the destination; `--no-hub` runs locally. `--dry-run` prints commands without creating outputs or contacting Hugging Face. `--processes 2` uses torchrun with microbatch 2 per GPU; validation covers each image once.

A direct training command also starts fresh:

```bash
uv run --frozen python train.py --model cm --connection-mode separate --data-root /path/to/vizwiz --num-epochs 30 --batch-size 4 --lr 0.00001 --seed 42 --output-dir outputs/my_cm
```

## Saved research artifacts

```text
outputs/<timestamp>/
├── <run-id>/
│   ├── best.pt
│   ├── run.json
│   ├── history.jsonl
│   ├── val/
│   │   ├── metrics.json
│   │   └── logits/
│   │       ├── index.json
│   │       └── <image>.npz
│   └── test/                       # Same result files as val/
└── summary/
    ├── summary.json
    └── runs.csv
```

- `best.pt`: selected weights, architecture/settings, selected epoch, validation mIoU, and source/software information. It has no optimizer state.
- `run.json`: resolved settings, model/component choice, seed, parameter counts, source commit, software versions, and selected epoch/score.
- `history.jsonl`: per-epoch total/segmentation/membership losses and validation mIoU. Membership loss is logged unweighted.
- `metrics.json`: native-resolution overall results and one record per image containing pixel counts, IoU, precision, recall, area ratio, and relative area error. It references the run record and identifies the selected experiment/epoch in plain text.
- `.npz`: only losslessly compressed float32 logits at the model output resolution and original `[height, width]`. The split index maps filenames and records input size, split, selected model/epoch, dtype, and interpolation settings. Resize logits with bilinear interpolation and `align_corners=False` before applying a threshold.
- Summary JSON/CSV: individual test results, per-model means and sample standard deviations, missing runs, and paired CM-Separate-minus-FTC seed differences. Fractions remain in 0–1 units; severe failures are counts. Single-seed configurations have no estimated standard deviation.

Completed-run uploads contain the checkpoint, settings/history, and validation/test metrics/logits. The summary folder uploads after all selected models finish. There are no continuation checkpoints, backup/status manifests, custom artifact versions, application hashes, or automatic ZIPs.

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

Question diagnostics intervene on the actual text, including repeated, shuffled, generic, and empty questions; identical shuffled texts are excluded. Preserve the shared donor mapping for comparable conditions. Relation-role diagnostics report include/exclude/neutral statistics and mask errors by five target-size bins; they require membership-enabled CM. Benchmarking covers FTC, CM-Shared, CM-Separate, capacity control, and the replicated U-Net, reporting full/decoder latency, memory, throughput, and parameters. FP16 is a separate measurement via `--precision fp16`. Decoder timing caches encoder/fusion outputs; full timing includes text encoding and excludes input-file loading/native resizing.

## Tests and repository layout

```bash
uv run --frozen python -m unittest discover -s tests -v
```

Offline tests use tiny encoders and synthetic masks. They cover reference CM/FTC behavior, ablations, gradients, capacity counts, native metrics, validation selection, lossless logits, calibration, fresh campaigns, research-only uploads, and notebook setup/interruption. CUDA measurements and the full study require the target hardware.

Root scripts/shared files remain directly in the repository, with `models/`, `experiments/`, `diagnostics/`, `configs/`, `docs/`, and `tests/` as folders. See [structure](docs/structure.md), [Kaggle](docs/kaggle.md), [Vast](docs/vast.md), and [verification](docs/verification.md).
