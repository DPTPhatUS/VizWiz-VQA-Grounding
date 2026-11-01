# Repository structure

Implementation scripts and shared files live directly at the source root. There is no enclosing application package or installation step for the source itself.

| Files/folders | Purpose |
|---|---|
| `vast.ipynb` | Two code cells: settings, then fresh setup/training. Requires a new workspace. |
| `train.py` | Fresh Adam training, per-epoch validation, selected-model saving. |
| `eval.py`, `IoU.py` | Validation/test metrics and float32 logits; optional binary masks. |
| `predict_save.py`, `visualize_predictions.py` | One-image prediction or evidence overlay. |
| `predictions.py` | Logit export/index, native reconstruction, optional PNG export. |
| `dataset.py` | Question-only inputs, image/mask alignment, loaders and collating. |
| `losses.py` | Segmentation BCE, occupancy targets, membership loss. |
| `metrics.py` | Native-grid counts, image metrics, aggregate scores. |
| `validation.py` | Non-padding distributed validation, preserving training modes and RNG. |
| `checkpoints.py` | Selected-model saving and strict architecture/tensor reconstruction. |
| `runtime.py` | Device transfer, parameter counts, source commit and software information. |
| `artifacts.py` | Atomic JSON writing and output-directory creation. |
| `preflight.py` | Optional standalone data/model check; not an automatic launch stage. |
| `bake_exif.py` | Optional EXIF preparation utility. |
| `models/` | CM/FTC implementations, component networks, encoders, factory, benchmark U-Net. |
| `experiments/launch.py` | Fresh timestamped campaign: train → validation/test evaluation → upload → summary. |
| `experiments/vast.py` | Fresh dataset download, extraction, ZIP deletion, single-GPU launch. |
| `experiments/hub.py` | Plain uploads of completed research artifacts and study summaries. |
| `experiments/summarize.py` | Per-seed comparisons, sample deviations, paired CM-Separate–FTC differences. |
| `experiments/export.py` | Explicit manual archive export. |
| `diagnostics/` | Threshold calibration, question dependence, relation-role analysis, efficiency. |
| `configs/high_priority.json` | Thirteen model/seed configurations and matched 30-epoch settings. |
| `docs/` | Vast/Kaggle setup and verification. |
| `tests/` | Offline behavior tests, reference fixtures, optional hardware smoke tools. |

Saved research artifacts are organized as `outputs/<timestamp>/<run-id>/`, with `best.pt`, `run.json`, `history.jsonl`, and `val/`/`test/` metrics and logits. Study summaries live at `outputs/<timestamp>/summary/`. Each NPZ holds logits and native image dimensions; the split index holds common interpretation settings and filename mapping. Results identify the model by its campaign/run name and selected epoch.

Training always begins from the experiment's seeded initialization and pretrained encoders. The source commit and dependency versions support reproducibility. Only the selected model is retained; there is no optimizer-state saving or cross-attempt state. Existing datasets, model files, result archives, and paper sources are not moved or deleted by this refactor.
