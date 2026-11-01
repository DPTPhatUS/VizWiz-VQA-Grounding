# Verification — 2 November 2026

The branch now implements a fresh-start research workflow. The Vast notebook requires a new workspace and always clones the source, installs the environment, downloads/extracts data, deletes the ZIP, and trains selected models from scratch. If an attempt fails, the user deletes the workspace and starts again.

The experiment matrix remains thirteen runs at 30 epochs: FTC, CM-Shared, and CM-Separate each use seeds 42/123/456; component ablations and the capacity control use seed 42. Training/validation math, native evaluation, initialization, and decoder parameter counts are preserved. Only validation-selected `best.pt` is saved.

## Completed checks

- **35 offline tests passed.** Coverage includes exact CM/FTC reference behavior, shared/separate initialization, bypasses and gradient paths, capacity counts, zero auxiliary weight, native-grid metrics, validation-only selection and earlier-epoch ties, validation preserving training, question interventions/donors, role targets/borders, selected-model round trips, fresh training/evaluation/prediction, and actual command entry points.
- Saved logits are lossless float32 arrays and reproduce native-grid counts across all 17 thresholds on CPU. Calibration reads them without loading model weights. Optional PNG export is checked against the native predictions.
- Two consecutive synthetic campaigns both train and evaluate from scratch in different timestamped folders. A simulated upload failure stops the attempt; the next attempt trains again. Selected-model tensors match across same-seed fresh runs.
- In-memory Hugging Face tests check the exact uploaded research files and final summary. No continuation checkpoint, optimizer state, backup/status manifest, custom artifact version, application hash, or automatic archive is generated.
- Notebook tests check unconditional clone/install, a new workspace requirement, fresh dataset download/extraction with ZIP deletion, token inheritance without command/config exposure, and interruption of the foreground process group.
- The notebook passed nbformat validation and a real local Jupyter execution with a tiny CPU model. Source/environment setup and cloud transfers were simulated. Actual training, validation/test export, and study summarization ran; simulated uploads contained the selected model, settings/history, both splits' metrics/logits, and summary.

Decoder parameter counts remain FTC 690,753; CM-Shared 942,953; CM-Separate 1,120,585; FTC capacity 1,120,845. The capacity control differs from CM-Separate by 260 parameters.

Full-suite output: `/tmp/cm-fresh-full-tests.log`. Targeted behavior checks: `/tmp/cm-fresh-targeted.log`. Jupyter smoke output: `/tmp/cm-fresh-jupyter.log`, with artifacts in `/tmp/cm-fresh-jupyter/`.

No scientific training campaign, real GPU measurement, or live Hugging Face upload was launched locally. Those remain for the intended Vast/Kaggle hardware. Existing datasets, checkpoints, result archives, and manuscript files were preserved.
