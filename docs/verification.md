# Implementation verification — 1 November 2026

Implementation target: `exp/context-membership`, based on commit `6a1736850729d3e90871345b3bc08c887ebef77e`. Changes are in this worktree. The full training campaign has not been launched.

The file reorganization starts from `a1db5e6ce3b49ad1e08dca24324c80fc58a0a5b1`. Implementation scripts and shared files remain directly in the repository root; `models/`, `experiments/`, `diagnostics/`, `configs/`, `docs/`, and `tests/` are root-level folders. See [structure.md](structure.md) for every file's purpose.

## Current workflow

- One checkpoint format, `format_version=1`, records explicit model/component settings and a persistent run UUID. Old checkpoint adapters, raw-state inference, metadata normalization, and old-schema round-trip tests have been removed. Original checkpoint files remain untouched.
- CM/FTC reference fixtures preserve architecture initialization/output checks independently of checkpoint-format support. Disabling components still preserves common initial weights.
- The thirteen-run preset uploads **after each completed run**, following selected-checkpoint test evaluation, to the public model repository `Last-Bullet/vizwiz-cm-aciids-2027`. There are no automatic intermediate-epoch uploads. Manual intermediate backup and restoration are available.
- The preset uses 30 epochs for all thirteen runs. FTC, CM-Shared, and CM-Separate each use seeds 42, 123, and 456; the component ablations and capacity control use seed 42. The cohort and remote prefix are `matched-30ep-v1`; direct training also defaults to 30 epochs.
- Kaggle and Vast.ai use the same launcher, lockfile, and training recipe. The Vast notebook fixes execution to one GPU/process; Kaggle retains its two-GPU commands. The launcher checks Hub write access before training; credentials come from the training machine's environment or Hugging Face login.

## Checks completed

- **38 offline tests passed.** Coverage includes CM/FTC reference outputs, component bypasses and gradients, capacity counts, zero auxiliary loss, current checkpoint reconstruction and rejection of incomplete metadata, nonpersistent CLIP position buffers, training/resume, validation logging, native metrics, validation-only threshold selection, question interventions, and fractional/empty/full role targets. Command tests exercise real subprocess entry points, manifest launch/resume/summary, and matching script/module native predictions. Four Vast tests cover the fixed ZIP layout and interrupted extraction, notebook-only source bootstrap and process-group interruption, and a real tiny CPU train/test/upload-retry/restore pipeline. The notebook skips the separate preflight without changing training or evaluation.
- **`vast.ipynb` passed nbformat schema validation and a real local Jupyter execution.** Cloud downloads and environment installation were simulated, while the foreground runner performed one actual tiny CPU training epoch, validation/test evaluation, summary, and record export against an in-memory Hub. This verifies notebook execution/process handling, not real instance provisioning, GPU capacity, or remote transfers. The tracked notebook contains no execution outputs.
- **Ten tiny-encoder model variants retained bit-identical outputs and losses:** both parameter-sharing modes with all context/membership combinations, FTC, and the capacity control. Inputs, weights, outputs, and losses were captured before the reorganization and checked after it. Current-format checkpoint metadata and parameter counts are unchanged.
- **Hub behavior passed against an in-memory, commit-addressed Hub:** one upload after training and test evaluation; inclusion of latest/best, hashes, metrics, and manifest records; exclusion of image PNGs; unchanged-upload detection; interrupted transfers and retry; manual intermediate snapshots; pinned historical restoration; corrupt-download rejection; run-identity protection; rejection of older progress; and continuation from a restored checkpoint. No live model repository or checkpoints were published by these tests.
- **Synthetic launcher pipeline passed:** preflight → two-epoch tiny FTC training → validation-selected checkpoint → native-grid test metrics → one mocked final Hub commit. A completed-run `--resume` skipped training/evaluation and did not create a duplicate commit.
- **Two-process CPU/Gloo validation passed through the relocated smoke command:** on a three-image synthetic split, shards `[0, 2]` and `[1]` gave the same mIoU as single-process evaluation, `0.20419414062862992`.
- **Two-process training passed through the final root-level launcher:** FTC and CM-Separate each completed two epochs, per-epoch validation, rolling checkpoints, and native-grid test export. Their trained tensors were bit-identical across the tested intermediate/final file layouts. Uploading occurs after the distributed training process exits.
- **Diagnostic CLI checks passed with the final paths:** validation threshold selection and fixed-threshold test evaluation, question conditions, role diagnostics, tiny CPU benchmark, seed-summary JSON/CSV, and record export. Those values are smoke results, not scientific findings.
- The dependency lock check passed offline with 65 packages, including an explicit dependency on the already-locked Hugging Face Hub version.

CPU tests used the workspace's existing Python environment: Python 3.12.10, PyTorch 2.12.1, torchvision 0.27.1, Transformers 5.13.0, Hugging Face Hub 1.22.0, NumPy 2.5.1, Pillow 12.3.0, and Matplotlib 3.11.0. No CUDA device was available. The local DDP checks required localhost socket access outside the default network-restricted sandbox.

## Parameter checks

| Decoder | Registered parameters |
|---|---:|
| FTC | 690,753 |
| CM-Shared | 942,953 |
| CM-Separate | 1,120,585 |
| FTC capacity control | 1,120,845 |

The capacity control adds 430,092 active parameters to FTC and differs from CM-Separate by 260 parameters. Ablations report registered/trainable counts separately because disabled modules stay registered for stable initialization.

## Cleanup and remaining work

The hard-coded `countnum.py` was replaced by preflight dataset checks, and the blanket `tests/` ignore rule was removed. Shared inference/metric/backup helpers replace duplicate paths. `IoU.py`, prediction/overlay tools, and the EXIF utility are retained. Datasets, original checkpoints, result archives, the manuscript, and unrelated workspace changes were preserved.

The former `revision/` folder is split into experiment orchestration and diagnostics. Its manifest moves to `configs/`. `utils.py` is split into checkpoint, runtime, and artifact responsibilities. Root script names, the current checkpoint format, and output-directory conventions are retained. Documentation links, Python source parsing, the dependency lock, and `git diff --check` passed.

Live Hub authentication/large-file transfers, CUDA/NCCL training, GPU decoder checks, FP16/FP32 latency and memory measurements, and the full thirteen-run study remain to be exercised on the intended Kaggle/Vast.ai machines. No minimum GPU-memory requirement or actual cloud runtime is inferred from CPU smoke tests.

Current offline test output is recorded temporarily in `/tmp/cm-vast-simple-full-tests.log`. The simplified-notebook Jupyter smoke log is `/tmp/cm-vast-simple-jupyter.log`, with artifacts under `/tmp/cm-vast-simple-jupyter/`. Earlier structure/distributed/diagnostic logs use `/tmp/cm-structure-*.log`, with synthetic artifacts under `/tmp/cm-structure-smoke/`. The pre-refactor comparison inputs are in `/tmp/cm-structure-reference.pt`. Earlier `/tmp/cm-revision-smoke/` artifacts predate the checkpoint-format change and should not be used with this workflow.
