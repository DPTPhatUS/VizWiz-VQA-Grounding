# Implementation verification — 1 November 2026

Implementation target: `exp/context-membership`, based on commit `6a1736850729d3e90871345b3bc08c887ebef77e`. Changes are in this worktree. The full training campaign has not been launched.

## Current workflow

- One checkpoint format, `format_version=1`, records explicit model/component settings and a persistent run UUID. Old checkpoint adapters, raw-state inference, metadata normalization, and old-schema round-trip tests have been removed. Original checkpoint files remain untouched.
- CM/FTC reference fixtures preserve architecture initialization/output checks independently of checkpoint-format support. Disabling components still preserves common initial weights.
- The ten-run preset uploads **after each completed run**, following selected-checkpoint test evaluation, to the public model repository `Last-Bullet/vizwiz-cm-aciids-2027`. There are no automatic intermediate-epoch uploads. Manual intermediate backup and restoration are available.
- Kaggle and Vast.ai guides use the same launcher, lockfile, and two-process settings. The launcher checks Hub write access before training; credentials come from the training machine's environment or Hugging Face login.

## Checks completed

- **31 offline tests passed.** Coverage includes CM/FTC reference outputs, component bypasses and gradients, capacity counts, zero auxiliary loss, current checkpoint reconstruction and rejection of incomplete metadata, nonpersistent CLIP position buffers, training/resume, validation logging, native metrics, validation-only threshold selection, question interventions, and fractional/empty/full role targets.
- **Hub behavior passed against an in-memory, commit-addressed Hub:** one upload after training and test evaluation; inclusion of latest/best, hashes, metrics, and manifest records; exclusion of image PNGs; unchanged-upload detection; interrupted transfers and retry; manual intermediate snapshots; pinned historical restoration; corrupt-download rejection; run-identity protection; rejection of older progress; and continuation from a restored checkpoint. No live model repository or checkpoints were published by these tests.
- **Synthetic launcher pipeline passed:** preflight → two-epoch tiny FTC training → validation-selected checkpoint → native-grid test metrics → one mocked final Hub commit. A completed-run `--resume` skipped training/evaluation and did not create a duplicate commit.
- **Two-process CPU/Gloo validation previously passed:** on a three-image synthetic split, shards `[0, 2]` and `[1]` gave the same mIoU as single-process evaluation, `0.20419414062862992`. Two-process context-only CM training also completed two epochs with validation and rolling checkpoints. Uploading now occurs after the distributed training process exits.
- The initial synthetic CLI pass also exercised threshold selection/test evaluation, question conditions, role diagnostics, CPU benchmark, seed-summary exports, and result archives. Those values are smoke results, not scientific findings.
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

Live Hub authentication/large-file transfers, CUDA/NCCL training, GPU decoder checks, FP16/FP32 latency and memory measurements, and the full ten-run study remain to be exercised on the intended Kaggle/Vast.ai machines. No minimum GPU-memory requirement or actual cloud runtime is inferred from CPU smoke tests.

Current offline test output is recorded temporarily in `/tmp/cm-platform-tests.log`. Earlier smoke artifacts are under `/tmp/cm-revision-smoke/`; they predate the format change and must be regenerated before use with this workflow.
