# Kaggle: matched Context–Membership study

Enable Internet and two T4 GPUs, and attach the VizWiz dataset. Use a notebook session to run one or a few chosen models; the ten-run study is not assumed to fit one session. This workflow trains fresh checkpoints and uploads each completed run to the public repository `Last-Bullet/vizwiz-cm-aciids-2027`.

## 1. Setup

After these changes have been pushed to `exp/context-membership`, clone that branch. For an unpushed working copy, upload the modified source archive instead.

```python
!git clone --single-branch --branch exp/context-membership https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen
```

Set the actual attached dataset path and this session's run IDs. Do not change the matched study's seeds or common settings between sessions.

```python
import os
from pathlib import Path

os.environ['CM_DATA_ROOT'] = '/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = '/kaggle/working/matplotlib-cache'
assert (Path(os.environ['CM_DATA_ROOT']) / 'train_grounding.json').is_file()
```

Create a Kaggle secret named `HF_TOKEN` containing a Hugging Face token with write access to the repository. Enable that secret for this notebook, then load it without displaying it:

```python
from kaggle_secrets import UserSecretsClient
os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
```

The launcher creates the public model repository if it does not exist and verifies write access before training. Credentials stay in the environment; run records contain only the repository and path. Dry runs and offline tests require no token.

## 2. Tests and command preview

```bash
%%bash
set -euo pipefail
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python -m revision.launch --data-root "$CM_DATA_ROOT" --processes 2 --runs context_only_s42 membership_only_s42 --dry-run
```

## 3. Launch selected runs

```bash
%%bash
set -euo pipefail
uv run --frozen python -m revision.launch --data-root "$CM_DATA_ROOT" --processes 2 --runs context_only_s42 membership_only_s42
```

Other IDs: `ftc_s42`, `ftc_s123`, `ftc_s456`, `cm_separate_s42`, `cm_separate_s123`, `cm_separate_s456`, `cm_noaux_s42`, and `ftc_capacity_s42`.

Each run validates every epoch and keeps local `latest.pt` and `best.pt`. After training and test evaluation finish, it uploads both checkpoints, records, and final metrics in one Hub commit. There are no automatic intermediate uploads. Check the printed resolved protocol: 20 epochs, global batch 4, learning rate 1e-5, and the intended model/flags. New local directories and unused remote run paths are required for fresh runs. Never initialize a fresh ablation from a trained full-CM checkpoint.

## 4. Resume after a session ends

For an unfinished run, stop training after an epoch completes and upload its stable checkpoint pair before the session ends:

```bash
%%bash
set -euo pipefail
uv run --frozen python -m revision.hub upload --run-dir outputs/revision/context_only_s42 --path-in-repo matched-20ep-v1/context_only_s42
```

In a new session, repeat setup and authentication, then download that backup. This also works when the previous session ran on Vast.ai:

```bash
%%bash
set -euo pipefail
uv run --frozen python -m revision.hub download --path-in-repo matched-20ep-v1/context_only_s42 --run-dir outputs/revision/context_only_s42
```

The downloader pins one remote commit and verifies hashes before exposing the restored folder. It refuses nonempty local directories. If the original working folder survived, skip the download and resume it directly:

```bash
%%bash
set -euo pipefail
uv run --frozen python -m revision.launch --data-root "$CM_DATA_ROOT" --processes 2 --runs context_only_s42 --resume
```

Retain the two-process setting. `best.pt` is inference-only; resume uses `latest.pt`. After a failed upload, both local checkpoints remain available. If training already completed, the launcher's `--resume` retries the final upload without retraining. An unfinished run has no remote checkpoint until explicitly backed up; local-only work is lost if the session storage is lost.

## 5. Selected-model diagnostics

Choose a completed fresh run; repeat with separate output names for the relevant controls and ablations. Older checkpoint formats are no longer supported.

```python
os.environ['CM_CKPT'] = '/kaggle/working/VizWiz-VQA-Grounding/outputs/revision/cm_separate_s42/best.pt'
os.environ['CM_LABEL'] = 'cm_separate_s42'
assert Path(os.environ['CM_CKPT']).is_file()
```

```bash
%%bash
set -euo pipefail
base="outputs/diagnostics/$CM_LABEL"
uv run --frozen python -m revision.thresholds validation --checkpoint "$CM_CKPT" --data-root "$CM_DATA_ROOT" --output-dir "${base}_val_thresholds"
uv run --frozen python -m revision.thresholds test --checkpoint "$CM_CKPT" --data-root "$CM_DATA_ROOT" --selection "${base}_val_thresholds/selection.json" --output-dir "${base}_test_thresholds"
uv run --frozen python -m revision.questions --checkpoint "$CM_CKPT" --data-root "$CM_DATA_ROOT" --donors outputs/diagnostics/donors_val_s2027.json --output-dir "${base}_questions"
```

For CM checkpoints only:

```bash
%%bash
set -euo pipefail
uv run --frozen python -m revision.roles --checkpoint "$CM_CKPT" --data-root "$CM_DATA_ROOT" --output-dir "outputs/diagnostics/${CM_LABEL}_roles"
```

## 6. Benchmark and summarize

Run benchmarks once per GPU/precision combination, without concurrent training:

```bash
%%bash
set -euo pipefail
uv run --frozen python -m revision.benchmark --device cuda --output outputs/benchmark_fp32.json
uv run --frozen python -m revision.benchmark --device cuda --precision fp16 --output outputs/benchmark_fp16.json
uv run --frozen python -m revision.summarize --root outputs/revision --output-dir outputs/study_summary
```

Gather completed run folders from all sessions before the final summary. Missing runs are listed; a partial summary is not a completed multi-seed experiment.

## 7. Export

Hub backups include checkpoints and run/test metrics. Save diagnostic reports, plots, and PNG predictions through the first archive below; they are not included in automatic Hub backups. The second archive provides an optional local checkpoint export. Choose a fresh archive name each time.

```bash
%%bash
set -euo pipefail
uv run --frozen python -m revision.export --root outputs --output /kaggle/working/cm_records.zip
uv run --frozen python -m revision.export --root outputs/revision --output /kaggle/working/cm_resume.zip --include-checkpoints
```

```python
from IPython.display import FileLink, display
display(FileLink('/kaggle/working/cm_records.zip'))
display(FileLink('/kaggle/working/cm_resume.zip'))
```

Manual intermediate backups remain in Hub history. See the [backup/storage notes](README.md#hugging-face-backups-and-machine-transfers). Use [VAST.md](VAST.md) for the equivalent SSH workflow.
