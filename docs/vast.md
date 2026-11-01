# Vast.ai: matched Context–Membership study

Use the same source revision, lockfile, manifest, and two-process protocol as Kaggle. The code uses ordinary CUDA/PyTorch and SSH commands; it does not depend on a Vast.ai SDK or create paid instances automatically.

## 1. Instance and source

Choose an SSH-capable PyTorch/CUDA instance with two visible GPUs and enough disk for the dataset, dependencies/CLIP cache, and outputs. Keep roughly 6 GB per retained run for latest plus best checkpoints, plus temporary space for another resumable checkpoint during atomic saving. GPU memory requirements still need a real-hardware smoke check; CPU tests do not establish a minimum VRAM size.

Container storage is deleted when a Vast.ai instance is destroyed. A local volume survives destruction but is tied to its host machine. Completed-run uploads to Hugging Face provide the off-machine copy; explicitly upload unfinished progress before releasing an instance. See Vast.ai's [storage documentation](https://github.com/vast-ai/docs/blob/main/guides/instances/storage/types.mdx).

Connect using the SSH command shown by Vast.ai, then prepare the working directory:

```bash
mkdir -p /workspace/cm-study
cd /workspace/cm-study
git clone --single-branch --branch exp/context-membership https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
cd VizWiz-VQA-Grounding
python -m pip install uv
uv python install 3.12
uv sync --frozen --python 3.12
```

The clone command requires these changes to have been pushed first. For an unpushed worktree, transfer its modified source archive instead. Keep the dataset outside the source tree if convenient, with `train_grounding.json`, `val_grounding.json`, `test_grounding.json`, split image directories, and `binary_masks_png/` under one data root.

## 2. Data, authentication, and checks

Set these in the shell that will launch training:

```bash
export CM_DATA_ROOT=/workspace/data/vizwiz
export HF_HOME=/workspace/cache/huggingface
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
nvidia-smi
uv run --frozen hf auth login
```

Use a write-enabled token for `Last-Bullet/vizwiz-cm-aciids-2027` in the interactive login prompt. Alternatively, inject `HF_TOKEN` through your secret-management mechanism. Do not put a token into a command, script, or source file. Set `HF_HOME` before login so later shells use the same credential/cache location. Verify environment variables inside a new SSH/tmux shell; Vast.ai documents shell-specific behavior in its [FAQ](https://console.vast.ai/faq/).

```bash
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python preflight.py --data-root "$CM_DATA_ROOT" --dataset-check-only
uv run --frozen python -m tests.smoke.decoder_gpu
uv run --frozen python -m experiments.launch --data-root "$CM_DATA_ROOT" --processes 2 --runs context_only_s42 --dry-run
```

The actual launcher runs a finite model forward/backward preflight and verifies Hub write access before each training job. The preset creates a **public model repository** if needed. The dry run does not contact the Hub or write outputs.

## 3. Run without depending on the SSH connection

If `tmux` is installed, start a session with `tmux new -s cm-study`. In that session, change back to the source directory and set the environment from step 2 if needed. Launch selected runs:

```bash
set -o pipefail
mkdir -p logs
uv run --frozen python -m experiments.launch --data-root "$CM_DATA_ROOT" --processes 2 --runs context_only_s42 membership_only_s42 2>&1 | tee logs/ablations.log
```

Detach with `Ctrl-b`, then `d`; reconnect using `tmux attach -t cm-study`. Omit `--runs` to run all thirteen jobs sequentially. Do not launch the same run ID on two machines concurrently.

Every epoch uses global batch 4, validates at probability 0.5, and saves local `latest.pt` and `best.pt`. After all epochs and selected-checkpoint test evaluation finish, one upload publishes both checkpoints, run records, and test metrics to `matched-30ep-v1/<run-id>/`. The next run waits for this upload. There are no automatic intermediate uploads. Dataset images and prediction PNGs are excluded. `hub_status.json` contains the confirmed commit SHA; failed uploads leave the completed local run intact and stop the launcher after retries.

One GPU is supported with `--processes 1`, retaining global batch 4. Use a separate complete study/cohort for that configuration: this workflow requires the same process count across the matched runs and across a run's resume. Two GPUs on both Vast.ai and Kaggle allow moving runs between those platforms with the existing preset.

## 4. Restore and resume on either platform

If the working directory survived, resume directly:

```bash
uv run --frozen python -m experiments.launch --data-root "$CM_DATA_ROOT" --processes 2 --runs context_only_s42 --resume
```

On a new instance, repeat setup/authentication, restore a completed run or manually backed-up intermediate epoch into an empty destination, then run the same resume command:

```bash
uv run --frozen python -m experiments.hub download --path-in-repo matched-30ep-v1/context_only_s42 --run-dir outputs/revision/context_only_s42
uv run --frozen python -m experiments.launch --data-root "$CM_DATA_ROOT" --processes 2 --runs context_only_s42 --resume
```

Restoration pins one Hub commit, checks every file's size/hash, and preserves the run identity and best-selection history. Work inside an unfinished epoch is repeated. `--revision COMMIT_SHA` selects an older snapshot, but uploads refuse to replace newer remote progress. A restored completed run skips training and already-recorded test evaluation.

Before releasing an instance with unfinished training, stop at a completed epoch and upload the stable checkpoint pair explicitly. The same command retries a failed final upload:

```bash
uv run --frozen python -m experiments.hub upload --run-dir outputs/revision/context_only_s42 --path-in-repo matched-30ep-v1/context_only_s42
```

For a new repetition, choose both a fresh local output root and unused remote prefix, for example `--output-root outputs/revision-v2 --hf-prefix matched-30ep-v2`. Existing remote run identities are protected even if the local directory was lost.

## 5. Results and storage

Gather completed run folders on one machine before summarizing. Hub restoration includes the test metrics and manifest record needed by the summarizer; it does not restore native prediction PNGs.

```bash
uv run --frozen python -m experiments.summarize --root outputs/revision --output-dir outputs/study_summary
uv run --frozen python -m experiments.export --root outputs --output /workspace/cm_records.zip
```

Download the archive through SCP/SFTP before destroying the instance. Diagnostic commands and GPU benchmarking are in [README.md](../README.md). Run benchmarks without concurrent training on one fixed GPU/precision configuration.

Manual intermediate backups add earlier versions to Hub history. No history is deleted automatically. Review Hugging Face's [storage guidance](https://huggingface.co/docs/hub/storage-limits) when planning the full campaign. A successful local save is not a completed remote backup: check `hub_status.json` or the repository commit before discarding the machine.
