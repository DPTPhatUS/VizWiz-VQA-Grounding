# Vast.ai notebook

Choose the NVIDIA CUDA template, one GPU, and sufficient memory, CPU, and disk. Upload only [vast.ipynb](../vast.ipynb) to `/workspace`, open it in Jupyter, edit the settings cell, and **Run All**.

## Settings

- `HF_TOKEN`: a Hugging Face write token, or leave it to read the instance's `HF_TOKEN` environment variable. Keep any notebook containing a token private; tokens are not passed in commands or saved in experiment records.
- `RUN_IDS`: `[]` runs all thirteen experiments. A list such as `["cm_shared_s42"]` selects individual runs.
- `WORK_DIR`: `/workspace/cm-study` by default.
- `GIT_URL` and `GIT_BRANCH`: source to clone automatically. The updated branch must be pushed before the first run; no source-folder upload is needed. Reruns reuse the checkout.
- `HF_REPO` and `HF_PREFIX`: public destination `Last-Bullet/vizwiz-cm-aciids-2027`, under `matched-30ep-v1/<run-id>/`. Use a new workspace and prefix for a separate study.

## Execution

The run cell clones the branch, installs `uv`, creates the locked Python 3.12 environment, and downloads `vizwiz.zip` from `Last-Bullet/VizWiz-VQA-Grounding`. The ZIP contains `vizwiz/` directly, so no directory discovery is needed. After successful extraction, the ZIP is deleted to free space. A completion marker lets reruns reuse the extracted data; interrupted extraction keeps the ZIP for retry.

The notebook runs one model at a time on one GPU, with the manifest's global batch of four, 30 epochs, per-epoch validation, and best-validation checkpoint selection. After test evaluation, it uploads the latest and best checkpoints plus records/metrics, then starts the next model. Intermediate epochs stay local. The model, training, native-resolution metrics, and checkpoint selection are unchanged.

There are no hardware guards, dataset scans, test suites, fit probes, dry runs, interactive prompts, detached workers, or monitoring stages. The notebook passes `--skip-preflight` to the shared launcher; ordinary CLI launches retain their default preflight. Training progress appears directly in the run cell. Interrupting that cell terminates its process group.

## Rerunning and results

Use **Run All** again after resolving a failure:

- Extracted data are reused without downloading the ZIP again.
- Training resumes from local `latest.pt`; work inside an unfinished epoch is repeated.
- A failure before the first saved epoch starts that model again, retaining its partial records under `interrupted/`.
- Completed local runs skip training and existing test evaluation. A failed final upload is retried.
- On a new instance, existing Hub backups are restored automatically. Only completed runs have automatic remote backups.

Files under `WORK_DIR`:

```text
source/                 Automatically cloned source and its .venv
cache/                  Dependency/model caches
downloads/              Temporary ZIP download; archive removed after extraction
data/vizwiz/            Extracted dataset
outputs/<run-id>/       latest.pt, best.pt, history, test metrics/predictions
reports/<report-id>/    Summary JSON, seed CSV, and records.zip
interrupted/            Records from attempts that failed before saving an epoch
```

The records ZIP excludes checkpoint payloads. Completed-run checkpoints and metrics are available in the Hugging Face repository. Diagnostics and optional manual backup commands remain documented in [README.md](../README.md).
