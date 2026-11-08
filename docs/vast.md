# Vast.ai: fresh training

Choose your NVIDIA CUDA template and one GPU with sufficient resources. Upload only [vast.ipynb](../vast.ipynb) to `/workspace`, open it in Jupyter, enter the settings, and **Run All**.

Settings:

- `HF_TOKEN`: a write token, or read it from the instance environment. Keep notebooks containing tokens private.
- `RUN_IDS=["cm_affinity_s42"]`: one new control by default. Change the list for the next run; `[]` selects all 27 experiments, including older configurations.
- `SAVE_MASKS=False`: logits are always saved; enable this for additional PNGs.
- `WORK_DIR=/workspace/cm-study`: a new directory for this attempt.
- `GIT_URL` / `GIT_BRANCH`: source cloned automatically. Push the updated branch before using it remotely.
- `HF_REPO` / `HF_PREFIX`: public destination `Last-Bullet/vizwiz-cm-aciids-2027`, prefix `matched-30ep-v1`.

Set `HF_REPO` to your current destination. Recommended new-run order is `cm_affinity_s42`, `cm_affinity_s123`, `cm_affinity_s456`, then `cm_two_role_s42`, `cm_two_role_s123`, `cm_two_role_s456`. Multiple IDs run sequentially. This notebook uses one GPU; use the Kaggle/Linux launcher with `--processes 2` if matching an existing two-GPU cohort.

Setup always clones the source, installs Python 3.12 and locked dependencies, downloads the dataset ZIP, and extracts its `vizwiz/` directory. The ZIP is deleted after successful extraction. Models and data caches belong to this workspace. Nothing from an earlier experiment is restored or continued.

Training runs one model at a time for 30 epochs, global batch 4. Each epoch validates at threshold 0.5; the highest validation mIoU selects `best.pt`. After training, that checkpoint produces validation/test metrics and float32 logits, with one `logits.npz` per split. Completed-model results upload before the next model starts. The study summary uploads when the queue finishes.

```text
/workspace/
├── vast.ipynb
└── cm-study/
    ├── source/                     # Source and its .venv
    ├── python/                     # Python installed for this attempt
    ├── cache/                      # Dependency and encoder downloads
    ├── downloads/                  # ZIP removed after extraction
    ├── data/vizwiz/
    └── outputs/<timestamp>/
        ├── <run-id>/
        │   ├── best.pt
        │   ├── run.json
        │   ├── history.jsonl
        │   ├── val/                # metrics.json + logits.npz
        │   └── test/               # Same artifacts; optional PNGs
        └── summary/                # summary.json + runs.csv
```

Hugging Face uses `matched-30ep-v1/<timestamp>/<run-id>/` and `<timestamp>/summary/`. A new timestamp separates each campaign from earlier results. Settings/history, selected model, metrics, logits, and summary are uploaded; optional PNGs remain local. Each run uploads seven files, including the two split-level NPZs.

If anything goes wrong, interrupt the run cell, **delete `/workspace/cm-study` yourself, and Run All from the beginning**. The notebook requires a new workspace and does not delete existing files for you. All selected models train again from scratch. There are no resume options, restoration, backup/status manifests, hashes, or automatic study-wide ZIPs.

Research artifact definitions and optional diagnostics are in [README.md](../README.md).

After collecting timestamp folders locally, `python -m experiments.summarize --root /path/to/outputs --output-dir /path/to/new_summary` combines them and pairs each control with available CM-Separate seeds. Keep one result per run ID under that root.
