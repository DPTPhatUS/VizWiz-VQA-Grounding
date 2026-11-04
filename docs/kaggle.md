# Kaggle: fresh matched experiments

Use a GPU notebook with Internet access and attach the VizWiz dataset. Clone the updated branch and install its dependencies:

```python
!git clone --single-branch --branch exp/context-membership https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd VizWiz-VQA-Grounding
!python -m pip install uv
!uv python install 3.12
!uv sync --frozen --python 3.12
```

Set `CM_DATA_ROOT` to the extracted `vizwiz/` folder and provide a Hugging Face write token through Kaggle Secrets:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
os.environ['CM_DATA_ROOT'] = '/kaggle/input/YOUR_DATASET/vizwiz'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['PYTHONUNBUFFERED'] = '1'
```

Launch fresh runs; use `--processes 2` when two GPUs are enabled, or `1` for one GPU:

```python
!uv run --frozen python -m experiments.launch --data-root "$CM_DATA_ROOT" --processes 2 --runs ftc_s42 cm_shared_s42 cm_separate_s42
```

Omit `--runs` for all thirteen experiments. Every invocation creates `outputs/<timestamp>/` and starts each selected model from scratch. The protocol is 30 epochs, global batch 4, Adam at 1e-5, and per-epoch validation. Only the validation-selected model is saved. Both validation/test metrics and lossless logits are exported, with one `logits.npz` per split, then the completed run uploads seven files to the public repository `Last-Bullet/vizwiz-cm-aciids-2027`. The summary uploads when all selected runs finish.

There is no continuation from earlier sessions. Start a fresh checkout/session after failure. Previously uploaded results remain in their timestamped folders.

Saved artifacts, optional PNG exports, calibration, diagnostics, and manual ZIP export are described in [README.md](../README.md). The single-GPU upload-only notebook is covered in the [Vast guide](vast.md).
