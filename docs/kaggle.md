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

For the instructor's essential/high-priority experiments, select the review preset. Run `cm_symmetric_membership_s42`, `cm_symmetric_membership_s123`, and `cm_symmetric_membership_s456` first. Then run `cm_steps0`, `cm_steps1`, and `cm_steps3`, each with suffixes `_s42`, `_s123`, and `_s456`. For example:

```python
!uv run --frozen python -m experiments.launch --manifest configs/instructor_high_priority.json --data-root "$CM_DATA_ROOT" --processes 2 --runs cm_symmetric_membership_s42 --hf-repo OWNER/REPOSITORY
```

Replace `OWNER/REPOSITORY` with your current destination. Keep the same process count as the CM comparison runs. Supply several IDs to queue them sequentially. The review manifest has 12 new runs and three depth-two CM-Separate references: omitting `--runs` trains all 15 from scratch. Without `--manifest`, the original 27-run preset is used. Every invocation creates `outputs/<timestamp>/` and starts each selected model from scratch. The protocol is 30 epochs, global batch 4, Adam at 1e-5, and per-epoch validation. Only the validation-selected model is saved. Both validation/test metrics, including Boundary IoU, and lossless logits are exported, with one `logits.npz` per split, then the completed run uploads seven files. The summary uploads when all selected runs finish.

Use `python -m experiments.summarize --manifest configs/instructor_high_priority.json --root outputs --output-dir outputs/combined_summary` (through `!uv run --frozen`) to combine different timestamp folders, including earlier CM-Separate results. It reports means, sample standard deviations, and paired control differences for available seeds.

Measure efficiency on the same GPU without training another model:

```python
!uv run --frozen python -m diagnostics.benchmark --models ftc ftc-capacity cm-separate --membership-steps 0 1 2 3 --device cuda --output outputs/review_efficiency_fp32.json
```

Run again with `--precision fp16` and a different output filename for FP16. The report contains latency, memory, throughput, parameters, and convolution/matrix/attention FLOPs with explicit coverage. See the [README](../README.md#instructor-high-priority-experiments) for definitions and the symmetric-target formulation.

There is no continuation from earlier sessions. Start a fresh checkout/session after failure. Previously uploaded results remain in their timestamped folders.

Saved artifacts, optional PNG exports, calibration, diagnostics, and manual ZIP export are described in [README.md](../README.md). The single-GPU upload-only notebook is covered in the [Vast guide](vast.md).
