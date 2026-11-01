# Repository structure

Command scripts and shared implementation files live directly in the repository root. Root-level folders separate model implementations, experiment tools, diagnostics, study configuration, documentation, and tests. Run commands from the repository root; the source checkout needs no installation or `PYTHONPATH` changes.

## Files and responsibilities

```text
.
├── README.md                         Main workflow, protocol, controls, and runnable commands
├── pyproject.toml                    Python requirement and dependencies
├── uv.lock                           Locked dependencies shared by Linux, Kaggle, and Vast.ai
├── .gitignore                        Excludes datasets, generated results, and Python caches
│
├── train.py                          Argument resolution, Adam training, resume, best/latest selection
├── eval.py                           Selected-checkpoint evaluation, native masks, metric export
├── IoU.py                            Alias for eval.py
├── predict_save.py                   One image/question to a native-resolution binary mask
├── visualize_predictions.py          One image/question to an evidence overlay
├── preflight.py                      Dataset/options checks and a finite forward/backward step
├── bake_exif.py                      Optional EXIF baking utility, retaining its dry-run controls
│
├── dataset.py                        Image/mask alignment, question-only text, collating, loader seeding
├── losses.py                         Segmentation BCE, occupancy role targets, membership loss
├── metrics.py                        Native-grid counts, per-image metrics, dataset summaries
├── validation.py                     Non-padding DDP validation with training modes and RNG restored
├── checkpoints.py                    Strict format/model reconstruction, save/load, metadata inspection
├── runtime.py                        Project paths, device transfer, parameter counts, source/environment records
├── artifacts.py                      SHA-256 hashes, atomic JSON writing, protected output creation
│
├── models/
│   ├── __init__.py                   Exposes the model factory and principal model/encoder classes
│   ├── factory.py                    Single constructor for CM, FTC, and FTC capacity control
│   ├── cm.py                         CM normalization, encoders, question fusion, and decoder call
│   ├── ftc.py                        Exact FTC architecture and its conventional capacity blocks
│   ├── connections.py                Context aggregation, membership propagation, neighborhoods
│   ├── mask_decoder.py               CM projections, FiLM skips, fusion, unary prediction, readout
│   ├── image_encoder.py              CLIP vision features and the offline tiny vision encoder
│   ├── text_encoder.py               Frozen CLIP/tiny question encoding and token masking
│   └── benchmark_unet.py             Replicated U-Net adapter available only to the benchmark
│
├── experiments/
│   ├── __init__.py                   Experiment-tools folder description; no startup side effects
│   ├── launch.py                     Manifest selection, preflight/train/test processes, final upload
│   ├── summarize.py                  Seed metrics, sample standard deviations, paired CM–FTC changes
│   ├── hub.py                        Completed-run/manual Hub upload and verified pinned restore
│   └── export.py                     Result archives, optionally including checkpoint payloads
│
├── diagnostics/
│   ├── __init__.py                   Diagnostic-tools folder description; no startup side effects
│   ├── common.py                     Shared inference loading, target-area bins, and plotting
│   ├── thresholds.py                 Validation calibration and fixed-threshold test evaluation
│   ├── questions.py                  Actual-text interventions, fixed donors, sensitivity summaries
│   ├── roles.py                      Membership loss/confusion/frequencies across target-area bins
│   └── benchmark.py                  Forward/decoder latency, memory, throughput, parameter counts
│
├── configs/
│   └── high_priority.json            Ten-run matrix, common protocol, public Hub destination
│
├── docs/
│   ├── structure.md                  This guide: file ownership and command migration
│   ├── kaggle.md                     Notebook setup, secrets, launch, backup/restore, exports
│   ├── vast.md                       SSH setup, tmux, launch, backup/restore, storage handling
│   ├── verification.md               Completed checks and remaining hardware/live-transfer work
│   └── assets/
│       └── task.png                  Existing task illustration
│
└── tests/
    ├── __init__.py                   Makes test helpers and opt-in smoke commands importable
    ├── support.py                    Small synthetic train/validation/test datasets
    ├── test_mechanism.py             CM role semantics, gradients, borders, train/resume/inference
    ├── test_experiments.py           Model controls, objectives, metrics, diagnostics, selection, presets
    ├── test_hub.py                   Offline Hub, transfer failures, restoration, final-upload flow
    ├── test_commands.py              Script/module entry points and real subprocess launch behavior
    ├── fixtures/
    │   ├── cm_reference.pt           Tiny CM reference weights, inputs, and outputs
    │   └── ftc_reference.pt          Tiny FTC reference weights, inputs, and outputs
    └── smoke/
        ├── __init__.py               Opt-in hardware checks; not ordinary test discovery
        ├── ddp.py                    CPU/Gloo validation coverage and score agreement
        └── decoder_gpu.py            Full-size CM decoder CUDA gradients and memory smoke check
```

## How the files fit together

The launcher reads the manifest and runs `preflight.py`, `train.py`, and `eval.py`. Those scripts share the dataset, model factory, losses, validation, and native-grid metrics. The checkpoint loader reconstructs each architecture from explicit settings. Diagnostics use that same loader. The Hub uploader runs after training/test processes exit and publishes latest/best checkpoints, run records, and final metrics together.

Changes to CM computation belong in `models/connections.py` or `models/mask_decoder.py`. A new supported model choice belongs in `models/factory.py` and its implementation file. Study-matrix changes belong in `configs/high_priority.json`. Calibration and question-sensitivity changes belong in `diagnostics/`. Platform instructions belong in `docs/` and call the same root scripts and tools.

`utils.py` has been replaced by three files with specific jobs: `checkpoints.py` owns model/checkpoint identity and reconstruction, `runtime.py` owns execution/provenance records, and `artifacts.py` owns shared artifact IO. `models/model.py` has become `models/cm.py`; model selection now lives in `models/factory.py`.

Parameter names, tensor layouts, checkpoint format version, and model metadata are preserved. Current-format checkpoints created before this reorganization remain loadable. Older checkpoint formats stay unsupported. The architecture, training protocol, selection rules, native-resolution metrics, and completed-run upload policy retain their behavior.

## Commands after the move

Use the existing root commands directly, for example `uv run --frozen python train.py ...`. Root-folder tools run through `python -m ...`:

| Previous module/path | Current module/path |
|---|---|
| `train.py`, `eval.py`, `IoU.py` | Unchanged |
| `predict_save.py`, `visualize_predictions.py` | Unchanged |
| `preflight.py`, `bake_exif.py` | Unchanged |
| `-m revision.launch` | `-m experiments.launch` |
| `-m revision.summarize` | `-m experiments.summarize` |
| `-m revision.hub` | `-m experiments.hub` |
| `-m revision.export` | `-m experiments.export` |
| `-m revision.thresholds` | `-m diagnostics.thresholds` |
| `-m revision.questions` | `-m diagnostics.questions` |
| `-m revision.roles` | `-m diagnostics.roles` |
| `-m revision.benchmark` | `-m diagnostics.benchmark` |
| `revision/high_priority.json` | `configs/high_priority.json` |
| `tests/ddp_smoke.py` | `torchrun --standalone --nproc_per_node=2 --module tests.smoke.ddp DATA_ROOT` |
| `tests/gpu_smoke.py` | `python -m tests.smoke.decoder_gpu` |

The former `revision/` folder is removed. Update saved notebook/module commands using this table. The README and platform guides already use the new paths.

## Generated files and external data

Dataset paths and artifact locations retain their meaning. `data/`, `outputs*/`, and `results*/` are ignored by Git; existing datasets, checkpoints, and result archives are not moved. The default experiment root remains `outputs/revision/<run-id>/`:

| Artifact | Purpose |
|---|---|
| `run.json` | Resolved settings, persistent run UUID, parameter counts, source/environment record |
| `history.jsonl` | Per-epoch training/validation records |
| `latest.pt` | Resumable model, optimizer/scaler state, progress, best-selection history |
| `best.pt` | Validation-selected model for inference/test; excludes optimizer state |
| `*.sha256.json` | Saved-checkpoint integrity records |
| `manifest_run.json` | Manifest identity used by the study summarizer |
| `test/metrics.json` | Selected-checkpoint native-grid test metrics and checkpoint identity |
| `test/*.png` | Native binary predictions; remain local or in explicit result exports |
| `hub_status.json` | Confirmed remote commit or last failed-upload record |

Automatic uploads still occur once after each completed run. Intermediate epochs stay local unless explicitly uploaded. Diagnostics, exports, and the paper's existing files keep their chosen output locations.
