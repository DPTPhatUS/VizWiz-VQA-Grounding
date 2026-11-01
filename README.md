# Context–Membership grounding experiments

This branch contains the paper's question-only Context–Membership (CM) models, the exact frozen-text compact control (FTC), and a conventional convolutional capacity control. It provides one training/evaluation interface for the matched revision study. Training the full study is an explicit action; installing dependencies or running tests does not start it.

## Models and ablations

| Choice | Options | Decoder parameters |
|---|---|---:|
| FTC | `--model ftc` | 690,753 |
| FTC capacity control | `--model ftc-capacity` | 1,120,845 |
| CM-Shared | `--model cm --connection-mode shared` | 942,953 |
| CM-Separate | `--model cm --connection-mode separate` | 1,120,585 |
| Context-only | CM-Separate plus `--disable-membership` | Reports registered and trainable counts separately |
| Membership-only | CM-Separate plus `--disable-context` | Reports registered and trainable counts separately |
| CM without relation supervision | CM-Separate plus `--edge-loss-weight 0` | 1,120,585 |

The capacity control adds three residual blocks, each expanding 128 to 532 channels, applying a depthwise 3×3 convolution, and projecting back to 128, with GroupNorm and GELU. It adds 430,092 active parameters, only 260 more than CM adds to FTC. These blocks follow skip fusion and precede the first upsampling stage.

Only the CLIP text encoder is frozen. Images use CLIP normalization and 336×336 input; every model uses question-only text (`Q: ...`). Segmentation uses BCE. CM adds 0.1 times the class-balanced directed membership loss unless its weight is explicitly zero. No Dice/Lovász, detail pyramid, answer generator, or crop pathway is introduced.

Context-only retains context aggregation, unary prediction, and the common CM readout. Membership-only feeds uncontextualized features to unary prediction and retains local propagation. Disabling both preserves the unary/readout control; it is **not FTC**. Disabled modules remain registered to preserve initialization and checkpoint layout, but unused parameters are frozen. `shared`/`separate` controls parameter tying, independently of component switches.

## Install and check

Use Python 3.12 or newer and the committed lockfile:

```bash
uv sync --frozen
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python preflight.py --data-root /path/to/vizwiz --dataset-check-only
```

The dataset root contains `train_grounding.json`, `val_grounding.json`, `test_grounding.json`, image folders `train/`, `val/`, `test/`, and `binary_masks_png/<split>/`. Preflight checks filenames, questions, and EXIF-adjusted image/mask dimensions, then runs a finite forward/backward/optimizer step unless `--dataset-check-only` is set. CLIP's unused pooled-output normalization parameters are reported separately; disconnected prediction parameters fail preflight. Add `--tiny --device cpu` only for synthetic smoke data, never for publication runs.

For a model-specific preflight:

```bash
uv run --frozen python preflight.py --data-root /path/to/vizwiz --connection-mode separate --disable-context
```

## The matched 20-epoch study

`revision/high_priority.json` specifies ten fresh runs:

- FTC and full CM-Separate at seeds 42, 123, and 456;
- Context-only, membership-only, and no-auxiliary-loss CM at seed 42;
- FTC capacity control at seed 42.

Every run uses 20 epochs, global batch 4, Adam at 1e-5, zero weight decay, gradient clipping at 1.0, and validation after every epoch. The highest native-grid validation mIoU at threshold 0.5 selects `best.pt`; ties retain the earlier epoch. Only that checkpoint is evaluated on test. Seed-42 references are retrained so the selection procedure matches the new ablations and seeds.

Preview commands without creating output directories:

```bash
uv run --frozen python -m revision.launch --data-root /path/to/vizwiz --processes 2 --dry-run
```

Launch one selected run on two GPUs:

```bash
uv run --frozen python -m revision.launch --data-root /path/to/vizwiz --processes 2 --runs context_only_s42
```

Omit `--runs` to execute all ten sequentially, or list several IDs. Use `--device cpu --processes 1` only for smoke manifests. `--output-root` defaults to `outputs/revision`. A fresh launch refuses existing nonempty run directories. Two-process training uses a global batch of 4, hence microbatch 2 per GPU. Validation uses non-padding rank shards and restores training/RNG state.

The preset uploads each **completed run** to the **public** Hugging Face model repository `Last-Bullet/vizwiz-cm-aciids-2027`. Authenticate on the training machine with a write-enabled token before launching:

```bash
uv run --frozen hf auth login
```

Alternatively, supply `HF_TOKEN` through the platform's secrets/environment mechanism. Tokens are never command-line arguments or saved settings. Before training, the launcher creates the repository if needed, verifies write access and visibility, and rejects a remote path belonging to another run. `--dry-run` does not contact the Hub. Add `--no-hub` for local-only runs and offline smoke manifests.

The rolling policy writes:

- `latest.pt`: resumable model, optimizer/scaler state, settings, and best score/epoch;
- `best.pt`: selected model weights and metadata for inference; it cannot resume training;
- `run.json`, `history.jsonl`, SHA-256 sidecars, and `manifest_run.json`;
- `test/metrics.json` and native-resolution binary PNG predictions.

Preserve **both** checkpoint files across sessions. Full CLIP/Adam checkpoints are several GB; allow roughly 6 GB per active run for latest plus best, in addition to masks and other outputs. Atomic local saving also needs temporary room for another `latest.pt`. Copy completed runs off Kaggle between sessions as needed.

## Hugging Face backups and machine transfers

After training and selected-checkpoint test evaluation finish, the launcher uploads `latest.pt`, `best.pt`, SHA-256 hashes, run records, `test/metrics.json`, and the manifest record in one Hub commit under `matched-20ep-v1/<run-id>/`. Intermediate epochs remain local. The uploader uses an explicit allowlist: dataset images, prediction PNGs, and unrelated files are excluded. It waits for upload completion before launching the next run. After three failed commit attempts, the launcher stops with the completed local run intact; `--resume` retries without repeating completed training or evaluation.

The remote paths contain the final latest/best pair. Manual intermediate backups add versions to Hub history, increasing storage use; history is never deleted automatically. See the [storage guidance](https://huggingface.co/docs/hub/storage-limits). The integration uses Hugging Face's [commit API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#create_commit).

Restore a completed run, or an explicitly uploaded intermediate checkpoint, onto a new Kaggle or Vast.ai machine:

```bash
uv run --frozen python -m revision.hub download --path-in-repo matched-20ep-v1/context_only_s42 --run-dir outputs/revision/context_only_s42
uv run --frozen python -m revision.launch --data-root /path/to/vizwiz --processes 2 --runs context_only_s42 --resume
```

Downloads pin one remote commit, verify every file's hash, and refuse a nonempty destination. `--revision COMMIT_SHA` restores a specific historical backup. Resume on the same number of processes; keep two processes across Kaggle and Vast.ai when combining their results in this study.

Back up an unfinished run before ending a rental session, or retry an upload after a connection failure. Pause/stop training at a completed epoch first so the two checkpoint files are stable. `--dry-run` inspects the upload without contacting the Hub:

```bash
uv run --frozen python -m revision.hub upload --run-dir outputs/revision/context_only_s42 --path-in-repo matched-20ep-v1/context_only_s42
```

`hub_status.json` records the last confirmed commit. Uploads reject another run's destination or progress older than the remote backup. A genuinely fresh rerun needs a new local output root **and** a new Hub prefix (`--hf-prefix`). `--hf-repo` overrides the preset destination. The standalone Hub commands also accept `--repo-id`.

Resume selected runs in their existing output directories:

```bash
uv run --frozen python -m revision.launch --data-root /path/to/vizwiz --processes 2 --runs context_only_s42 --resume
```

Resume preserves the seed, global batch, learning rate, objective, architecture, validation policy, and process count. If the 20-epoch run is complete, the launcher proceeds to any missing test evaluation. It rejects stale test results from another checkpoint.

Summarize completed runs; missing runs are listed explicitly:

```bash
uv run --frozen python -m revision.summarize --root outputs/revision --output-dir outputs/study_summary
```

The JSON/CSV exports include per-seed metrics, sample standard deviations, and paired CM-minus-FTC differences. Percent-valued quantities are stored as fractions, except severe-failure counts. A singleton configuration has no estimated standard deviation.

## Direct training and checkpoint format

Direct commands default to full CM-Shared, 100 epochs, no automatic validation, and periodic checkpoints every ten epochs. The revision preset sets all study choices explicitly. For a direct matched no-auxiliary run:

```bash
uv run --frozen python train.py --data-root /path/to/vizwiz --connection-mode separate --edge-loss-weight 0 --num-epochs 20 --batch-size 4 --lr 0.00001 --seed 42 --val-every 1 --checkpoint-policy rolling --output-dir outputs/noaux_s42
```

The logged `membership` loss is unweighted and can remain nonzero when its objective weight is zero. Positive relation-loss weights are rejected if membership is disabled. CM-only flags are rejected for FTC.

For a direct training command, use `revision.hub upload` explicitly after training/evaluation. Automatic completed-run uploads are provided by `revision.launch`.

All training, evaluation, prediction, and diagnostics use one checkpoint format (`format_version=1`) with explicit model/component settings. Older checkpoints and raw state dictionaries are no longer supported; their files are left untouched. Architecture identity and tensor loading remain strict. Reconstruction avoids redundant pretrained CLIP downloads; the tokenizer is loaded when text is first encoded.

```bash
uv run --frozen python eval.py --checkpoint /path/to/best.pt --data-root /path/to/vizwiz --dataset val --output-dir outputs/model_val
uv run --frozen python predict_save.py --checkpoint /path/to/model.pt --image photo.jpg --question 'What does it say?' --output prediction.png
uv run --frozen python visualize_predictions.py --checkpoint /path/to/model.pt --image photo.jpg --question 'What does it say?' --output overlay.png
```

`IoU.py` remains an evaluation alias. `bake_exif.py` remains an optional dataset utility; the loader already applies EXIF orientation at runtime. The old hard-coded `countnum.py` is replaced by preflight.

## Threshold calibration

For each FTC/CM checkpoint, sweep probability thresholds 0.10–0.90 in increments of 0.05 on validation:

```bash
uv run --frozen python -m revision.thresholds validation --checkpoint /path/to/model.pt --data-root /path/to/vizwiz --output-dir outputs/calibration/model_val
uv run --frozen python -m revision.thresholds test --checkpoint /path/to/model.pt --data-root /path/to/vizwiz --selection outputs/calibration/model_val/selection.json --output-dir outputs/calibration/model_test
```

Validation exports overlap counts, mIoU, Acc@0.5, macro precision/recall, severe failures, mask-area ratio, relative area error, and PDF curves. Selection maximizes validation mIoU; ties favor the threshold closest to 0.5, then the lower threshold. `selection.json` binds the selected threshold to the checkpoint SHA. Test evaluates only that threshold and 0.5. A `--limit` smoke run cannot produce a calibration selection. Ratios with zero ground-truth area are null and excluded from area-ratio means, with support reported.

## Question dependence and relation roles

Generate a single validation donor mapping and reuse it for every model:

```bash
uv run --frozen python -m revision.questions --checkpoint /path/to/model.pt --data-root /path/to/vizwiz --donors outputs/diagnostics/donors_val_s2027.json --output-dir outputs/diagnostics/model_questions
uv run --frozen python -m revision.roles --checkpoint /path/to/cm_model.pt --data-root /path/to/vizwiz --output-dir outputs/diagnostics/model_roles
```

Question conditions are original, repeated original, a fixed no-self permutation (seed 2027), generic, and empty. Identical normalized donor questions are excluded from shuffled-condition means and counted. The intervention changes `text`, the input actually consumed by CM/FTC. Cached image features are checked against ordinary forward inference on the first batch; `--no-cache-images` disables reuse. Changed-question IoU against the original mask is descriptive sensitivity, not ordinary grounding accuracy.

Role diagnostics require CM with membership enabled and use validation only. They export include/exclude/neutral frequencies, mean probabilities, balanced soft loss, hard-target confusion and macro recall, and native mask-area error across the five paper target-size bins. Hard targets use argmax of the existing fractional role targets, with deterministic index-order ties. No background-clutter category is inferred.

## Efficiency

```bash
uv run --frozen python -m revision.benchmark --device cuda --output outputs/benchmark_fp32.json
uv run --frozen python -m revision.benchmark --device cuda --precision fp16 --output outputs/benchmark_fp16.json
```

The benchmark covers FTC, both full CM variants, the capacity control, and a benchmark-only replicated U-Net adapter. It uses pretrained CLIP and initialized decoder weights, a fixed question, batch 1, 336×336 input, 10 warm-ups, 50 measurements, and disabled TF32. This is structural timing, not an accuracy evaluation. Use the same GPU and settings for all models.

Model-forward timing includes tokenization/text encoding but excludes file loading, host-to-device input-image transfer, and native-output resizing. Decoder timing uses cached image/text/fusion inputs and excludes encoders and bottleneck cross-attention. Peak allocated/reserved CUDA bytes include resident model/input memory; decoder measurements also include cached inputs. Throughput is the reciprocal of mean batch-1 latency. CPU/tiny timings are smoke checks, not deployment measurements.

## Verification and exports

The offline tests preserve CM/FTC reference outputs and test current-format reconstruction, disabled paths and gradients, selection/resume, donor interventions, role/threshold metrics, and mocked Hub backup/recovery. Opt-in hardware checks:

```bash
uv run --frozen torchrun --standalone --nproc_per_node=2 tests/ddp_smoke.py /path/to/vizwiz
uv run --frozen python tests/gpu_smoke.py
```

The DDP validation smoke uses a tiny model and the provided validation split; CPU local sockets must be available. For a quick check, use the synthetic split generated by `tests.support.fixture_data`. `gpu_smoke.py` checks full-size CM decoder gradients without training CLIP.

Export records/results by default; opt into checkpoints when transferring a resumable session:

```bash
uv run --frozen python -m revision.export --root outputs/revision --output /path/outside/revision/results.zip
uv run --frozen python -m revision.export --root outputs/revision --output /path/outside/revision/resume.zip --include-checkpoints
```

See [KAGGLE.md](KAGGLE.md) and [VAST.md](VAST.md) for commands calling these same tools. Full dataset training, CUDA timing, and scientific results must be obtained on the intended hardware before updating the paper.

See [VERIFICATION.md](VERIFICATION.md) for completed checks and remaining GPU validation.
