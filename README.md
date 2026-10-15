# Latent-answer distillation

This branch trains a compact question-only student with a learnable **latent answer query** and **Q/A skip gating** using single-stage self-distillation (or an optional frozen teacher via `--teacher-checkpoint`). During training, the privileged `Q + A` branch supervises both the latent answer representation (via cosine + Smooth L1 alignment) and the spatial mask predictions (via answer-value gain-weighted KL distillation); at inference, only the question and image are needed.

The experiment uses the existing `train.py`, `eval.py`, `dataset.py`, and `models/` layout. CLIP input is fixed at 336 pixels; architecture and loss settings are defined in code.

## Kaggle

Clone branch `exp/latent-answer-distillation`. Enable a GPU and Internet for CLIP downloads. Use Kaggle's installed Torch/Torchvision and install missing dependencies:

```bash
pip install transformers tqdm pillow
```

The dataset root contains `train_grounding.json`, `val_grounding.json`, `test_grounding.json`, image directories `train/`, `val/`, `test/`, and masks in `binary_masks_png/<split>/`. Evaluation requires masks for the selected split.

Run in a `%%bash` cell, replacing the dataset path:

```bash
set -e
cd /kaggle/working/VizWiz-VQA-Grounding
DATA_ROOT=/kaggle/input/YOUR_DATASET/vizwiz
# Single-stage self-distillation (no separate teacher training required; optional --teacher-checkpoint or --init-checkpoint supported).
python train.py --data-root "$DATA_ROOT" --num-epochs 100 --batch-size 4 --num-workers 4 --seed 42 --output-dir /kaggle/working/outputs

python eval.py --data-root "$DATA_ROOT" --checkpoint /kaggle/working/outputs/model_final_epoch100.pt --dataset test --batch-size 4 --num-workers 4 --output-dir /kaggle/working/results
```

Training selects CUDA automatically and supports `torchrun`; batch size is global across GPUs. `--seed`, `--num-workers`, and the usual training/data/output controls are retained. Evaluation accepts an optional `--device`. Run either script with `--help` for its short argument list.

## Single-image prediction

```bash
python predict_save.py --checkpoint /path/to/model_final_epoch100.pt --image /path/to/image.jpg --question "What does it say?" --output /kaggle/working/mask.png
python visualize_predictions.py --checkpoint /path/to/model_final_epoch100.pt --image /path/to/image.jpg --question "What does it say?" --output /kaggle/working/overlay.png
```

`IoU.py` delegates to `eval.py`. Existing experiment weight keys are preserved. Raw weights and ordinary checkpoint dictionaries are accepted; model tensor names and shapes must match this branch. `docs/` and unrelated local tests remain ignored; the latent regression test is tracked. CPU smoke tests do not establish real-data accuracy or GPU performance.


## Training output and separate evaluation

Training reads only the training split: no validation dataset is loaded and no validation runs, including after the final epoch. This applies to both teacher and student training.

At completion, training saves exactly one raw `state_dict`: `model_final_epoch<N>.pt`, matching the baseline format. It contains model tensors only, with no optimizer, scaler, epoch wrapper, or RNG state. There are no automatic `last.pt` or `best.pt` saves, and `--validate-every` is removed. `--save-every` defaults to **10** and saves resumable `checkpoint_epoch10.pt`, `checkpoint_epoch20.pt`, etc., containing model, optimizer, scaler, and completed epoch. Use `--save-every 0` to disable these periodic checkpoints while still saving the final raw model.

Raw weights use this branch's default model settings. Resumable checkpoints retain their own `run_config` and `experiment_config`; no standalone configuration file is written or read. `history.jsonl` contains training losses, teacher/gain diagnostics, and the objective version. Run `eval.py` separately with `--dataset val` or `--dataset test`; the commands above show the final weight filename for their epoch count.

To train further from these weights, use `--init-checkpoint /path/to/model_final_epoch<N>.pt`. This starts a new optimizer and epoch counter; `--num-epochs` is the number of additional epochs. Default self-distillation requires no separate teacher; repeat `--teacher-checkpoint` only for external-teacher runs. To resume a periodic checkpoint, use `--resume-checkpoint /path/to/checkpoint_epoch10.pt` with its existing output directory or a fresh one; optimizer/scaler state and completed epoch are restored, and `--num-epochs` is the final total epoch count. Repeat the original batch size, seed, learning rate, and experiment-specific training arguments when continuing a run. Older supported full checkpoints remain loadable.

Existing output directories are accepted for all training modes. Matching model filenames are overwritten; other model files remain. History is appended to `history.jsonl`, creating it if absent. To continue in a fresh Kaggle notebook, only the resumable checkpoint is needed to restore training state; no previous history or separate configuration file is required. The training dataset must still be available. Default student continuation remains single-stage and needs no external teacher. Only runs started with `--teacher-checkpoint` need that same external teacher for continuation; optional teacher continuation uses `train_teacher.py`.


## Code layout

- `models/model.py`: the compact question-only student and joint question/answer teacher.
- `models/image_encoder.py` and `models/text_encoder.py`: CLIP feature extraction.
- `models/mask_decoder.py`: the decoder used by this experiment.
- `train.py`: construct the model and dataset, run training, save checkpoints and final weights.
- `eval.py`: construct the model, load weights, evaluate and export masks.
- `utils.py`: batch transfer and a few plain weight-loading helpers.

The separate backbone, experiment, and checkpoint modules and model factory are removed. Training still uses no validation, saves resumable checkpoints every 10 epochs by default, and writes final raw weights.

For standalone teacher weights, use `--teacher` with `eval.py`, `predict_save.py`, or `visualize_predictions.py`. Student inference is the default.

Training shows a batch progress bar for each epoch (rank 0 under `torchrun`), and evaluation shows a batch progress bar. Install `tqdm` alongside Transformers and Pillow in the Kaggle notebook.


## Corrected single-stage objective (latent-gain-v2)

Default training is still one run: the inferred-answer student, real-answer oracle, and Q-only reference share one model and one optimizer. The reference removes the answer token, uses question-only skip conditioning, and receives BCE + 0.5 Dice supervision. It no longer substitutes a question vector for an answer inside an untrained reference branch. The real-answer oracle receives its own segmentation supervision. Its KD targets and gains are detached; its segmentation loss and the Q-reference loss retain gradients.

The default total loss is `student_seg + 0.5 * oracle_seg + 0.5 * question_seg + KD + 0.5 * latent_alignment + 0.2 * attention_guidance`, where each segmentation term uses BCE + 0.5 Dice. Attention guidance pools foreground gain and foreground support to the patch grid, normalizes each independently, and mixes 75% gain with 25% foreground prior. When foreground gain is absent, it falls back to the foreground prior. Empty foreground and missing-answer samples contribute zero attention loss. Missing real answers are masked out of the oracle answer token and latent/KD alignment.

`mean_kd_weight_mass` replaces the misleading `active_fraction`; `kd_pixel_fraction` measures pixel coverage. Teacher QA/Q IoU and their difference are diagnostics on resized training masks, not validation scores. `attention_gain_image_fraction` reports how often foreground gain is available. Epoch history and resumable checkpoints identify `latent-gain-v2`; raw final weights remain model tensors only.

For the matched architecture/loss control, use `--supervised-only`: it trains the same inferred-answer student with BCE + 0.5 Dice, without oracle/Q-reference losses, latent alignment, KD, or attention supervision. Repeat this flag when continuing the control. All experiments should use matched seeds, batch sizes, epoch budgets, native-resolution evaluation, and validation checkpoint selection.

## Checkpoint compatibility and retraining

No model parameters were added, removed, or reshaped. Complete latent checkpoints from the previous version still load strictly for inference/resume, and default inference is unchanged. Existing periodic optimizer/scaler states also remain compatible. However, these checkpoints were trained with the old objective. Train from scratch to report results for `latent-gain-v2`; using `--init-checkpoint` or `--resume-checkpoint` with old weights is a warm start or mixed-objective continuation and should be labeled accordingly. Keep outputs separate from old runs.

Legacy compact checkpoints lacking the latent-answer/gating tensors are permitted only for explicit student `--init-checkpoint` initialization followed by training. Inference, resume, and frozen external-teacher loading require complete model tensors and reject such partial checkpoints. The optional external teacher must have the full latent architecture trained with answer dropout. The default self-distillation run does not need this teacher or a separate training stage.

Run offline regression checks with `python -m unittest discover -s tests -p test_latent_distillation.py`. These cover gain-sensitive attention targets, Q-reference supervision, missing-answer masking, strict teacher loading, and old raw/full checkpoint inference and optimizer compatibility with tiny encoders. They do not establish real-data accuracy or GPU execution.

## Paired latent-answer diagnostic (no retraining)

`diagnose.py` evaluates the **whole validation split** using three views of the
same checkpoint. It does not modify weights, architecture, training or eval.py:

- `question`: remove the answer token and use Q-only skip conditioning.
- `predicted_answer`: infer A′ from image/Q, matching ordinary student inference.
- `oracle_answer`: encode the real answer separately, pool it into the answer
  vector and pass it through the existing oracle path. This is not joint Q+A
  tokenization, a separate teacher, or deployable Q/I inference.

```bash
uv run python diagnose.py \
    --checkpoint outputs/checkpoint_epoch20.pt \
    --data-root /kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz \
    --dataset val \
    --batch-size 4 \
    --num-workers 2 \
    --device cuda \
    --top-k 8 \
    --seed 42 \
    --save-masks \
    --output-dir outputs/diagnostic_latent_ep20
```

Use a single GPU process, not torchrun. Lower batch size if needed. The output
directory must be empty; use a new name for each checkpoint/run. Raw final
weights and full legacy latent checkpoints both work without retraining.
Weights lacking latent/gating tensors are rejected rather than initialized.
The saved objective version is reported when available; raw/older checkpoints
are labelled unknown, not assumed to be corrected `latent-gain-v2` runs.

Outputs:

- `summary.json`: native mIoU, paired gains, helped/hurt counts, GT-area groups,
  descriptive cosine correlations and protocol metadata.
- `per_image.csv` / `per_image.json`: question/answer, three IoUs, paired
  differences, cosine, Smooth L1, vector norms, GT fraction and aspect ratio.
- `selected_cases.json`: highest A′ gains, worst A′ harms, largest oracle–A′
  gaps, failures with nonempty GT under 1% of the image, high-alignment/low-IoU
  cases (cosine >= 0.8 and IoU < 0.5), and a seeded random group. These thresholds
  are descriptive review rules, not answer-correctness criteria.
  `--top-k` is examples per group. Images may appear in several groups.
- `gallery.html` + `cases/`: aggregate scores and image/GT/three-mask overlays
  for selected cases. Download the entire output folder, then open the HTML.
- `masks/{question,predicted_answer,oracle_answer}/`: all native binary PNG
  masks when `--save-masks` is enabled. Omit the flag to export only review overlays.

No manual image selection is needed for aggregate diagnostic results. To also
review specific images, put filenames in a UTF-8 file, one per line, and add
`--review-list filenames.txt`. This adds a gallery group; metrics still cover
the whole split. Ranked examples explain individual failures, not their
prevalence. The random group provides an additional less selective visual check.

Missing-answer records have null oracle/alignment fields and no oracle PNG.
Compare all three mIoUs on the **same answer-available subset**, which is shown
separately from whole-split Q/A′ scores. Cosine is representation alignment,
not answer accuracy; correlations are descriptive, not causal evidence.
The Q-only view is a same-checkpoint counterfactual, not a separately trained
Q-only baseline. An oracle can score worse than A′; it is not a mathematical
upper bound. Use validation for diagnosis/model selection, not test tuning.

All scoring resizes logits to native GT dimensions before thresholding at
logit 0; empty unions score 1, matching eval.py. Gallery rendering preserves
the scoring batch composition, and display thumbnails do not affect metrics.
The diagnostic always uses FP32 inference, without AMP.

Run `python -m unittest discover -s tests -p 'test_latent*.py' -v` for offline
checks covering no real-answer leakage, prediction equivalence, checkpoint
loading, missing answers, native mask sizes and CLI/gallery output. These are
tiny-model CPU checks, not real-data diagnostic findings.

## Step 3: matched training recipes

This adds training choices, **not architecture choices**. Parameter names,
counts, shapes, default inference, eval.py and diagnose.py are unchanged.
Complete existing latent weights remain loadable without retraining.

For a new controlled comparison, choose a recipe and use `--loss-version stable`.
The ordinary `--help` shows these choices alongside data/optimizer settings;
`--help-advanced` also shows loss weights, detachment, warm-up schedule and old
aliases. Every existing flag is still accepted. This is a help/documentation
cleanup, not a change to training defaults or checkpoint settings.

Start with `grounding-only` versus `alignment`. Add `warmup` to test the effect
of oracle-first training; use `full` to compare the complete existing objective.
`full` does **not** include warm-up. More active losses do not imply better masks.
The literature comparison and conditional next steps are in
[LATENT_RELATED_WORK.md](LATENT_RELATED_WORK.md).

| `--training-recipe` | Training objective |
|---|---|
| `full` (fresh default) | Existing inferred-answer segmentation + oracle/Q-reference segmentation + gain-weighted KD + answer alignment + attention guidance |
| `grounding-only` | Inferred-answer segmentation only; same as the old `--supervised-only` alias |
| `alignment` | Inferred-answer segmentation + answer-vector alignment, without oracle/Q-reference/KD/attention losses |
| `warmup` | Answer alignment throughout; grounding begins with real answers, then gradually shifts to inferred answers |

All recipes use image/Q at inference. `grounding-only` still has the latent
predictor and is not a separately trained architecture without that predictor.
Keep CLIP input at 336, segmentation loss, epochs, batch, LR and seed matched.
NaFlex flags from other branches do not apply to this branch.

Fresh runs without new flags retain **`full + legacy` (`latent-gain-v2`)**.
`--loss-version stable` explicitly selects the audited calculations below;
use the same loss version across a new ablation suite. Full/stable records
`latent-gain-v3`, alignment/stable records `latent-alignment-v2`, and
warmup/stable records `latent-warmup-v2`. Detachment appends `-detached`.
Grounding-only's segmentation formula does not depend on loss version.

Example 20-epoch warm-up experiment on two GPUs:

```bash
uv run torchrun --nproc_per_node=2 train.py \
    --data-root /kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz \
    --num-epochs 20 \
    --batch-size 4 \
    --num-workers 2 \
    --lr 0.00001 \
    --seed 42 \
    --save-every 10 \
    --training-recipe warmup \
    --loss-version stable \
    --output-dir outputs/latent_warmup_stable_seed42
```

This uses the default 5 warm-up + 5 transition epochs and default loss weights.
For the other runs, replace the recipe and output directory. No per-loss flags
are needed for the initial comparison. Full-only weights have defaults:
`--oracle-weight 0.5 --question-weight 0.5 --kd-weight 1 --attn-weight 0.2
--gain-mix 0.75`. Full-only weight flags are rejected for other recipes.
Start each comparison from the same pretrained initialization, with its own
output directory; do not resume one recipe into another and call it a fresh result.

For 5 warm-up + 5 transition epochs, oracle/predicted segmentation weights are:

- Epochs 1–5: 100% oracle, 0% predicted, on answer-available samples.
- Epochs 6–10: predicted fraction 0.2, 0.4, 0.6, 0.8, 1.0.
- Epochs 11–20: 100% predicted.

The mixture is between **per-image losses**, not between A/A′ vectors.
Missing-answer samples always use predicted grounding. Alignment remains active
throughout. Set transition epochs to 0 for an abrupt switch. Training must reach
the predicted-only phase. These schedule lengths are starting hypotheses, not
validated optimal settings.

`--detach-predicted-answer` is an optional additional ablation, default off.
It prevents grounding/KD gradients passing directly through A′ into the answer
predictor, while retaining answer-alignment gradients. In full mode, attention
guidance also still trains the predictor. Shared vision/text encoders still
receive grounding gradients through their other paths; this is not total
independence between tasks. Detachment is rejected for grounding-only, or if
no predictor-supervision term remains. Test warm-up vs alignment with the same
detachment setting, then change detachment separately.

Recipe, loss version, weights, detachment and schedule are saved in periodic
checkpoint `run_config.training_config` and history records. Resume restores
these settings and rejects conflicting explicit flags or objective versions.
The epoch counter is absolute, so resume does not restart warm-up. Continue to
repeat the original batch, seed and LR. Relocated external-teacher runs must
point to the same teacher weights; external teachers apply only to full mode.
Raw final weights remain plain state dictionaries and cannot record a recipe
or resume optimizer state. Prefer periodic checkpoints for provenance and resume.
Use `--init-checkpoint` for intentional mixed-objective warm starts, and label
them separately. New training outcomes require new runs; diagnostic of old
weights does not change them into stable-loss results.

## Loss audit and opt-in stable version

**Correct existing mechanics:**

- Segmentation uses BCEWithLogits + per-image soft Dice (default weight 0.5).
  Logits are resized before loss; GT masks remain binary. BCE is an ordinary
  pixel mean, not foreground-balanced. Dice supplies an additional overlap term.
- Mask KD is Bernoulli `KL(p_oracle || p_student)` at each pixel:
  `BCEWithLogits(student, p_oracle) - H(p_oracle)`. Its gradient is the same
  as soft-target BCE. It is not a softmax distribution over all image pixels.
- Oracle targets, entropy and gain/confidence/correctness weights are detached.
  Oracle segmentation and Q-reference segmentation keep their own gradients.
- Patch-attention guidance uses forward KL from the normalized gain/foreground
  target to the predictor's spatial attention, not the reverse direction.

**`--loss-version stable` changes only training calculations:**

1. Mask KL uses stable log-sigmoid probabilities for foreground/background.
   Equal logits give exactly zero, and extreme logits remain finite. Very small
   negative values from float rounding can still occur near equality; they do
   not indicate a mathematically negative KL. The loss is not arbitrarily clamped.
2. Attention adds a small epsilon and renormalizes the predicted distribution
   before log/KL. The legacy floor did not renormalize and cut gradients below
   the floor. The stable target-entropy term handles `0 log 0` explicitly.
3. Answer alignment and attention losses average over valid targets, rather than
   adding zero losses for missing-answer samples to the denominator. With DDP,
   counts are global and compensate for DDP's gradient averaging; ranks with no
   valid answers do not dilute another rank's supervision.
4. Full-recipe oracle segmentation is masked to answer-available samples.
   Legacy mode also applied oracle segmentation to the missing-answer Q fallback,
   effectively adding another Q loss there. Student/Q-reference segmentation
   still covers every image. No-valid-target batches have differentiable zero
   privileged losses.

These are explicitly versioned because their gradient scales differ. The legacy
default is preserved for old experiments. Do not compare total train_loss across
recipes as an accuracy measure: full contains more losses. Weighted loss components
and answer-availability fraction are logged; epoch components are means of batch
objectives, not separate dataset-wide valid-target statistics. Use diagnostic
and native validation mIoU for model comparison.

**Remaining design limitations, not silently changed:**

- The answer target is stop-gradient, but uses the same trainable CLIP text
  encoder. Its representation can drift as Q encoding is optimized; it is not
  a frozen teacher. No duplicate text tower was added.
- Target pooling includes the `A:` prefix and compresses answer tokens into one
  vector. High cosine is not answer correctness or evidence of useful grounding.
- Gain/confidence weights intentionally shrink KD when oracle and Q-only are
  similar or uncertain. Tiny `mean_kd_weight_mass` means weak effective KD;
  increasing KD weight automatically would also amplify unreliable targets.
- Foreground attention guidance and alignment are hypotheses, not guarantees
  of improved masks. The new recipes allow their contribution to be ablated.
- Warm-up does not eliminate incorrect A′ at inference; it only lets the decoder
  adapt to predicted representations before deployment.

Checks cover mathematical KL direction/gradients, zero/extreme probabilities,
missing-answer masks, explicit detachment, schedule endpoints, full training
resume equivalence and two-process CPU/Gloo DDP with uneven valid-answer counts.
Pre-change tiny-model snapshots match all three diagnostic logits, legacy loss
and gradients exactly. CUDA/NCCL/full pretrained accuracy have not been tested.

Formula references: [PyTorch KLDivLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)
and [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.BCEWithLogitsLoss.html).
