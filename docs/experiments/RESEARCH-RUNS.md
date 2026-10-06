# Answer-value distillation

Branch: `exp/answer-value-distillation`. Uses the corrected controls protocol below.
The inference network is the compact model with **no extra parameters or teacher dependency**.
This is an experimental hypothesis, not a demonstrated accuracy improvement.

First train a teacher on `exp/grounding-controls` using `--architecture joint --text-mode dropout --answer-dropout 0.5`. Its two views share weights; incremental KD rejects answer-only teachers because their question-only view is not trained. Train a corrected compact question-only control as the student initialization if comparing matched fine-tuning schedules.

From this branch:

```bash
python train_research.py --data-root /path/to/data/vizwiz --teacher-checkpoint /absolute/path/teacher/best.pt --init-checkpoint /absolute/path/compact/best.pt --kd-mode incremental --kd-weight 1 --num-epochs 100 --seed 42 --output-dir outputs-avd-s42
python eval_research.py --data-root /path/to/data/vizwiz --checkpoint outputs-avd-s42/best.pt --dataset test --output-dir results-avd-s42
python train_research.py --data-root /path/to/data/vizwiz --teacher-checkpoint /absolute/path/teacher/best.pt --kd-mode incremental --kd-weight 1 --num-epochs 150 --seed 42 --output-dir outputs-avd-s42 --resume-checkpoint outputs-avd-s42/last.pt
```

Ablations: `--kd-mode none|ordinary|confidence|error|incremental`; none does not need a teacher. All KD variants use identical foreground/background balancing, teacher, initialization, seed and training schedule. Ordinary uses unit weights; confidence uses one minus normalized Bernoulli entropy; error uses exp(-teacher BCE); incremental uses positive BCE(question teacher)-BCE(answer teacher), multiplied by answer confidence. Each nonempty foreground/background region receives equal total mass; pixels with zero gain stay zero. KL is averaged over the entire batch, so examples with missing answers or no positive gain contribute zero KD. Segmentation supervision always remains active. The teacher is frozen/eval and both teacher views reuse image features.

The student always consumes the question view, including training. The checkpoint stores only student weights and the teacher path/hash as training provenance. Evaluation works after moving/deleting the teacher file. Resume requires the same teacher file contents. Checkpoints must use the versioned research format, matching image size and real/tiny mode.

Repeat seeds 42/43/44. Report mean/std, matched training steps, teacher training cost separately, and runtime measured on the same hardware. Validation selects checkpoints; keep test results out of model/weight selection. Sweep KD weight on validation (e.g. 0.25, 0.5, 1). There is no claim of publication novelty without experimental comparison to existing privileged-information/distillation methods.

CPU verification covers frozen teacher gradients, beneficial/harmful answer weighting, no-answer/no-gain behavior, train/resume, and original-size evaluation after teacher removal. CUDA training and real-data accuracy have not been run here.

---

# Corrected grounding controls

The following teacher/control commands run in the `exp/grounding-controls` worktree.

Branch: `exp/grounding-controls`. Derived from `exp/compact-decoder`.
Use the new research entry points below; legacy `train.py`/`eval.py` retain their historical behavior.

Run commands from this worktree with the repository's environment (or install its dependencies on your training machine). Data is not duplicated: provide an absolute `--data-root`.

```bash
python train_research.py --data-root /path/to/data/vizwiz --architecture compact --text-mode question --num-epochs 100 --seed 42 --output-dir outputs-control-s42
python train_research.py --data-root /path/to/data/vizwiz --architecture joint --text-mode dropout --answer-dropout 0.5 --num-epochs 100 --seed 42 --output-dir outputs-teacher-s42
python eval_research.py --data-root /path/to/data/vizwiz --checkpoint outputs-control-s42/best.pt --dataset test --output-dir results-control-s42
```

Repeat with seeds 43 and 44. Alternatives: `--architecture baseline|residual|joint|compact`, `--dice-weight 1`, `--encoder-lr 1e-6`, `--freeze-encoders`. These define distinct runs; use separate output directories. The baseline preserves its original attention-only bottleneck; other controls use residual fusion.

The default input is 336, with CLIP channel normalization in the model. Cross-attention masks padding while keeping valid BOS/EOS. No random geometric augmentation is enabled. Question-only validation runs every epoch and selects `best.pt` by mean per-image IoU at original resolution. `--metric-resolution local` defines a separate 336-resolution protocol. Masks are resized as logits before thresholding at zero. Evaluation requires ground-truth masks, including the publicly released test masks.

```bash
torchrun --standalone --nproc_per_node=2 train_research.py --data-root /path/to/data/vizwiz --batch-size 4 --output-dir outputs-ddp
python train_research.py --data-root /path/to/data/vizwiz --output-dir outputs-control-s42 --resume-checkpoint outputs-control-s42/last.pt --num-epochs 150
```

Resume requires the same architecture/training protocol, global batch, seed and world size. `--num-epochs` is the final epoch, not additional epochs. Training uses constant learning rates; increasing that cap does not redefine a scheduler. `--init-checkpoint` loads matching research weights into a fresh run. Only load trusted checkpoints. Historical checkpoints are deliberately rejected because they were trained under a different input protocol; they are not silently treated as corrected baselines.

`last.pt` and `best.pt` contain model, optimizer, scaler, epoch, best score, run configuration, RNG states and annotation hashes. The output directory also contains `config.json` and `history.jsonl`. `eval_research.py` constructs the model from checkpoint metadata; it does not need training-time teacher files.

Offline verification:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

`--tiny --image-size 28 --num-workers 0` uses a small offline test model, not CLIP. It exists for integration tests and must never be reported as a trained research architecture.

Strict continuation requires the checkpoint to reside in the same `--output-dir`; use weight initialization for a new run/directory. Provenance hashes both JSON annotations and their binary mask contents (including paired masks). CPU checkpoints can resume on CUDA with a fresh AMP scaler; CUDA execution itself is unverified here.

Strict resume accepts `last.pt` only, avoiding stale best-checkpoint selection when rewinding within a run. Start a new initialized run to reuse older/best weights.
