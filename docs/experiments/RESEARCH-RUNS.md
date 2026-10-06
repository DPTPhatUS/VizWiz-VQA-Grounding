# Corrected grounding controls

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
