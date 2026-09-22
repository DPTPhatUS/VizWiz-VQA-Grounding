# Grounding architecture experiments

Base commit: `db29b302bf6c3bc49ab743dfd1bd4bd796d7a4b6`.
This branch defaults to **separate-qa-skips**: residual fusion plus FiLM driven by separately pooled question and answer tokens, mixed by a learned low-rank gate; missing/truncated answers fall back to the question.

Measured parameter change relative to the original model: **+596,865**. Encoder weights are unchanged. Counts include all added trainable parameters, including residual fusion where present.

## Run

From this branch checkout, with the project's dependencies installed:

```bash
python train.py --data-root /absolute/path/to/data/vizwiz --output-dir outputs/separate-qa-skips --seed 42 --validate-every 0
python eval.py --data-root /absolute/path/to/data/vizwiz --dataset val --checkpoint outputs/separate-qa-skips/checkpoint_epoch100.pt --output-dir results/separate-qa-skips
python -m unittest discover -s tests -v
```

Worktrees do not contain the ignored datasets or virtual environment. Use absolute dataset paths and the original checkout's `.venv/bin/python`, or create your own environment. Use separate output directories for each architecture and seed. Start each architecture from pretrained CLIP and a fresh decoder; an original grounding checkpoint cannot be strictly resumed into a changed architecture. Training checkpoints from this same branch resume with `--resume-checkpoint` and the same model options. Final exports carry weights and experiment identity but no optimizer, and are intended for evaluation. Legacy raw state dictionaries are accepted only if their keys/shapes match strictly; they have no identity metadata.

## Comparisons

- `exp/residual-fusion`: original model + scalar residual fusion.
- `exp/joint-text-skips`: residual fusion + joint-text FiLM on three skips.
- `exp/separate-qa-skips`: joint-text variant with separate question/answer pooling and a learned mixture.
- `exp/detail-refinement`: original (non-residual) fusion + RGB detail refinement, independently of FiLM.
- `exp/compact-decoder`: residual fusion + 128-channel decoder and joint-text conditioning by default; separate Q/A selectable.

Compare residual against original main, joint against residual, separate against joint, detail against original main, and compact against its matching wide conditioned model. The compact default is not an empirically selected winner. Repeat seeds and keep preprocessing, data, epochs, batch size and learning rate fixed. First use clean RGB, then separately evaluate YOLO overlays. No accuracy improvements have been established by implementation tests.

## Preserved baseline limitations

To avoid mixing unrelated corrections into architectural ablations, image normalization, original cross-attention padding behavior, losses and training augmentation are unchanged. New pooling excludes padding and BOS/EOS. `train.py`'s built-in validation still inherits random augmentation; use `eval.py` on saved checkpoints for deterministic validation (and keep `--validate-every 0`). Any future normalization, augmentation or attention-mask correction must be applied equally to the original baseline and every variant and reported as a separate control. Answer text is used only when supplied in dataset metadata, exactly as in the original pipeline.

## Verification scope

CPU unit tests use small encoder fixtures to avoid pretrained downloads, exercising real fusion/conditioning/decoder modules. They cover gradients, shape contracts, text masks, identity initialization, state-dict round trips and experiment identity rejection. Full-resolution shape checks use meta tensors where appropriate. Full training, CUDA throughput and mean IoU must be measured separately.

## Text spans

One CLIP text-encoder pass produces all tokens. Joint pooling excludes padding and BOS/EOS; separate pooling also excludes the `Q:` / `A:` markers. Separate mode uses the rightmost ` A: ` delimiter from the existing dataset format. Extremely long questions retain the original 77-token truncation policy; when no answer tokens survive, conditioning falls back to the question. Literal delimiter text inside an answer is ambiguous in the existing string format; structured question/answer inputs would be a separate data-interface change.

## Review and verification record

Implemented with CPU forward/backward tests, production-channel meta-tensor checks, strict checkpoint round trips and CLI smoke checks. A separate code review identified single-string tokenization and legacy checkpoint-loading compatibility issues; both were addressed with the shared text-input regression test and checkpoint loader integration. No GPU training or accuracy evaluation was run.
