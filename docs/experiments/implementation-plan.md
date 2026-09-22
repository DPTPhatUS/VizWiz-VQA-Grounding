# Lightweight grounding experiments

Base: db29b302bf6c3bc49ab743dfd1bd4bd796d7a4b6. User authorized implementation and five GitHub branches.

## Design and execution plan

Implement inline in isolated worktrees; keep main unchanged. Share only text feature extraction, checkpoint identity checking, tests, and documentation. Keep image preprocessing, loss, original cross-attention padding behavior, and augmentation unchanged so unrelated fixes do not confound ablations. New pooling always excludes padding and special tokens.

1. exp/residual-fusion: original model plus learnable scalar residual fusion initialized to 0.01. Test exact fusion, gradients, shape, state-dict round trip.
2. exp/joint-text-skips: residual fusion plus identity-initialized FiLM on three skips, pooled joint text -> 64 -> per-skip scale/shift. Test identity, conditioning after learning, padding exclusion and parameter budget.
3. exp/separate-qa-skips: replace joint pooling with separate question/answer pooling and a learned 768-dimensional gate; shared reduction -> FiLM. Missing/truncated answer falls back to question. Test token masks, absent answers, learned Q/A sensitivity and budget.
4. exp/detail-refinement: original fusion plus RGB 16/32-channel depthwise-separable detail branch at quarter resolution and semantic gating; zero-initialized residual-logit head preserves original output. Test odd input sizes, gradients and parameter budget.
5. exp/compact-decoder: residual fusion plus projected 128-channel features, joint/separate Q/A conditioning, depthwise-separable aggregation and a narrow progressive upsampler. Default joint; --conditioning separate selects Q/A. No empirical winner claimed. Test output sizes, gradients and parameter reduction.

## Files and interfaces

models/text_encoder.py adds optional return_features=True yielding tokens plus valid/question/answer masks while retaining tensor return by default. models/experiment.py contains each branch's added modules. models/model.py retains GroundingModel(image, text) interface and uses branch modules. models/checkpoint.py validates saved experiment identity before strict loading. train.py and eval.py record/validate configuration and expose --conditioning only where applicable. tests use unittest and CPU tensors; pretrained downloads are replaced at encoder boundaries. Real module gradients, mask handling, checkpoint round trips and parameter budgets are checked. CLI --help and compile checks cover scripts. No full training or accuracy claims without actual runs.

## Completion checks

Run unittest discovery, CLI smoke checks, diff checks and parameter counts on all five worktrees. Review across branches. Commit each branch, push only the five new refs to origin, verify remote heads. No merge, PR or training job requested.
