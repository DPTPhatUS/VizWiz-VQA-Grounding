# Research experiments design

Implements the three proposals approved in the October 6 research discussion.

Create exp/grounding-controls from exp/compact-decoder; branch exp/answer-value-distillation, exp/gain-guided-refinement and exp/evidence-extent from that shared commit. Preserve historical entry points and branches. New entry points are train_research.py and eval_research.py.

Shared protocol: CLIP-normalized vision input, padding-masked cross-attention, question-only deployment and validation, optional answer-dropout teacher training, no augmentation by default, BCE plus configurable Dice, original-resolution evaluation, best validation mIoU selection, strict resume vs explicit weight initialization, CPU/single GPU/torchrun support, seeded loaders, provenance/configuration in checkpoints and metrics. Inputs remain raw RGB until the model normalizes them.

A: frozen answer-dropout teacher, paired question/question-answer views, detached incremental BCE-improvement weights; ordinary/confidence/error KD ablations. Student is compact. No teacher in evaluation.

B: compact global model plus shallow shared high-resolution crop refiner, global regular candidate coverage and question-aware expected-IoU-gain router. Separate refiner and router stages; freeze coarse reference, then freeze refiner during router training. Fixed/random/uncertainty/relevance/gain controls. Negative-gain crops may be skipped. No oracle in standard evaluation.

C: compact features plus location and extent tokens, support envelope supervision, final mask and area supervision. Optional explicitly annotated same-image question pairs; signed prediction difference and verified paraphrase consistency. Require pair file when paired loss is enabled; never invent pair masks.

Training is not launched as part of implementation. Offline tests use small actual PyTorch encoders/fixtures, gradients, checkpoint roundtrips, synthetic image datasets, one-epoch training and original-size evaluation. GPU throughput and model accuracy require later training.
