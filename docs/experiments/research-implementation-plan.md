# Grounding Research Experiments Implementation Plan

> Execute inline with superpowers:executing-plans; final independent review with superpowers:requesting-code-review.

**Goal:** runnable, isolated branches for the three approved research proposals.
**Architecture:** shared corrected compact training/evaluation package, then independent variant modules with model, objective and CLI hooks. Keep legacy model entry points usable.
**Tech Stack:** Python, PyTorch, Transformers, PIL; unittest without network downloads.
**Spec:** research-design.md

## Global Constraints
- Question-only validation and deployment; privileged answers only for explicit teacher/training modes.
- No model downloads during tests; no training on real data in this task.
- Resume strictly validates configuration; weight initialization is separate.
- Image, crop and paired masks use aligned geometry.

## Review Focus
- Padding and batch composition, empty text, and missing answers.
- DDP validation coverage, synchronization, and staged frozen-module gradients.
- Original-size masks and high-resolution crop alignment.
- Resume configuration, optimizer, random state and best-checkpoint semantics.
- Unavailable paired annotations and accidental answer/test-label leakage.

### Task 1: Shared controls
Files: research/{data,model,losses,checkpoint,engine,variant}.py; train_research.py; eval_research.py; tests/test_research_controls.py.
Interfaces: model(batch)->dict(logits, optional features); Objective(model,batch)->(loss, metrics); variant build_model(args), build_objective(args,model,device), add_arguments(parser), validate_args(args).
- [ ] Write and run failing data/mask/loss/checkpoint tests.
- [ ] Implement shared protocol, train/eval CLI, provenance and resume.
- [ ] Run offline unit and end-to-end smoke checks; commit controls.

### Task 2: Answer-value distillation
Files: research/distillation.py, research/variant.py; tests/test_distillation.py; run documentation.
- [ ] Fail tests for useful-answer weights, teacher detachment, missing answers, KD modes.
- [ ] Implement frozen teacher objective and branch-specific arguments.
- [ ] Verify gradients, training/evaluation and checkpoint behavior; commit.

### Task 3: Gain-guided crops
Files: research/refinement.py, research/variant.py; tests/test_refinement.py; run documentation.
- [ ] Fail tests for crop coordinates, gain targets, selection and stage freezing.
- [ ] Implement crop refiner, router, two stages and ablations.
- [ ] Verify two-stage smoke training and question-only evaluation; commit.

### Task 4: Evidence extent
Files: research/extent.py, research/variant.py; tests/test_extent.py; paired-data documentation.
- [ ] Fail tests for query-token gradients, envelope/area losses, verified pairs and signed changes.
- [ ] Implement two-token decoder and optional paired supervision.
- [ ] Verify paired/unpaired training and evaluation; commit.

### Task 5: Review and handoff
- [ ] Independent review of shared code and all variants, then fix material findings with regression checks.
- [ ] Verify clean branches, documented commands, full offline suites and synthetic CLI runs.
- [ ] Deliver branch names and limitations without merging or pushing.
