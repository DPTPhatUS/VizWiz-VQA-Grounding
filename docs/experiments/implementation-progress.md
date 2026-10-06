# Implementation ledger
Plan: research-implementation-plan.md

Ruling: user explicitly requested implementation of the presented proposals; proceed through the shared controls and three branches without another approval round.
Ruling: preserve legacy scripts and add research entry points, so historical checkpoints retain their original input semantics.
Pre-flight: all variants consume the common model(batch) and objective(model,batch) contracts; variant-specific architecture configuration is persisted separately from training configuration.

Task 1 complete: 15 inherited tests and 5 new tests pass, including synthetic CLI training, resume, and original-resolution evaluation. CPU only; CUDA unavailable. Both new test modules were observed failing before implementation.

Task 3 complete in `exp/gain-guided-refinement` (not yet committed): `research/refinement.py` adds two-scale global candidates, high-resolution residual crops, mean overlap blending, question-aware gain regression, diversity/budget/negative-gain routing, and stage-specific frozen-module evaluation modes. `variant.py` adds staged CLI/objective and strict initialization requirements; all DDP gradients originate in `model.forward`. Dataset adds aligned original-source `detail_mask`; shared engine gains optional evaluation hooks with inference settings in metrics.

Red/green evidence: six geometry/gain/selection/model tests initially failed for missing implementation; random filename invariance separately failed before correction. Two staged CLI integration tests failed before plugin implementation. Final branch suite: 28 tests discovered, 27 pass and one inherited controls CLI test intentionally skipped in favor of this branch's staged integration. Original-resolution outputs, exact frozen weights, router resume and configuration mismatch rejection verified on CPU; no CUDA or real-data training. Production head parameter count measured offline: refiner 190,977; router 115,265. Detailed commands, ablation semantics and limitations are in RESEARCH-RUNS.md.

Final review: independent review completed; no remaining blocking architecture findings. Shared resume fixes enforce same-directory last.pt continuation, hash actual mask contents, and handle disabled AMP scaler state. Full CPU suite: 30 passed, one inherited control-only CLI test intentionally skipped in favor of variant integration tests. Two-process shared-control CPU DDP with an empty validation shard also passed. Real-data/CUDA accuracy and latency remain unmeasured. User requested separate branches, so preserve branches/worktrees without merging or pushing.
