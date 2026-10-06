# Implementation ledger
Plan: research-implementation-plan.md

Ruling: user explicitly requested implementation of the presented proposals; proceed through the shared controls and three branches without another approval round.
Ruling: preserve legacy scripts and add research entry points, so historical checkpoints retain their original input semantics.
Pre-flight: all variants consume the common model(batch) and objective(model,batch) contracts; variant-specific architecture configuration is persisted separately from training configuration.

Task 1 complete: 15 inherited tests and 5 new tests pass, including synthetic CLI training, resume, and original-resolution evaluation. CPU only; CUDA unavailable. Both new test modules were observed failing before implementation.

Final review: independent review completed; no remaining blocking architecture findings. Shared resume fixes enforce same-directory last.pt continuation, hash actual mask contents, and handle disabled AMP scaler state. Full CPU suite: 23 passed. Two-process shared-control CPU DDP with an empty validation shard also passed. Real-data/CUDA accuracy and latency remain unmeasured. User requested separate branches, so preserve branches/worktrees without merging or pushing.
