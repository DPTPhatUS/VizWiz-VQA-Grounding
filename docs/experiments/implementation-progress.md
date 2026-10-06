# Implementation ledger
Plan: research-implementation-plan.md

Ruling: user explicitly requested implementation of the presented proposals; proceed through the shared controls and three branches without another approval round.
Ruling: preserve legacy scripts and add research entry points, so historical checkpoints retain their original input semantics.
Pre-flight: all variants consume the common model(batch) and objective(model,batch) contracts; variant-specific architecture configuration is persisted separately from training configuration.

Task 1 complete: 15 inherited tests and 5 new tests pass, including synthetic CLI training, resume, and original-resolution evaluation. CPU only; CUDA unavailable. Both new test modules were observed failing before implementation.

Task 4 implemented (not committed by implementer): `research/extent.py`, branch-specific `research/variant.py`, `tests/test_extent.py`, `tests/test_extent_cli.py`. Two low-rank question/vision-attending tokens produce bounding-rectangle support and extent residual; final mask and area losses include empty regions. Optional paired objectives use explicit own-mask labels, signed changes, and declared same-region consistency. Added configurable evidence width and strict architecture metadata.

Shared data correction on this branch: validate same-region masks at original resolution, because downsampling can erase real annotation disagreements. The regression was observed failing before the fix. Initial missing architecture/validation tests failed before implementation; width CLI was likewise tested before implementation.

Verification command: `OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/vizwiz-mpl /home/dptphat/research/VizWiz-VQA-Grounding/.venv/bin/python -m unittest discover -s tests -v` → 31 collected, 30 passed, one expected controls-CLI skip. Includes tiny CPU paired CLI training, unpaired in-process training, resume, changed-objective rejection, and question-only original-resolution evaluation after deleting pair files. Added production head count: 148,545 parameters at width64/CLIP-L dimensions, measured with standalone head modules; tiny count 2,833. No CUDA/download/real-data training performed. Full experiment commands, pair schema, ablations and limitations are in RESEARCH-RUNS.md.

Review notes for shared controls: parent fixed resume provenance to include pair-mask and ordinary-mask contents and reject changed output directories; paired losses now normalize all summed view/pair contributions by original batch size, so unequal pair counts retain equivalent shard weighting. The combined-versus-sharded regression was observed failing for change and same pairs before correction, then passes. No actual multi-process execution claimed. Parent notified. Final implementation diff inspected; no other worktree modified.

Final review: independent review completed; no remaining blocking architecture findings. Shared resume fixes enforce same-directory last.pt continuation, hash actual mask contents, and handle disabled AMP scaler state. Full CPU suite: 31 passed, one inherited control-only CLI test intentionally skipped in favor of variant integration tests. Two-process shared-control CPU DDP with an empty validation shard also passed. Real-data/CUDA accuracy and latency remain unmeasured. User requested separate branches, so preserve branches/worktrees without merging or pushing.
