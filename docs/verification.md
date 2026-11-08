# Verification — 8 November 2026

The manifest contains 27 matched 30-epoch runs: the previous 21 configurations/seeds plus symmetric-affinity and two-role controls at seeds 42, 123, and 456. New controls preserve the common CM-Separate initialization, global context branch, unary/readout path, and training/selection protocol. Only validation-selected `best.pt` is saved.

## Completed checks

- **52 offline tests passed.** The full suite includes exact original CM/FTC reference behavior, bypasses and gradient paths, validation selection and RNG preservation, native metrics, lossless logits, fresh campaigns, uploads with a mocked Hub, and notebook setup/interruption.
- New controls: exact common initial tensors and RNG state; full decoder parameter counts; symmetric opposite-edge scores and gradients; valid borders; foreground/background agreement targets; fractional occupancy and empty/full masks; conditional two-role supervision; cancellation; finite gradients through every trainable tiny-model parameter; zero auxiliary weight; strict checkpoint round trips and option rejection.
- Both new controls completed a synthetic single-process training campaign through the actual launcher, selected and reloaded checkpoints, and exported native-resolution validation/test metrics and logits. The summary combines timestamp folders and computes paired CM-Separate-minus-control differences; duplicate run IDs are rejected.
- Both new controls passed full-resolution decoder forward/backward/Adam steps on CPU using 24×24 CLIP-shaped inputs, a 48×48 refinement grid, and 384×384 outputs. These checks use synthetic features, not a CLIP backbone.
- Both new controls completed two-process CPU/Gloo training and validation smoke checks. Non-padding validation shards cover three synthetic images exactly once, and distributed scores match single-process scores. The initial sandbox attempt lacked local socket access; the permitted retry passed. PyTorch emitted advisory DDP unused-parameter-search/gradient-stride warnings, without failures.
- All six new run IDs resolve in a two-process launcher dry run. Tiny CPU benchmark entry points work for `cm-affinity` and `cm-two-role`. These timings are smoke checks, not research efficiency measurements.
- The Vast notebook code cells compile, and its existing mocked setup/interruption tests pass with the new default `RUN_IDS=["cm_affinity_s42"]`. `nbformat` is not installed locally, so schema validation was not repeated.

Decoder parameters: FTC 690,753; CM-Shared 942,953; CM-Separate 1,120,585; symmetric affinity 1,120,550; two-role CM 1,120,568; FTC capacity 1,120,845. The new controls differ from full CM by 35 and 17 parameters, respectively; no unused head rows are retained.

Test output: `/tmp/cm-controls-full-tests.log`. Full-resolution decoder checks: `/tmp/cm-controls-decoder.log`. Distributed checks: `/tmp/cm-controls-ddp-retry.log`. Launcher dry run: `/tmp/cm-controls-dry-run.jsonl`.

CUDA is unavailable in this environment, so GPU/AMP checks and full CLIP training remain for Kaggle/Vast hardware. No scientific training campaign or live Hugging Face upload was launched. Existing datasets, checkpoints, results, and manuscript files were preserved. The three-role diagnostic explicitly rejects the two new controls; its metrics have not been redefined for their different targets.
