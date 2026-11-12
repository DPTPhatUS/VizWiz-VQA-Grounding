# Verification — 12 November 2026

The instructor's essential/high-priority preset contains 12 new runs: symmetric three-role membership and trained refinement depths zero, one, and three, each at seeds 42, 123, and 456. Three CM-Separate entries provide the depth-two references. All use the existing 30-epoch protocol. Selecting a reference starts fresh training; existing reference results can instead be included during analysis. The original 27-run preset is unchanged.

## Completed checks

- **60 offline tests passed.** This includes the original CM/FTC behavior, component bypasses, gradient paths, validation selection and RNG preservation, native metrics, lossless logits, fresh campaigns, mocked Hub uploads, and notebook setup/interruption.
- Symmetric three-role membership preserves exact initial tensors, construction RNG state, and the full decoder parameter count. Tests verify reciprocal probabilities, valid borders, fractional/empty/full occupancy targets, conserved class mass, and loss/gradient equivalence to constraining predictions under the original directional objective.
- Depths zero through three preserve initial tensors and use the expected signed update. Tests cover finite gradients, frozen unused update rates at zero steps, strict checkpoint reconstruction, and rejection of conflicting configurations and CLI options.
- A synthetic single-process campaign ran all five review configurations through the actual launcher. It selected and reloaded checkpoints, exported native validation/test mIoU and Boundary IoU with logits, and produced paired summaries against CM-Separate.
- Boundary-band calculation matches an independent iterative erosion implementation, including image borders. Existing threshold and native-resolution metric tests still pass. Boundary metrics do not affect checkpoint selection.
- The symmetric three-role control and depths zero, one, and three passed full-resolution decoder forward/backward/Adam steps on CPU: 24×24 CLIP-shaped inputs, a 48×48 refinement grid, and 384×384 outputs. These checks use synthetic features, not a CLIP backbone.
- The same four variants completed two-process CPU/Gloo training and validation checks. Non-padding validation shards cover three synthetic images exactly once, and distributed scores match single-process scores. The sandbox initially denied local socket access; the permitted retry passed. PyTorch emitted advisory unused-parameter-search and gradient-stride warnings, without failures.
- All 15 entries resolve in a two-process launcher dry run. Tiny CPU benchmarking covers FTC, FTC capacity, and all four depths of both directed and symmetric three-role CM. Tests verify the FLOP multiply-add convention and restoration of attention settings, including after an exception. These timings are smoke checks, not research efficiency measurements.
- Both Vast notebook code cells compile and contain no saved outputs. Existing mocked setup/interruption tests pass. The default selection is now `cm_symmetric_membership_s42` from `configs/instructor_high_priority.json`.

The full CM-Separate and symmetric three-role decoders each register 1,120,585 parameters. Zero-step CM retains that count but freezes its two unused update-rate parameters. Original decoder counts remain FTC 690,753; CM-Shared 942,953; symmetric affinity 1,120,550; two-role CM 1,120,568; FTC capacity 1,120,845.

The efficiency report separates synchronized latency/memory from FLOP counting. The counter covers convolution, matrix multiplication, and attention, with two FLOPs per multiply-add. It does not count elementwise operations, graph reductions, normalization, interpolation, or tokenization; measured latency captures costs outside that coverage.

Local checks used Python 3.12.10 and PyTorch 2.13.0+cu130. Missing test dependencies were installed under `/tmp`, without changing the project lockfile. Logs: `/tmp/cm-review-full-tests.log`, `/tmp/cm-review-decoder-smoke.log`, `/tmp/cm-review-ddp-verified.log`, and `/tmp/cm-review-dry-run.jsonl`.

CUDA is unavailable here. GPU/AMP checks, full-CLIP profiling with the locked environment, and scientific training remain for Kaggle/Vast hardware. No scientific training campaign or live Hugging Face upload was launched. Existing datasets, checkpoints, results, and manuscript files were preserved. Lower-priority experiments were not implemented.
