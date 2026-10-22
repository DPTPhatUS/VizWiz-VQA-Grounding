# Context–membership grounding

One branch implements two single-stage architecture experiments:

- `--connection-mode shared`: model 3, one connection network reused for context and membership.
- `--connection-mode separate`: model 4, independent context and membership connection networks.

Start a fresh run from pretrained CLIP. No trained grounding checkpoint, crop refiner, router, teacher, or extra annotation is required. This is an unvalidated architecture experiment, not a claim of improved accuracy.

## Inherited components selected by the user

The base is `exp/grounding-controls` at `2755690`: compact decoder, question-conditioned skip FiLM, and CLIP image/text encoders. Only CLIP text is frozen; image encoding and the grounding head train jointly. The old visual-plus-attention residual fusion remains removed: the bottleneck receives cross-attention output directly. Ordinary decoder skips and the new context update are separate operations.

All examples use question-only input (`Q: ...`). Image preprocessing uses CLIP normalization and 336-pixel input. Segmentation retains the control's binary cross-entropy loss and original-resolution evaluation.

## Exact architectural comparison

Both variants have the same decoder, output heads, token inputs, edge labels, loss weights, and two membership updates. Their sole structural difference is whether connection-building parameters are tied.

A connection network includes visual/text projections, per-location attention over individual valid question tokens, position embeddings, receiver/sender projections, and relative-position pair features. The context output head predicts a scalar reading score. The membership output head predicts three roles: include, exclude, and no foreground-source message.

In `shared`, the same connection network is called at both spatial resolutions. The two output heads are still distinct; shared does not mean identical final attention matrices. In `separate`, the membership network is an independent copy initialized with the context network's initial values. It has no shared parameter storage. With the same seed, all common parameters and initial predictions are identical between the two modes; optimization then differs.

For a normal 336-pixel CLIP input:

1. The compact decoder combines the four 24×24 CLIP depth features and question-conditioned skips.
2. The first upsampling block produces 64-channel features on a 48×48 grid.
3. Context reading uses a pooled 24×24 grid, dense directed connections, and softmax over source locations. It aggregates visual values, upsamples them, and adds them through a learned gate to the 48×48 features.
4. Membership uses eight local neighbors per receiver on the 48×48 grid. Invalid border edges are masked; no edge wraps to another row.
5. A separate token-attention unary head predicts initial mask scores from contextualized features and the question.
6. Two membership updates add supporting messages and subtract suppressing messages, retaining the unary scores at each update. Message strengths are bounded between zero and two; valid-neighbor averaging prevents degree-dependent scale changes.
7. Contextualized unary features and updated mask scores are concatenated and projected back to 64 channels. The remaining three compact upsampling blocks and final mask head produce 384×384 logits. Training resizes logits to the target grid; evaluation resizes logits to the original image before thresholding at zero.

This final head predicts a fresh mask; it does not add a correction to a frozen coarse model. The recurrent updates are two layers in one forward pass, not separate training stages.

## Membership supervision

Training targets use fractional foreground occupancy obtained by area averaging the training mask to the 48×48 grid. For receiver `i` and sender `j` with occupancies `yi` and `yj`, the soft three-class targets are:

```text
include = yi * yj
exclude = (1 - yi) * yj
no foreground-source message = 1 - yj
```

The loss excludes invalid border edges, normalizes each present class separately within each image, and averages the present classes and images. Empty/full masks are supported. These labels describe mask membership, not object identity or annotated semantic roles. Context connections have no extra region labels; they learn through final segmentation loss.

```text
loss = binary_cross_entropy(final_mask, target) + edge_loss_weight * membership_loss
```

`--edge-loss-weight` defaults to `0.1`. Set it to zero for a loss ablation. Both connection modes use exactly the same objective. There is no area penalty, bounding-box loss, answer distillation, or crop objective.

In shared mode, the membership loss directly trains the shared connection network. In separate mode, it directly trains the membership network, not the context network. Both paths receive final segmentation gradients. Shared upstream decoder features still receive gradients from both losses.

## Train

From this branch's checkout, install the usual PyTorch/torchvision, Transformers, Pillow, NumPy and tqdm dependencies. The dataset root must contain the grounding JSONs, split image directories, and `binary_masks_png/<split>`.

```bash
python train.py --connection-mode shared --data-root /path/to/vizwiz --num-epochs 100 --batch-size 1 --num-workers 2 --seed 42
python train.py --connection-mode separate --data-root /path/to/vizwiz --num-epochs 100 --batch-size 1 --num-workers 2 --seed 42
```

The default directories are `outputs/context-membership/shared` and `outputs/context-membership/separate`. Use `--output-dir` for distinct seeds or loss variants. A fresh run with no mode specified defaults to shared.

Training selects CUDA automatically and retains torchrun DDP support with global batch size. It reads only training data; run validation separately. The startup record reports total, trainable, and connection-module parameter counts. `history.jsonl` records total loss, segmentation loss, and unweighted membership loss.

Periodic full checkpoints are saved every ten epochs by default (`--save-every`). They include architecture mode, model weights, optimizer/scaler state, epoch and run settings. There is no additional automatic final save. `--save-every 0` disables checkpoints. Choose checkpoints using validation, not test results.

Resume without repeating the mode; the mode and, unless overridden, edge-loss weight are restored:

```bash
python train.py --data-root /path/to/vizwiz --resume-checkpoint outputs/context-membership/separate/checkpoint_epoch20.pt --num-epochs 100 --batch-size 1 --num-workers 2 --seed 42
```

Repeat the original optimizer/batch/seed settings on resume, as in grounding-controls. A contradictory explicit mode is rejected. `--init-checkpoint` starts a fresh optimizer from a checkpoint of this architecture and mode. Other experiment checkpoints and metadata-free raw weights are not supported. There is no compatibility loader or silent partial initialization.

## Evaluate and predict

Mode is reconstructed from checkpoint metadata; evaluation needs no mode flag.

```bash
python eval.py --checkpoint outputs/context-membership/shared/checkpoint_epoch100.pt --data-root /path/to/vizwiz --dataset val --batch-size 1 --num-workers 2 --output-dir results/context-membership/shared/val-e100
python eval.py --checkpoint outputs/context-membership/separate/checkpoint_epoch100.pt --data-root /path/to/vizwiz --dataset val --batch-size 1 --num-workers 2 --output-dir results/context-membership/separate/val-e100
python predict_save.py --checkpoint outputs/context-membership/separate/checkpoint_epoch100.pt --image /path/to/image.jpg --question "What does it say?" --output mask.png
python visualize_predictions.py --checkpoint outputs/context-membership/separate/checkpoint_epoch100.pt --image /path/to/image.jpg --question "What does it say?" --output overlay.png
```

`IoU.py` also delegates to this evaluator. Evaluation writes original-resolution binary PNGs and a metrics JSON recording the architecture configuration.

## Parameters and limits

No dataset training or accuracy evaluation of these new models has been performed.

| Component parameters | Shared | Separate |
|---|---:|---:|
| Compact decoder including new mechanism | 942,953 | 1,120,585 |
| Context/membership module within decoder | 247,976 | 425,608 |

Separate adds 177,632 parameters. These counts exclude CLIP and the existing 1024-dimensional cross-attention fusion. The variants are not parameter matched; accuracy differences must not be attributed solely to separation without a capacity control.

## Files

- `models/connections.py`: question-conditioned pair networks, context reading, and membership updates.
- `models/mask_decoder.py`: integrates the mechanism into the compact decoder.
- `models/model.py`: encoders, fusion, and explicit experiment configuration.
- `losses.py`: segmentation and balanced directed-membership supervision.
- `train.py`, `eval.py`, `predict_save.py`, `utils.py`: mode selection, checkpoints, training and inference.
