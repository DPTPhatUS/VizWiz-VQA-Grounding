# Latent-Answer Grounding: Region/Pixel & Multi-Scale Detail Decoder (`exp/spatial-detail-decoder`)

This branch implements a **Region/Pixel & Multi-Scale Detail Decoder**. It combines the **Spatial Evidence-Decoder Interaction** from `exp/spatial-evidence-decoder` with a shallow **Pixel Detail Stem** and **Semantic-Guided Region-Pixel Detail Fusion** modules, adapting multi-level aggregation ideas from [DDVT (Zhang et al., 2026)](https://arxiv.org/html/2607.23921v1#S3.SS2).

Student inference requires only the image and question. CLIP input resolution is fixed at 336 pixels.

---

## Architecture: DDVT-Inspired Detail Decoder

### 1. Motivation and Problem Formulation

The CLIP ViT-L/14-336 backbone uses $14 \times 14$ pixel patches, producing a $24 \times 24$ feature grid. Adding a shallow image path gives the decoder access to local spatial features before this patch compression.

Inspired by DDVT (Zhang et al., 2026), the model uses a dual-stream vision representation:
1. **Semantic Stream**: High-level semantic features modulated by latent answer evidence $A'$ across the decoder hierarchy.
2. **Detail Stream**: A shallow convolutional stem operating directly on the normalized, resized $336 \times 336$ image without further downsampling.

```
Resized RGB Image (3 x 336 x 336), then CLIP normalization
  │
  ├──► CLIP ViT Encoder ──► Patch Grid (24x24) ──► Latent Answer Interaction (A') ──► Semantic Stream (X_sem)
  │                                                                                          │
  └──► PixelDetailStem  ──► High-Freq Features (16 x H x W) (Detail Stream D)               │
                                   │                                                         │
                                   ▼                                                         ▼
                         [ Semantic Guidance Mask: M_guide = σ(Conv_1x1(X_sem)) ]
                                   │
                                   ▼
                         [ Gated Detail: D_tilde = D ⊙ M_guide ]
                                   │
                                   ▼
                         [ Separable Refinement: ΔX = Refine([X_sem, D_tilde]) ]
                                   │
                                   ▼
                         [ Residual Fusion: X_out = X_sem + residual_scale * ΔX ] (residual_scale = 0 at step 0)
```

---

### 2. Core Components

#### A. Pixel Detail Stem (`PixelDetailStem`)
A lightweight 2-layer convolutional stem operating on the normalized, resized input image:
- **Layer 1**: $\text{Conv}_{3\times 3}(3 \to 16, \text{pad}=1) \to \text{GroupNorm}(4, 16) \to \text{GELU}$
- **Layer 2**: $\text{Conv}_{3\times 3}(16 \to 16, \text{pad}=1) \to \text{GroupNorm}(4, 16) \to \text{GELU}$
- **Output**: Local feature map $D \in \mathbb{R}^{B \times 16 \times 336 \times 336}$ trained end-to-end with the segmentation loss.

#### B. Semantic-Guided Region-Pixel Fuser (`RegionPixelFuser`)
A semantic-guided fusion module that gates high-resolution detail features using the decoder's semantic stream:
1. **Semantic Guidance Mask**: Visual features $X_{\text{sem}}$ (which already integrate latent answer evidence) generate a continuous 2D spatial guidance mask:
   $$M_{\text{guide}} = \sigma(\text{guide\_scale} \cdot \text{Conv}_{1\times 1}(X_{\text{sem}}) + \text{guide\_bias}) \in (0, 1)^{B \times 1 \times H_i \times W_i}$$
2. **Spatial Gating**: Detail features $D$ are resampled to the stage resolution $(H_i, W_i)$ and gated by the guidance mask:
   $$\widetilde{D} = D \odot M_{\text{guide}}$$
3. **Depthwise-Separable Refinement**:
   $$\Delta X = \text{Refine}([\,X_{\text{sem}} \,\|\, \widetilde{D}\,])$$
4. **Zero-Init Identity Residual**:
   $$X_{\text{out}} = X_{\text{sem}} + \text{residual\_scale} \cdot \Delta X$$
   `residual_scale` is initialized to $0.0$, preserving the spatial-evidence decoder's initial output when shared weights and inputs match. Scale parameters receive gradients on the first backward pass, and gradients flow into the detail stem as `residual_scale` moves away from zero.

The default detail path adds **4,504 trainable parameters**.

---

### 3. Modular Upsampling Hierarchy & Defaults

The decoder has 4 progressive upsampling stages:
- **Stage 0**: $48 \times 48$, 64 channels
- **Stage 1**: $96 \times 96$, 32 channels
- **Stage 2**: $192 \times 192$, 16 channels
- **Stage 3**: $384 \times 384$, 16 channels

#### Modular Configuration
- **Default Stages**: Region-pixel detail fusion is applied at the **last 2 upsampling stages** (`detail_upsample_stages=(2, 3)`, corresponding to $192 \times 192$ and $384 \times 384$), where channel width is narrow and boundary detail is finest.
- **Configurable Stages**: Detail fusion can be enabled on any subset of the 4 stages:

```python
from models.model import GroundingModel

# Default: detail fusion at last 2 stages (2, 3) -> 192x192 & 384x384
model_default = GroundingModel()

# All 4 upsampling stages enabled for detail fusion
model_4stages = GroundingModel(detail_upsample_stages=(0, 1, 2, 3))

# Disable detail fusion (reverts to pure spatial-evidence decoder)
model_no_detail = GroundingModel(spatial_detail=False)

# Custom stage combinations
model_last_only = GroundingModel(detail_upsample_stages=(3,))
```

#### CLI Flags
- `--no-spatial-evidence`: Disable all cosine spatial interaction, including the bottleneck, while keeping the latent-answer token, skip conditioning, and detail path. Use this for the alignment + detail comparison on a fresh run with the matched alignment, `stable` loss, and frozen-text settings.
- `--spatial-detail` / `--no-spatial-detail`: Enable or disable detail fusion (default: enabled).
- `--detail-upsample-stages 2 3`: Upsampling stages for detail fusion (default: `2 3`).
- `--detail-dim 16`: Channel dimension for the pixel detail stem; must be a positive multiple of 4 (default: `16`).
- `--spatial-upsample-stages 0 1`: Spatial evidence interaction stages (default: `0 1`).

---

### 4. Checkpoint Compatibility & Auto-Detection

- **Inference and resume**: Entry points reconstruct the saved architecture, including the detail channel width, directly from the checkpoint. Omitted architecture flags restore checkpoint settings, while conflicting explicit flags on resume raise an error.
- **Weight initialization**: `initialize_weights(path_or_saved, model)` can load compatible compact checkpoints from the latent or spatial-evidence branches while initializing newly added detail modules. The U-Net architecture on `main` uses a different backbone/decoder layout and is not compatible.
- **Automatic detection**: `checkpoint_model_kwargs(saved)` inspects checkpoint metadata and state dict keys to instantiate the matching architecture.
- **Buffer serialization**: Active stage IDs are stored in persistent buffers (`decoder.detail_stage_ids`, `decoder.spatial_stage_ids`) for exact architecture reconstruction without external config files.
- **Feature ownership**: `encode_image` returns an `ImageFeatures` container that bundles detail features with the image pyramid while preserving standard 4-feature iteration and indexing. Callers passing a plain tuple/list pyramid must pass `detail_features` explicitly when detail fusion is enabled.

---

## Training

The dataset root expects `train_grounding.json`, `val_grounding.json`, image directories `train/` and `val/`, and ground-truth masks under `binary_masks_png/<split>/`.

### Matched Single-Stage Comparison (Detail Decoder Enabled)

This command matches the alignment recipe, `stable` loss version, and frozen text encoder used in the spatial-evidence comparison. Checkpoints are saved directly to `outputs/`.

```bash
mkdir -p outputs/configs
printf '%s\n' '{"freeze_text_encoder": true, "dice_weight": 0.5, "latent_weight": 0.5}' > outputs/configs/spatial_detail_alignment_frozen.json
uv run torchrun --nproc_per_node=2 train.py \
    --data-root /path/to/vizwiz \
    --num-epochs 20 \
    --batch-size 4 \
    --num-workers 2 \
    --lr 0.00001 \
    --seed 42 \
    --save-every 10 \
    --training-recipe alignment \
    --loss-version stable \
    --config outputs/configs/spatial_detail_alignment_frozen.json \
    --output-dir outputs
```

### Training with Custom Detail Stages (e.g., all 4 stages)

```bash
uv run torchrun --nproc_per_node=2 train.py \
    --data-root /path/to/vizwiz \
    --num-epochs 20 \
    --batch-size 4 \
    --training-recipe alignment \
    --loss-version stable \
    --config outputs/configs/spatial_detail_alignment_frozen.json \
    --spatial-upsample-stages 0 1 \
    --detail-upsample-stages 0 1 2 3 \
    --output-dir outputs/spatial_detail_4stages
```

`--batch-size` is **global across GPUs** (batch size 4 across 2 GPUs gives 2 images per GPU). Training selects CUDA automatically, enables AMP on CUDA, and runs without per-epoch validation.
The default CLI values for `--training-recipe` and `--loss-version` remain `full` and `legacy` for backward compatibility, so pass `--training-recipe alignment --loss-version stable` explicitly for matched comparisons. When architecture flags are omitted on a fresh run, detail stages `(2, 3)` with 16 channels are enabled by default.

| `--training-recipe` | Objective |
|---|---|
| `full` | Predicted-answer mask + oracle/Q-reference masks + gain-weighted mask KD + answer alignment + attention guidance; no warm-up |
| `grounding-only` | Predicted-answer mask supervision only; the latent predictor remains present |
| `alignment` | Predicted-answer mask supervision + answer-vector alignment |
| `warmup` | Answer alignment throughout; mask supervision transitions from real A to predicted A' |

Passing `--loss-version stable` uses the updated KL calculation and valid-answer reductions.

---

## Checkpoints and Evaluation

`--save-every 10` saves resumable checkpoints (`checkpoint_epoch10.pt`, `checkpoint_epoch20.pt`, etc.), and training also writes final weights to `model_final_epoch<N>.pt`.

```bash
uv run python eval.py \
    --checkpoint outputs/checkpoint_epoch20.pt \
    --dataset val \
    --data-root /path/to/vizwiz \
    --batch-size 4 \
    --num-workers 2 \
    --output-dir outputs/eval_val_detail_ep20
```

Evaluation scores and exports masks at native image resolution using only the image and question.

---

## Diagnostics & Probes

All diagnostic scripts support the detail decoder:
- `diagnose.py`: Compares Q-only, predicted-answer, and real-answer masks.
- `probe_latent_conditioning.py`: Runs interventions, embedding retrieval, and conditioning traces.
- `probe_predicted_answer.py`: Caches predicted vectors and runs answer swaps or zeroing.
- `predict_save.py` & `visualize_predictions.py`: Single-image inference and mask overlay visualization.

## Encoder freezing and optimization

Omitting the options below keeps the branch's default training configuration: Adam with LR `1e-5`, zero weight decay, constant LR, and the default encoder freezing policy. Encoder freezing is opt-in, while the detail stem and decoder always remain trainable. Model state-dict keys and evaluation preprocessing are unchanged.

| Option | Values / default |
| --- | --- |
| `--freeze-backbone [BOOL]` | Set both pretrained encoders; omitted preserves existing policy |
| `--freeze-vision-encoder [BOOL]` | Set vision encoder only |
| `--freeze-text-encoder [BOOL]` | Set text encoder only |
| `--optimizer` | `adam` (default), `adamw` |
| `--lr` | Main learning rate; default `0.00001` |
| `--backbone-lr` | Optional separate LR for unfrozen encoders; default uses `--lr` |
| `--weight-decay` | Nonnegative; default `0`. Biases, norm vectors and token/position embeddings receive no decay when parameter groups are enabled |
| `--lr-scheduler` | `none` (default), `cosine`, `linear`, `poly` |
| `--warmup-ratio` | Fraction of planned optimizer updates, in `[0, 1)`; default `0` |
| `--min-lr-ratio` | Final LR / starting LR for decay schedules; default `0.1` |
| `--poly-power` | Positive exponent for `poly`; default `0.9` |
| `--config` | Flat JSON with training options; optional |

`BOOL` accepts `True` or `False`; passing a freeze flag without a value sets it to `True`. The negative aliases `--no-freeze-backbone`, `--no-freeze-vision-encoder`, and `--no-freeze-text-encoder` set the corresponding flag to `False`. Within the same configuration source, a specific encoder flag overrides `freeze_backbone`. Precedence follows **JSON config > explicit CLI > branch defaults**, so a JSON shorthand also overrides component flags passed on the CLI. Recipe and loss settings can share the same JSON file, and unrecognized keys raise an error immediately. Run `train.py --help` to see all branch-specific flags.

Example CLI flags to freeze the backbone and use AdamW with a cosine schedule:

```bash
--freeze-backbone True \
--optimizer adamw \
--lr 0.0001 \
--weight-decay 0.01 \
--lr-scheduler cosine \
--warmup-ratio 0.05 \
--min-lr-ratio 0.1
```

To freeze only the vision encoder while keeping the default text policy, pass `--freeze-vision-encoder True`. To train both encoders explicitly, pass `--freeze-backbone False`. Branches that rely on a fixed reference stage (such as gain-guided refinement) reject encoder unfreezing. Unused CLIP/SigLIP pooled-output parameters remain frozen when token encoders are unfrozen.

Equivalent JSON configuration (`--config path/to/train.json`):

```json
{
  "freeze_backbone": true,
  "optimizer": "adamw",
  "lr": 0.0001,
  "weight_decay": 0.01,
  "lr_scheduler": "cosine",
  "warmup_ratio": 0.05,
  "min_lr_ratio": 0.1
}
```

Frozen encoders stay in `eval` mode during training and are excluded from the optimizer. Learning-rate schedules step once per completed optimizer update (accounting for gradient accumulation) and do not advance when AMP skips an update due to gradient overflow. Checkpoints record the resolved optimization config, trainable parameter layout, and scheduler state; resuming restores those settings and rejects conflicting overrides. When resuming a scheduled run, keep `--num-epochs`, dataloader length, and gradient accumulation unchanged so the LR schedule trajectory stays consistent.

Existing checkpoints remain compatible with evaluation and legacy constant-LR resume. Changing encoder freezing, optimizer type, weight decay, or LR scheduling requires starting a new run (or using weights-only initialization where supported).
