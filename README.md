# Latent-Answer Grounding: Region/Pixel & Multi-Scale Detail Decoder (`exp/spatial-detail-decoder`)

This branch tests a **Region/Pixel & Multi-scale Detail Decoder**. It combines the **Spatial Evidence-Decoder Interaction** from `exp/spatial-evidence-decoder` with a shallow **Pixel Detail Stem** and **Semantic-Guided Region-Pixel Detail Fusion** modules. This is a custom fusion design inspired by multi-level aggregation in [DDVT (Zhang et al., 2026)](https://arxiv.org/html/2607.23921v1#S3.SS2), rather than a reproduction of its CMA module.

Student inference requires only image + question. CLIP input is fixed at 336 pixels.

---

## Architecture: DDVT-Inspired Detail Decoder

### 1. Motivation and Problem Formulation

The CLIP ViT-L/14-336 backbone uses $14 \times 14$ pixel patches, producing a $24 \times 24$ feature grid. A shallow image path lets the decoder access local features before this spatial compression. Whether it improves small-object grounding and boundary accuracy must be measured on validation data.

Inspired by DDVT (Zhang et al., 2026), this architecture introduces a dual-level vision representation:
1. **Semantic Stream**: High-level semantic features modulated by latent answer evidence $A'$ across the decoder hierarchy.
2. **Detail Stream**: A shallow convolutional stem operating on the same normalized, resized $336 \times 336$ image, without further downsampling. It cannot recover information already lost when the original image was resized.

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
A lightweight, shallow 2-layer convolutional network operating directly on normalized, resized input images:
- **Layer 1**: $\text{Conv}_{3\times 3}(3 \to 16, \text{pad}=1) \to \text{GroupNorm}(4, 16) \to \text{GELU}$
- **Layer 2**: $\text{Conv}_{3\times 3}(16 \to 16, \text{pad}=1) \to \text{GroupNorm}(4, 16) \to \text{GELU}$
- **Output**: Learned local feature map $D \in \mathbb{R}^{B \times 16 \times 336 \times 336}$. No explicit high-pass or edge supervision is imposed.

#### B. Semantic-Guided Region-Pixel Fuser (`RegionPixelFuser`)
Custom semantic-guided fusion. DDVT's Cross-Modal Multi-Scale Aggregation (CMA) also uses deformable alignment and CondConv; those operations are not implemented here.
1. **Semantic Guidance Mask**: Visual features $X_{\text{sem}}$ (which already integrate latent answer evidence) generate a continuous 2D spatial guidance mask:
   $$M_{\text{guide}} = \sigma(\text{guide\_scale} \cdot \text{Conv}_{1\times 1}(X_{\text{sem}}) + \text{guide\_bias}) \in (0, 1)^{B \times 1 \times H_i \times W_i}$$
2. **Spatial Gating**: Detail features $D$ are dynamically resampled to stage resolution $(H_i, W_i)$ and selectively gated:
   $$\widetilde{D} = D \odot M_{\text{guide}}$$
   The mask is trained indirectly by segmentation loss. Suppression of clutter and selection of useful detail are hypotheses to evaluate, rather than guaranteed properties.
3. **Depthwise-Separable Refinement**:
   $$\Delta X = \text{Refine}([\,X_{\text{sem}} \,\|\, \widetilde{D}\,])$$
4. **Zero-Init Identity Residual**:
   $$X_{\text{out}} = X_{\text{sem}} + \text{residual\_scale} \cdot \Delta X$$
   `residual_scale` is initialized to $0.0$, preserving the spatial-evidence decoder's output when common weights, inputs, and execution settings match. Scale parameters receive gradients first; gradients reach the detail stem as scales move away from zero.

The default detail path adds **4,504 trainable parameters**. Its dense spatial activations still add computation and memory; parameter count alone does not establish runtime cost.

---

### 3. Modular Upsampling Hierarchy & Defaults

The decoder features 4 progressive upsampling stages:
- **Stage 0**: $48 \times 48$, 64 channels
- **Stage 1**: $96 \times 96$, 32 channels
- **Stage 2**: $192 \times 192$, 16 channels
- **Stage 3**: $384 \times 384$, 16 channels

#### Modular Configuration
- **Strict Default**: Region-pixel detail fusion is applied to the **last 2 upsampling stages** (`detail_upsample_stages=(2, 3)` corresponding to $192 \times 192$ and $384 \times 384$), exactly where channels are narrow and boundary sharpness is most critical.
- **Configurable Stages**: Can be set to any subset of stages or all 4 stages:

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
- `--spatial-detail` / `--no-spatial-detail`: Enable/disable detail fusion (default: enabled).
- `--detail-upsample-stages 2 3`: Upsampling stages for detail fusion (default: `2 3`).
- `--detail-dim 16`: Channel dimension for pixel detail stem; a positive multiple of 4 (fresh-run default: `16`).
- `--spatial-upsample-stages 0 1`: Spatial evidence interaction stages from Hướng 1 (default: `0 1`).

---

### 4. Checkpoint Compatibility & Auto-Detection

- **Inference/resume**: Existing entry points reconstruct the saved architecture, including detail width. Omitted architecture flags restore checkpoint settings; conflicting explicit resume flags raise an error.
- **Weight initialization**: `initialize_weights(path_or_saved, model)` can add complete new modules to compatible compact checkpoints from the latent/spatial-evidence branches. It starts a new experiment. The U-Net architecture on `main` is not interchangeable with this compact architecture.
- **Automatic detection**: `checkpoint_model_kwargs(saved)` inspects checkpoint metadata and state dict keys to automatically instantiate matching architectures.
- **Buffer serialization**: Active stage IDs are stored in persistent buffers (`decoder.detail_stage_ids`, `decoder.spatial_stage_ids`), ensuring exact architecture reconstruction without external config files.
- **Feature ownership**: `encode_image` returns an `ImageFeatures` object that keeps detail features with their image pyramid. Existing four-feature iteration/indexing is preserved. Callers with a plain pyramid must pass matching `detail_features` explicitly when detail fusion is active.

---

## Training

The dataset root contains `train_grounding.json`, `val_grounding.json`, image folders `train/`, `val/`, and masks in `binary_masks_png/<split>/`.

### Matched Single-Stage Comparison (Detail Decoder Enabled)

Keep the alignment/stable recipe and frozen text encoder used by the spatial-evidence comparison. Checkpoints below are written directly to `outputs/`.

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

### Training with Custom Detail Stages (e.g. all 4 stages)

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

Batch size is **global across GPUs**: 4 on two GPUs means 2 per GPU. Training chooses CUDA automatically, uses AMP on CUDA, and does not run validation.
The historical CLI defaults remain `full` and `legacy`; the matched comparison above selects its recipe explicitly. Architecture flags omitted on a fresh run enable detail stages `(2, 3)` with 16 channels.

| `--training-recipe` | Objective |
|---|---|
| `full` | Predicted-answer mask + oracle/Q-reference masks + gain-weighted mask KD + answer alignment + attention guidance; no warm-up |
| `grounding-only` | Predicted-answer mask supervision only; the latent predictor remains present |
| `alignment` | Predicted-answer mask supervision + answer-vector alignment |
| `warmup` | Answer alignment throughout; mask supervision transitions from real A to predicted A' |

For new matched comparisons, explicitly select `--loss-version stable`. This uses audited KL calculations and valid-answer reductions.

---

## Checkpoints and Evaluation

`--save-every 10` writes resumable `checkpoint_epoch10.pt`, etc. Training also writes final raw weights as `model_final_epoch<N>.pt`.

```bash
uv run python eval.py \
    --checkpoint outputs/checkpoint_epoch20.pt \
    --dataset val \
    --data-root /path/to/vizwiz \
    --batch-size 4 \
    --num-workers 2 \
    --output-dir outputs/eval_val_detail_ep20
```

Evaluation scores and exports masks at native resolution by default. Student predictions do not use ground-truth answers.

---

## Diagnostics & Probes

All diagnostic tools seamlessly support the detail decoder:
- `diagnose.py`: Compares Q-only, predicted-answer, and real-answer masks.
- `probe_latent_conditioning.py`: Interventions, embedding retrieval, and conditioning traces.
- `probe_predicted_answer.py`: Caches predicted vectors and runs answer swaps/zeroing.
- `predict_save.py` & `visualize_predictions.py`: Single-image inference and mask overlay generation.
