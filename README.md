# VizWiz VQA Grounding: Fixed SigLIP 2 Base NaFlex

Branch `exp/siglip2-naflex-baseline` derives from `exp/baseline-fixes` and fixes the vision encoder to `google/siglip2-base-patch16-naflex` (768 channels, 12 transformer layers). The companion branch `exp/siglip2-so400m-naflex-baseline` fixes the vision encoder to So400m (1152 channels, 27 layers). Neither branch includes a runtime vision-backbone switch.

`models/image_encoder.py` manages the vision encoder and patch grids, `models/text_encoder.py` manages text encoding, and `models/model.py` defines cross-attention, the decoder, and mask prediction. Both branches retain three feature skips from hidden states 4, 7, and 9 (all on the same spatial patch grid), final-state fusion, decoder channels `1024/512/256/128`, Adam optimization, and per-image BCE supervision.

## Text and image inputs

CLIP text (`openai/clip-vit-large-patch14-336`) is the default text encoder. Passing `--text-encoder siglip2` selects the text tower from this branch's fixed SigLIP2 checkpoint instead. The SigLIP2 tokenizer pads sequences to a fixed length of 64 tokens, whereas CLIP uses dynamic padding up to 77 tokens. Linear projections into the fusion layer adapt to the active text and vision channel widths. The unused SigLIP pooled text head stays frozen, and token features drive grounding.

`--text-mode qa` formats the prompt as a single sequence (`Q: <question> A: <answer>`), while `--text-mode question` trains a question-only control. Changing the text encoder or text mode starts a separate experiment.

Images pass directly from raw RGB PIL into the SigLIP2 processor, which resizes them while approximately preserving aspect ratio within `--max-num-patches`, extracts patches, rescales pixel values by `1/255`, and normalizes with RGB mean and standard deviation `[0.5, 0.5, 0.5]` (`(pixel/255 - 0.5)/0.5`). No 336x336 square resize or CLIP normalization is applied beforehand. Saved processor metadata is verified when loading checkpoints. Ground-truth binary masks are resized to the processed patch grid via nearest-neighbor interpolation, and padding tokens are stripped before CNN decoding and loss computation. See the [official Base processor configuration](https://huggingface.co/google/siglip2-base-patch16-naflex/blob/main/preprocessor_config.json) for upstream details.

The default patch budget on this branch is 512 (compared to 256 in the upstream processor). To evaluate a 1024-patch budget, run a separate experiment with a new output directory. Because replacing CLIP with SigLIP2 NaFlex changes both the pretrained visual representation and the input resolution, metric differences reflect both changes together.

## Setup and training

```bash
uv sync --locked
```

Expected data layout under `--data-root`: `train_grounding.json`, `val_grounding.json`, `test_grounding.json`, `train/`, `val/`, `test/`, and `binary_masks_png/{train,val,test}/`. The first run downloads the pretrained encoder weights, processor, and tokenizers; Hugging Face caches the full checkpoint file even when only one tower is used.

Suggested two-GPU training command:

```bash
uv run torchrun --nproc_per_node=2 train.py \
    --data-root /kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz \
    --num-epochs 20 \
    --batch-size 2 \
    --gradient-accumulation-steps 2 \
    --lr 0.00001 \
    --num-workers 2 \
    --max-num-patches 512 \
    --text-encoder clip \
    --text-mode qa \
    --gradient-checkpointing \
    --amp \
    --validate-every 1 \
    --save-every 10 \
    --seed 42 \
    --output-dir outputs/naflex_base_clip_512
```

`--batch-size` specifies the global micro-batch across GPUs, so `--batch-size 2` on 2 GPUs places 1 image per GPU. With `--gradient-accumulation-steps 2`, the effective batch size is `2 * 2 = 4` images per optimizer step (except on the final partial window). `--num-workers` is set per process. `--amp` enables FP16 mixed precision with gradient scaling on CUDA, and `--gradient-checkpointing` recomputes encoder activations during the backward pass to lower memory usage.

Keep the micro-batch size and gradient accumulation steps identical across branches when running matched comparisons. Because NaFlex decoding groups images by patch grid dimensions `(H, W)` and applies local BatchNorm per group, different DDP ranks can see different group counts, making per-group SyncBN unsafe. Gradient accumulation also produces different BatchNorm statistics than a larger per-step micro-batch.

So400m requires substantially more memory than Base, and enabling `--text-encoder siglip2` adds a larger text tower and its optimizer state. On 16 GB GPUs, check memory headroom before launching long runs, and start with CLIP text to isolate the vision encoder change.

All training flags except `--resume-checkpoint` appear in the example above. Use `--no-amp` or `--no-gradient-checkpointing` to disable those features. `--save-every 0` disables periodic checkpoints while still writing the final model, and `--validate-every 0` disables per-epoch validation. In-training validation logs BCE loss and uses a distributed sampler that may pad the final batch; run `eval.py` to compute exact validation mIoU over the unpadded split.

## Checkpoints, evaluation and resume

```bash
uv run eval.py --checkpoint outputs/naflex_base_clip_512/checkpoint_epoch20.pt --dataset val --data-root /kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz --output-dir outputs/eval_naflex_base_clip_512
uv run train.py --data-root data/vizwiz --resume-checkpoint outputs/naflex_base_clip_512/checkpoint_epoch20.pt --num-epochs 30 --output-dir outputs/naflex_base_clip_512
```

Checkpoints store the model architecture, encoder configurations, preprocessing metadata, patch budget, text encoder, text mode, and training arguments. Both `eval.py` and `--resume-checkpoint` restore these settings automatically (requiring cached or downloadable tokenizer files). Resuming rejects changes to the vision backbone, text encoder, text mode, or patch budget. Micro-batch size, gradient accumulation, AMP, and gradient checkpointing default to the saved values unless explicitly overridden on the CLI. When resuming, `--num-epochs` specifies the target total epoch count.

Periodic checkpoints (`checkpoint_epoch*.pt`) include optimizer and GradScaler states, whereas final model files contain weights and metadata only (resuming from a final model file initializes a fresh optimizer). Existing Base NaFlex checkpoints trained with CLIP text preserve their parameter keys and shapes, and checkpoints missing text-encoder metadata default to CLIP. Load checkpoints using their matching experiment branch; CLIP-vision checkpoints from `exp/baseline-fixes` are incompatible with this branch.

By default, `eval.py` evaluates at native ground-truth resolution by resizing predicted logits to the original image dimensions, thresholding, scoring, and saving the resulting mask. Passing `--metric-resolution resized` scores against legacy 336x336 resized ground-truth masks instead. Always compare models under the same resolution and text mode. Evaluating a QA-trained model without answers serves as a sensitivity diagnostic rather than a true question-only baseline. The chosen evaluation protocol is recorded in the output metadata.

```python
from models import GroundingModel
model = GroundingModel.from_checkpoint('outputs/naflex_base_clip_512/checkpoint_epoch20.pt')
result = model.predict('image.jpg', question='What does this label say?', answer='example')
result['pred_pil'].save('mask.png')  # original image dimensions
```

Update the hardcoded file paths and questions in the example scripts before running them directly. For dataset-wide metrics and mask exports, use `eval.py`.

## Verification

```bash
uv run python -m unittest discover -s tests -v
```

The 12 CPU unit tests verify rectangular patch grids, ground-truth mask alignment, image normalization, padding removal, gradient flow, equal-image loss weighting, pretrained tower extraction, native-resolution prediction, train/eval/resume lifecycles, text encoder switching, and partial gradient accumulation. The suite also tests two-process Gloo DDP with mismatched per-rank grid groups and gradient checkpointing using small synthetic encoders.

## Project

Research fork of [yjh9929/VizWiz-VQA-Grounding](https://github.com/yjh9929/VizWiz-VQA-Grounding), licensed under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/). Figures in `images/` are inherited from the upstream repository.

## Encoder freezing and optimization

Omitting the flags below keeps this branch's default training configuration: Adam with learning rate `1e-5`, zero weight decay, a constant learning rate, and the branch's default encoder freezing policy. Encoder freezing is opt-in, while the detail CNN and mask decoder always remain trainable. Model state-dict keys and evaluation preprocessing are unchanged.

| Option | Values / default |
| --- | --- |
| `--freeze-backbone [BOOL]` | Freeze or unfreeze both pretrained encoders; omitting keeps the branch default |
| `--freeze-vision-encoder [BOOL]` | Freeze or unfreeze the vision encoder only |
| `--freeze-text-encoder [BOOL]` | Freeze or unfreeze the text encoder only |
| `--optimizer` | `adam` (default), `adamw` |
| `--lr` | Main learning rate; default `0.00001` |
| `--backbone-lr` | Optional separate learning rate for unfrozen encoders; defaults to `--lr` |
| `--weight-decay` | Nonnegative weight decay; default `0`. Biases, normalization parameters, and token/position embeddings are excluded from decay when parameter groups are active |
| `--lr-scheduler` | `none` (default), `cosine`, `linear`, `poly` |
| `--warmup-ratio` | Fraction of total optimizer updates used for linear warmup, in `[0, 1)`; default `0` |
| `--min-lr-ratio` | Final LR divided by initial LR for decay schedules; default `0.1` |
| `--poly-power` | Positive exponent for the `poly` scheduler; default `0.9` |
| `--config` | Path to a flat JSON file with training options; optional |

`BOOL` accepts `True` or `False`, and passing a freeze flag without an argument sets it to `True`. The negative aliases `--no-freeze-backbone`, `--no-freeze-vision-encoder`, and `--no-freeze-text-encoder` set the corresponding option to `False`. Within the same configuration source, a component-specific flag (`--freeze-vision-encoder` or `--freeze-text-encoder`) overrides `freeze_backbone`. Precedence follows **JSON config > explicit CLI flags > branch defaults**, so a `freeze_backbone` entry in `--config` overrides component flags passed on the CLI. Branch-specific recipe and loss settings can be included in the same JSON file, and unknown keys raise an error immediately. See `train.py --help` for the full list of CLI flags.

Example flags to append to a training command:

```bash
--freeze-backbone True \
--optimizer adamw \
--lr 0.0001 \
--weight-decay 0.01 \
--lr-scheduler cosine \
--warmup-ratio 0.05 \
--min-lr-ratio 0.1
```

To freeze only the vision encoder while keeping the branch's text encoder policy, pass `--freeze-vision-encoder True`. To train both encoders, pass `--freeze-backbone False`. Architectures with a fixed reference stage (such as gain-guided refinement) reject encoder unfreezing. Unused CLIP and SigLIP pooled-output parameters remain frozen even when token encoders are unfrozen.

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

When frozen, encoders are placed in `eval` mode and omitted from the optimizer. Learning rate schedulers step once per completed optimizer update (accounting for gradient accumulation) and do not advance when AMP skips a step due to gradient overflow. Checkpoints record the resolved optimization configuration, trainable parameter groups, and scheduler state. Resuming restores these settings and rejects conflicting overrides; keep `--num-epochs`, dataset length, and gradient accumulation unchanged when resuming a scheduled run so the learning rate schedule stays aligned.

Older checkpoints remain directly loadable for evaluation. Resuming a legacy checkpoint preserves its original optimizer layout and constant learning rate; changing encoder freezing, the optimizer, weight decay, or the learning rate schedule requires starting a new run (or initializing from weights where supported). Resuming also follows each branch's existing RNG and data-ordering behavior.
