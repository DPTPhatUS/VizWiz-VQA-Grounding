# Parent-unseen utility evidence edits

Branch: `exp/utility-evidence-edits`.

Goal: improve small (<1% GT area) and medium (1-10%) masks while limiting damage
to large (>=10%) masks. Improvement is a hypothesis, not a measured result.
This branch preserves the existing pyramid architecture and adds a small
question-conditioned decision head. It does not regenerate OCR masks with a CNN.

## Why this experiment differs

The previous Transport head learned useful word scores, but the learned mask
and scope gate lost their benefit. On 145 val images satisfying its GT scope
criterion, learned evidence IoU was about 28.7%, versus about 57.3% using the same
selection scores and original polygon unions. The original gate still failed.
These are diagnostic subgroup results, not deployable performance claims.

Three methodological connections motivate this implementation:

- [Mask Scoring R-CNN, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html)
  learns mask quality separately from classification confidence. Here the head
  predicts the **advantage of an exact candidate mask over the parent**, rather
  than OCR purity. We do not reproduce its network or COCO claims.
- [Consistent Estimators for Learning to Defer, ICML 2020](https://proceedings.mlr.press/v119/mozannar20b.html)
  frames expert selection as cost-sensitive learning. Here identity/parent is
  an explicit zero-advantage action. Our Huber-plus-expected-regret objective
  is an adaptation, **not the paper's exact surrogate or consistency theorem**.
- [Learn then Test](https://arxiv.org/abs/2110.01052) separates learning from risk
  testing. We select a single policy on tune, then audit it on independent
  safety images with an exact binomial bound. This is a restricted fixed-policy
  application, not a reproduction of its full multiple-testing framework.

## Architecture and two matched objectives

```text
I + Q --> frozen pyramid --> native parent mask P
                |
                +--> visual feature map + full question tokens
OCR strings --> frozen text encoder --> word vectors
OCR polygons --> singles / nearby line groups / all-word union
                |
        exact native actions:
        P; R; P intersect R; P union R; P minus R
                |
word/Q attention + set context + candidate visual/geometry + parent reference
                |
        shared advantage scorer (64 hidden channels)
                |
choose one action if its score exceeds the selected threshold; otherwise P
```

`R` is a polygon union. Nearby groups use existing deterministic geometry-based
proposals; no GT or answer influences proposal membership/order. There are at
most `4*(2*N+1)+1` actions (197 at N=24), often fewer because identity edits are
removed. Candidate features are cached. Mask geometry/IoU labels remain native.
The scorer adds **170,577 parameters** at CLIP dimensions (about 0.68 MB FP32 weights).
The scorer has no backbone-specific text/visual width; dimensions come from the
manifest. A new backbone still requires a compatible extractor and new cache.

Both experiments have the **same actions, architecture, initialization and image
order**, differing only in objective:

| Recipe | Supervised utility | Training emphasis |
|---|---|---|
| `gain` | `Delta = IoU(action, GT) - IoU(parent, GT)` | expected IoU improvement |
| `risk` | `Delta - 3 * is_large_GT * max(-Delta, 0)` | stronger cost for harming large GT; small/medium image losses weighted 2 |

For each image, loss is valid-edit mean Huber regression on utility/0.1 plus
expected regret under `softmax(scores/0.1)`. Regret is best utility minus the
policy's expected utility. Padding is excluded and slot zero is exactly zero.
Native labels use integer intersections/unions and float64 ratios. Sensitive loss
operations use FP32; no FP16 mask-area sums. Safety's strict 5-point IoU-loss
comparison uses integer cross-multiplication, including at threshold boundaries.
The resulting scores are **utilities, not calibrated probabilities**. We select
a complete binary mask; no multiplication of gate/word/pixel confidences.

GT area appears only in training costs, tuning metrics and safety auditing;
there is no GT area/answer input during inference.

## Parent-unseen protocol (required)

An old full-train pyramid cannot supply honest train-side utility labels: it has
already learned those images. This pilot therefore requires **one newly trained
parent**, with image membership recorded in its checkpoint.

1. Hash-split official train into 80% parent-fit and 20% head-only images, keeping identical upright pixel hashes
   in the same group. One representative per exact duplicate is retained for
   the head cohorts; exclusions are recorded.
2. Train the parent on parent-fit only, for 20 epochs.
3. Freeze the entire parent. Split head-only images, independently of labels,
   into 60% fit / 20% tune / 20% safety.
4. Train each head on fit for 20 epochs. Select checkpoint 5/10/15/20 and threshold
   on tune, maximizing the equally weighted small/medium IoU gain, subject to
   nonnegative overall and large-group mean gain on tune.
5. Audit the ONE chosen policy on safety. Do not search another threshold after
   inspecting safety. If it fails, deploy identity/parent.
6. Evaluate proposed and audited policies on **untouched official val/test**.
   They are not used for checkpoint/threshold fitting in this protocol.

With 6,494 train images, the split is approximately 5,195 parent-fit / 779 head-fit
/ 260 tune / 260 safety. Code checks exact lists, parent overlap, cache hashes,
and data annotations. All images remain in their original directories.

This is a matched pilot against **its own 80%-trained parent**. The old full-train
pyramid remains an external reference, not a matched control. A positive pilot
would still need a larger/out-of-fold study and repeated seeds before a paper
claim. Do not hide the reduced parent training set or extra head training budget.

### What the safety check actually means

For each fixed proposed policy it audits
`P(IoU loss > 0.05 | GT area >= 0.10)` over all large images, not only switches.
A one-sided Clopper-Pearson upper bound must be <=0.05. Confidence error is 0.025
per recipe (Bonferroni for two recipes: joint error <=0.05). Safety also requires
nonnegative **observed** mean gain overall and on large images.

This is **not a guarantee of nondecreasing population mean IoU**, nor a guarantee
that no large image is harmed. The bound requires an independent safety sample,
fixed policy and IID images from the target distribution. Distribution shift or
repeated redesign after peeking at safety invalidates that interpretation. Too
few safety large images can prevent certification. Identity has exactly zero
relative harm by construction, even without a statistical estimate.

`policy.json` records proposed/deployed thresholds and the audit. A null deployed
threshold means **exact parent fallback**, not a successful new method.

## Kaggle Session 1: train the disjoint parent

Use the usual 2-GPU notebook, internet enabled, with VizWiz and the existing OCR
cache attached. Do not use any of the old `PYRAMID_CKPT` or feature-cache files.

### Setup

```python
!pip install -q uv
!git clone --single-branch --branch exp/utility-evidence-edits https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen

import os
from pathlib import Path
os.environ['DATA_ROOT'] = '/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz'
os.environ['OCR_CACHE'] = '/kaggle/input/datasets/khoahoang0512/vizwiz-ocr-cache/ocr_cache'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
```

```bash
%%bash
set -euo pipefail
uv run prepare_parent_split.py --data-root "$DATA_ROOT" \
    --output-dir splits --holdout-fraction 0.2 --cache-root "$OCR_CACHE" --seed 42
uv run torchrun --standalone --nproc_per_node=2 train_joint.py \
    --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
    --output-dir outputs --variant pyramid --train-image-list splits/parent_fit.json \
    --num-epochs 20 --batch-size 2 --gradient-accumulation-steps 2 \
    --num-workers 2 --lr 0.00001 --optimizer adam --weight-decay 0 \
    --lr-scheduler none --warmup-ratio 0 \
    --freeze-vision-encoder False --freeze-text-encoder True \
    --max-candidates 24 --ocr-dropout 0.2 --text-dropout 0.2 \
    --gradient-checkpointing --amp --save-every 10 --seed 42
```

`batch-size` here is the global microbatch; effective global batch is 4. Parent
snapshots/resume are saved at 10/20, with no in-training validation. Two snapshots
plus replacing full Adam resume have about 9.93 GB transient tensor footprint;
startup checks space including additional reserve. No checkpoint deletion.

Download **`outputs/checkpoint_epoch20.pt`** and **`splits/refiner_fit.json`** and
upload them together as a new Kaggle input dataset. Keep that exact parent file.
Also keep the small metadata ZIP:

```python
import zipfile
with zipfile.ZipFile('utility_parent_metadata.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for path in sorted(Path('splits').glob('*.json')):
        z.write(path, path.name)
    for name in ('run_config.json', 'history.jsonl'):
        z.write(Path('outputs') / name, name)
```

If recovering this parent session, use a new empty output folder and the same
parent-fit list with `--resume-checkpoint INPUT/latest_resume.pt`; num-epochs 20
is the total target. Retain the split files; never generate a different split.

## Kaggle Session 2: one cache, both lightweight heads and all evaluation

One GPU is sufficient; **do not use torchrun for head training**. Attach data,
OCR cache, and the NEW parent plus `refiner_fit.json` from Session 1. Existing
`evidence_transport_feature_cache` cannot be reused because parent and labels
have changed.

Use the same clone/sync/environment setup, then set:

```python
os.environ['PARENT_INPUT'] = '/kaggle/input/YOUR_NEW_PARENT_DATASET'
os.environ['PYRAMID_CKPT'] = os.environ['PARENT_INPUT'] + '/checkpoint_epoch20.pt'
os.environ['HEAD_LIST'] = os.environ['PARENT_INPUT'] + '/refiner_fit.json'
os.environ['UTILITY_CACHE'] = 'outputs/utility_cache'
Path('outputs').mkdir(exist_ok=True)
assert Path(os.environ['PYRAMID_CKPT']).is_file()
assert Path(os.environ['HEAD_LIST']).is_file()
```

### Cache (frozen parent once)

```bash
%%bash
set -euo pipefail
uv run cache_utility_evidence.py \
    --checkpoint "$PYRAMID_CKPT" --head-image-list "$HEAD_LIST" \
    --data-root "$DATA_ROOT" --cache-root "$OCR_CACHE" \
    --output-dir "$UTILITY_CACHE" --max-candidates 24 \
    --batch-size 2 --num-workers 2 --seed 42
```

Parent exposure causes a clear error. Fit/tune/safety labels are separate from
features. Val/test cache construction does not open GT masks. No dense FP32
768-grid cache is stored: native parent masks are losslessly bit-packed,
proposals are deterministic polygon edits, and only small descriptors are FP16.
No old 8.3-GB logit cache or RGB crop CNN cache is needed. Actual cache size varies
with native resolution/candidate count; each write preserves a 2 GiB notebook
reserve and never deletes existing files. The cache must be new/empty.

### Train + select + audit + evaluate (automatic)

```bash
%%bash
set -euo pipefail
uv run run_utility_evidence.py \
    --feature-cache "$UTILITY_CACHE" --data-root "$DATA_ROOT" \
    --output-dir outputs --recipe both --num-epochs 20 \
    --batch-size 16 --num-workers 2 --lr 0.0003 --weight-decay 0.01 \
    --seed 42 --amp
```

This command includes **all evaluation**, not training alone:

- Each head independently trains 20 epochs on fit; the parent never backprops.
- Weights-only snapshots at 5/10/15/20; one tiny replacing full resume every epoch.
- All four checkpoints and the fixed threshold grid are evaluated on tune.
- The chosen checkpoint/threshold is audited once on safety.
- Native masks, per-image/size-group mean IoU and global IoU are exported on
  official val/test for BOTH `proposed` and `certified` views. If safety rejects,
  `certified` is the parent exactly, while `proposed` exposes the failed candidate.

Outputs: `outputs/gain/` and `outputs/risk/`, each with `policy.json`, history,
tuning results, safety rows, snapshots and `eval_{val,test}_{proposed,certified}`.
No teacher checkpoint, answer input, new OCR inference model, native crop CNN or
repeated backbone training per head. Both runs share feature cache, image order,
architecture and initialization. Input dimensions are inferred from the manifest.
GPU training time/VRAM/accuracy are not locally measured; epoch seconds are logged.

### ZIP results (no checkpoints or feature cache)

```python
import zipfile
root = Path('outputs')
with zipfile.ZipFile('utility_evidence_two_runs_ep20_seed42.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for recipe in ('gain', 'risk'):
        for path in sorted((root / recipe).rglob('*')):
            if path.is_file() and path.suffix.lower() not in ('.pt', '.tmp', '.zip'):
                z.write(path, path.relative_to(root))
    z.write(Path(os.environ['UTILITY_CACHE']) / 'manifest.json', 'utility_manifest.json')
```

Stop here for normal export. Do **not** ZIP the large shared cache on working
disk. Reuse it by attaching this notebook's output as Kaggle Input, without
copying it. Head checkpoints and `policy.json` are small; download them separately
if needed. Keep the exact parent checkpoint and original OCR cache for deployment.

## Evaluate existing heads without training

Read the selected epoch from `policy.json`, then use the corresponding snapshot:

```bash
uv run eval_utility_evidence.py \
    --checkpoint /kaggle/input/YOUR_HEAD/checkpoint_epoch5.pt \
    --policy /kaggle/input/YOUR_HEAD/policy.json \
    --feature-cache "$UTILITY_CACHE" --data-root "$DATA_ROOT" \
    --output-dir outputs_eval_gain --dataset both --view both \
    --batch-size 16 --num-workers 2
```

This does not retune thresholds or reopen safety selection. The checkpoint hash
must match the policy. The chosen epoch may differ from this example.

## Resume a head

Use a completed cache attached as Input and a fresh output folder:

```bash
uv run run_utility_evidence.py \
    --feature-cache /kaggle/input/YOUR_UTILITY_CACHE --data-root "$DATA_ROOT" \
    --output-dir outputs --recipe risk --num-epochs 20 \
    --batch-size 16 --num-workers 2 --lr 0.0003 --weight-decay 0.01 --seed 42 --amp \
    --resume-checkpoint /kaggle/input/YOUR_HEAD/latest_resume.pt \
    --previous-checkpoint-dir /kaggle/input/YOUR_HEAD
```

Upload earlier snapshots alongside the resume, since the selected checkpoint
may be from the previous session. Epoch target is total. If latest resume is
already epoch 20, this command finalizes selection/audit/evaluation without more
training. Cache, recipe, source and optimizer settings must match. Do not use
weights-only snapshots to resume AdamW. Never use a previously inspected safety
cohort to iteratively redesign/tune the model and retain the original risk claim.

## GT-free online inference

```bash
uv run predict_utility_evidence.py \
    --checkpoint /path/SELECTED_HEAD.pt --policy /path/policy.json \
    --parent-checkpoint "$PYRAMID_CKPT" \
    --image /path/image.jpg --question "What does the label say?" \
    --cache-root "$OCR_CACHE" --output prediction.png
```

Put the image's OCR JSON under `OCR_CACHE/single/IMAGE_STEM.json`. The policy,
parent and OCR protocol are verified; the same feature builder and exact native
edit operations are used online. `--no-ocr` replaces `--cache-root` for the exact
parent fallback. There is no GT/answer/GT-size input.

## Interpretation and validation boundary

Develop further only if the proposed policy has meaningful small/medium gains
without overall/large degradation on untouched validation, and the audit accepts
nontrivial edits. Report test separately, without tuning from it. An identity-only
result is a failed improvement experiment, even though it passes safety. If the
raw proposal is bad, do not relax safety to manufacture a gain.

CPU fixtures cover native rendering/IoU labels, empty OCR and empty parent,
FP16 finite gradients, parent-exposure rejection, cache/online parity, no val/test
GT access during extraction, two-worker epoch resume, exact binomial bounds,
size-conditioned auditing and full train/select/eval paths. No GPU speed, accuracy,
mean-IoU safety guarantee or cross-dataset/backbone transfer is established.

The only changes to old public scripts are an optional train-image-list in
`train_joint.py` and optional holdout-fraction in `prepare_parent_split.py`.
Defaults retain all train images and the old 15% filename split respectively.
The optional OCR-cache split groups exact pixel duplicates; the new guide enables it. Legacy model
keys/checkpoints/objectives are unchanged. Old guides remain on their branches.
