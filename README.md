# Crop support: four controlled continuations in two Kaggle sessions

Branch: `exp/crop-recall-preservation`, from context-crop-pyramid `2b40528`.
All runs start INDEPENDENTLY from the FULL B/mild_focus counter5 checkpoint.
They retain Adam/scaler state, LR1e-5, frozen vision/text, group weights, sampler,
seed42 and batch6. Train counters6-10 only: five more epochs,30total with parent20.
No run starts from another run, the crop10 checkpoint, C/D, or original parent20.

No new inference parameters or inputs. No inference crops, teacher or GT are
needed. Reference state exists only in training/full checkpoints of retain runs.

## Why these experiments

The completed crop25 run improved tiny/medium validation IoU by2.51/2.45pp versus
matched B10, but lost3.17pp on large regions; large recall88.28->83.42%.
Unchanged B5->B10 training reduced small/medium accuracy. These runs test the
crop dose and two targeted responses to missed large foreground, not more epochs.

| Session | Folder / --variant | Crop probability | Large/full retention | Large/full final overlap |
|---|---|---:|---|---|
|1|low_crop|0.10|None|Dice (unchanged)|
|1|retain|0.25|One-sided B5 foreground penalty, weight1|Dice (unchanged)|
|2|recall|0.25|None|Tversky: FP0.3 / FN0.7|
|2|retain_recall|0.25|Same retention, weight1|Same Tversky|

The optional `control` variant replays the previous crop25 recipe; it is NOT run
by default because its result already exists. This is a fixed set of hypotheses,
not a hyperparameter sweep or an established winning combination.

### Scope and exact objectives

Group labels/weights use the ORIGINAL uncropped nearest-resized TRAIN mask.
The support scope S is source foreground area>=10% AND crop_applied==0. Therefore
an originally tiny crop that becomes large is NEVER treated as a large image.
Large images that really are cropped use the original objective as well.

Baseline per-image loss L_B: final BCE+0.5Dice plus the original auxiliary
BCE+Dice heads. Only the final overlap on S changes for recall variants:

    T = (TP+1)/(TP + 0.3*FP + 0.7*FN + 1)
    D = (2*TP+1)/(2*TP+FP+FN+1)
    L_recall = L_B + 0.5*(D-T) on S; L_B elsewhere

This is a REPLACEMENT of final Dice, not an extra duplicate BCE/Tversky loss.
All auxiliary terms stay exactly unchanged. These are soft pixel sums in FP32,
with the existing loss's smoothing constants. More FN penalty can also increase
false positives; large precision/IoU must be checked along with recall.
[Tversky](https://arxiv.org/abs/1706.05721) provides the established FP/FN tradeoff.

For retention, K is GT-positive AND frozen B5 logit>0 AND S:

    keep(image) = mean_K relu(sigmoid(B5)-sigmoid(student))^2
    L_retain = L_B + 1.0*keep
    L_retain_recall = L_recall + 1.0*keep

Empty K gives0; normalize per image. Teacher false positives/missed positives are
not protected. No penalty for exceeding parent confidence. This encourages a
probability floor, not guaranteed binary recall or perfect localization.
The idea is related to [Learning without Forgetting](https://arxiv.org/abs/1606.09282);
it is not a claim to invent distillation. The prior C result remains negative
for its tested recipe; applying it during crop drift is a NEW hypothesis.

The B5 reference copies ONLY post-encoder modules (~6.65M parameters), excluding
CLIP vision/text. Encoder outputs are shared within the student's ONE forward.
Reference runs eval/no_grad, outside DDP/optimizer/model_state_dict. Its AMP-shared
features can differ near thresholds from separate FP32 evaluation. Resume restores
the ORIGINAL B5 body from support_state, never reanchors to the current student.
Low_crop/recall allocate no teacher. All variants keep the same inference model.

### Crop policy

Same crop geometry as the completed crop25 experiment: choose a native-RGB
window preserving all TRAIN GT components, source aspect ratio (pixel rounding
aside), bbox context and jittered position. Side fraction is between35% and90%,
with bbox/context constraints; infeasible or empty masks fall back to full image.
RGB/mask coordinates match BEFORE336/768views and existing CLIP normalization.
Only crop probability differs for low_crop. Actual crop rate is below requested
rate because most large GT extents cannot satisfy the context window.

## Cell 1 — Setup (both sessions)

Use a fresh Kaggle session with Internet, two GPUs, VizWiz and the FULL B5
checkpoint attached. Result ZIPs exclude .pt; raw weights cannot resume Adam.
Optionally attach previous A/B10 and crop10 RESULTS ZIPs for matched comparison.
Keep checkpoints mounted in /kaggle/input rather than copying into outputs.

```python
!git clone --single-branch --branch exp/crop-recall-preservation https://github.com/DPTPhatUS/VizWiz-VQA-Grounding.git
%cd /kaggle/working/VizWiz-VQA-Grounding
!uv sync --frozen
```

Already cloned on this branch: `!git pull --ff-only`, then `!uv sync --frozen`.
Run `train_group.py`, not `train.py`.

## Cell 2 — Choose session and B5

Set SESSION=1 for low_crop+retain. In a SECOND fresh session, set SESSION=2 for
recall+retain_recall and repeat cells1-5. Auto-detect B5 by metadata, not filename;
set B_CKPT explicitly if several copies are attached. Load only trusted files.

```python
import gc
import json
import os
from pathlib import Path
from training.crop_support import RECIPES
from utils import read_checkpoint

SESSION = 1  # Session2: change this to2; all other steps stay the same.
os.environ['DATA_ROOT'] = '/kaggle/input/datasets/phtphantun/vizwiz-vqa-grounding/vizwiz'
os.environ['INPUT_ROOT'] = '/kaggle/input'
os.environ['B_CKPT'] = ''  # Optional exact full B/mild_focus checkpoint_epoch5.pt.
os.environ['RUN_ROOT'] = 'outputs'
os.environ['EVAL_BATCH'] = '4'
os.environ['EVAL_TEST'] = '1'  # Use0 if local test GT is unavailable.
os.environ['BASELINE_RESULTS_ZIP'] = ''  # Optional size_weighted_pyramid_two_runs_ep10_seed42.zip.
os.environ['CROP_RESULTS_ZIP'] = ''  # Optional context_crop_pyramid_ep10_seed42.zip.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
assert SESSION in (1, 2)
variants = ('low_crop', 'retain') if SESSION == 1 else ('recall', 'retain_recall')
os.environ['RUN_VARIANTS'] = ' '.join(variants)
os.environ['RUN_NAME'] = f'crop_support_s{SESSION}_ep10_seed42'
paths = ([Path(os.environ['B_CKPT'])] if os.environ['B_CKPT'] else
         sorted(Path(os.environ['INPUT_ROOT']).rglob('checkpoint_epoch5.pt')))
choices = []
for path in paths:
    saved = read_checkpoint(path)
    config = saved.get('run_config', {})
    if (saved.get('epoch') == 5 and config.get('focus_alpha') == 0.25
            and config.get('objective_version') == 'size-weighted-updates-v1'):
        assert saved.get('optimizer_state_dict', {}).get('state'), 'Use a FULL checkpoint.'
        assert 'scaler_state_dict' in saved
        choices.append((path, config, saved.get('training_controls', {}).get('config')))
    del saved
    gc.collect()
if len(choices) != 1:
    raise ValueError(f'Select one B5 using B_CKPT: {[str(x[0]) for x in choices]}')
path, source_config, controls = choices[0]
assert controls is not None
for key, expected in {
    'batch_size': 6, 'gradient_accumulation_steps': 1, 'world_size': 2,
    'seed': 42, 'lr': 0.00001, 'gradient_checkpointing': True, 'project_updates': False,
}.items():
    assert source_config[key] == expected, key
for key, expected in {
    'optimizer': 'adam', 'lr': 0.00001, 'weight_decay': 0.0,
    'lr_scheduler': 'none', 'warmup_ratio': 0.0,
    'freeze_vision_encoder': True, 'freeze_text_encoder': True,
}.items():
    assert controls[key] == expected, key
assert source_config['training_config']['overlap_loss'] == 'dice'
os.environ['B_CKPT'] = str(path)
root = Path(os.environ['RUN_ROOT'])
for variant in variants:
    folder = root / variant
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f'Use a fresh output folder: {folder}')
(root / 'configs').mkdir(parents=True, exist_ok=True)
for variant in variants:
    config = {**source_config['training_config'], 'focus_alpha': 0.25,
              'project_updates': False, 'variant': variant}
    (root / 'configs' / f'{variant}.json').write_text(json.dumps(config, indent=2))
(root / 'session.json').write_text(json.dumps({
    'session': SESSION, 'variants': list(variants), 'B5_checkpoint': str(path),
    'source_config': source_config, 'optimizer_controls': controls,
    'recipes': {name: RECIPES[name].config for name in variants},
}, indent=2))
print('B5:', path)
print('Runs:', os.environ['RUN_VARIANTS'])
```

One experiment selector: `--variant`. JSON overrides CLI then defaults. Presets
replace the previous standalone crop-probability flag; probability/retention/FN
weights are fixed and versioned to avoid an accidental sweep or hidden overrides.
The existing optimizer/LR/freeze flags and JSON precedence remain available.
Do not change these training settings between the prescribed runs.

## Cell 3 — Two independent runs, then evaluate

```bash
%%bash
set -euo pipefail
for variant in $RUN_VARIANTS; do
    run_dir="$RUN_ROOT/$variant"
    common=(
        --data-root "$DATA_ROOT"
        --resume-checkpoint "$B_CKPT"
        --config "$RUN_ROOT/configs/${variant}.json"
        --variant "$variant"
        --batch-size 6
        --gradient-accumulation-steps 1
        --lr 0.00001
        --optimizer adam
        --weight-decay 0
        --lr-scheduler none
        --warmup-ratio 0
        --freeze-vision-encoder True
        --freeze-text-encoder True
        --no-project-updates
        --gradient-checkpointing
        --seed 42
    )
    uv run --frozen torchrun --standalone --nproc_per_node=2 preflight_group.py \
        "${common[@]}" --num-workers 0 --steps 3
    uv run --frozen torchrun --standalone --nproc_per_node=2 train_group.py \
        "${common[@]}" --output-dir "$run_dir" \
        --num-epochs 10 --num-workers 2 --save-every 10 --no-save-final-weights
    uv run --frozen python eval.py \
        --checkpoint "$run_dir/checkpoint_epoch10.pt" --dataset val \
        --data-root "$DATA_ROOT" --device cuda \
        --batch-size "$EVAL_BATCH" --num-workers 2 --output-dir "$run_dir/eval_val_ep10"
    uv run --frozen python diagnose.py \
        --predictions "$run_dir/eval_val_ep10" --data-root "$DATA_ROOT" \
        --output-dir "$run_dir/diagnostic_ep10"
    if [[ "$EVAL_TEST" == '1' ]]; then
        uv run --frozen python eval.py \
            --checkpoint "$run_dir/checkpoint_epoch10.pt" --dataset test \
            --data-root "$DATA_ROOT" --device cuda \
            --batch-size "$EVAL_BATCH" --num-workers 2 --output-dir "$run_dir/eval_test_ep10"
    fi
done
# Evaluate the unchanged B5 reference once per session, not once per variant.
uv run --frozen python eval.py \
    --checkpoint "$B_CKPT" --dataset val --data-root "$DATA_ROOT" --device cuda \
    --batch-size "$EVAL_BATCH" --num-workers 2 --output-dir "$RUN_ROOT/reference/eval_val_ep5"
uv run --frozen python diagnose.py \
    --predictions "$RUN_ROOT/reference/eval_val_ep5" --data-root "$DATA_ROOT" \
    --output-dir "$RUN_ROOT/reference/diagnostic_ep5"
```

Commands are sequential independent processes. Preflight updates are discarded;
training reloads the SAME B5 for every variant, including its optimizer/scaler.
`num-epochs10` means five new epochs, not ten new epochs or fresh training.
No validation inside training; full-image FP32 eval/diagnostic runs afterward.
Use EVAL_BATCH=1 if batch4 exceeds memory. Test needs local GT; choose using val.

Each session writes ONLY two new `outputs/<variant>/checkpoint_epoch10.pt` files.
No checkpoints6-9, copied5, duplicate final weights or checkpoint deletion.
Retain checkpoints include only an extra post-encoder reference (~26.6MB FP32
parameters plus small buffers), not a second backbone. Both frozen encoders are
still stored in every full checkpoint; leave enough disk for both files.

If interrupted before10, restart the failed variant from B5 in a fresh folder;
there is no intermediate checkpoint by design. If the first run completes, set
RUN_VARIANTS to only the failed second run before rerunning Cell3. The reference
may be reevaluated. For an own-branch checkpoint resume, keep the SAME variant;
its original reference is restored from support_state, not reset. Previous
crop/C/D optimizer checkpoints are rejected; old weights still evaluate normally.

Two five-epoch runs per session are the intended budget. CPU checks do not prove
Kaggle runtime/VRAM/disk fit; if necessary run one variant per available session.
Do not shorten one run only and call it a matched comparison.

## Cell 4 — Compare both runs with B5 and existing B10/crop25

Attach the existing result ZIPs and set BASELINE_RESULTS_ZIP/CROP_RESULTS_ZIP
in Cell2, or set those environment variables here before running this cell.
They are read directly; no checkpoint loads, extra inference or extraction.
Without them the table is provisional: B5 has five fewer training epochs.

```python
import csv
from zipfile import ZipFile

root = Path(os.environ['RUN_ROOT'])
entries = []
def local_result(folder, epoch):
    summary = json.loads((folder / f'eval_val_ep{epoch}/metrics.json').read_text())['summary']
    diag = json.loads((folder / f'diagnostic_ep{epoch}/summary.json').read_text())
    path = folder / f'eval_test_ep{epoch}/metrics.json'
    test = json.loads(path.read_text())['summary'] if path.is_file() else {}
    return summary, diag, test
entries.append(('B5_reference', 5, *local_result(root / 'reference', 5)))
for variant in variants:
    summary, diag, test = local_result(root / variant, 10)
    assert summary['objective_version'] == 'crop-support-updates-v1'
    assert summary['training_protocol']['support_recipe'] == RECIPES[variant].config
    entries.append((variant, 10, summary, diag, test))
for variable, label, prefix, expected_version in [
    ('BASELINE_RESULTS_ZIP', 'B10_control', 'mild_focus', 'size-weighted-updates-v1'),
    ('CROP_RESULTS_ZIP', 'crop25_control', 'context_crop', 'context-crop-updates-v1'),
]:
    if not os.environ[variable]:
        continue
    with ZipFile(os.environ[variable]) as archive:
        def read_result(suffix, required=True):
            matches = [name for name in archive.namelist() if name.endswith(suffix)]
            if not matches and not required:
                return {}
            if len(matches) != 1:
                raise ValueError(f'Expected one {suffix}: {matches}')
            return json.loads(archive.read(matches[0]))
        summary = read_result(f'{prefix}/eval_val_ep10/metrics.json')['summary']
        diag = read_result(f'{prefix}/diagnostic_ep10/summary.json')
        test = read_result(f'{prefix}/eval_test_ep10/metrics.json', required=False).get('summary', {})
        assert summary['objective_version'] == expected_version
        entries.append((label, 10, summary, diag, test))
rows = []
for label, epoch, summary, diag, test in entries:
    protocol = summary['training_protocol']
    for key in ('initial_checkpoint', 'mask_group_sha256', 'group_counts',
                'proposal_weights', 'updates_per_epoch', 'focus_alpha'):
        assert protocol[key] == source_config['group_protocol'][key], (label, key)
    assert summary['optimization_config'] == source_config['optimization_config']
    assert summary['training_config'] == source_config['training_config']
    recipe = protocol.get('support_recipe', {})
    rows.append({
        'run': label, 'counter': epoch, 'total_training_epochs': 20 + epoch,
        'crop_probability': protocol.get('context_crop', {}).get('probability', 0),
        'retention_weight': recipe.get('retention_weight', 0),
        'large_fp_weight': recipe.get('large_fp_weight'),
        'val_miou_pct': 100 * summary['mean_iou'],
        'val_global_iou_pct': 100 * summary['global_iou'],
        **{f'{name}_miou_pct': 100 * diag['by_gt_area'][group]['mean_iou']
           for name, group in [('small', 'small_lt_1pct'),
                               ('medium', 'medium_1_to_10pct'),
                               ('large', 'large_ge_10pct')]},
        'large_recall_pct': 100 * diag['by_gt_area']['large_ge_10pct']['mean_recall'],
        'large_precision_pct': 100 * diag['by_gt_area']['large_ge_10pct']['mean_precision'],
        'test_miou_pct': 100 * test['mean_iou'] if test else None,
    })
comparison = root / 'support_comparison.csv'
with comparison.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(comparison.read_text())
```

### Decision rule

Evaluate tiny AND medium IoU, large recall/precision/IoU and overall mean IoU.
Compare low_crop with B10 and crop25 to understand the dose tradeoff. Compare
retain/recall/retain_recall directly with crop25, and with each other, at SAME
30total epochs. B5 is the working checkpoint reference, not a matched-budget
control. Prior seed42 values:

| Reference | Val mean | Tiny | Medium | Large | Large recall |
|---|---:|---:|---:|---:|---:|
| B5 |66.1693|19.5605|47.8303|79.4138|87.5861|
| B10 |66.0497|18.8439|46.8101|79.7930|88.2752|
| crop25+10 |64.9561|21.3502|49.2588|76.6254|83.4239|

A useful candidate should recover the crop large-region deficit while retaining
small/medium gains. Neither greater recall alone nor global IoU alone is success.
Do not select using test, declare a winner from one seed, or extend epochs just
because train loss falls. No new sample annotations or validation-based training
weights are used. Four exploratory comparisons increase selection optimism;
additional matched seeds are needed after selecting a recipe.

## Cell 5 — ZIP results, exclude checkpoints

```python
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(os.environ['RUN_ROOT'])
archive = Path('/kaggle/working') / f"{os.environ['RUN_NAME']}.zip"
with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() not in {'.pt', '.tmp', '.zip'}:
            bundle.write(path, path.relative_to(root))
print('Download results:', archive)
print('Keep the two full checkpoint_epoch10.pt files separately.')
```

ZIP includes configs/session plan, histories6-10, val/test masks and metrics,
diagnostics, B5 reference and comparison; no .pt/.tmp/nested.zip files.
Save each session ZIP under its distinct s1/s2 name.

## Logging, compatibility and implementation checks

`train_loss` / `proposal_loss` include the active support terms. The original
mixed full/crop segmentation loss remains available as
`base_segmentation_objective` and `loss_<group>`. The relationship is:

    train_loss = base_segmentation_objective
                 + retention_objective_contribution + recall_objective_delta

recall_objective_delta is SIGNED (Tversky replacement can be below Dice). A lower
scalar objective alone is not evidence of better segmentation. Retention support
logs `retention_shortfall_large`, `protected_coverage_large`, and
`support_fraction_large`; teacher-free recipes have zero protected coverage.
Crop request/applied/input-area logs retain their original source-group meaning.
Source mask groups, all recipe constants, B5 file hash, reference state, data and
source-code hashes are saved. Subsequent resume validates recipe/policy and
requires support_state; it cannot silently drop or reset a reference.

The optional control preset must reproduce old crop25 training (same model,
optimizer and scaler) on a fixed CPU fixture. The model encoder/body split keeps
state keys and FP32 outputs unchanged. Reference forwards never re-run frozen
encoders. ContextCrop remains independent of backbone feature dimensions.
All checks are CPU/synthetic execution checks, not real Kaggle accuracy/fit or
cross-backbone validation. No changes are made to prior experimental branches.