# Latent-answer grounding: related work and next experiments

Reviewed on 2026-10-15 against `exp/latent-answer-distillation`, training
implementation `bd290c5`. This is a focused method comparison, not a systematic
review or a current leaderboard. No new VizWiz training results were produced.

## What the current mechanism already does

At deployment the model sees image + question. During training, real answers
provide privileged supervision. `infer_latent_answer` pools the question into
one vector, attends to image patches, and predicts one answer vector A'. The
grounder receives question tokens + that vector, with three conditioned skips.
The oracle substitutes a separately encoded answer vector; it does not use the
baseline's joint Q+A tokenization.

The full recipe already contains mask distillation, answer alignment,
foreground/gain attention guidance, and supervised oracle/Q-reference masks.
Two-way cross-attention, skip gating, and output-level KD are therefore **not
missing components**. Adding them again would not constitute a new experiment.
The current skip gate blends Q/A features, but is not a calibrated predictor of
whether A' will improve the final mask.

Two limitations follow from the implementation, rather than experimental proof:

- `no_grad()` stops gradients through each answer target computation, but the
  shared text encoder is updated through other paths. The target can move.
- Cosine + Smooth L1 asks one vector to match a pooled answer embedding,
  including the `A:` prefix. This does not directly measure whether that vector
  supplies the spatial information the decoder needs.

These are hypotheses to investigate, not established explanations for lower mIoU.

## Closest primary sources

### 1. DaVI: same answer-grounding task

Pan et al., *Tell Me the Evidence? Dual Visual-Linguistic Interaction for Answer
Grounding*, CVPR 2022 VizWiz Workshop.
[Paper](https://arxiv.org/abs/2207.05703).
Read: full two-page report; method and Table 1 on page 2.

DaVI conditions question-token representations on visual features, then uses
those representations for answer decoding and visual-mask decoding. It does
not require a generated answer string as input to the mask decoder. It reports
68.52% test IoU for the single model and 70.57% with an ensemble; the ablation
without answering is 64.28%. Multiple interactions change in that ablation,
so the difference cannot be assigned solely to an auxiliary answer loss.
The report does not specify detailed loss weights or a warm-up schedule.

**Implication for us:** learning image-conditioned linguistic evidence can be
more appropriate than treating accurate answer reconstruction as a mandatory
intermediate task. Our model already interacts in both directions; the concrete
difference worth testing is its single pooled-vector bottleneck versus retaining
several tokens. DaVI's scores are not directly comparable to our validation mIoU.

### 2. Cho et al.: the closest missing-answer training condition

Cho et al., *Dealing with Missing Modalities in the Visual Question
Answer-Difference Prediction Task through Knowledge Distillation*,
CVPR 2021 MULA Workshop.
[Paper](https://arxiv.org/abs/2104.05965).
Read: architecture, Section 3.6, and Tables 4–5, pages 3–7.

Teachers can use I/Q/A while the student receives I/Q. Their final method
distills task features and predictions from multimodal, image, and question
teachers. It does not reconstruct an answer embedding as a compulsory step.
In their VizWiz experiments, an answer-only teacher is unhelpful; the large
multimodal teacher alone also falls below the no-KD baseline. The chosen
combination of teachers and intermediate features improves performance.
This is multi-label prediction of answer-difference reasons, **not evidence-mask
segmentation**; its numbers must not be read as grounding IoU.

**Implication for us:** access to A does not by itself make its transferred signal
useful or learnable from I/Q. Compare trained recipes and ordinary KD controls;
do not assume full KD beats alignment because it has more losses.

### 3. Modality hallucination: relevant feature-supervision precedent

Hoffman, Gupta and Darrell, *Learning with Side Information through Modality
Hallucination*, CVPR 2016.
[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Hoffman_Learning_With_Side_CVPR_2016_paper.pdf).
Read: Sections 3.1–3.2 and optimization details, pages 2–4.

This RGB detection model uses depth only during training. A hallucination
branch matches intermediate depth features and also receives detection losses.
The depth extractor is frozen up to the feature-matching layer. The RGB,
hallucination, and remaining task layers are jointly optimized after initial
RGB/depth training. This is not a paper demonstrating our one-stage answer
method, and it adds a separate inference branch.

**Implication for us:** distinguish a fixed target extractor from stop-gradient
on a moving extractor, and supervise a representation at the task interface.
Neither their three-branch architecture nor their loss-weight heuristic should
be copied wholesale into this compact model.

### 4. DDVT: same task, with explicit spatial modeling

Zhang et al., *DDVT: Dynamic Dual-level Vision Transformer Fusion Network for
Answer Grounding in Visual Question Answering*, arXiv preprint, July 2026.
[Paper](https://arxiv.org/html/2607.23921v1).
Read: Sections 3–4, including losses and discussion.

DDVT combines region features, question-guided resolution selection, and
pixel/region aggregation. Its supervision includes mask, edge, answer, and
compute-budget terms. It reports examples where localization is correct but
the generated answer is wrong. This is an arXiv preprint in the checked source;
its reported results do not establish that it is the current challenge leader.

**Implication for us:** answer correctness and spatial localization are distinct
axes. A bigger answer branch alone cannot recover small details removed by the
336-pixel input pipeline. Our NaFlex experiments should remain a separate
vision/resolution control, not be mixed into the first latent ablation suite.

### 5. VGOQ: nearby compact grounding architecture, different target task

Everaert et al., *Visual Grounding for Object Questions*, CVPR 2026.
[Proceedings](https://openaccess.thecvf.com/content/CVPR2026/html/Everaert_Visual_Grounding_for_Object_Questions_CVPR_2026_paper.html),
[author project and architecture diagram](https://martin-ev.github.io/vgoq/).
Read: proceedings abstract and author project, including model diagram and
limitations. The main PDF could not be retrieved in this pass; no detailed
training-loss conclusions are drawn from it.

The author diagram uses frozen CLIP encoders and a compact multi-level grounding
transformer with FiLM, segmentation, and relevance heads. Its main new task
targets evidence/context for general object questions, with synthetic data.
VizWiz-VGOQ is not the original answer-grounding benchmark. The project's
small parameter count describes the trained lightweight component; it should
not be confused with the total size of the underlying CLIP towers.

**Implication for us:** compact visual-language grounding is a serious baseline
family. Merely combining frozen encoders, skips, and a small head is not a new
learning principle; a claim should identify which information is transferred
and demonstrate where that transfer improves masks.

### Adjacent control-design evidence

*What Does Privileged Information Add to On-Policy Self-Distillation?*,
arXiv preprint, September 2026.
[Paper](https://arxiv.org/abs/2609.20612).
Read: method, comparisons, and failure analysis in the primary PDF.

This studies LLM mathematics, not vision. It separates the effect of distillation
itself from the incremental contribution of privileged information, finding
benefits that depend on the student and training setup. We take only the
control-design lesson: compare ordinary/self-KD and answer-privileged KD under
matched conditions. Its token-level algorithm and empirical gains do not
transfer directly to dense binary masks.

## Recommended sequence, with decision criteria

### A. Use the existing diagnostic before adding another mechanism

Run `diagnose.py` on the existing checkpoint, over the full validation split.
Inspect the matched answer-available subset for oracle comparisons. Compare:

- oracle A minus same-checkpoint Q-only;
- predicted A' minus same-checkpoint Q-only;
- oracle A minus predicted A';
- high-cosine/poor-mask cases, small-GT cases, and examples where A' hurts.

The Q-only diagnostic is a counterfactual view of the same checkpoint, not a
separately trained Q-only baseline. A weak oracle result can indicate an
ineffective conditioning interface or inadequate learned use of A; it does not
prove answers contain no useful information. In particular, our pooled-answer
oracle is not identical to the baseline's joint-token Q+A interface.

### B. First trained comparison: grounding-only versus alignment

Use identical initialization, seed, segmentation loss, epochs, global batch,
LR, image pipeline and validation protocol. The difference is whether A'
receives answer-alignment supervision. Both still infer A' at deployment.
Start with default weights and no detachment.

Then compare warmup against alignment if the oracle has a useful margin.
Warm-up tests a training schedule, not a fundamentally different inference
model. Include full as the complete-objective comparison, not as a presumed
winner. Do not search every loss weight before this basic comparison is clear.
Use multiple seeds for confirmation after an initial screening; report mask
quality and compute rather than total training loss across recipes.

### C. Preferred next mechanism: task-feature supervision

**Proposal, not implemented or validated.** If the oracle helps while A' has a
large gap, compare matching a grounding feature with matching the raw answer
vector. For example, let F be the existing decoder's final feature before its
mask output. Train with a small, region-balanced feature loss between
F(I,Q,A') and stop-gradient F(I,Q,A), on answer-available examples where the
oracle actually improves the supervised prediction. Normalize channel vectors
and average foreground and background regions separately.

This asks A' to reproduce the *effect on grounding*. Existing mask KD instead
matches one output probability per pixel; it does not supervise the decoder
feature channels. Do not simply stack another term onto full: first **replace**
answer-vector alignment with feature supervision in a matched recipe, then
test whether both are needed. Use ordinary mask KD as an additional control
before attributing any improvement to feature transfer.

Reusing existing decoder features can keep parameter names/counts unchanged;
the oracle forward still adds training compute. Feature alignment can also
overconstrain the student or encourage branch collapse. Check that the oracle
margin and feature variance remain nontrivial. Updating a shared branch with
stop-gradient targets is still not a frozen teacher. No gain is guaranteed.

### D. Only if the vector bottleneck remains: a few evidence tokens

**Architecture proposal, not implemented.** Retain question-token/image
interaction and produce a small token set (for example 4), rather than forcing
all answer-related evidence into one pooled vector. The grounding head can
attend to these tokens directly. Keep the same vision/text towers and compact
decoder, with task-feature supervision rather than requiring exact answer
strings. The current code already has both attention directions; the tested
change would be token capacity and supervision, not merely adding cross-attention.

Compare 1 versus 4 tokens with a matched recipe and report actual parameter,
memory, and speed changes. This alters latent tensor shapes and needs a new
experiment/checkpoint contract. It cannot restore image detail that the
encoder never saw. A larger backbone, OCR system, or extra annotated mask is
not required for this specific hypothesis.

## Keep target stability as a controlled check

Measure answer-target drift over fixed training examples before assuming it
causes failure. If substantial, first compare recipes with the same CLIP text
encoder frozen in both runs. This changes trainable parameters, but avoids a
second tower and stabilizes both Q and A in the same space. Cached A from a
different frozen encoder while Q keeps changing is a different design with a
representation mismatch to account for; it is not automatically a safe fix.
Do not add a new freeze/EMA/cache switch to every recipe without this evidence.

## A smaller user-facing configuration

Only two method choices are needed for the initial comparison:

```bash
--training-recipe grounding-only --loss-version stable
--training-recipe alignment --loss-version stable
```

Optional follow-ups:

```bash
--training-recipe warmup --loss-version stable
--training-recipe full --loss-version stable
```

Warmup defaults to 5 oracle epochs followed by a 5-epoch transition. Full has
no warm-up. These are experiment definitions, not proven optimal schedules.
Use separate output directories. `--help-advanced` exposes loss weights and
schedule overrides when a specific ablation needs them. No extra preset alias,
configuration framework, or automatic loss tuning was added.

The CLI cleanup changes help visibility and examples only. Legacy defaults,
all accepted old flags, resume settings, model weights, diagnostic inference
and native-resolution scoring remain unchanged. New loss/mechanism results
require new training; existing checkpoints remain useful for diagnostic.

Validation of this cleanup: the 12 existing training-recipe CPU checks pass,
including checkpoint loading, resume equivalence, and two-process Gloo valid
target reductions. Manual parser checks confirm identical advanced-option
values in both help modes, and that advanced help exits before device/dataset
checks. This validates CLI compatibility, not a research improvement or a GPU
training run.
