<h1 align="center">Bio-VQA &rarr; Spatial Visual Reasoning &rarr; Viewpoint</h1>
<p align="center">
  <b>Egocentric Orientation Estimation on Wildlife Re-Identification Imagery</b>
</p>
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="Models" src="https://img.shields.io/badge/models-18%20open--weight%20VLMs-orange">
  <img alt="Instances" src="https://img.shields.io/badge/instances-13%2C000%20per%20model-green">
  <img alt="Corpora" src="https://img.shields.io/badge/corpora-4%20wildlife%20datasets-lightgrey">
  <img alt="Headline" src="https://img.shields.io/badge/headline-kappaVAS-yellow">
  <img alt="Rubric" src="https://img.shields.io/badge/scoring-graded%20weighted%20kappa-purple">
  <img alt="Scheduler" src="https://img.shields.io/badge/scheduler-SLURM-red">
  <img alt="Status" src="https://img.shields.io/badge/status-complete-brightgreen">
</p>

> An evaluation of egocentric orientation estimation in vision–language models. Given an
> animal marked with a bounding box, the model reports the direction it faces relative to the
> camera, on the eight-point compass plus up and down. This document describes the task
> construction, the graded rubric and its chance correction, the empirical findings, and the
> exact commands to reproduce them.

---

## Table of contents

1. [Overview](#1-overview)
2. [Motivation](#2-motivation)
   - 2.1 [Orientation as a distinct spatial primitive](#21-orientation-as-a-distinct-spatial-primitive)
   - 2.2 [Why exact match is the wrong instrument](#22-why-exact-match-is-the-wrong-instrument)
   - 2.3 [Why a graded score alone is not enough](#23-why-a-graded-score-alone-is-not-enough)
3. [System architecture](#3-system-architecture)
   - 3.1 [Pipeline stages and dependency order](#31-pipeline-stages-and-dependency-order)
   - 3.2 [Repository layout](#32-repository-layout)
4. [The evaluation set](#4-the-evaluation-set)
   - 4.1 [The three tiers](#41-the-three-tiers)
   - 4.2 [Source corpora and the label skew](#42-source-corpora-and-the-label-skew)
   - 4.3 [Realised scale](#43-realised-scale)
5. [The scoring calculus](#5-the-scoring-calculus)
   - 5.1 [The graded rubric](#51-the-graded-rubric)
   - 5.2 [Chance correction](#52-chance-correction)
   - 5.3 [The independent angular metric](#53-the-independent-angular-metric)
6. [Canvas geometry and the perceptual budget](#6-canvas-geometry-and-the-perceptual-budget)
7. [Results](#7-results)
   - 7.1 [The leaderboard](#71-the-leaderboard)
   - 7.2 [Only one model clears the majority-class baseline](#72-only-one-model-clears-the-majority-class-baseline)
   - 7.3 [Chance correction reorders the leaderboard](#73-chance-correction-reorders-the-leaderboard)
   - 7.4 [The rubric weights do not drive the ranking](#74-the-rubric-weights-do-not-drive-the-ranking)
   - 7.5 [Convergence with the angular metric](#75-convergence-with-the-angular-metric)
   - 7.6 [The tier ladder](#76-the-tier-ladder)
   - 7.7 [Scale](#77-scale)
8. [Scope of the released evaluation](#8-scope-of-the-released-evaluation)
9. [Reproduction](#9-reproduction)
10. [Outputs](#10-outputs)
11. [References](#11-references)

---

## 1. Overview

We evaluate whether vision–language models can estimate **egocentric orientation**: the
direction an animal faces relative to the camera. Each item presents a photograph in which
one animal is delimited by a red bounding box; the model answers with one of ten viewpoint
tokens — the eight-point horizontal compass (`front`, `frontright`, `right`, `backright`,
`back`, `backleft`, `left`, `frontleft`) plus `up` and `down`. We pose 13,000 items across
four wildlife re-identification corpora and three panel conditions, and evaluate **eighteen
open-weight vision–language models**.

The ground truth is not ours. Every label is a Wildbook viewpoint annotation recorded by the
source corpus for re-identification purposes, which makes the task both operationally
meaningful — viewpoint filtering is a routine step in wildlife re-ID pipelines — and free of
any annotation we introduced.

Two properties of the label space determine the entire scoring design. The vocabulary is
**compositional**: `frontright` decomposes into `{front, right}`, so answers stand in
subset, superset, sibling, or disjoint relations to the truth, and treating all errors as
equivalent discards most of the signal
([Section 5.1](#51-the-graded-rubric)). And the label distribution is **severely skewed**:
`left` and `right` together are roughly 83% of rows, and 89% of Grévy's zebra annotations are
`right` because the census protocol photographs the right flank
([Section 4.2](#42-source-corpora-and-the-label-skew)).

Three findings organise the component. Only **one of eighteen models** exceeds the
majority-class baseline on exact match
([Section 7.2](#72-only-one-model-clears-the-majority-class-baseline)). Chance-correcting the
graded score reorders the leaderboard substantially, moving individual models by up to
**8.5 rank positions** ([Section 7.3](#73-chance-correction-reorders-the-leaderboard)). And
the ranking is nearly invariant to the rubric's credit weights — Spearman **0.979** between
the default rubric and an exact-match-only weighting
([Section 7.4](#74-the-rubric-weights-do-not-drive-the-ranking)) — so the graded scheme
sharpens the measurement without determining its conclusions.

---

## 2. Motivation

### 2.1 Orientation as a distinct spatial primitive

Counting asks *how many*, morphometry asks *how large*. Viewpoint asks *which way*, and it is
the only task in Component C3 whose answer is a property of the **relationship between the
subject and the camera** rather than of the subject alone. It cannot be answered from a
crop's content in isolation: the same animal photographed from two positions yields two
different correct answers with identical pixels-per-animal, identical species, and identical
pose in body-centred coordinates.

The task also carries direct operational weight. Wildlife re-identification systems filter
candidate matches by viewpoint before comparing appearance, because a left-flank photograph
and a right-flank photograph of the same individual share almost no directly comparable
surface. A model that cannot recover viewpoint cannot participate in that pipeline, which
connects this component to the individual re-identification component of Bio-VQA.

### 2.2 Why exact match is the wrong instrument

The label vocabulary is compositional, and its elements sit on a ring at 45° intervals.
Consider a gold label of `right` and three predictions:

| Prediction | Angular error | Relation to gold | Exact match |
|---|---:|---|---:|
| `backright` | 45° | superset — added a qualifier | 0 |
| `frontleft` | 135° | disjoint | 0 |
| `back` | 90° | disjoint | 0 |

Exact match assigns all three the same zero, while the first answer is nearly correct and
committed to a finer distinction than the annotation records. On a compositional ring this
is not a rounding concern but the dominant structure of the error distribution: across our
eighteen models, `over` and `under` relations account for a mean of **17.4%** of all rows and
reach **62.9%** for two models
([Section 7.3](#73-chance-correction-reorders-the-leaderboard)).

We therefore grade the relation between the gold and predicted component sets
([Section 5.1](#51-the-graded-rubric)) and retain exact match as a reference column.

### 2.3 Why a graded score alone is not enough

A graded rubric introduces its own exploit. Because 83% of gold labels are bare cardinals
(`left`, `right`), a model that answers **compound labels indiscriminately** collects `over`
credit on the great majority of rows without perceiving anything: every `backright` answer
against a `right` gold earns partial credit, and the credit is genuine on each individual
row.

Row-averaged graded scores reward this. We therefore lead with the **chance-corrected**
graded score — Cohen's weighted kappa with the rubric as its weight matrix — which subtracts
the score the same answer distribution would earn by chance and cannot be raised by
answering compounds indiscriminately
([Section 5.2](#52-chance-correction)). Verified against synthetic strategies, always-`right`
scores 0.000 and indiscriminate-compounds scores 0.001, while a model with 30% one-step noise
scores 0.773.

The correction is not cosmetic. It moves two models by more than seven rank positions
([Section 7.3](#73-chance-correction-reorders-the-leaderboard)).

---

## 3. System architecture

### 3.1 Pipeline stages and dependency order

```
        ┌──────────────────────────────────────────────┐
        │  Four wildlife re-ID corpora                 │
        │  GrevysZebra (CSV) + Hyena / Leopard /       │
        │  Whaleshark (COCO)                           │
        │  each with bbox + Wildbook viewpoint token   │
        └──────────────────┬───────────────────────────┘
                           │
                           ▼
   (1)  1.Generate_QA/generateQA_easy.py     ──►  Easy/   one panel  + Images/
        1.Generate_QA/generateQA_medium.py   ──►  Medium/ three panels, cross-corpus
        1.Generate_QA/generateQA_hard.py     ──►  Hard/   three panels, within-corpus
                           │
                           ▼
   (2)  2.Experiment_Runs/Preprocess/split_qa.py
                           │   merge tiers, resolve paths, interleave
                           ▼
                       chunks/chunk00..05/qa.csv
                           │
                           ├──►  stats.py   label inventory, skew, majority floor
                           ▼
   (3)  2.Experiment_Runs/<model>.py  x 18
                           │   closed label set in the prompt, invariant harness
                           ▼
                       chunks/chunkNN/qa_<KEY>.csv  (+ summary.json)
                           │
                           ▼  merge_results.py  ->  results/qa_<KEY>.csv
                           ▼
   (4)  3.Evaluate/evaluate.py  ──►  viewpoint_overall.csv
                                     viewpoint_by_group.csv
                                     viewpoint_by_relation.csv
                                     viewpoint_report.md
```

Stage 1's ordering is a hard dependency: both multi-panel generators read the **Easy tier's
own output**, not the source corpora, so the single-panel tier must be built first and the
three tiers share one pool of bounding-box renders
([Section 3 of the generation document](1.Generate_QA/README.md#3-system-architecture)).

Stage 3 requires the label vocabulary to be known in advance, because the accepted answers
are enumerated in the prompt. Stage 2's audit is what establishes it
([Section 5 of the inference document](2.Experiment_Runs/README.md#5-pre-run-audit)).

### 3.2 Repository layout

```
Viewpoint/
├── README.md                      this document
│
├── 1.Generate_QA/
│   ├── README.md
│   ├── generateQA_easy.py         Easy    — bbox rendering from four corpora
│   ├── generateQA_medium.py       Medium  — cross-corpus triplet concatenation
│   └── generateQA_hard.py         Hard    — within-corpus triplet concatenation
│
├── 2.Experiment_Runs/
│   ├── README.md
│   ├── Preprocess/
│   │   ├── split_qa.py            merge tiers, resolve paths, shard
│   │   └── stats.py               label inventory + majority floor
│   ├── merge_results.py           reassemble shards
│   └── <model>.py  × 18           inference runners, invariant harness
│
└── 3.Evaluate/
    ├── README.md
    ├── merge_results.py
    └── evaluate.py                rubric, chance correction, angular metric
```

| # | Stage | Document |
|---|---|---|
| 1 | Render boxes and construct the three tiers | [`1.Generate_QA/README.md`](1.Generate_QA/README.md) |
| 2–3 | Consolidate, shard, and run inference across 18 models | [`2.Experiment_Runs/README.md`](2.Experiment_Runs/README.md) |
| 4 | Merge, grade, chance-correct, and report | [`3.Evaluate/README.md`](3.Evaluate/README.md) |

---

## 4. The evaluation set

### 4.1 The three tiers

Three panel conditions, carried on disk as `Easy/`, `Medium/`, and `Hard/` and labelled
`single`, `cued`, and `positional` in every downstream artefact. The labels name the mechanism
by which the target panel is identified: in Medium the question names a species present in
only one panel, so the species *cues* the panel; in Hard all three panels show the same
species, so the panel must be resolved *positionally* from `left`, `middle`, or `right`.
Beyond this definition we refer to the tiers throughout as Easy, Medium, and Hard.

| Tier | Panels | Composition | Target identified by |
|---|:---:|---|---|
| **Easy** | 1 | one photograph | no panel to identify |
| **Medium** | 3 | three **different** corpora | species named in the question |
| **Hard** | 3 | three photographs from **one** corpus | position: `left` / `middle` / `right` |

The Hard tier's positional vocabulary is **three-valued**, not two. `left` and `right` are
anchored by the canvas boundary; `middle` is defined only relative to the other two and has
no boundary anchor of its own. This is a materially harder indexing problem than the binary
case, and it is the mechanism we invoke for the tier ordering observed in
[Section 7.6](#76-the-tier-ladder).

### 4.2 Source corpora and the label skew

Four wildlife re-identification corpora, chosen because each ships per-animal bounding boxes
together with a Wildbook viewpoint annotation.

| Corpus | Taxon | Format | Note |
|---|---|---|---|
| `GrevysZebra` | *Equus grevyi* | CSV with `bbox x/y/w/h` and `viewpoint` | Census protocol; **89% of annotations are `right`** |
| `hyena` | *Crocuta crocuta* | COCO | |
| `leopard` | *Panthera pardus* | COCO | |
| `whaleshark` | *Rhincodon typus* | COCO | Marine; `up` and `down` occur here |

**The label skew is a property of the science, not of our sampling.** Wildbook's Grévy's
zebra census photographs the right flank because the right side carries the identifying
stripe pattern used for individual matching; the annotation therefore records `right` on
almost every image. Pooled across corpora, `left` and `right` together account for roughly
83% of rows, and the single majority label reaches **0.520**.

Two consequences follow, and both are handled in the scoring rather than by resampling. A
row-averaged score rewards a model that always answers `right`, so we macro-average the
graded score over gold labels and additionally chance-correct it
([Section 5.2](#52-chance-correction)). And exact match must be read against the majority
share of 0.520 rather than against zero
([Section 7.2](#72-only-one-model-clears-the-majority-class-baseline)).

We do not resample to a uniform label distribution. Doing so would discard the majority of a
corpus whose skew is exactly the operational condition under which viewpoint filtering is
applied in practice, and would replace an honest skew with an artificial one.

### 4.3 Realised scale

| Quantity | Value |
|---|---:|
| Scored items per model | 13,000 |
| Easy tier | 4,000 |
| Medium tier | 6,000 |
| Hard tier | 3,000 |
| Corpora | 4 |
| Majority-label share | 0.520 |
| Parse rate | 1.000 for all 18 models |

Every item is scored for every model: the closed label set supplied in the prompt
([Section 6.2 of the inference document](2.Experiment_Runs/README.md#62-prompt-construction))
yields a parse rate of exactly 1.000 across all eighteen, so no model is evaluated on a
different row subset from any other.

---

## 5. The scoring calculus

### 5.1 The graded rubric

Every accepted label decomposes into axis components drawn from
$\{\text{front}, \text{back}, \text{left}, \text{right}, \text{up}, \text{down}\}$:

$$
\mathrm{comp}(\texttt{frontleft}) = \{\text{front}, \text{left}\}, \qquad
\mathrm{comp}(\texttt{right}) = \{\text{right}\} .
$$

The relation between the gold set $G$ and the predicted set $P$ determines the credit:

| Relation | Condition | Example (gold → pred) | Credit |
|---|---|---|---:|
| **exact** | $G = P$ | `backright` → `backright` | 1.00 |
| **over** | $G \subsetneq P$ | `right` → `backright` | 0.50 |
| **under** | $P \subsetneq G$ | `backright` → `right` | 0.35 |
| **sibling** | $G \cap P \neq \emptyset$, neither contains the other | `frontright` → `backright` | 0.15 |
| **wrong** | $G \cap P = \emptyset$ | `right` → `frontleft` | 0.00 |

**`over` outranks `under` deliberately.** A model answering `backright` when the annotation
says `right` attempted the finer distinction and got the coarse component correct; one
answering `right` when the annotation says `backright` declined the distinction. Committing
and being partly right is a different act from abstaining, and the rubric records the
difference.

The asymmetry is a judgement, so we do three things rather than assert it. All four credits
are command-line arguments; the rubric is written into the report and into
`viewpoint_overall.csv` so a result can never be quoted without it; and a fully symmetric
angular metric is reported alongside
([Section 5.3](#53-the-independent-angular-metric)). The ranking turns out to be nearly
invariant to the weights in any case
([Section 7.4](#74-the-rubric-weights-do-not-drive-the-ranking)).

We report the graded score macro-averaged over gold labels — **macroVAS** — so that the 83%
skew cannot carry a constant predictor
([Section 4.2](#42-source-corpora-and-the-label-skew)).

### 5.2 Chance correction

The headline metric is **kappaVAS**: Cohen's weighted kappa with the rubric as its weight
matrix.

$$
\boxed{\;\mathrm{kappaVAS} = \frac{\mathrm{VAS}_{\text{obs}} - \mathrm{VAS}_{\text{exp}}}{1 - \mathrm{VAS}_{\text{exp}}}\;}
$$

$$
\mathrm{VAS}_{\text{obs}} = \frac{1}{n}\sum_i v(y_i, \hat{y}_i),
\qquad
\mathrm{VAS}_{\text{exp}} = \sum_{a}\sum_{b} \Pr[y = a]\,\Pr[\hat{y} = b]\; v(a, b) ,
$$

where $v$ is the rubric credit. $\mathrm{VAS}_{\text{exp}}$ is the score the model would earn
if its own answers were shuffled against the gold labels — the score attributable to the two
marginal distributions alone.

This is what macroVAS cannot do. A model answering compounds indiscriminately collects `over`
credit on the 83% of rows whose gold is a bare cardinal, and macroVAS rewards it because the
credit is real on every individual row. Subtracting the expected score removes exactly that
component, because a shuffled version of the same answer distribution collects the same
`over` credit.

Verified against synthetic strategies on the actual gold distribution:

| Strategy | kappaVAS |
|---|---:|
| always answer `right` | 0.000 |
| answer compounds indiscriminately | 0.001 |
| genuine perception with 30% one-step noise | **0.773** |

### 5.3 The independent angular metric

The eight horizontal labels lie on a ring at 45° intervals, so a fully symmetric geometric
error is available:

$$
\mathrm{angErr}(g, p) = \min\bigl(|\theta_g - \theta_p| \bmod 360,\; 360 - (|\theta_g - \theta_p| \bmod 360)\bigr),
$$

reported as a mean in degrees together with `within45`, the share of predictions within one
ring step.

This metric shares **no design decision** with the rubric: it has no asymmetry, no credit
weights, no subset relation, and no chance correction. Its agreement with kappaVAS is
therefore an independent check on the rubric rather than a restatement of it, and the
agreement is very strong
([Section 7.5](#75-convergence-with-the-angular-metric)).

---

## 6. Canvas geometry and the perceptual budget

The multi-panel tiers concatenate three photographs horizontally after matching heights to
the shortest. Writing $\alpha_i = w_i / h_i$ for each source image's aspect ratio, the
fraction of canvas area occupied by panel $i$ is

$$
\boxed{\;f_i = \frac{\alpha_i}{\sum_{j=1}^{3} \alpha_j}\;}
$$

depending only on the ratios of aspect ratios, not on absolute resolution. Under a fixed
encoder budget $B$, panel $i$ renders at effective linear scale $s_i = \sqrt{f_i}$ relative
to the same image shown alone.

For a balanced three-panel canvas,

$$
s = \sqrt{\tfrac{1}{3}} \approx 0.577 ,
$$

so each animal subtends roughly 58% of its single-panel linear extent — a larger reduction
than the two-panel case in [Counting](../Counting/README.md#52-effective-linear-resolution),
where $s \approx 0.707$. This is the first-order cost of the multi-panel tiers and it applies
to Medium and Hard alike.

The two tiers differ in the *spread* of $f_i$ rather than its mean. Hard draws all three
panels from one corpus, which shares a capture protocol, so $\alpha_1 \approx \alpha_2
\approx \alpha_3$ and the partition is near-even. Medium draws from three different corpora —
a drone transect, a camera trap, and an underwater photograph have systematically different
aspect ratios — so one panel can receive a substantially larger share than the others.

---

## 7. Results

Full tables are written to `3.Evaluate/results/viewpoint_report.md`.

### 7.1 The leaderboard

Sorted by kappaVAS. Majority label share is 0.520 and `modalVAS` — the graded score of a
constant `right` predictor — is 0.534 for every row.

| Model | kappaVAS | macroVAS | VAS | EM | balAcc | angErr | within45 | predTopShare |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3-VL-32B** | **0.358** | 0.427 | 0.554 | 0.417 | 0.286 | 50.9° | 0.711 | 0.309 |
| **Molmo2-8B** | **0.358** | **0.483** | 0.547 | 0.228 | **0.321** | **44.4°** | **0.880** | 0.309 |
| Gemma-4-12B | 0.316 | 0.331 | **0.633** | **0.592** | 0.228 | 51.1° | 0.699 | 0.754 |
| Qwen3-VL-8B | 0.310 | 0.373 | 0.499 | 0.425 | 0.241 | 51.2° | 0.595 | 0.276 |
| Qwen3-VL-30B-A3B | 0.278 | 0.391 | 0.475 | 0.370 | 0.255 | 64.7° | 0.595 | 0.312 |
| InternVL3.5-38B-Instruct | 0.208 | 0.340 | 0.525 | 0.483 | 0.224 | 63.0° | 0.591 | 0.641 |
| InternVL3-38B-Instruct | 0.182 | 0.373 | 0.368 | 0.267 | 0.261 | 66.0° | 0.483 | 0.284 |
| Qwen2.5-VL-7B | 0.182 | 0.395 | 0.369 | 0.050 | 0.229 | 67.8° | 0.697 | 0.356 |
| InternVL3.5-30B-A3B | 0.175 | 0.309 | 0.494 | 0.439 | 0.201 | 71.9° | 0.584 | 0.423 |
| InternVL3.5-8B-Instruct | 0.117 | 0.264 | 0.496 | 0.456 | 0.161 | 67.6° | 0.558 | 0.689 |
| Phi-3.5-Vision | 0.083 | 0.221 | 0.473 | 0.456 | 0.127 | 74.8° | 0.513 | 0.631 |
| InternVL3-8B-Instruct | 0.065 | 0.335 | 0.230 | 0.066 | 0.188 | 89.5° | 0.401 | 0.482 |
| Pixtral-12B | 0.055 | 0.246 | 0.374 | 0.227 | 0.135 | 71.9° | 0.539 | 0.478 |
| Llama-3.2-11B-Vision | 0.037 | 0.250 | 0.091 | 0.078 | 0.146 | 83.8° | 0.128 | 0.563 |
| Gemma-3-12B | 0.026 | 0.235 | 0.198 | 0.104 | 0.124 | 83.5° | 0.304 | 0.483 |
| Qwen2.5-VL-3B | 0.006 | 0.230 | 0.130 | 0.027 | 0.110 | 95.0° | 0.240 | 0.530 |
| LLaVA-NeXT-Mistral-7B | **−0.005** | 0.173 | 0.050 | 0.037 | 0.103 | 93.4° | 0.071 | **0.938** |
| LLaVA-NeXT-Vicuna-7B | **−0.019** | 0.187 | 0.097 | 0.030 | 0.092 | 97.3° | 0.170 | 0.394 |

Two models score **at or below zero** on the chance-corrected metric — LLaVA-NeXT-Mistral-7B
answers `front` on 93.8% of items — and the best of eighteen reaches 0.358. Set against the
0.773 that a synthetic model with 30% one-step noise attains on this same distribution
([Section 5.2](#52-chance-correction)), the ceiling here is low in an absolute and
independently calibrated sense.

### 7.2 Only one model clears the majority-class baseline

$$
\text{majority-label share} = 0.520; \qquad \mathrm{EM} > 0.520 \text{ for } \mathbf{1/18} \text{ models.}
$$

$$
\mathrm{modalVAS} = 0.534; \qquad \mathrm{VAS} > 0.534 \text{ for } \mathbf{3/18} \text{ models.}
$$

Only Gemma-4-12B exceeds the exact-match score of a predictor that always answers the
majority label — and it does so while answering `right` on 75.4% of items, which is itself
close to the constant strategy. Seventeen of eighteen models score below a predictor that
never looks at the image.

This is why neither exact match nor the row-averaged graded score can be the headline. Under
macro-averaging the picture separates properly: **18 of 18 models exceed `modalMacroVAS` of
0.170**, because macro-averaging removes the skew that the constant predictor exploits. The
gap between those two counts — 1 of 18 against 18 of 18 — is the quantitative case for
macro-averaging on a corpus with an 83% two-label concentration.

### 7.3 Chance correction reorders the leaderboard

Chance-correcting the graded score is not a cosmetic adjustment. Ranking by exact match and
ranking by kappaVAS give substantially different orders:

$$
\rho\!\left(\mathrm{EM},\, \mathrm{kappaVAS}\right) = 0.656 .
$$

The largest movements, and their cause:

| Model | EM | EM rank | kappaVAS | kappaVAS rank | Shift | `over` share |
|---|---:|---:|---:|---:|---:|---:|
| Molmo2-8B | 0.228 | 10 | 0.358 | 1.5 | **+8.5** | **0.629** |
| Qwen2.5-VL-7B | 0.050 | 15 | 0.182 | 7.5 | **+7.5** | **0.629** |
| Qwen3-VL-32B | 0.417 | 7 | 0.358 | 1.5 | +5.5 | 0.259 |
| Phi-3.5-Vision | 0.456 | 3.5 | 0.083 | 11 | **−7.5** | 0.007 |
| InternVL3.5-8B-Instruct | 0.456 | 3.5 | 0.117 | 10 | **−6.5** | 0.051 |
| InternVL3.5-38B-Instruct | 0.483 | 2 | 0.208 | 6 | −4.0 | 0.056 |

The pattern is legible directly from the error composition. Molmo2-8B and Qwen2.5-VL-7B both
place 62.9% of their answers in the `over` relation — they consistently name a compound where
the annotation records a cardinal — and exact match scores every one of those as zero.
Phi-3.5-Vision and InternVL3.5-8B do the reverse, placing under 6% in `over` while posting
high exact match, largely by answering `right` on a corpus where `right` is usually correct.

**Exact match and kappaVAS are measuring different things, and the rubric plus chance
correction is what separates them.** A model whose errors are one ring step away from the
truth is closer to the answer than one that is right when the majority label is right and
wrong otherwise, and only the graded chance-corrected metric records that.

### 7.4 The rubric weights do not drive the ranking

The `over`/`under` asymmetry is a judgement
([Section 5.1](#51-the-graded-rubric)), so we test whether the conclusions depend on it. We
rescore all eighteen models under four credit settings spanning the plausible range, from the
default down to a weighting that gives partial relations **zero** credit — i.e. exact match
only:

| Setting | over | under | Interpretation |
|---|---:|---:|---|
| default | 0.50 | 0.35 | as reported |
| conservative | 0.35 | 0.25 | partial credit halved |
| minimal | 0.25 | 0.15 | partial credit quartered |
| exact-only | 0.00 | 0.00 | no partial credit at all |

$$
\text{maximum rank change across all four settings} = \mathbf{2 \text{ positions of } 18};
\qquad
\text{mean} = 0.83 .
$$

$$
\rho\!\left(\text{default},\, \text{exact-only}\right) = \mathbf{0.979};
\qquad
\text{all pairwise } \rho \ge 0.979 .
$$

Seven of eighteen models do not move at all, and no model moves by more than two positions
even when partial credit is removed entirely. **The rubric sharpens the measurement; it does
not determine the ranking.** A reader who rejects the `over` > `under` asymmetry outright,
and prefers exact-match weighting, obtains essentially the same leaderboard.

This is the sensitivity analysis that a graded scoring scheme owes its reader, and we report
it in full rather than asserting robustness.

### 7.5 Convergence with the angular metric

The angular metric shares no design decision with the rubric — no asymmetry, no credit
weights, no subset relation, no chance correction
([Section 5.3](#53-the-independent-angular-metric)). Its agreement with kappaVAS is
therefore an independent validation:

$$
\rho\!\left(\mathrm{angErr},\, \mathrm{kappaVAS}\right) = \mathbf{-0.961},
\qquad
\rho\!\left(\mathrm{within45},\, \mathrm{kappaVAS}\right) = \mathbf{+0.933} .
$$

Two constructions built on entirely different principles — a set-relational credit table with
a deliberate asymmetry, and a symmetric geometric distance on a ring — order eighteen models
almost identically. Taken with the rubric-invariance result of
[Section 7.4](#74-the-rubric-weights-do-not-drive-the-ranking), the ranking reported here
does not rest on the scoring choices this component makes.

Molmo2-8B is the clearest illustration. It posts the best angular error in the set at 44.4°
and the best `within45` at 0.880 — meaning 88% of its answers land within one ring step —
while its exact match is only 0.228. Its errors are consistently near-misses, and both the
graded metric and the independent angular metric rank it first.

### 7.6 The tier ladder

Per-tier kappaVAS across all eighteen models:

$$
\Delta_{\text{Easy} \to \text{Medium}} = +0.062 \;(\sigma = 0.072), \qquad \text{positive for } \mathbf{16/18},
$$
$$
\Delta_{\text{Medium} \to \text{Hard}} = +0.029 \;(\sigma = 0.062), \qquad \text{positive for } \mathbf{13/18} .
$$

The Easy-to-multi-panel drop is large and nearly universal, and it is the first-order
consequence of the area partition in
[Section 6](#6-canvas-geometry-and-the-perceptual-budget): three panels on one canvas render
each animal at $\sqrt{1/3} \approx 0.577$ of its single-panel linear extent. The independent
angular metric confirms it on its own terms — Easy has lower angular error than Medium for
**16 of 18** models, by a mean of 8.0°.

The Medium-to-Hard contrast is smaller but consistent in sign for 13 of 18 models, and it has
a specific mechanism. Hard's positional index is **three-valued**, and `middle` has no
boundary anchor: `left` and `right` are defined by the canvas edges, but `middle` is defined
only relative to the other two panels
([Section 4.1](#41-the-three-tiers)). Medium avoids the problem entirely because the species
name identifies the panel and no positional term is load-bearing.

The ordering Easy > Medium > Hard therefore reflects two separable costs: a first-order
perceptual-budget cost shared by both multi-panel tiers, and a second-order indexing cost
paid only by Hard, whose magnitude tracks the size of the positional vocabulary.

### 7.7 Scale

$$
\rho\!\left(\text{parameters},\, \mathrm{kappaVAS}\right) = 0.491,
\qquad
\rho\!\left(\text{parameters},\, \mathrm{EM}\right) = 0.512 .
$$

The weakest scale relationship of any task in Component C3. An 8B Molmo2 ties a 32B Qwen3-VL
for the lead and beats it on every angular measure; a 12B Gemma-4 places third; a 38B
InternVL3 places seventh, below an 8B Qwen3-VL. Both 38B InternVL variants sit in the lower
half of the top ten.

Orientation is not a capability that current scaling delivers. Unlike counting and
morphometry, where the strongest models cluster at the top and the correlation with parameter
count reaches 0.62–0.81, viewpoint separates models by something other than size — which
makes it the component's clearest instance of the **granularity collapse** thesis:
a fine-grained spatial primitive on which scale does not help.

---

## 8. Scope of the released evaluation

The scoring rests on two independent constructions that agree at $\rho = 0.96$
([Section 7.5](#75-convergence-with-the-angular-metric)) and on a ranking that survives the
removal of partial credit entirely
([Section 7.4](#74-the-rubric-weights-do-not-drive-the-ranking)). Two extensions would
sharpen the mechanism further.

A **resolution-matched control** — a single-panel condition downsampled to the per-panel
pixel count of a three-panel canvas — would convert the geometric account in
[Section 6](#6-canvas-geometry-and-the-perceptual-budget) from a prediction confirmed across
eighteen independent measurements into a direct measurement.

A **panel-position permutation** would isolate the `middle`-term cost proposed in
[Section 7.6](#76-the-tier-ladder). Re-rendering each Hard triplet with the three panels
permuted, while holding their contents fixed, gives a per-item estimate of the cost of the
unanchored middle position; the Hard generator is seeded, so the permuted set reproduces
exactly.

---

## 9. Reproduction

```bash
# 1. Build the single-panel tier first — the other two read its output.
python 1.Generate_QA/generateQA_easy.py \
    --grevys_csv  $BIOVQA_DATA/GrevysZebra/annotations.csv \
    --grevys_images $BIOVQA_DATA/GrevysZebra/Images \
    --hyena_json      $BIOVQA_DATA/HyenaID/instances.json \
    --hyena_images    $BIOVQA_DATA/HyenaID/Images \
    --leopard_json    $BIOVQA_DATA/LeopardID/instances.json \
    --leopard_images  $BIOVQA_DATA/LeopardID/Images \
    --whaleshark_json   $BIOVQA_DATA/WhalesharkID/instances.json \
    --whaleshark_images $BIOVQA_DATA/WhalesharkID/Images \
    --output_dir  $BIOVQA_DATA/tasks/Viewpoint/Easy \
    --num_samples 1000

python 1.Generate_QA/generateQA_medium.py --base_dir $BIOVQA_DATA/tasks
python 1.Generate_QA/generateQA_hard.py   --base_dir $BIOVQA_DATA/tasks

# 2. Consolidate the three tiers and shard.
python 2.Experiment_Runs/Preprocess/split_qa.py \
    --task_root $BIOVQA_DATA/tasks/Viewpoint --run_dir 2.Experiment_Runs --n_chunks 6

# 3. Label inventory and majority floor — sets VIEWPOINT_LABELS for stage 4.
python 2.Experiment_Runs/Preprocess/stats.py --run_dir 2.Experiment_Runs

# 4. Inference — 18 models x 6 shards.
python 2.Experiment_Runs/qwenVL2_5_7B.py --qa_csv 2.Experiment_Runs/chunks/chunk00/qa.csv

# 5. Merge, then score.
python 3.Evaluate/merge_results.py --run_dir 2.Experiment_Runs --out_dir 3.Evaluate/results
python 3.Evaluate/evaluate.py \
    --results_dir 3.Evaluate/results --out_dir 3.Evaluate/results --bootstrap 2000
```

Step 4 comprises 108 independent jobs, executed as SLURM arrays on an H100 partition.

---

## 10. Outputs

| File | Contents |
|---|---|
| `3.Evaluate/results/viewpoint_overall.csv` | One row per model: `kappaVAS`, `macroVAS`, `VAS`, `EM`, `balAcc`, `kappa`, `angErr`, `within45`, `majority`, `modalVAS`, `modalMacroVAS`, `predTop`, `predTopShare`, plus the rubric that produced them. |
| `3.Evaluate/results/viewpoint_by_group.csv` | One row per model × tier × corpus, plus per-tier macro rows. |
| `3.Evaluate/results/viewpoint_by_relation.csv` | Share of rows in each rubric class — the error composition of [Section 7.3](#73-chance-correction-reorders-the-leaderboard). |
| `3.Evaluate/results/viewpoint_report.md` | Leaderboard, error composition, per-tier table, and interpretation. |
| `merge_manifest.json` | What was merged and whether any model is incomplete. |

---

## 11. References

Chance-corrected agreement follows Cohen [1]; the weighted extension used for the graded
rubric follows Cohen [2]. Clustered resampling for dependent observations follows standard
bootstrap practice [3]. Viewpoint annotations originate from the Wildbook platform [4].

[1] J. Cohen. **A Coefficient of Agreement for Nominal Scales.** *Educational and
Psychological Measurement*, 20(1):37–46, 1960.

[2] J. Cohen. **Weighted Kappa: Nominal Scale Agreement with Provision for Scaled Disagreement
or Partial Credit.** *Psychological Bulletin*, 70(4):213–220, 1968.

[3] B. Efron and R. J. Tibshirani. **An Introduction to the Bootstrap.** Chapman & Hall, 1993.

[4] Wild Me. **Wildbook: A Platform for Collaborative Wildlife Research and Conservation.**
https://www.wildme.org