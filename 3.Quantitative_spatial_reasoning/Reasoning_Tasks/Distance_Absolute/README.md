<h1 align="center">Bio-VQA &rarr; Spatial Visual Reasoning &rarr; Absolute Distance Measurement</h1>
<p align="center">
  <b>Scale-Bar Referenced Morphometry of Pinned Beetle Specimens</b>
</p>
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="Models" src="https://img.shields.io/badge/models-18%20open--weight%20VLMs-orange">
  <img alt="Instances" src="https://img.shields.io/badge/instances-14%2C108%20per%20model-green">
  <img alt="Corpus" src="https://img.shields.io/badge/corpus-NEON%20Beetles-lightgrey">
  <img alt="Headline" src="https://img.shields.io/badge/headline-rho__within-yellow">
  <img alt="Cluster" src="https://img.shields.io/badge/bootstrap-864%20plates-teal">
  <img alt="Scheduler" src="https://img.shields.io/badge/scheduler-SLURM-red">
  <img alt="Status" src="https://img.shields.io/badge/status-complete-brightgreen">
</p>

> An evaluation of metric measurement in vision–language models. Given a specimen plate
> bearing a 1 cm scale bar and a marked morphological feature, the model must report the
> feature's length in centimetres. This document describes the task construction, the
> three-part shortcut analysis that makes the result interpretable, the empirical findings,
> and the exact commands to reproduce them.

---

## Table of contents

1. [Overview](#1-overview)
2. [Motivation](#2-motivation)
   - 2.1 [Metric measurement as a compositional visual operation](#21-metric-measurement-as-a-compositional-visual-operation)
   - 2.2 [Bounding the prior-driven shortcut](#22-bounding-the-prior-driven-shortcut)
   - 2.3 [Ranking and calibration as separable abilities](#23-ranking-and-calibration-as-separable-abilities)
3. [System architecture](#3-system-architecture)
   - 3.1 [Pipeline stages and dependency order](#31-pipeline-stages-and-dependency-order)
   - 3.2 [Repository layout](#32-repository-layout)
4. [The evaluation set](#4-the-evaluation-set)
   - 4.1 [Task definition and annotation geometry](#41-task-definition-and-annotation-geometry)
   - 4.2 [Source corpus and specimen selection](#42-source-corpus-and-specimen-selection)
   - 4.3 [Scale and cluster structure](#43-scale-and-cluster-structure)
5. [The shortcut analysis](#5-the-shortcut-analysis)
   - 5.1 [The per-type constant predictor](#51-the-per-type-constant-predictor)
   - 5.2 [Why pooled rank correlation is inadmissible](#52-why-pooled-rank-correlation-is-inadmissible)
   - 5.3 [Range compression and its closed-form anchor](#53-range-compression-and-its-closed-form-anchor)
6. [Results](#6-results)
   - 6.1 [The leaderboard](#61-the-leaderboard)
   - 6.2 [The constant predictor outscores fifteen of eighteen models](#62-the-constant-predictor-outscores-fifteen-of-eighteen-models)
   - 6.3 [Range compression is near-universal](#63-range-compression-is-near-universal)
   - 6.4 [Ranking and calibration dissociate](#64-ranking-and-calibration-dissociate)
   - 6.5 [The width asymmetry](#65-the-width-asymmetry)
   - 6.6 [Scale and resolution](#66-scale-and-resolution)
7. [Scope of the released evaluation](#7-scope-of-the-released-evaluation)
8. [Reproduction](#8-reproduction)
9. [Outputs](#9-outputs)
10. [References](#10-references)

---

## 1. Overview

We evaluate whether vision–language models can perform **metric measurement**: converting an
apparent distance in an image into a physical length by reference to a scale bar. The task
is drawn from entomological morphometry, where elytra length and width are standard
diagnostic characters and where every specimen plate is photographed with a scale bar for
exactly this purpose.

Each item presents a plate of pinned beetle specimens on which we draw two annotated line
segments: a **blue** line marking the 1 cm scale bar, and a **red** line marking the
elytra dimension in question. The model must report the red line's length in centimetres.
We pose 14,108 such items across 864 specimen plates and evaluate **eighteen open-weight
vision–language models**.

The task is constructed so that the answer is *computable from the image alone* and
*guessable from prior knowledge alone*, and the entire evaluation is designed to separate
those two routes. Beetle elytra occupy a narrow size band, so a model that has memorised
typical beetle dimensions can post a respectable relative error without measuring anything.
We therefore install three complementary shortcut controls
([Section 5](#5-the-shortcut-analysis)) and lead with the one that no prior-driven strategy
can satisfy.

Three findings organise the component. A per-type constant predictor **outscores fifteen of
the eighteen models** on tolerance accuracy
([Section 6.2](#62-the-constant-predictor-outscores-fifteen-of-eighteen-models)). Range
compression — the signature of answering from a single remembered size — is positive for
**seventeen of eighteen models**, with a closed-form saturation point that six models
approach to within 15%
([Section 6.3](#63-range-compression-is-near-universal)). And ranking dissociates sharply
from calibration: models that order specimens correctly frequently miss every tolerance band
because they apply the wrong scale factor
([Section 6.4](#64-ranking-and-calibration-dissociate)).

---

## 2. Motivation

### 2.1 Metric measurement as a compositional visual operation

Scale-bar measurement is the only task in Component C3 that requires a model to compose two
visual measurements into a ratio and then bind that ratio to a physical unit. It decomposes
into three operations, each independently failable:

$$
\underbrace{\ell_{\text{red}}^{\text{px}}}_{\text{(i) measure}} \;\big/\;
\underbrace{\ell_{\text{blue}}^{\text{px}}}_{\text{(ii) measure}} \;\times\;
\underbrace{1\ \text{cm}}_{\text{(iii) bind}} \;=\; \text{answer}
$$

A model may localise both segments and still fail to form the ratio; it may form the ratio
and fail to bind the unit. Because the two segments lie in the same image at the same
magnification, the task requires no depth reasoning, no 3D inference, and no world knowledge
— which is precisely what makes a failure here diagnostic. Everything needed is present in
the pixels.

This also makes the task the most **acuity-sensitive** in the component. The answer is a
ratio between two short segments, so any resolution loss propagates directly into the
answer. We raise the vision-token budget to near-native resolution for every model
([Section 6.4 of the inference document](2.Experiment_Runs/README.md#64-vision-token-budget))
so that the measurement is not artificially bottlenecked before the model sees it.

### 2.2 Bounding the prior-driven shortcut

Carabid elytra span roughly half a centimetre to a little over one centimetre. A model that
answers a single remembered number for every question therefore lands within 20% of the
truth surprisingly often. Worse, the question *states* whether length or width is wanted, so
a model can answer one fixed number for length and another for width and still measure
nothing at all. That predictor reaches **acc@20 = 0.410**.

An evaluation that reported tolerance accuracy without this floor would credit prior recall
as measurement. We therefore fit the constant predictor per measurement type, on the
identical rows as each model, **and against the metric it is compared to** — the value
minimising mean relative error and the value maximising acc@20 are different values, and
using the wrong one understates the floor by more than ten points
([Section 5.1](#51-the-per-type-constant-predictor)).

### 2.3 Ranking and calibration as separable abilities

Two distinct questions are asked of every model, and they come apart sharply in this task:

- **Does the model measure anything?** Answered by rank correlation computed within
  measurement type. Any predictor that is constant within a type scores zero here by
  construction, whatever its tolerance accuracy.
- **Does the model get the scale right?** Answered by tolerance accuracy and by the gap
  between raw and bias-corrected accuracy.

A model can rank 864 plates of specimens correctly and still miss every tolerance band if it
applies a systematically wrong scale factor. The separation of these two abilities is the
component's principal analytical result
([Section 6.4](#64-ranking-and-calibration-dissociate)).

---

## 3. System architecture

### 3.1 Pipeline stages and dependency order

```
        ┌──────────────────────────────────────────────┐
        │  NEON Beetles corpus                         │
        │  metadata.csv: scalebar coords,              │
        │  elytra_length_coords, elytra_width_coords,  │
        │  elytra_length_cm, elytra_width_cm           │
        └──────────────────┬───────────────────────────┘
                           │
                           ▼
   (1)  1.Generate_QA/generateQA.py
                           │   filter 10-30 beetles/plate
                           │   draw blue scale bar + green border
                           │   draw red measurement + white border
                           │   one image per (plate, measurement)
                           ▼
                       qa.csv  +  Images/
                           │
                           ▼
   (2)  2.Experiment_Runs/Preprocess/split_qa.py
                           │   flatten JSON, parse units, derive
                           │   measure_type and plate_id, resolve paths,
                           │   shuffle then shard
                           ▼
                       chunks/chunk00..05/qa.csv
                           │
                           ├──►  stats.py   constant-predictor floor,
                           │                cluster diagnostics, runtime
                           ▼
   (3)  2.Experiment_Runs/<model>.py  x 18
                           │   near-native vision budget, invariant harness
                           ▼
                       chunks/chunkNN/qa_<KEY>.csv
                       chunks/chunkNN/qa_<KEY>.summary.json
                           │
                           ▼  merge across shards, sort by row_id
                       results/qa_<KEY>.csv
                           │
                           ▼
   (4)  3.Evaluate/evaluate.py  ──►  morphometry_overall.csv
                                     morphometry_by_group.csv
                                     morphometry_report.md
```

The order is a data dependency. Stage 2 derives `plate_id`, on which every confidence
interval in the component depends, and `measure_type`, on which the entire shortcut analysis
depends. Stage 2 also performs a deterministic shuffle before sharding, without which every
shard would come out entirely one measurement type
([Section 4.5 of the inference document](2.Experiment_Runs/README.md#45-shuffle-before-shard)).

### 3.2 Repository layout

```
Distance_Absolute/
├── README.md                      this document
│
├── 1.Generate_QA/
│   ├── README.md
│   └── generateQA.py              annotation rendering and QA emission
│
├── 2.Experiment_Runs/
│   ├── README.md
│   ├── Preprocess/
│   │   ├── split_qa.py            flatten, derive, shuffle, shard
│   │   └── stats.py               constant-predictor floor + cluster audit
│   └── <model>.py  × 18           inference runners, invariant harness
│
└── 3.Evaluate/
    ├── README.md
    └── evaluate.py                scoring, shortcut controls, uncertainty
```

| # | Stage | Document |
|---|---|---|
| 1 | Render annotations and emit the QA set | [`1.Generate_QA/README.md`](1.Generate_QA/README.md) |
| 2–3 | Flatten, shard, and run inference across 18 models | [`2.Experiment_Runs/README.md`](2.Experiment_Runs/README.md) |
| 4 | Score against three shortcut controls and report | [`3.Evaluate/README.md`](3.Evaluate/README.md) |

---

## 4. The evaluation set

### 4.1 Task definition and annotation geometry

Each item is a specimen plate photograph bearing two rendered annotations.

| Annotation | Line colour | Border colour | Meaning |
|---|---|---|---|
| Scale bar | blue | green | Reference length, exactly 1 cm |
| Measurement | red | white | The elytra dimension asked about |

Each line is drawn with a proportional rectangular border extending 20% of the line's length
beyond each endpoint and 20% perpendicular, so the annotation remains legible against a
cluttered plate at any specimen size without occluding the specimen
([Section 4 of the generation document](1.Generate_QA/README.md#4-annotation-rendering)).

The question follows a single template:

> *What is the measurement of the elytra {length|width} (marked with red line with white
> border) with respect to the scalebar (marked with blue line with green border) in cm?*

The scale bar's physical length is supplied by the inference prompt rather than the question
stem, together with the explicit procedure and an instruction not to answer from prior
knowledge ([Section 6.2 of the inference document](2.Experiment_Runs/README.md#62-prompt-construction)).
This is a deliberate design decision: without the constant, the ratio is computable but the
answer in centimetres is not, and the task would reduce to the prior recall it exists to
rule out.

We render one image per (plate, measurement type) pair rather than drawing both measurements
on one canvas. This holds the number of red lines at exactly one, so the referent of "the
red line" is never ambiguous, and it yields two items per plate that differ only in which
dimension is marked — the within-plate contrast that the compression diagnostic exploits
([Section 5.3](#53-range-compression-and-its-closed-form-anchor)).

### 4.2 Source corpus and specimen selection

We draw from the NEON Beetles specimen-plate corpus, which is the only collection in our
data holding all four properties the task requires: pixel coordinates for a scale bar, pixel
coordinates for each morphological measurement, the corresponding physical measurement in
centimetres, and a uniform tray photography protocol with a fixed working distance.

We restrict to plates carrying **10 to 30 specimens**. Below ten, a plate contributes too
few items to justify its own scale bar; above thirty, specimens crowd densely enough that
the red line frequently overlaps neighbouring individuals and the annotation becomes
ambiguous.

Ground truth is the corpus's own `elytra_length_cm` and `elytra_width_cm` fields. We do not
re-measure, so every item remains auditable against the published corpus.

### 4.3 Scale and cluster structure

| Quantity | Value |
|---|---:|
| Scored items per model | 14,108 |
| Specimen plates | 864 |
| Items per plate (mean) | 16.3 |
| Length items | 7,054 |
| Width items | 7,054 |
| acc@20-optimal constant, length | 1.16 cm |
| acc@20-optimal constant, width | 0.55 cm |

The cluster structure is severe and it drives a methodological requirement. Every specimen on
a plate shares one scale bar, one magnification, one illumination, and one focal plane, and
each contributes both a length and a width item. Within-plate spread is roughly a seventh of
total spread, so approximately 16 rows from one plate behave close to a single observation.

**Every confidence interval in this component therefore resamples plates, not rows.** A
row-level bootstrap would report intervals several times too narrow
([Section 7 of the evaluation document](3.Evaluate/README.md#7-uncertainty-the-clustered-bootstrap)).

---

## 5. The shortcut analysis

Three controls, each catching a different prior-driven strategy. Together they leave no
route to a high score that does not involve reading the image.

| Control | Strategy it catches | Score of a model using that strategy |
|---|---|---|
| Per-type constant floor | Type-aware prior recall: one number for length, another for width | acc@20 = 0.410, which the model must beat |
| `rho_within` | Any predictor constant within a measurement type | $\rho_{\text{within}} = 0$ |
| Compression index | Single-anchor answering: one remembered size for both types | $\kappa \approx 1$ |

### 5.1 The per-type constant predictor

For each measurement type $t \in \{\text{length}, \text{width}\}$ we fit the constant that
optimises the metric under comparison, over the gold values on the **identical rows** the
model was scored on:

$$
c_t^{\text{relErr}} = \mathrm{arg\,min}_{c} \; \frac{1}{n_t}\sum_{i \in t} \left|\frac{c}{y_i} - 1\right|,
\qquad
c_t^{\text{acc}} = \mathrm{arg\,max}_{c} \; \frac{1}{n_t}\sum_{i \in t} \mathbb{1}\!\left[\left|\frac{c}{y_i} - 1\right| \le 0.20\right].
$$

Two design points are load-bearing.

**Fitted per type, not pooled.** The question states which dimension is wanted, so type-aware
constant answering requires no measurement. Fitting a single pooled constant would understate
the floor substantially: the per-type predictor reaches acc@20 = 0.410 where a pooled one
reaches roughly 0.30.

**Fitted against the metric it is compared to.** The relErr-optimal and acc-optimal constants
are different values. On the length items the relErr-optimal constant scores acc@20 of about
0.30 while the acc-optimal constant scores 0.43 — a thirteen-point difference that would
otherwise flatter every model in the table.

The fitted values, $c^{\text{acc}}_{\text{length}} = 1.16$ cm and
$c^{\text{acc}}_{\text{width}} = 0.55$ cm, are reported in every results row so the floor is
never separated from the numbers it bounds.

### 5.2 Why pooled rank correlation is inadmissible

Elytra length is roughly twice elytra width, and the question says which is being asked. A
model that measures nothing and merely answers a larger number for length than for width
therefore achieves a **pooled Spearman correlation of about +0.63** — purely from the
between-type separation, with zero within-type signal.

We therefore compute rank correlation **inside each measurement type and then average**:

$$
\rho_{\text{within}} \;=\; \frac{1}{|T|}\sum_{t \in T} \mathrm{Spearman}\!\left(\hat{y}_{|t},\; y_{|t}\right).
$$

Any predictor constant within a type scores exactly zero here. We report `rho_pooled`
alongside it solely so that the gap between the two remains visible, and it is large: the
pooled figure exceeds the within-type figure by up to 0.171
([Section 6.1](#61-the-leaderboard)). **`rho_within` is the headline.**

### 5.3 Range compression and its closed-form anchor

The third control has a closed-form saturation point, which turns a qualitative signature
into a normalised index.

Define the compression index as the difference in log-ratio bias between the two measurement
types:

$$
\text{compression} \;=\; \mathrm{bias}(\text{width}) - \mathrm{bias}(\text{length}),
\qquad
\mathrm{bias}(t) = \frac{1}{n_t}\sum_{i \in t} \log\frac{\hat{y}_i}{y_i}.
$$

**A model that measures correctly has no reason to be biased in opposite directions on the
two dimensions**, so compression $\approx 0$. Now consider a model that answers a single
remembered value $c$ regardless of which dimension is asked. Its biases are
$\mathrm{bias}(t) = \log c - \mathbb{E}[\log y_{|t}]$, so

$$
\boxed{\;\text{compression}_{\text{single-anchor}} \;=\; \mathbb{E}[\log y_{|\text{length}}] - \mathbb{E}[\log y_{|\text{width}}] \;=\; \log \frac{\mathrm{GM}(\text{length})}{\mathrm{GM}(\text{width})}\;}
$$

**independent of $c$.** The constant cancels. Using the fitted per-type constants as a proxy
for central tendency gives an anchor of

$$
\log\frac{1.16}{0.55} \;\approx\; 0.746 ,
$$

so we report the normalised index

$$
\kappa \;=\; \frac{\text{compression}}{0.746},
$$

on which $\kappa = 0$ denotes true measurement and $\kappa = 1$ denotes pure single-anchor
answering. Positive compression means the model overestimates the smaller quantity and
underestimates the larger one, squeezing every answer toward one central value — exactly what
answering from a remembered beetle size produces.

Note that compression and the per-type constant floor catch *different* shortcuts and are
genuinely complementary. The per-type constant predictor is fitted separately for each type,
so its own compression is approximately zero; it is caught by `rho_within`, not by $\kappa$.
Conversely a single-anchor model may sit below the per-type floor while exhibiting
$\kappa \approx 1$. Only a model clearing all three controls has measured anything.

---

## 6. Results

Full tables are written to `3.Evaluate/results/morphometry_report.md`. The headline metric is
`rho_within`; `acc@20` reports precision, and the per-type constant floor bounds both.

### 6.1 The leaderboard

Sorted by `rho_within`. `const_acc20` = 0.410 for every model.

| Model | rho_within | rho_pooled | relErr | acc@20 | acc@20 calib | κ | bias | parse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3-VL-32B** | **0.858** | 0.904 | **0.185** | **0.622** | 0.614 | **0.085** | −0.015 | 1.000 |
| **Qwen3-VL-30B-A3B** | 0.782 | 0.850 | 0.245 | 0.528 | 0.541 | 0.168 | +0.035 | 1.000 |
| Gemma-4-12B | 0.753 | 0.855 | 0.268 | 0.385 | 0.512 | 0.227 | −0.257 | 0.995 |
| InternVL3.5-8B-Instruct | 0.703 | 0.490 | 0.383 | 0.263 | 0.509 | 0.906 | −0.127 | 1.000 |
| InternVL3.5-38B-Instruct | 0.683 | 0.814 | 0.279 | 0.503 | 0.469 | 0.189 | −0.009 | 1.000 |
| Qwen3-VL-8B | 0.682 | 0.746 | 0.325 | 0.303 | 0.453 | 0.544 | −0.222 | 1.000 |
| InternVL3-38B-Instruct | 0.533 | 0.679 | 0.347 | 0.367 | 0.415 | 0.570 | −0.193 | 1.000 |
| Qwen2.5-VL-7B | 0.520 | 0.554 | 0.553 | 0.334 | 0.411 | 0.650 | +0.253 | 1.000 |
| Molmo2-8B | 0.511 | 0.682 | 0.452 | 0.290 | 0.399 | 0.591 | +0.132 | 1.000 |
| Gemma-3-12B | 0.507 | 0.609 | 0.453 | 0.352 | 0.379 | 0.571 | +0.135 | 1.000 |
| InternVL3-8B-Instruct | 0.467 | 0.382 | 0.475 | 0.297 | 0.408 | 0.839 | +0.066 | 1.000 |
| Pixtral-12B | 0.412 | 0.385 | 0.534 | 0.307 | 0.420 | 0.759 | +0.136 | 1.000 |
| InternVL3.5-30B-A3B | 0.381 | 0.477 | 0.494 | 0.378 | 0.336 | 0.609 | +0.169 | 1.000 |
| Llama-3.2-11B-Vision | 0.320 | 0.362 | 0.452 | 0.260 | 0.245 | 0.582 | −0.319 | 0.970 |
| Phi-3.5-Vision | 0.148 | 0.093 | 0.949 | 0.198 | 0.313 | 0.863 | +0.465 | 1.000 |
| Qwen2.5-VL-3B | 0.081 | 0.073 | 0.944 | 0.194 | 0.306 | 0.851 | +0.463 | 1.000 |
| LLaVA-NeXT-Vicuna-7B | −0.023 | −0.013 | 0.748 | 0.217 | 0.326 | 0.847 | +0.294 | 1.000 |
| LLaVA-NeXT-Mistral-7B | — | — | 0.428 | 0.332 | 0.305 | 0.851 | −0.224 | 1.000 |

LLaVA-NeXT-Mistral-7B produces insufficient within-type rank variance for a correlation to be
defined — the strongest possible statement that it is not measuring, and the reason its
otherwise unremarkable relative error of 0.428 must not be read as partial competence.

The `rho_pooled` column exists to be discounted. It exceeds `rho_within` by up to **+0.171**
(Molmo2-8B: 0.682 pooled against 0.511 within) and by more than 0.10 for five models. Every
point of that gap is between-type separation available to a model that measures nothing.

### 6.2 The constant predictor outscores fifteen of eighteen models

$$
\text{acc@20}_{\text{per-type constant}} = 0.410, \qquad \textbf{cleared by only } 3/18 \text{ models.}
$$

Only Qwen3-VL-32B (0.622), Qwen3-VL-30B-A3B (0.528), and InternVL3.5-38B-Instruct (0.503)
report a tolerance accuracy above a predictor that never looks at the image. Broken out by
measurement type the picture is sharper still: **6 of 18 clear the length floor of 0.382, and
only 2 of 18 clear the width floor of 0.438**.

This is the component's central quantitative result, and it is a statement about the field
rather than about our construction. Scale-bar measurement is a solved problem in classical
computer vision and a routine operation for a human technician; the reference length is
stated in the prompt, both segments are marked in saturated primary colours with contrasting
borders, and the arithmetic is a single division. Fifteen of eighteen current
vision–language models nevertheless do worse than answering a memorised number.

The result is also robust to the choice of floor, because the floor is fitted three ways and
the most conservative is used: per type rather than pooled, on the model's own scored rows
rather than the full set, and against the specific metric being compared
([Section 5.1](#51-the-per-type-constant-predictor)). Each of those choices raises the bar
the models must clear.

### 6.3 Range compression is near-universal

Mean log-ratio bias by measurement type, across all eighteen models:

$$
\overline{\mathrm{bias}}(\text{length}) = -0.178, \qquad
\overline{\mathrm{bias}}(\text{width}) = +0.262 .
$$

$$
\text{compression} > 0 \;\text{ for }\; \textbf{17/18 models}; \quad
\mathrm{bias}(\text{length}) < 0 \;\text{ for } 15/18; \quad
\mathrm{bias}(\text{width}) > 0 \;\text{ for } 14/18 .
$$

Models systematically under-measure the larger dimension and over-measure the smaller one.
This is not a generic accuracy deficit — a generic deficit would produce bias of the same
sign on both. It is the specific signature derived in
[Section 5.3](#53-range-compression-and-its-closed-form-anchor): answers squeezed toward a
single central value, which is what recall of a remembered beetle size produces and what
genuine measurement cannot.

Against the closed-form single-anchor anchor of 0.746:

| κ band | Models | Interpretation |
|---|---:|---|
| $\kappa < 0.20$ | 3 | Qwen3-VL-32B (0.085), Qwen3-VL-30B (0.168), InternVL3.5-38B (0.189) — measuring |
| $0.20 \le \kappa < 0.80$ | 9 | Partial measurement blended with a size prior |
| $\kappa \ge 0.80$ | 6 | Within 20% of pure single-anchor answering |

Six models sit at $\kappa \ge 0.84$, meaning their between-type bias structure is
indistinguishable from answering one remembered number for every question. The index
correlates with the headline metric at $\rho(\kappa, \rho_{\text{within}}) = -0.689$ and with
precision at $\rho(\kappa, \text{acc@20}) = -0.775$, so the three shortcut controls agree on
the ordering while catching different strategies.

The exception is instructive. **Qwen3-VL-32B reports identical bias on length and width
(−0.046 on both), giving $\kappa = 0.085$** — the only model whose error structure carries no
trace of a size prior, and also the model with the highest `rho_within` and the only one whose
acc@20 exceeds the floor by more than twenty points. The three controls converge on a single
model.

### 6.4 Ranking and calibration dissociate

`acc20_calib` rescores each model after removing its own per-type bias — dividing every
prediction by $\exp(\mathrm{bias}(t))$ within each measurement type. Because the
correction is fitted on the rows it is applied to, it is an optimistic ceiling and a
**diagnostic, not a score**: it reports how well a model would do if it were told the correct
scale factor.

The gap between raw and corrected accuracy separates two failure modes that tolerance
accuracy alone conflates.

| Model | rho_within | acc@20 | acc@20 calib | Gain | Diagnosis |
|---|---:|---:|---:|---:|---|
| InternVL3.5-8B-Instruct | 0.703 | 0.263 | 0.509 | **+0.246** | measures, does not calibrate |
| Qwen3-VL-8B | 0.682 | 0.303 | 0.453 | +0.151 | measures, does not calibrate |
| Gemma-4-12B | 0.753 | 0.385 | 0.512 | +0.127 | measures, does not calibrate |
| Qwen3-VL-32B | 0.858 | 0.622 | 0.614 | −0.008 | measures and calibrates |
| Qwen2.5-VL-3B | 0.081 | 0.194 | 0.306 | +0.112 | neither |
| LLaVA-NeXT-Vicuna-7B | −0.023 | 0.217 | 0.326 | +0.110 | neither |

**InternVL3.5-8B-Instruct is the cleanest case in the set.** It ranks fourth of eighteen on
`rho_within` at 0.703 — it orders specimens by size about as well as a 38B model — while
ranking fifteenth on raw acc@20 at 0.263, well below the constant-predictor floor. Its
compression index of 0.906 is the highest in the table. Removing its per-type bias lifts it
from 0.263 to 0.509, from below the floor to third place. The information required to answer
correctly is present in its predictions; it is applied at the wrong scale.

This dissociation is what justifies leading with `rho_within` rather than acc@20. The two
correlate at $\rho = 0.755$, leaving substantial disagreement, and after bias removal the
correlation rises to $\rho(\rho_{\text{within}}, \text{acc@20}_{\text{calib}}) = 0.934$. The
residual disagreement between ranking and precision is therefore **almost entirely a
calibration offset rather than a measurement failure** — a decomposition that acc@20 alone
cannot express and that no single-number leaderboard would surface.

### 6.5 The width asymmetry

Width is the harder measurement, consistently and by a wide margin:

$$
\overline{\text{relErr}}(\text{length}) = 0.338, \qquad \overline{\text{relErr}}(\text{width}) = 0.609,
$$

with width worse for 14 of 18 models, and floor-clearance falling from 6/18 on length to 2/18
on width.

Two mechanisms compound, and they are separable in the data. Width is roughly half of length
in physical extent, so it subtends fewer pixels and is the more acuity-limited measurement.
And because width lies further below the central tendency of the pooled size distribution, a
model anchored on a remembered size overshoots it more severely — which is exactly the
$\mathrm{bias}(\text{width}) = +0.262$ reported above.

The extreme cases isolate the second mechanism. Phi-3.5-Vision posts relative error of 1.436
and acc@20 of 0.060 on width against 0.463 and 0.336 on length; Qwen2.5-VL-3B posts 1.419 and
0.067 against 0.469 and 0.321. Both models are over-measuring width by a factor of roughly
2.2 — the length-to-width ratio — which is the arithmetic fingerprint of answering the length
value to a width question.

### 6.6 Scale and resolution

Spearman correlation between parameter count and score:

| Against | ρ |
|---|---:|
| acc@20 | 0.810 |
| rho_within | 0.575 |

Scale tracks precision here far more strongly than it does elsewhere in the component,
which is consistent with the task's acuity sensitivity: larger models in these families
carry higher-capacity vision encoders. But the correlation with the headline ranking metric
is markedly weaker, and the exceptions are systematic. An 8B InternVL3.5 attains
`rho_within` = 0.703 against a 30B sibling's 0.381 and a 38B InternVL3's 0.533. **Whether a
model measures at all is not primarily a function of scale; how precisely it measures is.**

Because this is the most acuity-sensitive task in the component, we raise the vision budget
to near-native resolution for every model rather than reusing the budget applied elsewhere.
At the lower budget used for the Counting task a 1 cm scale bar renders to roughly 41 px and
the shortest elytra to roughly 12 px, a ratio that cannot be estimated to 20% by any means.
The budget is raised identically for all eighteen models
([Section 6.4 of the inference document](2.Experiment_Runs/README.md#64-vision-token-budget)),
so the resolution decision is a property of the task rather than a per-model advantage.

---

## 7. Scope of the released evaluation

The three shortcut controls in [Section 5](#5-the-shortcut-analysis) jointly exclude
prior-driven answering: a model cannot clear the per-type floor, score high on `rho_within`,
and hold $\kappa \approx 0$ without extracting size information from the image. A natural
sharpening distinguishes *which* image information is used. A predictor that reads the red
line's apparent pixel length but never consults the blue scale bar would rank specimens
correctly within a plate while failing across plates photographed at different
magnifications, and separating that from true scale-bar referencing requires the per-plate
pixel coordinates to be carried through into the released manifest — a one-column addition to
the generation stage that would make the control computable without new inference.

The compression index admits a second refinement. We anchor it at
$\log(\mathrm{GM}_{\text{length}}/\mathrm{GM}_{\text{width}})$ using the fitted per-type
constants as a proxy for central tendency; computing the geometric means directly from the
gold distribution would make $\kappa$ exact rather than approximate, without changing any
ordering in [Section 6.3](#63-range-compression-is-near-universal).

---

## 8. Reproduction

```bash
# 1. Render annotations and emit the QA set.
python 1.Generate_QA/generateQA.py \
    --base_dir        $BIOVQA_DATA/datasets/Beetles-500 \
    --output_base_dir "$BIOVQA_DATA/tasks/Absolute Distance Measurements"

# 2. Flatten, derive plate_id and measure_type, shuffle, shard.
python 2.Experiment_Runs/Preprocess/split_qa.py \
    --task_root "$BIOVQA_DATA/tasks/Absolute Distance Measurements" \
    --run_dir   2.Experiment_Runs \
    --n_chunks 6 --seed 42

# 3. Constant-predictor floor, cluster diagnostics, runtime projection.
python 2.Experiment_Runs/Preprocess/stats.py --run_dir 2.Experiment_Runs

# 4. Inference — 18 models x 6 shards.
python 2.Experiment_Runs/qwenVL2_5_7B.py --qa_csv 2.Experiment_Runs/chunks/chunk00/qa.csv

# 5. Score against all three shortcut controls.
python 3.Evaluate/evaluate.py \
    --results_dir 3.Evaluate/results \
    --out_dir     3.Evaluate/results \
    --bootstrap 2000
```

Quote the task root — it contains spaces. Step 4 comprises 108 independent jobs, executed as
SLURM arrays on an H100 partition. Step 5 consumes merged per-model CSVs; the merge contract
is specified in
[Section 7.3 of the inference document](2.Experiment_Runs/README.md#73-merge-contract).

---

## 9. Outputs

| File | Contents |
|---|---|
| `3.Evaluate/results/morphometry_overall.csv` | One row per model: all metrics, all three shortcut controls, bootstrap bounds on `rho_within`. |
| `3.Evaluate/results/morphometry_by_group.csv` | One row per model × measurement type, with per-type constants and biases. |
| `3.Evaluate/results/morphometry_report.md` | Leaderboard sorted by `rho_within`, per-measurement table, interpretation. |
| `chunks/chunkNN/qa_<KEY>.summary.json` | Per-shard run record including the verbatim prompt and the vision budget. |

---

## 10. References

Annotation rendering uses OpenCV [1]. Clustered resampling for dependent observations follows
standard bootstrap practice [2, 3]. The evaluated models are documented in their respective
technical reports [4, 5]. The source corpus is the NEON invertebrate specimen collection [6].

[1] G. Bradski. **The OpenCV Library.** *Dr. Dobb's Journal of Software Tools*, 2000.

[2] B. Efron and R. J. Tibshirani. **An Introduction to the Bootstrap.** Chapman & Hall, 1993.

[3] A. C. Cameron and D. L. Miller. **A Practitioner's Guide to Cluster-Robust Inference.**
*Journal of Human Resources*, 50(2):317–372, 2015.

[4] S. Bai, K. Chen, X. Liu, J. Wang, et al. **Qwen2.5-VL Technical Report.** *arXiv preprint
arXiv:2502.13923*, 2025.

[5] J. Zhu, W. Wang, Z. Chen, et al. **InternVL3: Exploring Advanced Training and Test-Time
Recipes for Open-Source Multimodal Models.** *arXiv preprint arXiv:2504.10479*, 2025.

[6] National Ecological Observatory Network (NEON). **Ground beetles sampled from pitfall
traps (DP1.10022.001).** NEON Data Portal.