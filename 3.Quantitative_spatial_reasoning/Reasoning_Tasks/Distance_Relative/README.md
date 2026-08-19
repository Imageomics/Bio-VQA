<h1 align="center">Bio-VQA &rarr; Spatial Visual Reasoning &rarr; Relative Size</h1>
<p align="center">
  <b>Within-Plate Comparative Morphometry of Pinned Beetle Specimens</b>
</p>
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="Models" src="https://img.shields.io/badge/models-18%20open--weight%20VLMs-orange">
  <img alt="Instances" src="https://img.shields.io/badge/instances-3%2C976%20per%20model-green">
  <img alt="Plates" src="https://img.shields.io/badge/plates-207-lightgrey">
  <img alt="Chance" src="https://img.shields.io/badge/chance-0.250-red">
  <img alt="Headline" src="https://img.shields.io/badge/headline-balAcc%20%2B%20posBias-yellow">
  <img alt="Scheduler" src="https://img.shields.io/badge/scheduler-SLURM-teal">
  <img alt="Status" src="https://img.shields.io/badge/status-complete-brightgreen">
</p>

> An evaluation of comparative visual measurement. Four beetles on one plate are each
> marked with a coloured line along the elytra; the model names the largest. Because all
> four share a single magnification, on-screen line length is exactly proportional to real
> size and no scale bar is required. This document describes the task construction, the
> position-bias controls, the empirical findings, and the exact commands to reproduce them.

---

## Table of contents

1. [Overview](#1-overview)
2. [Motivation](#2-motivation)
   - 2.1 [Comparison isolated from calibration](#21-comparison-isolated-from-calibration)
   - 2.2 [Position bias as a pre-registered control](#22-position-bias-as-a-pre-registered-control)
   - 2.3 [The margin asymmetry as an internal validity check](#23-the-margin-asymmetry-as-an-internal-validity-check)
3. [System architecture](#3-system-architecture)
   - 3.1 [Pipeline stages and dependency order](#31-pipeline-stages-and-dependency-order)
   - 3.2 [Repository layout](#32-repository-layout)
4. [The evaluation set](#4-the-evaluation-set)
   - 4.1 [Task definition and annotation scheme](#41-task-definition-and-annotation-scheme)
   - 4.2 [Why no scale bar is needed](#42-why-no-scale-bar-is-needed)
   - 4.3 [The separation margin](#43-the-separation-margin)
   - 4.4 [Scale and cluster structure](#44-scale-and-cluster-structure)
5. [The control set](#5-the-control-set)
   - 5.1 [Chance, balanced accuracy, and kappa](#51-chance-balanced-accuracy-and-kappa)
   - 5.2 [The position-bias index](#52-the-position-bias-index)
6. [Results](#6-results)
   - 6.1 [The leaderboard](#61-the-leaderboard)
   - 6.2 [A narrow band above chance](#62-a-narrow-band-above-chance)
   - 6.3 [Systematic avoidance of the last option](#63-systematic-avoidance-of-the-last-option)
   - 6.4 [The margin asymmetry inverts](#64-the-margin-asymmetry-inverts)
   - 6.5 [Position bias predicts accuracy](#65-position-bias-predicts-accuracy)
   - 6.6 [Scale](#66-scale)
7. [Scope of the released evaluation](#7-scope-of-the-released-evaluation)
8. [Reproduction](#8-reproduction)
9. [Outputs](#9-outputs)
10. [References](#10-references)

---

## 1. Overview

We evaluate whether vision–language models can perform **comparative measurement**: given
four morphological features marked in a single photograph, identify the largest. Each item
presents one plate of pinned beetle specimens on which four individuals are marked with a
coloured dotted box and a coloured line drawn along the elytra; the model answers A, B, C,
or D. We pose 3,976 such items across 207 specimen plates and evaluate **eighteen
open-weight vision–language models**.

The task is the within-plate counterpart to
[Absolute Distance](../Distance_Absolute/README.md). Because all four specimens appear in
one photograph at one magnification, and because the lines are drawn before any resizing,
**on-screen line length is exactly proportional to real elytra length**. No scale bar, no
unit conversion, and no prior knowledge of beetle dimensions can help or hinder: the answer
is a pure ordering of four line segments visible in the image
([Section 4.2](#42-why-no-scale-bar-is-needed)).

That construction makes the results unusually interpretable. Chance is exactly 0.250, the
answer key is near-uniform over the four options by construction, and the best of eighteen
models reaches **0.443 balanced accuracy** — 0.193 above chance
([Section 6.2](#62-a-narrow-band-above-chance)). Two models sit at chance exactly, answering
the same letter on every item.

Two findings organise the component. Models systematically avoid the last option: the mean
predicted share of D is **0.077 against a gold share of 0.248**, with six models never
answering D at all ([Section 6.3](#63-systematic-avoidance-of-the-last-option)). And the
generator's own margin asymmetry inverts: width items carry double the separation margin of
length items and are therefore strictly easier by construction, yet only 5 of 18 models do
better on width ([Section 6.4](#64-the-margin-asymmetry-inverts)) — direct evidence that
most models are not comparing line lengths.

---

## 2. Motivation

### 2.1 Comparison isolated from calibration

[Absolute Distance](../Distance_Absolute/README.md) asks a model to convert an apparent
length into centimetres, which composes three operations: measure the target, measure the
reference, and bind the ratio to a unit. That task shows the three dissociate — several
models rank specimens correctly while applying the wrong scale factor.

Relative Size removes the last two operations entirely. There is no reference segment and no
unit; the model must only order four segments that are simultaneously visible. It is
therefore the **lower bound** of the measurement family: whatever a model can do on absolute
morphometry, it should be able to do here, because this task is a strict sub-problem. A
model that fails here cannot be said to measure at all, whatever its absolute-task numbers
show.

### 2.2 Position bias as a pre-registered control

Four-alternative multiple choice invites a failure mode that accuracy alone cannot detect: a
model that always answers the same letter scores whatever share that letter holds in the
key. Llama-3.2-Vision exhibits a documented 63% first-option preference elsewhere in this
project, so we treat option-position preference as a **pre-registered control** rather than
a post-hoc observation, and report the full predicted-letter distribution for every model
alongside three statistics that a constant predictor cannot pass
([Section 5](#5-the-control-set)).

The control earns its place. Two models answer A on 100% of items, and the mean predicted
share of the last option is under a third of its gold share
([Section 6.3](#63-systematic-avoidance-of-the-last-option)).

### 2.3 The margin asymmetry as an internal validity check

The generator accepts a quartet only when the largest specimen exceeds the runner-up by a
clear margin: **5% on length, 10% on width**
([Section 4.3](#43-the-separation-margin)). That asymmetry was introduced to equalise yield
across the two measurement types, and it has a useful consequence. Width items carry twice
the separation margin, so under any account in which the model is comparing line lengths,
width must be the easier stratum.

This gives the component an **internal validity check that requires no additional
annotation**: a model doing better on length than on width is doing something other than
comparing line lengths. The check is decisive
([Section 6.4](#64-the-margin-asymmetry-inverts)).

---

## 3. System architecture

### 3.1 Pipeline stages and dependency order

```
        ┌──────────────────────────────────────────────┐
        │  NEON Beetles corpus                         │
        │  metadata.csv  : elytra coords + cm values   │
        │  Detections.csv: per-specimen bounding boxes │
        └──────────────────┬───────────────────────────┘
                           │
                           ▼
   (1)  1.Generate_QA/generateQA.py
                           │   filter 10-30 specimens/plate
                           │   sample 4 specimens, pick length or width
                           │   enforce the separation margin
                           │   draw 4 coloured dotted boxes + lines
                           ▼
                       qa.csv  +  Images/
                           │
                           ▼
   (2)  2.Experiment_Runs/Preprocess/split_qa.py
                           │   flatten JSON, resolve the option letter,
                           │   derive measure_type / plate_id / set_index,
                           │   resolve paths, shuffle, shard
                           ▼
                       chunks/chunk00..05/qa.csv
                           │
                           ├──►  stats.py   key balance, cluster ratio, runtime
                           ▼
   (3)  2.Experiment_Runs/<model>.py  x 18
                           │   near-native vision budget, invariant harness
                           ▼
                       chunks/chunkNN/qa_<KEY>.csv  (+ summary.json)
                           │
                           ▼  merge_results.py  ->  results/qa_<KEY>.csv
                           ▼
   (4)  3.Evaluate/evaluate.py  ──►  relsize_overall.csv
                                     relsize_by_group.csv
                                     relsize_confusion.csv
                                     relsize_report.md
```

Stage 2 derives `plate_id`, on which every confidence interval depends, and `measure_type`,
on which the internal validity check depends. It also performs a seeded shuffle before
sharding: rows arrive grouped by plate and, within a plate, in set order, so a plain stride
would hand each shard a systematic slice
([Section 4.5 of the inference document](2.Experiment_Runs/README.md#45-shuffle-before-shard)).

### 3.2 Repository layout

```
Distance_Relative/
├── README.md                      this document
│
├── 1.Generate_QA/
│   ├── README.md
│   └── generateQA.py              quartet sampling, margin filter, annotation rendering
│
├── 2.Experiment_Runs/
│   ├── README.md
│   ├── Preprocess/
│   │   ├── split_qa.py            flatten, derive, shuffle, shard
│   │   └── stats.py               key balance + cluster audit
│   └── <model>.py  × 18           inference runners, invariant harness
│
└── 3.Evaluate/
    ├── README.md
    ├── merge_results.py           reassemble shards into per-model CSVs
    └── evaluate.py                scoring, position-bias controls, uncertainty
```

| # | Stage | Document |
|---|---|---|
| 1 | Sample quartets and render annotations | [`1.Generate_QA/README.md`](1.Generate_QA/README.md) |
| 2–3 | Flatten, shard, and run inference across 18 models | [`2.Experiment_Runs/README.md`](2.Experiment_Runs/README.md) |
| 4 | Merge, score against the control set, and report | [`3.Evaluate/README.md`](3.Evaluate/README.md) |

---

## 4. The evaluation set

### 4.1 Task definition and annotation scheme

Each item is one plate photograph on which four specimens are marked. Every marked specimen
receives two annotations in a single colour: a **dotted bounding box** delimiting the
individual, and a **solid line** drawn along the elytra dimension in question.

| Option | Colour | BGR |
|---|---|---|
| Beetle A | red | `(0, 0, 255)` |
| Beetle B | blue | `(255, 0, 0)` |
| Beetle C | green | `(0, 255, 0)` |
| Beetle D | white | `(255, 255, 255)` |

The question names all four colours explicitly:

> *Among the four highlighted beetle specimens, which one has the largest elytral
> {length|width}? Beetle A (red dotted box/line), Beetle B (blue dotted box/line), Beetle C
> (green dotted box/line), and Beetle D (white dotted box/line)*

The dotted box and the solid line serve different purposes and both are necessary. The box
establishes *which specimen* each colour refers to on a plate holding ten to thirty
individuals; the line establishes *what is being measured* on that specimen. Drawing the
line alone would leave the option-to-specimen binding ambiguous in dense regions; drawing
the box alone would leave the measurement axis unspecified, since elytra length and width
are different dimensions of the same object.

### 4.2 Why no scale bar is needed

This is the property that makes the task a clean comparison rather than a measurement.

All four specimens lie on **one plate**, photographed at **one magnification**, and each
coloured line is drawn from the corpus's own pixel coordinates **before any resizing**.
Writing $\lambda$ for the plate's pixels-per-centimetre factor and $\ell_i$ for specimen
$i$'s true elytra dimension, the on-screen line length is

$$
L_i^{\text{px}} = \lambda \, \ell_i ,
$$

with $\lambda$ identical across all four. Therefore

$$
\operatorname*{arg\,max}_i \, L_i^{\text{px}} \;=\; \operatorname*{arg\,max}_i \, \ell_i
\qquad \text{exactly, for any } \lambda > 0 .
$$

**The ordering of on-screen lengths is the ordering of real sizes, and any subsequent
uniform resize preserves it.** No scale bar is required, no unit conversion is possible, and
no knowledge of typical beetle dimensions can help — the four candidates are all beetles of
the same family on the same plate. The inference prompt states this explicitly, because a
model that assumed the panels came from different photographs might attempt a scale
correction that does not exist
([Section 6.2 of the inference document](2.Experiment_Runs/README.md#62-prompt-construction)).

### 4.3 The separation margin

A quartet is accepted only when the largest value exceeds every other value by a margin:

$$
\text{accept} \iff \max_i v_i > \frac{\max_j v_j}{\gamma} \text{ for all } i \neq \arg\max,
\qquad
\gamma = \begin{cases} 0.95 & \text{length} \\ 0.90 & \text{width} \end{cases}
$$

equivalently, the runner-up must fall below 95% of the winner on length and below 90% on
width. Ties are rejected outright.

The asymmetry was chosen to equalise yield: elytra width has a narrower absolute spread than
length, so a common margin would have rejected almost every width quartet. Its analytical
consequence is that **width items are strictly easier by construction** — twice the
separation to resolve — which converts the generator's own parameter into a falsifiable
prediction ([Section 6.4](#64-the-margin-asymmetry-inverts)).

### 4.4 Scale and cluster structure

| Quantity | Value |
|---|---:|
| Scored items per model | 3,976 |
| Specimen plates | 207 |
| Items per plate (mean) | 19.2 |
| Maximum question sets per plate | 20 |
| Length items | 3,437 (86%) |
| Width items | 539 (14%) |
| Gold letter shares | A 0.257, B 0.249, C 0.246, D 0.248 |

The answer key is near-uniform over the four options, so plain accuracy is meaningful and
chance is 0.250. This is the one task in Component C3 with no gold skew to correct for, and
it is why the control set targets the *model's* answer distribution rather than the key's
([Section 5](#5-the-control-set)).

Clustering is severe. Up to twenty question sets are drawn from a single plate, **reusing
the same specimens in different combinations**, so rows within a plate share not only their
photograph and magnification but often several of their four candidates. Every confidence
interval therefore resamples plates, not rows
([Section 7 of the evaluation document](3.Evaluate/README.md#7-uncertainty-the-clustered-bootstrap)).

---

## 5. The control set

### 5.1 Chance, balanced accuracy, and kappa

Three statistics, each of which a constant predictor fails.

$$
\text{chance} = 0.250,
\qquad
\mathrm{balAcc} = \frac{1}{|\mathcal{L}|}\sum_{\ell \in \mathcal{L}} \Pr\!\left[\hat{y} = \ell \mid y = \ell\right],
\qquad
\kappa = \frac{p_o - p_e}{1 - p_e} .
$$

**Balanced accuracy** averages recall over the four gold letters. A predictor that always
answers one letter recalls that letter perfectly and the other three not at all, scoring
exactly $1/4 = 0.250$ **whatever the gold distribution happens to be** — the property plain
accuracy lacks.

**Cohen's kappa** corrects for both marginals and scores 0 for a constant predictor.

**`predTop` and `predTopShare`** report the model's own most frequent answer and its share.
This is the first column to read: near 1.00 means a constant predictor and nothing else in
the row is interpretable.

### 5.2 The position-bias index

Accuracy and bias are different quantities, and a model can post respectable accuracy while
being strongly biased. We summarise the bias as the **total variation distance** between the
predicted and gold letter distributions:

$$
\mathrm{posBias} = \tfrac{1}{2} \sum_{\ell \in \{A,B,C,D\}} \bigl| \Pr[\hat{y} = \ell] - \Pr[y = \ell] \bigr| .
$$

It is 0 when the answer mix matches the key's and reaches its maximum of $0.75$ when the
model always answers one letter against a uniform key. Reporting it beside balanced accuracy
separates *how often the model is right* from *how it distributes its answers*, and the full
$4 \times 1$ predicted distribution is written to `relsize_confusion.csv` so the bias can be
inspected letter by letter rather than only as a scalar.

---

## 6. Results

Full tables are written to `3.Evaluate/results/relsize_report.md`.

### 6.1 The leaderboard

Chance is 0.250 for every row.

| Model | balAcc | acc | kappa | posBias | predTop | predTopShare | parse |
|---|---:|---:|---:|---:|:---:|---:|---:|
| **Gemma-4-12B** | **0.443** | 0.442 | **0.257** | **0.133** | C | 0.324 | 1.000 |
| **Qwen3-VL-32B** | **0.443** | 0.443 | **0.257** | 0.258 | C | 0.489 | 1.000 |
| Qwen3-VL-30B-A3B | 0.423 | 0.423 | 0.231 | 0.289 | B | 0.538 | 1.000 |
| InternVL3-38B-Instruct | 0.363 | 0.364 | 0.151 | 0.241 | B | 0.364 | 1.000 |
| InternVL3.5-38B-Instruct | 0.360 | 0.357 | 0.145 | 0.436 | C | 0.529 | 1.000 |
| Molmo2-8B | 0.358 | 0.355 | 0.142 | 0.306 | D | 0.500 | 1.000 |
| Qwen3-VL-8B | 0.345 | 0.346 | 0.127 | 0.208 | A | 0.363 | 1.000 |
| Qwen2.5-VL-7B | 0.337 | 0.334 | 0.115 | 0.363 | C | 0.610 | 1.000 |
| InternVL3.5-30B-A3B | 0.333 | 0.334 | 0.111 | **0.126** | A | 0.339 | 1.000 |
| Llama-3.2-11B-Vision | 0.317 | 0.325 | 0.091 | 0.463 | A | 0.722 | **0.642** |
| InternVL3-8B-Instruct | 0.304 | 0.304 | 0.073 | 0.301 | B | 0.479 | 1.000 |
| Pixtral-12B | 0.304 | 0.309 | 0.073 | 0.437 | A | 0.694 | 1.000 |
| Qwen2.5-VL-3B | 0.300 | 0.304 | 0.068 | 0.412 | A | 0.627 | 1.000 |
| InternVL3.5-8B-Instruct | 0.299 | 0.303 | 0.066 | 0.473 | A | 0.575 | 1.000 |
| Gemma-3-12B | 0.280 | 0.283 | 0.041 | 0.390 | A | 0.529 | 1.000 |
| Phi-3.5-Vision | 0.256 | 0.262 | 0.008 | 0.625 | A | 0.882 | 1.000 |
| LLaVA-NeXT-Vicuna-7B | **0.250** | 0.257 | **0.000** | **0.743** | A | **1.000** | 1.000 |
| LLaVA-NeXT-Mistral-7B | **0.250** | 0.257 | **0.000** | **0.743** | A | **1.000** | 1.000 |

The two LLaVA-NeXT models answer **A on every one of 3,976 items**. Their accuracy of 0.257
is exactly the gold share of A, their balanced accuracy is exactly chance, their kappa is
exactly zero, and their position bias is 0.743 against a theoretical maximum of 0.750. This
is the constant-predictor failure mode in its pure form, and the control set identifies it
unambiguously from three independent directions.

### 6.2 A narrow band above chance

$$
\text{chance} = 0.250, \qquad \text{best} = 0.443 \;(+0.193), \qquad \text{worst} = 0.250 \;(+0.000).
$$

The entire measurable band is **0.193 wide**, and three models sit within 0.01 of chance.
For a four-way forced choice in which the correct answer is directly visible — four coloured
lines in one photograph, no scale conversion, no unit, no prior knowledge in play — this is
a strikingly low ceiling.

The task admits no confound that would explain it away. The answer key is near-uniform by
construction, so there is no skew to exploit and no floor above chance to clear. There is no
reference measurement to mis-calibrate, unlike
[Absolute Distance](../Distance_Absolute/README.md#64-ranking-and-calibration-dissociate),
where several models were shown to rank correctly while scaling wrongly. And the separation
margin guarantees a resolvable difference in every item
([Section 4.3](#43-the-separation-margin)). What remains is the comparison itself.

### 6.3 Systematic avoidance of the last option

The gold letter shares are near-uniform: A 0.257, B 0.249, C 0.246, D 0.248. The models'
predicted shares are not.

$$
\overline{\Pr}[\hat{y} = A] = 0.441 \;\;(\text{gold } 0.257),
\qquad
\overline{\Pr}[\hat{y} = D] = 0.077 \;\;(\text{gold } 0.248).
$$

The last option is under-selected by a factor of **3.2×** and the first over-selected by
**1.7×**. The median ratio $\Pr[\hat{y}=A]/\Pr[\hat{y}=D]$ across models is **15.4**.

| Predicted share of D | Models |
|---|---:|
| exactly 0.000 | **6** |
| below 0.05 | **12 / 18** |
| below 0.10 | **14 / 18** |
| at or above the gold share of 0.248 | **2** |

Six models — Llama-3.2-11B-Vision, InternVL3.5-8B, both LLaVA-NeXT variants, Pixtral-12B,
and Phi-3.5-Vision — never answer D on any of 3,976 items, despite D being correct on
roughly a quarter of them. Several others answer it on under 1%: InternVL3-38B at 0.007,
InternVL3-8B at 0.009, InternVL3.5-38B at 0.014.

This is not a property of the images. Option D is white, the option letters are enumerated
in a fixed order in the question, and the winner's identity is assigned by the quartet
sample rather than by position, so nothing in the visual stimulus distinguishes D from the
others. The effect is a property of the **answer-slot ordering in the prompt**, and it
supplies quantitative confirmation of the first-option preference that motivated the control
([Section 2.2](#22-position-bias-as-a-pre-registered-control)).

Two models invert the pattern rather than escaping it: Molmo2-8B answers D on 50.0% of items
and A on 3.5%. A bias toward the last option is the same defect with the opposite sign, and
`posBias` scores it accordingly at 0.306.

### 6.4 The margin asymmetry inverts

The generator requires the winner to exceed the runner-up by 5% on length but **10% on
width** ([Section 4.3](#43-the-separation-margin)). Width items therefore carry twice the
separation and are strictly easier to resolve. Under any account in which the model compares
line lengths, width accuracy must exceed length accuracy.

$$
\text{width} > \text{length for only } \mathbf{5/18} \text{ models}; \qquad
\overline{\Delta}_{\text{width}-\text{length}} = -0.013 .
$$

Thirteen of eighteen models do **better on the harder stratum**. The inversion is largest in
exactly the models that are otherwise strongest: InternVL3.5-38B is 0.071 worse on width,
Qwen3-VL-32B 0.065 worse, Molmo2-8B 0.061 worse. Only Qwen3-VL-8B shows the predicted
direction at any magnitude (+0.071).

The width stratum is the smaller of the two (539 items against 3,437), so an individual
model's gap carries wide uncertainty. The reliable read is the group-level statistic: across
eighteen independent architectures, a construction-guaranteed doubling of the discriminative
margin produces **no aggregate improvement**. If models were resolving the four line lengths
and reporting the maximum, doubling the gap between first and second place could not fail to
help.

This is the component's most direct evidence about mechanism, and it costs no additional
annotation — the check was latent in a generator parameter chosen for an unrelated reason
([Section 2.3](#23-the-margin-asymmetry-as-an-internal-validity-check)).

### 6.5 Position bias predicts accuracy

The two controls are not independent, and their relationship is itself a finding:

$$
\rho\!\left(\mathrm{posBias},\, \mathrm{balAcc}\right) = -0.768,
\qquad
\rho\!\left(\mathrm{predTopShare},\, \mathrm{balAcc}\right) = -0.702 .
$$

Models that distribute their answers closest to the key perform best, and the relationship is
monotone across nearly the whole leaderboard. The two least biased models —
InternVL3.5-30B-A3B at 0.126 and Gemma-4-12B at 0.133 — include the joint leader; the two
most biased at 0.743 are the two constant predictors.

Gemma-4-12B is the informative case. It attains the top balanced accuracy at 0.443 *and* the
second-lowest position bias at 0.133, with a predicted-letter distribution
(A 0.164, B 0.304, C 0.324, D 0.208) that is the closest to uniform in the set. Qwen3-VL-32B
matches its accuracy exactly at 0.443 while carrying twice the bias at 0.258 and answering C
on 48.9% of items. **Identical accuracy, materially different answer behaviour** — which is
precisely why `posBias` is reported beside `balAcc` rather than folded into it.

### 6.6 Scale

$$
\rho\!\left(\text{parameters},\, \mathrm{balAcc}\right) = 0.684 .
$$

Positive and moderately strong, with systematic exceptions. A 12B Gemma-4 ties a 32B Qwen3-VL
for the lead; a 12B Gemma-3 (0.280) falls below a 3B Qwen2.5-VL (0.300); an 8B InternVL3.5
(0.299) trails an 8B InternVL3 (0.304) and both trail their 38B siblings by only 0.06. The
top three positions are held by 12B, 32B, and 30B models respectively, and the fourth by a
38B — capability here is not ordered by parameter count within the upper half of the table.

Parse rate is 1.000 for seventeen of eighteen models. Llama-3.2-11B-Vision reaches only
0.642, and its scored subset is correspondingly reduced to 2,553 items across 204 plates —
which the evaluation handles by recomputing every statistic on that subset rather than
comparing it against figures derived from the full set.

---

## 7. Scope of the released evaluation

The task isolates comparison from calibration, and the control set establishes that most
models are neither comparing reliably nor distributing their answers neutrally. Two
extensions would sharpen the mechanism further.

An **annotation-only control** is computable with a one-line change to the generator:
replace the plate photograph with flat noise while leaving the four coloured lines and boxes
exactly where they are. The question remains fully answerable from the annotation geometry
alone, so the gap between that condition and the real images measures how much of the score
derives from the specimens rather than from the drawing.

An **option-permutation control** would separate the position effect from the visual one
directly. Re-rendering each quartet with the colour-to-specimen assignment permuted, while
holding the image and the correct answer's visual identity fixed, converts the aggregate
bias reported in [Section 6.3](#63-systematic-avoidance-of-the-last-option) into a per-item
causal estimate. Both the quartet sampling and the colour map are seeded, so the permuted
set reproduces exactly.

---

## 8. Reproduction

```bash
# 1. Sample quartets and render annotations.
python 1.Generate_QA/generateQA.py \
    --base_dir        $BIOVQA_DATA/datasets/Beetles-500 \
    --output_base_dir $BIOVQA_DATA/tasks/Distance-Relative/Hard \
    --seed 42

# 2. Flatten, derive plate_id and measure_type, shuffle, shard.
python 2.Experiment_Runs/Preprocess/split_qa.py \
    --task_root $BIOVQA_DATA/tasks/Distance-Relative/Hard \
    --run_dir   2.Experiment_Runs \
    --n_chunks 6 --seed 42

# 3. Key balance, cluster diagnostics, runtime projection.
python 2.Experiment_Runs/Preprocess/stats.py --run_dir 2.Experiment_Runs

# 4. Inference — 18 models x 6 shards.
python 2.Experiment_Runs/qwenVL2_5_7B.py --qa_csv 2.Experiment_Runs/chunks/chunk00/qa.csv

# 5. Merge shards, then score.
python 3.Evaluate/merge_results.py --run_dir 2.Experiment_Runs --out_dir 3.Evaluate/results
python 3.Evaluate/evaluate.py \
    --results_dir 3.Evaluate/results --out_dir 3.Evaluate/results --bootstrap 2000
```

Step 4 comprises 108 independent jobs, executed as SLURM arrays on an H100 partition.

---

## 9. Outputs

| File | Contents |
|---|---|
| `3.Evaluate/results/relsize_overall.csv` | One row per model: `balAcc`, `acc`, `kappa`, `posBias`, `predTop`, `predTopShare`, `plates`, `parse_rate`, bootstrap bounds. |
| `3.Evaluate/results/relsize_by_group.csv` | One row per model × measurement type — the stratum carrying the margin check. |
| `3.Evaluate/results/relsize_confusion.csv` | Gold and predicted share of each option letter, per model. |
| `3.Evaluate/results/relsize_report.md` | Leaderboard, per-measurement table, answer-distribution table. |
| `chunks/chunkNN/qa_<KEY>.summary.json` | Per-shard run record including the verbatim prompt and vision budget. |

---

## 10. References

Annotation rendering uses OpenCV [1]. Chance-corrected agreement follows Cohen [2].
Clustered resampling for dependent observations follows standard bootstrap practice [3, 4].
The source corpus is the NEON invertebrate specimen collection [5].

[1] G. Bradski. **The OpenCV Library.** *Dr. Dobb's Journal of Software Tools*, 2000.

[2] J. Cohen. **A Coefficient of Agreement for Nominal Scales.** *Educational and
Psychological Measurement*, 20(1):37–46, 1960.

[3] B. Efron and R. J. Tibshirani. **An Introduction to the Bootstrap.** Chapman & Hall, 1993.

[4] A. C. Cameron and D. L. Miller. **A Practitioner's Guide to Cluster-Robust Inference.**
*Journal of Human Resources*, 50(2):317–372, 2015.

[5] National Ecological Observatory Network (NEON). **Ground beetles sampled from pitfall
traps (DP1.10022.001).** NEON Data Portal.