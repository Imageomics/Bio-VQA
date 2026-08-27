<h1 align="center"> Bio-VQA &rarr; C2 &rarr; Fine-Grained Attribute Grounding </h1>

<p align="center">
  <b>Four question categories on annotated birds, scored by a chance-adjusted graded score</b>
</p>

<p align="center">
  <img alt="Questions" src="https://img.shields.io/badge/questions-824%2C104-green">
  <img alt="Images" src="https://img.shields.io/badge/images-23%2C576-blue">
  <img alt="Source" src="https://img.shields.io/badge/source-CUB--200--2011-informational">
  <img alt="Metric" src="https://img.shields.io/badge/metric-CAGS-orange">
  <img alt="Validation" src="https://img.shields.io/badge/validated-3%20annotators-purple">
  <img alt="Range" src="https://img.shields.io/badge/CAGS%20range-.378%E2%80%93.593-yellow">
</p>

---

## Overview

Fine-grained attributes and morphological traits are central to trait-based
ecology. Recognising the traits of an organism is related to but distinct from
naming its taxon, and **C2 asks how well VLMs recognise fine-grained visual
attributes** — colour, pattern and shape.

CUB-200-2011 provides **312 binary attributes per image**, each named as a part,
a property and a value (`has_wing_color::brown`). Evaluating a model on 312
sparse yes/no questions is uninformative: most flags are absent, so a constant
predictor answering *no* scores well.

**The fix is to invert the mapping.** The part-and-property structure is parsed
back out of the attribute names, a part and a property are combined into a single
question (*wing colour*), and the marked value becomes the ground truth
(*brown*).

> **Design contract:** distractors are drawn from **other values attested for
> that same part and property within the dataset**, so every option is a real
> trait of some bird. Any value that is itself true of the image is excluded, so
> when a part carries several attested values **exactly one option is correct**.
> A model must ground its answer rather than eliminate.

---

## Table of Contents

- [Four Question Categories](#four-question-categories)
- [Handling Annotator Certainty](#handling-annotator-certainty)
- [Anatomical Localization: Two Controls](#anatomical-localization-two-controls)
- [The Metric: CAGS](#the-metric-cags)
- [Metric Validation](#metric-validation)
- [Results](#results)
- [What the Results Imply](#what-the-results-imply)
- [Limitations](#limitations)
- [FAQ](#faq)
- [Citation](#citation)

---

## Four Question Categories

```
   ┌────────────────────────────────────────────────────────────────────┐
   │  ONE annotated bird image  ──▶  four categories of question        │
   └────────────────────────────────────────────────────────────────────┘

   (a) MORPHOLOGICAL ATTRIBUTE       "What colour is the breast?"
       part + property → value        distractors: other attested breast colours

   (b) RELATIVE SALIENCE             "Which of these two parts is more
       two present traits             prominent?"

   (c) ANATOMICAL LOCALIZATION       "Which body part does the marker sit on?"
       VISUALLY PROMPTED              refers to a RENDERED MARKER, not a name

   (d) TAXONOMY & NAMING             open-ended species question
       the control category           tests naming rather than grounding
```

Because **each of the 11,788 source images is rendered** with a marker for
category (c), the component comprises **23,576 images** and **824,104
questions**.

Category (c) is the only visually-prompted question family in Bio-VQA: the
question refers to a rendered marker rather than to a named part, so a model
cannot answer from category knowledge alone.

---

## Handling Annotator Certainty

Each CUB attribute label carries a four-level annotator certainty — *not
visible*, *guessing*, *probably*, *definitely* — which is **retained alongside
the binary presence flag rather than collapsed into the question text**.

| Question category | Certainty handling |
| :--- | :--- |
| Morphological attribute | Condition on presence; the marked value is the ground truth |
| Relative salience | Restrict candidates to attributes marked present, prioritising *definitely* |
| Anatomical localization | Part keypoints carry **no** certainty field — filtered by their binary visibility flag instead |

---

## Anatomical Localization: Two Controls

A rendered marker is placed on the image and the model is asked which body part
it sits on. Two distractor markers are added:

| Distractor | Shares with the target | Does not share |
| :--- | :--- | :--- |
| **1** | colour | shape |
| **2** | shape | colour |

> **Neither cue identifies the marker alone.** A model must bind colour and shape
> together to answer. This is the construction that makes category (c) a
> grounding test rather than a colour-matching test.

---

## The Metric: CAGS

Two problems arise in C2, and a raw graded score fails both.

**Answer spaces have structure.** If a wing is brown, answering *red* is visually
closer than answering *green*.

**Answer spaces have uneven cardinality.** In the CUB attribute set there are
3 options for bill length, 5 for wing shape, and **15 for colours**. A raw graded
score confounds model ability with the dimensionality of the option set.

The **Chance-Adjusted Graded Score** handles both. Let $s(\hat{y}, y) \in [0,1]$
measure similarity between two answers, and $\mathbb{E}_0[s]$ the expected score
of a random guess:

$$
\mathrm{CAGS} = \frac{\mathbb{E}\bigl[s(\hat{y}, y)\bigr] - \mathbb{E}_{0}[s]}
                     {1 - \mathbb{E}_{0}[s]}
$$

| Behaviour | CAGS |
| :--- | :---: |
| random guess | **0** |
| perfect answer | **1** |

**regardless of the option count.**

### Where the similarity comes from

| Answer type | Similarity kernel |
| :--- | :--- |
| colour | **CIEDE2000** perceptual distance |
| part | **co-occurrence frequencies** in the annotations |

---

## Metric Validation

CAGS could in principle produce the model ordering by itself, so it is checked
two ways.

| Check | Result |
| :--- | :--- |
| **20-configuration sensitivity sweep** over kernel and hyperparameter settings | Model ranking almost unchanged, $\rho \ge .9959$ |
| **Three independent annotators** scoring a stratified sample | Agree with CAGS at **.908** of the rate at which they agree with each other |

> The first shows the ranking follows the models rather than the kernel. The
> second gives a human agreement ceiling to compare CAGS against.

---

## Results

**C2 is the only component that spans a wide range without saturating** — CAGS
from **.378 to .593**.

### Models see parts but confuse sides

C2 separates models that ground answers in the image from models that answer from
category knowledge. **Molmo 2-8B is the clearest case:**

| Model | Anatomical localization | Taxonomy & naming |
| :--- | ---: | ---: |
| Qwen3-VL-32B | .4754 | — |
| **Molmo 2-8B** | **.4741** *(statistically tied)* | **.3301** *(15th of 18)* |

An 8B model with pointing supervision **matches a 32B at finding body parts and
sits near the bottom at naming what it is looking at**. These are different
abilities, and a single score hides that.

### The sharpest failure is laterality

Conditional on **naming the correct part**:

| Finding | Value |
| :--- | ---: |
| Models at or below chance at telling left from right | **17 of 18** |
| Models significantly below chance | **9** |
| Mirror-confusion rate range | **21.8% – 54.0%** |
| Uniform baseline | 20% |
| Correlation with overall model strength | $\rho = .75$ |
| Molmo 2-8B, the only exception | **.689** |

> Stronger models are **better at eliminating the wrong part and no better at
> picking the correct side**, so the mirrored distractor is what they land on.
> The positive correlation with model strength is the counter-intuitive result:
> improving a model makes this particular error *more* likely, not less.

Laterality loads **lowest of any C2 task on the shared cross-component factor
(.154)**, which is consistent with it measuring something the rest of the
benchmark does not.

---

## What the Results Imply

**Grounding and naming are separable, and parameter count does not predict
both.** Molmo 2-8B moves 11 of 18 places between components — from the 3rd
percentile on C1 to the 79th on C2 — the largest movement of any model in the
benchmark. Pointing supervision buys grounding without buying taxonomy.

**Left/right confusion is a systematic failure, not noise.** It survives
conditioning on correct part identification, it is significantly below chance for
half the models, and it *increases* with model strength. That combination rules
out both "the model cannot find the part" and "the model is guessing".

**C2 correlates with C3 at .927, disattenuating to 1.015.** At the reliability
measurable here, fine-grained attribute grounding and quantitative spatial
reasoning are **one construct**. This is reported rather than argued around: the
two components are separable by task content but not, at this sample size, by
measured model behaviour.

---

## Limitations

| Limitation | Effect |
| :--- | :--- |
| **Single dataset, single clade** | CUB-200-2011 is birds only. Attribute grounding on other body plans is untested. |
| **Attribute vocabulary is CUB's** | Colour, pattern and shape values follow the original annotation scheme, including its granularity choices. |
| **CAGS depends on a kernel choice** | The 20-configuration sweep shows the ranking is stable ($\rho \ge .9959$), but the absolute values are kernel-dependent. |
| **Human ceiling is three annotators** | .908 agreement ratio is measured on a stratified sample, not the full corpus. |
| **Not separable from C3** | Disattenuated correlation of 1.015 means the two components cannot be distinguished at this reliability. |


<p align="center">
  <sub>Part of the <b>Bio-VQA</b> benchmark &middot; Component 2</sub>
</p>