<h1 align="center"> Bio-VQA &rarr; C3 &rarr; Quantitative Spatial Reasoning </h1>

<p align="center">
  <b>Seven atomic spatial tasks plus a non-spatial negative control, bounded below by image-blind floors</b>
</p>

<p align="center">
  <img alt="Questions" src="https://img.shields.io/badge/questions-141%2C516-green">
  <img alt="Images" src="https://img.shields.io/badge/images-78%2C139-blue">
  <img alt="Tasks" src="https://img.shields.io/badge/tasks-7%20%2B%201%20control-orange">
  <img alt="Datasets" src="https://img.shields.io/badge/source%20datasets-5-yellow">
  <img alt="Floor" src="https://img.shields.io/badge/bounded-image--blind%20floor-red">
  <img alt="Coherence" src="https://img.shields.io/badge/mean%20%CF%81-%2B0.703-purple">
</p>

---

## Overview

Spatial reasoning — counting, localization, measurement — is widely studied in
general VQA and is the most common operation in biodiversity fieldwork. **C3 asks
how well VLMs perform these tasks on biological subjects.**

Following the task-decomposition approach of AVA-Bench, spatial reasoning is
broken into **seven atomic tasks**, so a failure is attributed to a specific
ability rather than to spatial reasoning in general. An **eighth task, OCR, is
included as a negative control**.

| # | Task | Question | Items |
| :---: | :--- | :--- | ---: |
| 1 | Counting | *How many animals?* | 53,498 |
| 2 | Viewpoint | *Which way is it facing?* | 13,000 |
| 3 | Absolute distance | *How long is this trait, in cm?* | 14,108 |
| 4 | Relative distance | *Which specimen is larger?* | 3,976 |
| 5 | Relative counting | *Which panel has the most / fewest?* | 42,412 |
| 6 | Localization | *Where is the animal? Give a box.* | 3,988 |
| 7 | Relative position | *Where is the red one relative to the blue one?* | 1,000 |
| — | **OCR** *(negative control)* | *Transcribe the label.* | 522 |

> **Design contract:** every task is bounded below by an **image-blind floor** —
> the highest score a rule that never looks at the image can achieve. Two tasks
> fall **below** their floor, which no accuracy-only report would reveal.

---

## Table of Contents

- [Source Datasets](#source-datasets)
- [Why Each Task Runs Where It Does](#why-each-task-runs-where-it-does)
- [Metrics and the Floor](#metrics-and-the-floor)
- [The OCR Negative Control](#the-ocr-negative-control)
- [Results](#results)
- [Internal Coherence](#internal-coherence)
- [What the Results Imply](#what-the-results-imply)
- [Limitations](#limitations)
- [FAQ](#faq)
- [Citation](#citation)

---

## Source Datasets

Five ecological corpora, aggregated into **ten 500-image subsets**:

| Dataset | Content |
| :--- | :--- |
| WAID | Aerial wildlife — camel, cattle, kiang, seal, sheep, zebra |
| Boxes on Bees and Pollen | Hive frames, dense |
| 2018-NEON-beetles | Specimen trays with scalebars |
| DePondFi | Pond fish |
| GZCD | Grevy's zebra |

Selected on two bases: they provide **multi-organism images** required for
counting and relative position, and prior biological VQA works on
single-specimen imagery.

Subsets are built under two constraints:

| Constraint | Purpose |
| :--- | :--- |
| **Count stratification** | A balanced mix of sparse and dense scenes |
| **Verified annotation completeness** | Scoring a count against a partial annotation penalises a *correct* prediction |

---

## Why Each Task Runs Where It Does

Each task runs **only on subsets where the ground truth exists**.

| Task | Runs on | Reason |
| :--- | :--- | :--- |
| Absolute & relative distance | beetles | The only corpus with recorded trait and scalebar measurements |
| Relative position | beetles | Specimens lie flat in a lattice at a single focal depth, so *left of* and *above* are genuine 2-D structure rather than camera-perspective artefacts |
| **Localization & viewpoint** | **C4 wildlife catalogues** | See below |

### The task-isolation problem

In a crowded image, *"which way is the leftmost zebra facing?"* forces the model
to **locate** the target before it can determine the **angle** — breaking task
isolation, which is the entire point of the decomposition.

**Solution:** localization and viewpoint run on the C4 wildlife catalogues, where
a single subject dominates the frame and annotations explicitly record its
bounding box and viewing angle. This isolates the two abilities **without
introducing additional datasets**, and has the side benefit of making
image-matched C3 × C4 correlation possible.

---

## Metrics and the Floor

Exact match is reported for every task. Graded metrics quantify *how wrong* a
wrong answer is.

**Counting** uses bounded count agreement — predicting 14 when the truth is 15
shows stronger spatial reasoning than predicting 10 or 20:

$$
\mathrm{CAS}(\hat{n}, n) = \max\!\Bigl(0,\; 1 - \frac{|\hat{n} - n|}{n}\Bigr)
$$

Macro-averaged over source corpora, since the count distribution is not uniform
across datasets.

| Task | Metric |
| :--- | :--- |
| Counting, relative counting | CAS; balanced accuracy over four options |
| Localization | mean IoU against the gold box |
| Viewpoint | balanced accuracy over ten orientations, plus a chance-corrected agreement score |
| Relative distance | balanced accuracy, four options |
| Relative position | balanced accuracy, eight options |

### The image-blind floor

$\phi$ is the highest score an image-blind rule $g_0$ can reach using simple
strategies — the modal count, a per-type constant, the majority class:

$$
\phi = \max_{g_0 \in \mathcal{G}_0} \mathbb{E}\bigl[S(g_0, y)\bigr],
\qquad
\mathrm{lift} = \frac{S - \phi}{1 - \phi}
$$

> **A lift of 0 means the model does no better than an image-blind guess.
> A negative lift means it does worse.**

---

## The OCR Negative Control

Locating a specimen label is a spatial act, but the answer depends on
**transcribing characters** — no position, extent, direction or magnitude. OCR is
therefore run on the same imagery, the same models and the same pipeline, and
held **out of the C3 composite**.

The control answers the obvious objection to any coherence result:

> If the seven tasks agree merely because larger models are better at everything,
> then **any** eighth vision task should agree too.

| | Seven spatial | Plus the control |
| :--- | ---: | ---: |
| Mean inter-task $\rho$ | **+0.703** | +0.619 |
| Krippendorff $\alpha$ | **0.940** | 0.921 |
| Variance on PC1 | **75%** | 68% |
| Mean $\rho$ vs $\log$(params) | **+0.588** | **+0.028** |

OCR correlates **+0.366** with the seven, against **+0.703** among themselves.
Every spatial task individually beats its correlation with the control.

> **The last row is the argument.** The seven spatial tasks track parameter count
> at +0.588; the control tracks it at **+0.028**. The coherence is therefore not
> a general capability gradient — if it were, a task on the same imagery with the
> same models would ride the same gradient, and it does not.

The exclusion is decided on **construct grounds before any correlation is seen**,
which is what makes this a control rather than a dropped task.

---

## Results

### Counting is the clearest number

**No model exceeds .242 exact match.** The same model families answer **55–70%**
of CV-Bench's counting questions correctly on everyday objects — roughly a
**3× drop** when the objects are organisms.

> The comparison is indicative rather than controlled: their format is
> five-option multiple choice over small counts, ours is open numeric over dense
> scenes of near-identical animals. But that regime is exactly where counting is
> known to fail.

### Two tasks fall below their floor

| Task | Floor | Models below it |
| :--- | :--- | ---: |
| Viewpoint | majority class, .520 | **17 of 18** |
| Absolute distance | per-type constant | **15 of 18** |

The predictions on absolute distance are **well-formed numbers in the right units
with far too little spread**: models settle on one size per specimen type instead
of measuring.

> A benchmark reporting only accuracy would record these as partial competence.

### Splitting one image into two panels costs every model accuracy

| Contrast | Effect | Holds for |
| :--- | ---: | ---: |
| easy &rarr; medium | **+.064** ($\sigma = .037$) | **18 of 18** |
| medium &rarr; hard | −.005 ($\sigma = .027$) | 6 of 18 |

Halving the area a target occupies is enough to break counting, whichever model
does it. Drawing both panels from the same corpus makes no further difference.

### Per-task headline

| Task | Best model | Score | Floor / baseline |
| :--- | :--- | ---: | :--- |
| Counting | InternVL3-38B | macroCAS .799 | modal-count floor |
| Viewpoint | Qwen3-VL-32B | $\kappa$VAS .358 | 17/18 below majority .520 |
| Absolute distance | Qwen3-VL-32B | $\rho_{\text{within}}$ .858 | 15/18 below per-type constant |
| Relative distance | Gemma-4-12B | balAcc .443 | chance .250 |
| Relative counting | Qwen3-VL-32B | lift .403 | chance .250 (shortcut-free condition) |
| Localization | Qwen3-VL-32B | meanIoU .866 | mean-box .415 |
| Relative position | Qwen3-VL-32B | lift +.425 | chance .125 |
| *OCR (control)* | *Qwen3-VL-30B* | *suffixCharAcc .987* | *constant-string .349* |

---

## Internal Coherence

The seven spatial tasks cohere:

| Statistic | Value |
| :--- | ---: |
| Mean inter-task $\rho$ | **+0.703** |
| Range | +0.351 to +0.909 |
| Krippendorff $\alpha$ | **0.940** |
| Variance on PC1 | **75%** |

Tightest pairs: viewpoint × relative counting (+0.909), viewpoint × relative
position (+0.887) — both spatial-relation judgements.

Within the cross-component analysis, **relative position (.168) and localization
(.176) load lowest on the shared factor**, which is consistent with them
measuring something the rest of the benchmark does not.

---

## What the Results Imply

**Spatial competence on biological subjects is far below spatial competence on
everyday objects**, in the same models, at roughly 3× on counting. The
decomposition attributes this: it is not one weak ability but a consistent
deficit across seven of them.

**Floors are load-bearing.** Two of seven tasks sit below what a rule that never
looks at the image achieves. On absolute distance the failure has a specific
shape — correct units, correct format, no spread — which is invisible in an
accuracy number and diagnostic once the floor is computed.

**The composite measures space, not model quality.** The negative control
establishes this directly: OCR shares imagery, models and pipeline, and does not
join the factor. Its near-zero correlation with parameter count (+0.028), against
+0.588 for the seven, is the sharpest scaling result in the component.

---

## Limitations

| Limitation | Effect |
| :--- | :--- |
| **Localization is Easy-tier only** | 3,988 of the 13,000 three-tier corpus have been evaluated. See the note below. |
| **Relative position is one dataset, one condition** | Beetles-500, `distractor` only. No clean→distractor contrast. |
| **Some tasks run on one corpus** | Distance and relative position run on beetles alone, since that is where the ground truth exists. |
| **Counting comparison is indicative** | CV-Bench uses five-option MCQ over small counts; C3 uses open numeric over dense scenes. |
| **Not separable from C2** | Disattenuated correlation of 1.015 — at this reliability the two components are one construct. |

> ⚠️ **Item-count reconciliation.** The eight evaluated tasks sum to **132,504**
> items. The stated component total of **141,516** counts all three Localization
> tiers (13,000) rather than the Easy tier alone (3,988). The difference is
> exactly **9,012**. Either the component figure becomes ~132,516, or the Medium
> and Hard localization tiers need a sweep.


<p align="center">
  <sub>Part of the <b>Bio-VQA</b> benchmark &middot; Component 3</sub>
</p>