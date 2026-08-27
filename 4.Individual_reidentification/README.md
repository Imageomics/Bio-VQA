<h1 align="center"> Bio-VQA &rarr; C4 &rarr; Individual Re-identification </h1>

<p align="center">
  <b>Balanced same/not-same verification at three difficulty levels, scored by signal detection theory</b>
</p>

<p align="center">
  <img alt="Pairs" src="https://img.shields.io/badge/pairs-15%2C000-green">
  <img alt="Catalogues" src="https://img.shields.io/badge/catalogues-5-blue">
  <img alt="Conditions" src="https://img.shields.io/badge/conditions-easy%20%7C%20random%20%7C%20hard-orange">
  <img alt="Metric" src="https://img.shields.io/badge/metric-d%E2%80%B2%20%2B%20criterion%20c-yellow">
  <img alt="Mining" src="https://img.shields.io/badge/pair%20mining-DINOv2-purple">
  <img alt="Specialist" src="https://img.shields.io/badge/specialist-MegaDescriptor-informational">
</p>

---

## Overview

Animal re-identification is a highly specialised task, handled by purpose-built
matching models and field experts. **We are not aware of a benchmark that asks it
of a generalist VLM.**

The question C4 asks is narrower and sharper than "can a VLM re-identify
animals": it asks **whether a VLM attends to identity — a zebra's stripe pattern,
a leopard's spots — or to appearance similarity.**

The probe is minimal: two side-by-side images, one binary question.

> *Are these two photographs of the same individual?*

> **Design contract:** difficulty is set by **DINOv2 embedding similarity**, and
> the hard condition is **adversarial by construction** — its pairs agree on
> everything a general encoder measures and differ only in the markings that
> identify the animal. Appearance points to the *wrong* answer.

---

## Table of Contents

- [Source Catalogues](#source-catalogues)
- [Three Difficulty Conditions](#three-difficulty-conditions)
- [Why Retrieval Is Viewpoint-Constrained](#why-retrieval-is-viewpoint-constrained)
- [Metrics](#metrics)
- [Results](#results)
- [The Manipulation Check](#the-manipulation-check)
- [What the Results Imply](#what-the-results-imply)
- [Limitations](#limitations)
- [FAQ](#faq)
- [Citation](#citation)

---

## Source Catalogues

| Catalogue | Species | Identity signal |
| :--- | :--- | :--- |
| WhaleSharkID | whale shark | spot pattern |
| BelugaID | beluga whale | scarring, dorsal ridge |
| HyenaID | spotted hyena | spot pattern |
| LeopardID | leopard | rosette pattern |
| GZCD | Grevy's zebra | stripe pattern |

Each contributes **1,000 balanced pairs per condition** — 15,000 pairs in total.
Same-individual pairs are sampled **first**, since the catalogues hold fewer
repeat sightings than distinct animals.

These are the same catalogues that supply C3's localization and viewpoint tasks,
which makes image-matched C3 × C4 correlation available.

---

## Three Difficulty Conditions

Difficulty is set by DINOv2 embedding similarity, and the three conditions place
appearance and identity in three different relationships.

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │  EASY      appearance AGREES with identity                          │
   │            same    → an animal with its NEAREST match               │
   │                      (often frames taken seconds apart)             │
   │            diff    → animals that look nothing alike                │
   ├─────────────────────────────────────────────────────────────────────┤
   │  RANDOM    appearance carries NO signal                             │
   │            both classes sampled uniformly                           │
   ├─────────────────────────────────────────────────────────────────────┤
   │  HARD      appearance DISAGREES with identity     ← ADVERSARIAL     │
   │            diff    → drawn from each other's NEAREST neighbours     │
   │            same    → drawn from its FURTHEST                        │
   └─────────────────────────────────────────────────────────────────────┘
```

> The hard condition is adversarial **by construction**. Its pairs agree on
> everything a general encoder measures — pose, lighting, background, coat tone —
> and differ only in the markings that identify the animal. **A model reading
> resemblance rather than identity will score below chance**, and the direction
> of that failure is the measurement.

---

## Why Retrieval Is Viewpoint-Constrained

Retrieval runs **within viewpoint groups**, not across the whole catalogue.

**Viewpoint dominates the embedding space.** Without the constraint, an anchor's
most dissimilar neighbours would be almost entirely the opposite viewing angle —
so the "hard same" pairs would be a left-flank photograph paired with a
right-flank photograph, and the task would degenerate into viewpoint matching
rather than identity discrimination.

Constraining retrieval to a viewpoint group forces the embedding to select on
something other than pose, which is what makes the hard condition test what it is
meant to test.

---

## Metrics

C4 is a **verification task in the sense established by Labeled Faces in the
Wild**: balanced same and not-same pairs, scored by verification accuracy.
Accuracy is reported on that basis.

**But a VLM returns a yes/no decision rather than a similarity score**, so each
model sits at **one operating point**, and accuracy conflates discrimination with
bias: *a model answering yes to every pair scores 50%.*

Accuracy is therefore decomposed using signal detection theory. Writing $H$ for
the rate at which a model says *yes* when the pair is the same, and $F$ for the
rate when it is not:

$$
d' = z(H) - z(F),
\qquad
c = -\tfrac{1}{2}\bigl[z(H) + z(F)\bigr]
$$

where $z$ is the inverse normal CDF.

| Quantity | Measures |
| :--- | :--- |
| $d'$ | **how well** a model separates the two classes |
| $c$ | **which way** it leans |

A model that discriminates and one that simply answers one way no longer receive
the same score.

### The log-linear correction

If a model always answers *yes* or always *no*, $H$ or $F$ reaches 0 or 1, $z$
diverges, and $d'$ becomes infinite. The **standard log-linear correction** is
applied before computing both $d'$ and $c$.

**Reported:** accuracy, $d'$ and $c$ with 95% bootstrap confidence intervals over
2,000 item resamples of each model–dataset–condition cell. Balanced accuracy,
Matthews correlation and Cohen's $\kappa$ are given in the appendix.

**Specialist baseline: MegaDescriptor.**

---

## Results

### Verification is close to unsolved before any manipulation

| Condition | Best $d'$ | Approx. balanced accuracy |
| :--- | ---: | ---: |
| **random** | **.696** | ~64% |

**No model reaches $d' = 1$.**

### The adversarial condition answers the question it was built for

| Finding | Value |
| :--- | ---: |
| Models with **negative** $d'$ under hard pairing | **17 of 18** |
| Range of negative $d'$ | **−.932 to −.091** |
| The exception | answers the same way to everything |

> **Scoring below chance is not deliberate error**, which would require
> identifying the right answer in order to invert it. It means the model is
> following a cue **that we inverted**.

Once two different animals are matched on pose, lighting and background, **the
pairs that look most alike are the different ones** — and models keep calling
them the same. They are reading resemblance, not the markings that carry
identity.

### Response bias moves the same way

Criterion $c$ is **negative for 17 of 18 models** and grows **more negative as
pairs get harder**. Models answer *yes* more often as the task gets harder, not
less — which is the opposite of what a model tracking its own uncertainty would
do.

---

## The Manipulation Check

A difficulty manipulation has to be shown to work rather than asserted.

| Check | Result |
| :--- | ---: |
| Model–species cells showing the full **easy ≥ random ≥ hard** ordering | **77 of 90** |
| Probability a random ordering of three levels is monotone | $1/6$ |
| Exact binomial probability of the observed count | $1.9 \times 10^{-46}$ |

The ordering holds across species and across models, so the DINOv2-based
difficulty mining does what it was designed to do.

---

## What the Results Imply

**VLMs judge appearance, not identity.** This is the component's central finding,
and the adversarial condition is what licenses stating it as a mechanism rather
than as a performance gap. A model that were simply weak would score near chance
on hard pairs; scoring *below* chance requires following a cue, and the cue is
the one the construction inverted.

**The failure is not a bias artefact.** Criterion $c$ is reported separately from
$d'$ precisely so that a model answering *yes* to everything cannot be mistaken
for one that discriminates. The one model that does not show negative $d'$ under
hard pairing is exactly that case, and it is identified as such.

**C4 is the genuine dissociation in the benchmark.** It **loads lowest on the
shared cross-component factor across all five species (.091–.185)** — lower than
any task in any other component. Whatever the other three components share, C4
measures something else. That is the result the component was built to find.

---

## Limitations

| Limitation | Effect |
| :--- | :--- |
| **Five species** | All with strong pattern-based identity signals. Species identified by shape or scarring alone are untested. |
| **One operating point per model** | A VLM's yes/no answer gives no ROC curve, so $d'$ is estimated from a single $(H, F)$ pair. |
| **Log-linear correction is required** | Models answering one way throughout would otherwise give infinite $d'$. The correction is standard but does affect extreme cells. |
| **Pair mining depends on DINOv2** | A different embedding would define different "hard" pairs. The manipulation check shows the ordering holds, not that this embedding is optimal. |
| **Specialist baseline** | MegaDescriptor under random and hard pairing is the validity argument for the component and should be reported alongside. |

<p align="center">
  <sub>Part of the <b>Bio-VQA</b> benchmark &middot; Component 4</sub>
</p>