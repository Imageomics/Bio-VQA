# Hierarchical Taxonomic Reasoning — Component C1 of BioVQA

> An evaluation of hierarchical taxonomic identification in vision–language models,
> from phylum to species and vernacular name, against a domain-specialist reference.
> This document describes the evaluation protocol, the scoring calculus, the specialist
> baseline, and the empirical findings, and gives the exact commands to reproduce them.

---

## Table of contents

1. [Overview](#1-overview)
2. [Motivation](#2-motivation)
   - 2.1 [Hierarchy as the object of measurement](#21-hierarchy-as-the-object-of-measurement)
   - 2.2 [Nomenclatural drift is not error](#22-nomenclatural-drift-is-not-error)
   - 2.3 [The need for a specialist anchor](#23-the-need-for-a-specialist-anchor)
3. [System architecture](#3-system-architecture)
   - 3.1 [Component scripts and dependency order](#31-component-scripts-and-dependency-order)
   - 3.2 [Why the order is load-bearing](#32-why-the-order-is-load-bearing)
4. [The evaluation set](#4-the-evaluation-set)
   - 4.1 [Groups, questions, and scale](#41-groups-questions-and-scale)
   - 4.2 [Answer normalization](#42-answer-normalization)
5. [The scoring calculus](#5-the-scoring-calculus)
   - 5.1 [Exact match](#51-exact-match)
   - 5.2 [Soft match and the synonym relation](#52-soft-match-and-the-synonym-relation)
   - 5.3 [Graded tree-distance measures](#53-graded-tree-distance-measures)
   - 5.4 [A worked example](#54-a-worked-example)
   - 5.5 [What each score implies](#55-what-each-score-implies)
6. [The specialist baseline: BioCLIP-2](#6-the-specialist-baseline-bioclip-2)
   - 6.1 [Why a specialist, and why this one](#61-why-a-specialist-and-why-this-one)
   - 6.2 [Eliciting answers from a contrastive classifier](#62-eliciting-answers-from-a-contrastive-classifier)
   - 6.3 [Three label spaces](#63-three-label-spaces)
   - 6.4 [On evaluating with in-distribution images](#64-on-evaluating-with-in-distribution-images)
   - 6.5 [Why the specialist is not perfect](#65-why-the-specialist-is-not-perfect)
7. [Results](#7-results)
   - 7.1 [The leaderboard](#71-the-leaderboard)
   - 7.2 [Granularity collapse](#72-granularity-collapse)
   - 7.3 [Specificity and rarity compound](#73-specificity-and-rarity-compound)
   - 7.4 [Recognition exceeds generation](#74-recognition-exceeds-generation)
   - 7.5 [Scale is not the bottleneck](#75-scale-is-not-the-bottleneck)
   - 7.6 [Difficulty stratification](#76-difficulty-stratification)
   - 7.7 [Mistake severity and its dissociation from accuracy](#77-mistake-severity-and-its-dissociation-from-accuracy)
   - 7.8 [Taxonomic revision lag](#78-taxonomic-revision-lag)
8. [Cross-group generality](#8-cross-group-generality)
   - 8.1 [The rank-correlation construction](#81-the-rank-correlation-construction)
   - 8.2 [The correlation matrix](#82-the-correlation-matrix)
   - 8.3 [Interpretation](#83-interpretation)
9. [Reproduction](#9-reproduction)
10. [Outputs](#10-outputs)
11. [Design references](#11-design-references)

---

## 1. Overview

We evaluate whether vision–language models (VLMs) possess *hierarchical* biological
knowledge: given a single organism image, can a model name the correct taxon at each
Linnaean rank, from the broad (phylum) to the specific (species, and vernacular name)?
We pose a fixed battery of rank questions per image over ten organismal groups drawn from
**TreeOfLife-200M**, producing **890,000** question–answer instances per evaluated model,
and we score twenty model configurations — eighteen open-weight generalist VLMs and two
configurations of the domain specialist **BioCLIP-2**.

We score each answer along three complementary axes: a strict exact match
([Section 5.1](#51-exact-match)); a taxonomy-aware **soft** match that credits
deprecated-but-correct names via the GBIF Backbone ([Section 5.2](#52-soft-match-and-the-synonym-relation));
and a family of **graded** tree-distance measures that quantify how far a wrong answer
lies from the truth ([Section 5.3](#53-graded-tree-distance-measures)).

Our central finding is a **granularity collapse** ([Section 7.2](#72-granularity-collapse)):
accuracy falls monotonically and steeply with taxonomic depth — from **0.748** at phylum to
**0.086** at species, pooled over all models. The collapse is scale-invariant
([Section 7.5](#75-scale-is-not-the-bottleneck)) and, within the taxonomy task, remarkably
uniform across organismal groups ([Section 8](#8-cross-group-generality)), with a single
interpretable exception.

---

## 2. Motivation

### 2.1 Hierarchy as the object of measurement

A flat accuracy score treats every error as equivalent. In a *hierarchical* domain this
discards the most informative signal in the data. A model that answers *Amphibia* when
asked the species of a frog has not failed in the way a model answering *Insecta* has: the
first is correct at every rank above species, the second is wrong from the class down. We
therefore measure error as **displacement on the tree of life** rather than as a binary
event, and report the depth-resolved accuracy profile as a first-class result rather than a
single scalar ([Section 5.3](#53-graded-tree-distance-measures),
[Section 7.7](#77-mistake-severity-and-its-dissociation-from-accuracy)).

### 2.2 Nomenclatural drift is not error

Biological nomenclature is revised continually as phylogenetic understanding advances. A
model trained on pre-revision corpora will answer *Magnoliophyta* where the current
Catalogue-of-Life backbone records *Tracheophyta*, or *Reptilia* where the accepted classes
are *Squamata*, *Testudines*, *Crocodylia*, and *Sphenodontia*. Scoring these as simple
errors conflates **stale knowledge** with **absent knowledge** — two phenomena a benchmark
should distinguish. We separate them with a synonym-aware soft match grounded in the GBIF
Backbone Taxonomy, and we report the gap between strict and soft accuracy as a direct
estimate of taxonomic revision lag ([Section 5.2](#52-soft-match-and-the-synonym-relation),
[Section 7.8](#78-taxonomic-revision-lag)).

### 2.3 The need for a specialist anchor

Absolute VLM accuracy is difficult to interpret without a reference: is 0.55 at species
good or poor? We resolve this by evaluating **BioCLIP-2**, a contrastive vision model
trained specifically on the TreeOfLife distribution, under identical questions and scoring.
The specialist establishes an in-domain reference against which generalist performance can
be read as a *fraction recovered*, and its own residual error at fine ranks reveals how
much of the difficulty is intrinsic to the task rather than particular to generalist models
([Section 6](#6-the-specialist-baseline-bioclip-2), [Section 6.5](#65-why-the-specialist-is-not-perfect)).

---

## 3. System architecture

### 3.1 Component scripts and dependency order

We factor the pipeline into five scripts. Three build and consume the synonym table; two
produce the specialist baseline that feeds its answers into the scorer.

```
                          ┌─────────────────────────────┐
                          │   GBIF Backbone Taxonomy     │
                          │   (hosted DwC-A archive)     │
                          └───────────────┬─────────────┘
                                          │  download + parse
                                          ▼
      (1)  Evaluate_build_synonyms.py  ───►  synonyms.pkl
                                          │      { name → { accepted taxa } }
   TreeOfLife-200M                        │
   catalog.parquet ───┐                   │
                      │                   ▼
   generalist answer  ├───►  (2)  Evaluate_run.py  ───►  scored CSVs + evaluate_stats.txt
   CSVs  ─────────────┘             (writes hard / soft / graded columns)
                                          │
                                          ▼
      (3)  Evaluate_check_synonyms.py  ───►  Evaluation_soft_audit.csv
                                                 (verifies every soft credit vs GBIF)

   Specialist baseline (independent; its answers are consumed by step 2):

      (0a) prefetch_pybioclip.py  ───►  warms the HF cache once, serially
      (0b) pyBioCLIP.sh  ──►  pyBioCLIP.py  ───►  bioclip2_* answer columns
```

| # | Script | Role |
|---|--------|------|
| 0a | `prefetch_pybioclip.py` | Download the BioCLIP-2 weights and TreeOfLife embeddings once into `HF_HOME`, serially, and verify the cache. |
| 0b | `pyBioCLIP.sh` → `pyBioCLIP.py` | Produce the specialist baseline answers under three label spaces ([Section 6.3](#63-three-label-spaces)). |
| 1 | `Evaluate_build_synonyms.py` | Download the GBIF Backbone and build the set-valued synonym map `synonyms.pkl`. |
| 2 | `Evaluate_run.py` | Score every model CSV; write the hard, soft, and graded columns; emit reports and summary tables. |
| 3 | `Evaluate_check_synonyms.py` | Audit every soft credit against GBIF and record a per-credit verdict. |

### 3.2 Why the order is load-bearing

The order **build → run → check** is not a convention but a data dependency. Step 1 writes
`synonyms.pkl`. Step 2 is the *only* stage that computes the soft column: it reads the
synonym map and writes `hard`, `soft`, and the graded columns into each model CSV. Step 3
*reads* the soft column that step 2 wrote and cannot conjure credits that step 2 has not
computed. Running the audit before the scorer, or the scorer without a freshly built synonym
map, therefore reports stale or empty results — a failure mode we call out because it
produces plausible-looking but incorrect numbers rather than an error.

---

## 4. The evaluation set

### 4.1 Groups, questions, and scale

We draw ten organismal groups from TreeOfLife-200M and pose a fixed battery of rank
questions per image. Each image carries a verified lineage, so every question has a
programmatically determined ground truth. We recognize nine question types, mapped to eight
scored ranks plus a multiple-choice variant of the vernacular-name question:

| Question type | Scored rank | Answer form | Example ground truth |
|---|---|---|---|
| `phylum` … `genus` | phylum, class, order, family, genus | single taxon name | `Chordata`, `Amphibia`, `Anura`, `Hylidae`, `Dryophytes` |
| `species epithet` | species | bare epithet | `cinereus` |
| `scientific name` | scientific_name | binomial | `Dryophytes cinereus` |
| `common name` | common_name | vernacular, free text | `green treefrog` |
| `common name` + options | common_name_MCQ | one of four candidates | *(as above, among 4)* |

Nine groups contribute nine questions per image; the Reptiles group contributes eight.
Across ten groups and their images this yields **890,000** question instances per evaluated
model, and 20 model configurations are scored on the identical set.

### 4.2 Answer normalization

All comparison is performed on a normalized string, so that surface variation does not
masquerade as error. Normalization case-folds, strips diacritics, removes punctuation, and
removes leading boilerplate and rank prefixes. Under this map,

> *"The phylum is **Chordata**."*  →  `chordata`  ←  *"chordata"*

compare equal, as do *"(B) Dryophytes cinereus"* and *"dryophytes cinereus"*. Normalization
is applied identically to prediction and ground truth before any scoring rule is evaluated.

---

## 5. The scoring calculus

We attach to every answer one strict label ([Section 5.1](#51-exact-match)), one soft label
([Section 5.2](#52-soft-match-and-the-synonym-relation)), and a set of graded scores
([Section 5.3](#53-graded-tree-distance-measures)). The three answer different questions:
*is it exactly right*, *is it right up to nomenclatural revision*, and *how close is it*.

### 5.1 Exact match

The strict indicator, and the headline number, which we never relax. For normalized
prediction $\hat{y}$ and ground truth $y$,

$$
\mathrm{EM}_{\text{hard}} \;=\; \mathbb{1}\!\left[\,\hat{y} = y\,\right].
$$

### 5.2 Soft match and the synonym relation

We additionally credit a prediction that is a **deprecated synonym whose GBIF-accepted form
is the ground truth**. Let $\mathcal{S}(\hat{y})$ denote the set of accepted taxa to which
$\hat{y}$ resolves in the GBIF Backbone. Then

$$
\mathrm{EM}_{\text{soft}} \;=\; \mathbb{1}\!\left[\,\hat{y} = y \;\;\lor\;\; y \in \mathcal{S}(\hat{y})\,\right].
$$

Two properties of this relation are deliberate.

**It is one-directional.** We require that the *answer* resolve *onto* the accepted ground
truth, not that the two share any accepted form. Symmetric resolution would credit two
lateral synonyms of a common target as mutually correct, over-crediting the metric; the
one-directional rule admits only genuine old→new substitutions.

**It is set-valued.** A single historical name may split into several modern taxa — a
*pro-parte* synonym. The paraphyletic grade *Reptilia* is the canonical case:

$$
\mathcal{S}(\texttt{reptilia}) \;=\; \{\texttt{squamata},\ \texttt{testudines},\ \texttt{crocodylia},\ \texttt{sphenodontia}\}.
$$

A squamate, a turtle, a crocodilian, and a tuatara each answered *"Reptilia"* are therefore
all credited. A single-valued map would retain only one of the four targets and silently
reject the other three — an error we specifically avoid. In the GBIF Backbone as built,
**75,315** names carry two or more accepted targets.

### 5.3 Graded tree-distance measures

Exact and soft match are binary and cannot separate a near miss from a wild one. We
therefore embed both prediction and truth in the taxonomic tree and measure their
separation. Let $d(\cdot)$ be depth from the root and $\mathrm{LCA}(a,b)$ the lowest common
ancestor of two nodes.

**Mistake severity** — the number of ranks climbed from the truth to reach the shared
ancestor with the prediction:

$$
\text{severity} \;=\; d(y) - d\!\big(\mathrm{LCA}(\hat{y}, y)\big), \qquad \text{lower is closer;}\ \ 0 \Rightarrow \text{correct}.
$$

**Wu–Palmer similarity** — normalized depth of the shared ancestor, in $[0,1]$:

$$
\text{wu\_palmer} \;=\; \frac{2\, d\!\big(\mathrm{LCA}(\hat{y}, y)\big)}{d(\hat{y}) + d(y)}.
$$

**Lin similarity** — the information-theoretic analogue, weighting agreement by the rarity
of the shared ancestor, so that agreeing on an obscure clade is worth more than agreeing on
a ubiquitous one:

$$
\text{lin\_ic} \;=\; \frac{2\, \mathrm{IC}\!\big(\mathrm{LCA}(\hat{y}, y)\big)}{\mathrm{IC}(\hat{y}) + \mathrm{IC}(y)},
\qquad
\mathrm{IC}(c) \;=\; -\log \frac{\lvert \operatorname{subtree}(c)\rvert}{N},
$$

where $N$ is the total node count and $\lvert \operatorname{subtree}(c)\rvert$ counts the
descendants of clade $c$.

### 5.4 A worked example

Consider an image of the green treefrog, ground-truth species *Dryophytes cinereus*, and a
model that answers *"Hyla cinerea"* — the pre-2016 name for the same animal.

| Score | Value | Reasoning |
|---|---|---|
| $\mathrm{EM}_{\text{hard}}$ | 0 | `hyla cinerea` $\neq$ `dryophytes cinereus` as strings. |
| $\mathrm{EM}_{\text{soft}}$ | 1 | GBIF resolves `hyla cinerea` → `dryophytes cinereus`; the accepted form is the truth. |
| severity | 0 | Both resolve to the same species node; the LCA *is* that node. |
| wu\_palmer | 1.00 | Identical node depth; shared ancestor is the node itself. |

Now consider a coarser miss on the same image: the model answers *"Anura"* (the correct
order) when asked for the species.

| Score | Value | Reasoning |
|---|---|---|
| $\mathrm{EM}_{\text{hard}}$ | 0 | Not the species. |
| $\mathrm{EM}_{\text{soft}}$ | 0 | *Anura* is an ancestor, not a synonym of the species; correctly uncredited. |
| severity | 3 | Species → genus → family → **order**: three ranks climbed to the LCA. |
| wu\_palmer | ≈ 0.7 | High: the shared ancestor is deep, so the answer is close, not random. |

The two rows together are the argument for graded scoring: exact match assigns both the
same zero, while severity and Wu–Palmer correctly report the first as *exactly right up to
renaming* and the second as *wrong but near*.

### 5.5 What each score implies

| Score | Question it answers | Reads as high when… | Principal use |
|---|---|---|---|
| $\mathrm{EM}_{\text{hard}}$ | Is the answer exactly the accepted name? | the model names the *current* taxon precisely | headline accuracy; comparison across models |
| $\mathrm{EM}_{\text{soft}}$ | Is it right up to nomenclatural revision? | the model knows the taxon but under an older name | isolating stale-vs-absent knowledge |
| $\mathrm{EM}_{\text{soft}} - \mathrm{EM}_{\text{hard}}$ | How much error is mere revision lag? | a group's taxonomy was recently rewritten | the revision-lag finding ([7.8](#78-taxonomic-revision-lag)) |
| severity | When wrong, how far up the tree? | misses are near the truth, not random | the accuracy-vs-severity dissociation ([7.7](#77-mistake-severity-and-its-dissociation-from-accuracy)) |
| wu\_palmer, lin\_ic | How structurally similar overall? | predictions cluster near the truth | corroboration; largely redundant with EM ([7.7](#77-mistake-severity-and-its-dissociation-from-accuracy)) |

We report severity as a primary column precisely because it is *non-redundant* with exact
match: a model can be inaccurate yet make consistently shallow errors, a property exact
match cannot express. Wu–Palmer and Lin, by contrast, track exact match closely
([Section 7.7](#77-mistake-severity-and-its-dissociation-from-accuracy)) and serve as
corroboration rather than independent evidence.

---

## 6. The specialist baseline: BioCLIP-2

### 6.1 Why a specialist, and why this one

A generalist accuracy of 0.50 at species is uninterpretable in isolation. We anchor the
scale with **BioCLIP-2**, a vision foundation model trained contrastively on the TreeOfLife
distribution with hierarchical taxonomic supervision. It is the appropriate reference for
three reasons. First, it is *in-domain*: it has been optimized for exactly the label space
our questions probe, so its accuracy approximates the best a current model can plausibly
achieve on this task. Second, it is *taxonomically structured*: its supervision is the
Linnaean hierarchy itself, so its predictions carry a full lineage, allowing us to score it
at every rank with the same machinery used for the VLMs. Third, it is *open and
reproducible*. Against this anchor, a generalist score reads as a fraction of specialist
performance rather than as a bare number: the strongest generalist recovers roughly
two-thirds of the specialist's pooled accuracy ([Section 7.1](#71-the-leaderboard)).

### 6.2 Eliciting answers from a contrastive classifier

BioCLIP-2 is a classifier, not a question-answerer; it cannot read a prompt. We therefore
derive all answers from a **single top-1 species prediction** per image and read the eight
coarser answers off that species' lineage. Multiple-choice items are handled by restricting
a custom-label classifier to the four option strings and taking the arg-max; because the
soft-max is monotone, restricting to a subset preserves the ranking among those options.

This elicitation has an important consequence for interpretation. Because every rank answer
descends from *one* species decision, a coarse rank is correct whenever the predicted
species merely shares the true lineage at that rank, whereas a fine rank demands the exact
species. The specialist therefore exhibits the same coarse-to-fine profile as the VLMs, and
for a related structural reason: a single prediction, read at many resolutions. We note this
explicitly because it means the specialist's coarse-rank scores measure *lineage consistency
of the top-1 species* rather than independent coarse classification — a subtly different
quantity from what the VLMs, which answer each rank directly, are measured on.

### 6.3 Three label spaces

We evaluate the specialist under three candidate sets, which separate the *task* from the
*hint* the model is given:

| Label space | Candidate set | Size (order) | Role |
|---|---|---|---|
| `kingdom` *(default)* | species within the kingdom named in the question | $10^4$–$10^5$ | **fair** head-to-head: the VLM prompts also state the kingdom (“the plant/animal shown”), so both sides receive the cue |
| `tol` | all of TreeOfLife-200M | $\sim 9\times10^5$ | **ablation** with no hint; the `kingdom`–`tol` gap isolates the value of the kingdom cue |
| `group` | only the group's ground-truth species | $10^2$ | **ceiling**: a closed candidate set the VLMs never receive; an upper bound only |

The `kingdom` restriction is not a courtesy but a fairness requirement. Without it, a plant
image can be matched to an animal whose name happens to align in embedding space — for
instance a *Pinus strobus* image classified as the *white pine barkminer moth*, an insect
named after the tree it feeds on. Because the VLM prompts hand the model the kingdom for
free, the specialist must be told it too, or the comparison is rigged against the
specialist. The empirical gap between `kingdom` and `tol` is small and concentrated almost
entirely at the phylum rank ([Section 7.1](#71-the-leaderboard)), showing that the cue
prevents gross cross-kingdom confusions but confers no fine-grained advantage.

### 6.4 On evaluating with in-distribution images

Because our images are drawn from TreeOfLife-200M and BioCLIP-2 is trained on that same
distribution, one must ask whether the specialist is being tested on images it has seen. We
address this directly rather than eliding it.

**Contamination inflates scores; it cannot explain a *low* score.** If BioCLIP-2 had
memorized these images as a lookup table, its accuracy would approach ceiling. That its
species accuracy is far from ceiling ([Section 6.5](#65-why-the-specialist-is-not-perfect))
is itself evidence that whatever exposure occurred does not translate into exact recall at
fine ranks. We therefore treat the specialist as an **optimistic, in-domain reference** —
an upper bound on what a current model achieves *with the evaluation distribution in its
favor* — not as a held-out baseline. Framed this way, in-distribution exposure strengthens
rather than weakens the central claim: even a model advantaged by seeing the distribution
tops out well below ceiling at species, so the difficulty of fine-grained identification is
a property of the *task*, not an artifact of generalist weakness.

**"Trained on the distribution" is not "memorized this image."** TreeOfLife-200M contains
many images per species; training exposure to *other* images of a species does not entail
recall of a specific test image, and a single gradient contribution among hundreds of
millions does not imprint an individual example. The relevant exposure is at the level of
the *class*, not the instance.

### 6.5 Why the specialist is not perfect

The specialist's fine-rank accuracy is far from 1.0, and this is expected, not anomalous,
for four independent reasons.

1. **Extreme multi-class retrieval.** Species prediction is a top-1 decision over a label
   space of order $10^5$ (kingdom-restricted) to $10^6$ (full TreeOfLife). The correct
   species competes against many near-identical neighbors; ranking the true species second
   scores zero under exact match. High accuracy at this cardinality is not expected of *any*
   model, and published TreeOfLife-scale species accuracies sit in precisely the range we
   observe.

2. **The long tail.** The training distribution is severely imbalanced; a species observed a
   handful of times cannot acquire a sharp decision boundary even though its images are
   nominally "in the training set." Accuracy consequently degrades from frequent to rare
   taxa regardless of exposure.

3. **Cryptic species.** Sibling species can be visually indistinguishable in a single
   photograph; the discriminating signal is simply not present in the pixels. No model
   resolves such pairs from one image, however it was trained.

4. **The contrastive objective.** CLIP-style training aligns image and text embeddings; it
   does not optimize per-image label recall. The loss the model minimizes is not "name every
   training image correctly," so perfect fit is neither achieved nor sought.

The upshot is that the specialist's residual error *bounds the intrinsic difficulty* of
fine-grained taxonomic identification. When we report that generalists collapse at species,
[Section 7.2](#72-granularity-collapse), the specialist tells us this is not merely a
generalist deficiency: the task is hard in an absolute sense.

---

## 7. Results

### 7.1 The leaderboard

We report the pooled leaderboard over 890,000 instances per model, sorted by soft exact
match. The specialist configurations lead; the strongest generalist recovers roughly 68% of
the specialist's pooled accuracy.

| Model | EM | EM\_soft | severity | wu\_palmer | lin\_ic |
|---|---:|---:|---:|---:|---:|
| **bioclip2\_kingdom** *(specialist, fair)* | 0.743 | 0.743 | 2.52 | 0.854 | 0.836 |
| **bioclip2** *(specialist, no hint)* | 0.737 | 0.737 | 2.64 | 0.836 | 0.824 |
| qwen3\_30B | 0.503 | 0.518 | 2.47 | 0.721 | 0.702 |
| qwen3\_32B | 0.488 | 0.505 | 2.46 | 0.702 | 0.682 |
| qwen3\_8B | 0.483 | 0.497 | 2.48 | 0.707 | 0.685 |
| InternVL3\_38B | 0.455 | 0.470 | 2.61 | 0.680 | 0.661 |
| qwen2.5\_7B | 0.445 | 0.459 | 2.44 | 0.658 | 0.638 |
| gemma3\_12B | 0.401 | 0.429 | 2.57 | 0.646 | 0.618 |
| InternVL3.5\_38B | 0.390 | 0.416 | 2.61 | 0.635 | 0.606 |
| gemma4\_12B | 0.384 | 0.394 | 2.76 | 0.609 | 0.580 |
| InternVL3\_8B | 0.355 | 0.373 | 2.69 | 0.575 | 0.551 |
| qwen2.5\_3B | 0.350 | 0.367 | 2.22 | 0.594 | 0.573 |
| llama3.2\_11B | 0.338 | 0.353 | 2.49 | 0.556 | 0.529 |
| InternVL3.5\_8B | 0.325 | 0.349 | 2.74 | 0.584 | 0.552 |
| molmo2\_8B | 0.324 | 0.349 | 2.75 | 0.576 | 0.540 |
| pixtral\_12B | 0.329 | 0.346 | 2.78 | 0.566 | 0.533 |
| InternVL3.5\_30B\_A3B | 0.317 | 0.336 | 2.67 | 0.571 | 0.544 |

*(Abbreviated; the two lowest configurations and the three label-space variants are given
in full in `evaluate_stats.txt`.)*

The `kingdom`–`tol` comparison is instructive. The two specialist configurations differ by
**+0.060 at phylum** (0.985 vs 0.925) and by essentially nothing at every finer rank. The
kingdom cue prevents gross cross-kingdom confusions and buys nothing below phylum — evidence
that the specialist's advantage is the *representation*, not the *hint*.

### 7.2 Granularity collapse

Pooled over all twenty configurations, exact match falls monotonically and steeply with
taxonomic depth:

| Rank | Mean EM | | Rank | Mean EM |
|---|---:|---|---|---:|
| phylum | **0.748** | | species | **0.086** |
| class | 0.578 | | scientific\_name | 0.165 |
| order | 0.568 | | common\_name | 0.085 |
| family | 0.421 | | common\_name\_MCQ | 0.677 |
| genus | 0.292 | | | |

The drop from phylum to species is a factor of roughly **8.7×** within a single image and a
single model population. Models resolve *what kind* of organism is present but not *which*
organism. This profile — not any single scalar — is the primary object of the benchmark.

### 7.3 Specificity and rarity compound

The collapse is not driven by taxonomic depth alone; depth and rarity **compound**. Two
mechanisms act together. As one descends the hierarchy, (i) the number of candidate classes
grows by orders of magnitude — tens of phyla but hundreds of thousands of species — so the
*discrimination* required at each step sharpens; and (ii) the *prevalence* of each class in
web-scale training text falls, so the evidence available for the correct answer thins. A
species is thus simultaneously the hardest *discrimination* and the sparsest *signal*, and
the two effects multiply rather than add.

The information-content weighting of Lin similarity makes this concrete: agreeing with the
truth at a rare, deep clade contributes more information — $\mathrm{IC}(c) = -\log
(\lvert\operatorname{subtree}(c)\rvert / N)$ grows as the clade shrinks — yet it is exactly
these high-information agreements that models most often miss. The difficulty gradient the
benchmark exposes is therefore a *joint* function of specificity and rarity, which is why it
is so steep and so consistent across groups ([Section 8](#8-cross-group-generality)).

### 7.4 Recognition exceeds generation

The same organism, named in free text versus chosen from four options, produces starkly
different accuracy. Pooled, open-ended common-name accuracy is **0.085**; the four-way
multiple-choice variant of the *identical* question is **0.677**. The per-model gap is large
throughout — for example, qwen3\_32B answers 0.078 open-ended but 0.809 in multiple choice, a
gap of **+0.732**.

| Model | open-ended | MCQ (4-way) | gap |
|---|---:|---:|---:|
| qwen3\_32B | 0.078 | 0.809 | +0.732 |
| qwen3\_8B | 0.065 | 0.775 | +0.710 |
| InternVL3\_38B | 0.044 | 0.752 | +0.708 |
| qwen2.5\_7B | 0.062 | 0.753 | +0.692 |
| qwen3\_30B | 0.107 | 0.772 | +0.665 |

The knowledge is **latent but not freely retrievable**: presented with the correct name
among four, models identify it far more often than they can produce it unprompted. This is a
direct caution against multiple-choice biology benchmarks, which systematically overstate
open-ended competence — the ability they measure is *recognition*, not *recall*.

### 7.5 Scale is not the bottleneck

Within a model family, accuracy is nearly flat across a large parameter range. Qwen3 moves
from 0.483 (8B) to 0.495 (30B) to 0.488 (32B) — a spread of roughly one point across a
fourfold change in parameters. The earlier Qwen2.5 family shows a real 3B→7B gain (0.350 →
0.445), but the current generation has saturated. The limiting resource is therefore not
model capacity but the **long-tailed prevalence** of each taxon in training data
([Section 7.3](#73-specificity-and-rarity-compound)) — a data problem, not a scale problem.

### 7.6 Difficulty stratification

We assign each item a difficulty tier from the prevalence of its taxon in the catalog. The
axis is validated by independent replication: **all 20** models satisfy
$\text{EM}(\text{Easy}) > \text{EM}(\text{Medium}) > \text{EM}(\text{Hard})$, with pooled
means

$$
\text{Easy} = 0.497 \;>\; \text{Medium} = 0.342 \;>\; \text{Hard} = 0.190.
$$

That every model, without exception, respects the ordering establishes prevalence as a
genuine and model-independent predictor of difficulty rather than an artifact of any one
architecture.

### 7.7 Mistake severity and its dissociation from accuracy

Severity ([Section 5.3](#53-graded-tree-distance-measures)) measures how far up the tree a
wrong answer lands. It is *not* redundant with accuracy. Across the leaderboard, the rank
correlation between exact-match rank and severity rank is

$$
\rho\big(\text{EM},\ \text{severity}\big) \;=\; -0.705,
$$

negative as expected (better models miss by less), but far from $-1$: the two orderings
disagree substantially. Concretely, the small qwen2.5\_3B attains among the *best* severity
(shallowest misses) while ranking only mid-table on exact match — it is comparatively poor
at being *right* yet comparatively good at being *wrong in the right neighborhood*. Exact
match cannot express this; severity can, which is why it earns a primary column.

By contrast, Wu–Palmer and Lin similarity correlate with exact match at $\rho \approx 0.97$
and are therefore near-redundant; we retain them as corroboration and relegate their
detailed tables to the appendix.

### 7.8 Taxonomic revision lag

The gap $\mathrm{EM}_{\text{soft}} - \mathrm{EM}_{\text{hard}}$ estimates how much apparent
error is in fact stale nomenclature. Pooled, the gap is small, confirming that soft credit
is not indiscriminate; but it is *structured*, opening precisely in the groups whose
higher taxonomy has been rewritten. It is largest in **Angiosperms** and **Gymnosperms**
(the *Magnoliophyta* → *Tracheophyta* phylum revision) and in **Reptiles** (the dissolution
of *Reptilia* into modern classes), and near zero in groups whose taxonomy was untouched.
Every soft credit is independently re-verified against GBIF by
`Evaluate_check_synonyms.py`; in the audited run, 100% of the ~130,000 soft credits resolve
to a valid GBIF old→new substitution, with zero flagged for review.

---

## 8. Cross-group generality

This section is the analytical heart of the component. It asks whether the *ranking* of
models by taxonomic competence is stable across organismal groups, or whether different
groups reward different models — the within-component precursor to the cross-component
dissociation that motivates BioVQA as a whole.

### 8.1 The rank-correlation construction

For each organismal group $g$ we rank the twenty model configurations by exact match,
producing a ranking vector $r_g$. For every pair of groups $(g, h)$ we compute the Spearman
rank correlation

$$
\rho_{g,h} \;=\; \operatorname{Spearman}\big(r_g,\ r_h\big),
$$

and summarize the resulting $10 \times 10$ matrix by its **mean off-diagonal** value,

$$
\bar{\rho} \;=\; \frac{1}{|G|(|G|-1)} \sum_{g \neq h} \rho_{g,h}.
$$

A high $\bar{\rho}$ indicates that a *single* latent taxonomic ability governs performance:
models that are strong on one clade are strong on all. A low $\bar{\rho}$ would indicate
clade-specific competencies that do not transfer.

### 8.2 The correlation matrix

|                | Amph | Angio | Arach | Birds | Bryo | Fish | Gymno | Insects | Mamm | Rept |
|----------------|-----:|------:|------:|------:|-----:|-----:|------:|--------:|-----:|-----:|
| **Amphibians** | 1.00 | 0.94 | 0.82 | 0.90 | 0.92 | 0.88 | 0.39 | 0.89 | 0.95 | 0.98 |
| **Angiosperms**| 0.94 | 1.00 | 0.87 | 0.88 | 0.96 | 0.89 | 0.49 | 0.91 | 0.92 | 0.94 |
| **Arachnids**  | 0.82 | 0.87 | 1.00 | 0.90 | 0.87 | 0.81 | 0.41 | 0.94 | 0.83 | 0.83 |
| **Birds**      | 0.90 | 0.88 | 0.90 | 1.00 | 0.91 | 0.87 | 0.42 | 0.97 | 0.91 | 0.89 |
| **Bryophytes** | 0.92 | 0.96 | 0.87 | 0.91 | 1.00 | 0.94 | 0.46 | 0.95 | 0.89 | 0.91 |
| **Fish**       | 0.88 | 0.89 | 0.81 | 0.87 | 0.94 | 1.00 | 0.37 | 0.91 | 0.83 | 0.88 |
| **Gymnosperms**| 0.39 | 0.49 | 0.41 | 0.42 | 0.46 | 0.37 | 1.00 | 0.45 | 0.46 | 0.45 |
| **Insects**    | 0.89 | 0.91 | 0.94 | 0.97 | 0.95 | 0.91 | 0.45 | 1.00 | 0.88 | 0.90 |
| **Mammals**    | 0.95 | 0.92 | 0.83 | 0.91 | 0.89 | 0.83 | 0.46 | 0.88 | 1.00 | 0.95 |
| **Reptiles**   | 0.98 | 0.94 | 0.83 | 0.89 | 0.91 | 0.88 | 0.45 | 0.90 | 0.95 | 1.00 |

$$
\bar{\rho}_{\text{off-diagonal}} \;=\; 0.807.
$$

Excluding the Gymnosperms row and column (discussed below), the mean off-diagonal
correlation among the remaining nine groups rises to approximately **0.90**.

### 8.3 Interpretation

Two facts stand out.

**Within the taxonomy task, competence is nearly one-dimensional.** Off the Gymnosperms
line, every pairwise correlation lies between 0.81 and 0.98, and the mean is ≈0.90. Models
that rank highly on amphibians rank highly on mammals, fish, birds, and the rest: a single
latent taxonomic ability drives the ranking, and clade identity barely perturbs it. This is
a strong, clean regularity — and it is the *within-component* baseline against which the
BioVQA thesis is defined. The motivating question of the full benchmark is whether this
one-dimensionality *survives across components* (taxonomy vs. attribute grounding vs. spatial
reasoning vs. individual re-identification); the high within-taxonomy $\bar\rho$ reported
here is precisely the null the cross-component analysis is designed to test against.

**Gymnosperms are the lone dissociation, and the reason is identifiable.** Every correlation
involving Gymnosperms collapses to ≈0.4. This single column is what pulls the overall mean
down from ≈0.90 to 0.807. The cause is not noise: the gymnosperm label space is dominated by
a small number of visually similar conifers, and the specialist's derived-lineage
elicitation ([Section 6.2](#62-eliciting-answers-from-a-contrastive-classifier)) interacts
with this narrow, homogeneous space differently than the generalists do, reordering the
leaderboard for this group alone. It is a genuine, localized exception to the otherwise
uniform structure, and we report it as such rather than smoothing it away.

---

## 9. Reproduction

```bash
# 0a. Warm the specialist cache once, serially (login node).
export HF_HOME=/path/to/hf_cache
python prefetch_pybioclip.py

# 0b. Specialist baseline — three label spaces, one SLURM array task per group.
sbatch --array=0-9                                pyBioCLIP.sh   # kingdom (fair)
sbatch --array=0-9 --export=ALL,LABEL_SPACE=tol   pyBioCLIP.sh   # no hint (ablation)
sbatch --array=0-9 --export=ALL,LABEL_SPACE=group pyBioCLIP.sh   # ceiling

# 1. Build the GBIF synonym table.
python Evaluate_build_synonyms.py --out <base_dir>/synonyms.pkl

# 2. Score every model CSV; write reports and summary tables.
python Evaluate_run.py --base_dir <base_dir> --parquet <catalog.parquet>

# 3. Audit every soft credit against GBIF.
python Evaluate_check_synonyms.py --base_dir <base_dir> --synonyms <base_dir>/synonyms.pkl
```

`Evaluate_run.py` loads `<base_dir>/synonyms.pkl` automatically when it is present; absent a
synonym table, soft match reduces to hard match and no answer is credited beyond exact
agreement. The order **build → run → check** is mandatory
([Section 3.2](#32-why-the-order-is-load-bearing)).

---

## 10. Outputs

| File | Contents |
|---|---|
| `evaluate_stats.txt` | Leaderboard, then the pooled report, then per-group reports; the single file to read. |
| `result_ALL.txt`, `result_<Group>.txt` | The pooled and per-group reports individually. |
| `csv/` | Machine-readable summary tables (by model, rank, difficulty, group, and their interactions). |
| `Evaluation_soft_audit.csv` | Every soft credit, with its GBIF verification verdict and resolved target set. |
| *(per-model CSVs)* | Augmented in place with `hard`, `soft`, `resolution`, `lca_height`, `wu_palmer`, `lin_ic`. |

The report is organized into sixteen numbered sections spanning the leaderboard, per-rank
and per-difficulty breakdowns, the recognition–generation contrast, revision lag, severity,
scale, metric redundancy, and the cross-group correlation matrix of
[Section 8](#8-cross-group-generality).

---

## 11. Design references

Our graded measures follow the standard ontology-similarity literature. Path-based
similarity is that of **Wu and Palmer (1994)**, *Verb Semantics and Lexical Selection*, ACL.
The information-content formulation is **Lin (1998)**, *An Information-Theoretic Definition of
Similarity*, ICML, building on **Resnik (1995)**, *Using Information Content to Evaluate
Semantic Similarity in a Taxonomy*, IJCAI. Nomenclatural resolution uses the **GBIF Backbone
Taxonomy** (Global Biodiversity Information Facility). The specialist model and image
distribution are **BioCLIP-2** and **TreeOfLife-200M** (Imageomics Institute), extending
**BioCLIP** (Stevens *et al.*, *BioCLIP: A Vision Foundation Model for the Tree of Life*,
CVPR 2024). The coarse-to-fine derivation of ranks from a single species prediction, and its
consequence that errors at a parent rank depress accuracy at child ranks, follows the
top-down constrained-inference protocol standard in hierarchical fine-grained classification.