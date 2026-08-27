<h1 align="center"> Bio-VQA &rarr; C1 &rarr; Hierarchical Taxonomic Reasoning </h1>

<p align="center">
  <b>Eight Linnaean ranks &times; ten organismal groups, bounded above by a domain specialist</b>
</p>

<p align="center">
  <img alt="Questions" src="https://img.shields.io/badge/questions-~890K-green">
  <img alt="Images" src="https://img.shields.io/badge/images-100K-blue">
  <img alt="Groups" src="https://img.shields.io/badge/organismal%20groups-10-orange">
  <img alt="Ranks" src="https://img.shields.io/badge/ranks-8-yellow">
  <img alt="Ceiling" src="https://img.shields.io/badge/ceiling-BioCLIP--2-purple">
  <img alt="Source" src="https://img.shields.io/badge/source-TreeOfLife--200M-informational">
</p>

---

## Overview

Taxonomic identification is the most common biological visual task, and
contrastive encoders such as BioCLIP-2 are specialists at it. **C1 asks how far a
generalist VLM stands from the specialist on the same task**, at eight ranks
rather than at species alone, and across ten organismal groups rather than three.

| Stage | Script | Responsibility |
| :---: | :--- | :--- |
| 1 | `download.py` | Query the ToL-200M catalog, select species with complete taxonomy, download concurrently, emit a manifest. |
| 2 | `verify.py` | Audit every manifest for missing / `NaN` / empty ranks before anything downstream runs. |
| 3 | `generateQA.py` | Turn each verified row into open-ended and multiple-choice questions across eight ranks. |

> **Design contract:** a row exists in the manifest **if and only if** its image
> is on disk **and** every taxonomic rank used by the generator is non-null.
> Stage 2 exists to prove that contract holds.

---

## Table of Contents

- [Corpus Construction](#corpus-construction)
- [The Eight Ranks](#the-eight-ranks)
- [Difficulty Tiers](#difficulty-tiers)
- [Two Label Defects, and Their Fixes](#two-label-defects-and-their-fixes)
- [Metrics](#metrics)
- [Results](#results)
- [What the Results Imply](#what-the-results-imply)
- [Limitations](#limitations)
- [FAQ](#faq)
- [Citation](#citation)

---

## Corpus Construction

A 100K subset of TreeOfLife-200M spanning **626 families, 2,311 genera
(6&times; VLM4Bio) and 3,403 species (4&times; VLM4Bio)**.

The source is long-tailed, so a random sample would be dominated by a few
abundant taxa. **10K images are selected from each of ten groups**, under three
constraints:

| Constraint | Purpose |
| :--- | :--- |
| Complete lineage phylum &rarr; species, plus common name | No question is generated without a ground truth |
| Minimum resolution threshold | Excludes low-quality images |
| Per-species cap | Prevents abundant species dominating a group |

### The ten groups

| Group | Filter rank | Value |
| :--- | :--- | :--- |
| Amphibians | class | `Amphibia` |
| Angiosperms | class | `Magnoliopsida`, `Liliopsida` |
| Arachnids | class | `Arachnida` |
| Birds | class | `Aves` |
| Bryophytes | **phylum** | `Bryophyta` |
| Fish | class | `Actinopterygii`, `Teleostei`, `Elasmobranchii`, `Holocephali`, `Coelacanthi`, `Dipneusti` |
| Gymnosperms | class | `Pinopsida` |
| Insects | class | `Insecta` |
| Mammals | class | `Mammalia` |
| Reptiles | class | `Squamata`, `Testudines`, `Crocodylia`, `Sphenodontia` |

**Six of these groups do not appear in prior biological VQA work.**

Two groups need explanation, and both trace to the same cause &mdash; the GBIF
Backbone follows the Catalogue of Life, which eliminated paraphyletic groups:

**Fish spans six classes.** *Actinopterygii* is paraphyletic and does not exist
as a single usable class value covering all fishes. "Fish" is itself paraphyletic
as a concept, so the split is assembled explicitly.

**Reptiles carry no order.** COL promoted the former reptile orders &mdash;
Squamata, Testudines, Crocodylia &mdash; to **class** rank, retaining Reptilia
only as a synonym. Family therefore sits directly under class, and the `order`
column is empty for every reptile row. Reptiles are asked **seven ranks**, not
eight.

> ⚠️ Models answering **Reptilia** to *"what is the class?"* are marked wrong
> against this ground truth. The ground truth is what the source backbone says;
> the models are reproducing textbook Linnaean taxonomy. **That gap is a finding
> about training-data recency, not a labelling defect**, and the methods section
> states that the benchmark follows the ToL-200M/GBIF backbone.

---

## The Eight Ranks

```
   phylum ──▶ class ──▶ order ──▶ family ──▶ genus ──▶ species
                                                          │
                                                          ├──▶ scientific name
                                                          │
                                                          └──▶ common name
                                                                 │
                                                    ┌────────────┴────────────┐
                                                    ▼                         ▼
                                              open-ended              4-way multiple choice
                                                    └──── same image, same target ────┘
```

The common-name question is asked **twice on the same image** &mdash; once
open-ended, once as four-way multiple choice. This measures the
**recognition–production gap**: what a model recalls unprompted versus what it
recognises once given options.

### Distractor construction

Distractors are drawn from the **same rank as the target**. Writing $c(t)$ for
the class of taxon $t$ and $\mathcal{S}$ for the taxa in the sample:

$$
D(t) \sim \mathrm{Unif}_{3}\bigl(\{\, u \in \mathcal{S} : c(u) = c(t),\; u \neq t \,\}\bigr)
$$

Options are shuffled, so correct-answer position carries no signal.

---

## Difficulty Tiers

Items are tiered by how common the target taxon is in the source catalog. Writing
$n(t)$ for its image count and $\mathcal{T}_r$ for the taxa at rank $r$, the
**prevalence percentile** is:

$$
P(t) = \frac{\bigl|\{\, u \in \mathcal{T}_r : n(u) \le n(t) \,\}\bigr|}{|\mathcal{T}_r|}
$$

| Tier | Condition |
| :--- | :--- |
| **Easy** | $P(t) > 0.67$ |
| **Medium** | $0.33 < P(t) \le 0.67$ |
| **Hard** | $P(t) \le 0.33$ |

The tiers separate taxa a model has seen often from ones it has barely seen.

---

## Two Label Defects, and Their Fixes

### 1 &mdash; Non-English common names

GBIF vernacular records are **not filtered by language** and contain Swedish,
Danish and Norwegian names in the common-name field. A model answering with the
correct English name would be scored wrong.

**Fix:** every non-English common name is resolved to English against the
iNaturalist and Wikidata APIs. Taxa with no English vernacular have their
common-name and MCQ items excluded from scoring.

### 2 &mdash; Synonymy

Taxonomic names change. A cougar answered as *Felis concolor* is the same animal
as *Puma concolor*, and the current backbone accepts only the latter.

**Fix:** a synonym map is built from a **versioned GBIF Backbone snapshot**
&mdash; the same taxonomy ToL-200M draws its labels from &mdash; and any synonym
of the target is accepted.

---

## Metrics

Exact match is reported for every rank. Three graded metrics quantify *how wrong*
a wrong answer is.

For an image of a warbler, exact match gives the same zero to a model answering
*Turdidae* (a different songbird family) and one answering *Fungi*. The first
confuses two songbird families; the second fails to recognise the subject as an
animal.

Let $\ell = \mathrm{LCA}(\hat{y}, y)$ be the deepest clade containing both.

**Wu–Palmer similarity** measures the depth of that shared ancestor:

$$
\mathrm{wup}(\hat{y},y) = \frac{2\,\mathrm{depth}(\ell)}{\mathrm{depth}(\hat{y}) + \mathrm{depth}(y)}
$$

**Lin similarity** weights by information content,
$\mathrm{IC}(c) = -\log\bigl(|\mathrm{subtree}(c)| / N\bigr)$, so a rare genus
counts for more than a common one:

$$
\mathrm{lin}(\hat{y},y) = \frac{2\,\mathrm{IC}(\ell)}{\mathrm{IC}(\hat{y}) + \mathrm{IC}(y)}
$$

**Severity** &mdash; how far a wrong answer lands, in ranks:

$$
\mathrm{sev} = \mathbb{E}\bigl[\, r(y) - r(\ell) \;\bigm|\; \hat{y} \neq y \,\bigr]
$$

A severity of 1 means misses stay within the correct parent rank; higher values
mean the broader clade is missed too.

**Soft exact match** accepts any valid synonym for the gold answer.

**Ceiling: BioCLIP-2**, evaluated on the identical images.

---

## Results

### Accuracy falls with rank

Averaged over 18 models, exact match drops at **every** step:

| Rank | Mean EM |
| :--- | ---: |
| phylum | **.725** |
| class | .546 |
| family | .382 |
| genus | .247 |
| scientific name | **.123** |

A **5.9&times; fall** from the coarsest rank to the one biologists use. Every
model shows the same ordering, so it is not an averaging artefact.

Two things happen at once: the candidate set grows from tens of phyla to
thousands of species, and each answer appears less often in training data.

### Models recognise more than they can produce

| Format | Mean score |
| :--- | ---: |
| common name, **open-ended** | **.039** |
| common name, **4-way MCQ** | **.645** |

A **16.5&times; difference on identical images**. The knowledge is there; models
cannot retrieve it unprompted.

> Any benchmark that asks only multiple-choice questions reports a number that
> will not survive open-ended use.

### The gap to a specialist widens with rank

| Rank | Best VLM | BioCLIP-2 | Ratio |
| :--- | ---: | ---: | ---: |
| class | .731 | .874 | 1.20&times; |
| genus | .445 | .691 | 1.56&times; |
| scientific name | .269 | .544 | **2.02&times;** |

The generalist is close to the specialist while the answer stays coarse, and
falls behind as it sharpens.

### Scale buys the coarse end

| Family | Scaling | Gain at class | Gain at scientific name |
| :--- | :--- | ---: | ---: |
| InternVL3 | 8B &rarr; 38B | **+18.7** | +6.5 |
| InternVL3.5 | 8B &rarr; 38B | +10.5 | +5.0 |
| Qwen3-VL | 8B &rarr; 32B | +2.3 (family) | +2.0 (species) |

> Extra parameters buy **broader coverage of categories a model already partly
> knew**, not resolution among the ones it does not.

---

## What the Results Imply

**The collapse is a widening gap along one scale, not a separate failure.**
Coarse and fine ability correlate at $\rho = .769$ ($p = .0003$; $.597$ with
parameter count partialled out). A model good at phylum is good at species
*relative to other models* &mdash; it simply loses ground as the rank sharpens.

**The recognition–production gap is the largest single effect in the benchmark.**
16.5&times; between two formats of the same question on the same image. It is
also the effect most likely to be invisible in prior work, since prior
biological VQA is predominantly multiple-choice.

**Gymnosperms load lowest on the shared cross-component factor (.183)** &mdash;
the group with the thinnest coverage in the source catalog, and the one whose
generation required both knobs relaxed to reach 10K images.

---

## Limitations

| Limitation | Effect |
| :--- | :--- |
| **Backbone-dependent ground truth** | Reptile class answers follow GBIF/COL, not textbook Linnaean taxonomy. Stated in methods. |
| **Reptiles carry seven ranks** | `order` is unpopulated in the source. Tooling assuming six populated ranks will silently drop every reptile — allow skip-edges. |
| **Download is not reproducible** | Thread interleaving decides which images win the race to the budget. **The manifests are the artifact of record.** |
| **MCQ distractors are not difficulty-graded** | Pools are unioned across genus/family/order/class and sampled uniformly, so the class-level set dominates. Best read as a coarse recognition probe. |
| **Gymnosperm coverage** | ~1K accepted species worldwide; reaching 10K images required a 100/species cap and a 256 KB size floor. |



<p align="center">
  <sub>Part of the <b>Bio-VQA</b> benchmark &middot; Component 1</sub>
</p>