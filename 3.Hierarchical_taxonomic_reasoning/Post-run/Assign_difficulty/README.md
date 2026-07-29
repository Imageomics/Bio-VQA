<h1 align="center"> Hierarchical Taxonomic Reasoning &rarr; Difficulty Assignment </h1>

<p align="center">
  <b>Taxonomy-Aware Difficulty Stratification for the Hierarchical Taxonomic Reasoning Component</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/DuckDB-analytics-FFF000?logo=duckdb&logoColor=black" alt="DuckDB">
  <img src="https://img.shields.io/badge/scheduler-SLURM-4B8BBE" alt="SLURM">
  <img src="https://img.shields.io/badge/determinism-seed--free-success" alt="Deterministic">
  <img src="https://img.shields.io/badge/coverage-100%25%20resolved-brightgreen" alt="Coverage">
</p>

---

## Overview

Every question in the Hierarchical Taxonomic Reasoning component is assigned a **difficulty tier
derived from the taxonomy itself &mdash; never from model behaviour**. Difficulty is treated as a
property of the *question*, so it is computed once over the union of all answer keys and propagated
identically to every model's prediction file. Two models are therefore always compared on the same
stratification.

Two axes are used, because the two question formats measure different competencies:

| Format | Axis | Question it answers | Construct |
| :--- | :--- | :--- | :---: |
| Open recall | **Prevalence** | How much of the visual world does this taxon occupy? | [&sect;5](#5-construct-i--prevalence-tier) |
| 4-way MCQ | **Confusability** | How close is the nearest wrong option in the tree? | [&sect;6](#6-construct-ii--confusability-tier) |

> **Design contract:** a difficulty tier is a function of the answer key and the taxonomic backbone
> alone. No prediction file influences any tier, and every item resolves to exactly one node in the
> tree, with the resolution mechanism recorded per item so the derivation is auditable.

---

## Table of Contents

- [Motivation](#motivation)
- [Design Principles](#design-principles)
- [Pipeline](#pipeline)
- [Taxonomic Backbone](#taxonomic-backbone)
- [Construct I &mdash; Prevalence Tier](#5-construct-i--prevalence-tier)
- [Construct II &mdash; Confusability Tier](#6-construct-ii--confusability-tier)
- [Target Resolution](#target-resolution)
- [Vernacular Names and Rank Heterogeneity](#vernacular-names-and-rank-heterogeneity)
- [Tier Assignment, Degenerate Cells &amp; Reference Populations](#tier-assignment-degenerate-cells--reference-populations)
- [Emitted Columns](#emitted-columns)
- [Usage](#usage)
- [Cluster Execution](#cluster-execution)
- [Determinism and Reproducibility](#determinism-and-reproducibility)
- [Audit Output](#audit-output)
- [Methodological Grounding](#methodological-grounding)
- [Limitations](#limitations)
- [References](#references)

---

## Motivation

Aggregate accuracy on a taxonomic VQA benchmark conflates two very different competencies. A model
that names *Corvus corax* correctly and a model that names a rarely-photographed sedge correctly have
not demonstrated the same thing, and a four-way multiple-choice item whose distractors are congeners
is not the same task as one whose distractors come from another phylum. Reporting a single mean over
such a corpus hides exactly the structure a taxonomic benchmark exists to expose.

**Difficulty stratification restores that structure.** By partitioning items along axes defined
*before* any model sees them, per-tier accuracy becomes interpretable: a flat profile across tiers
indicates a model whose performance is driven by something other than taxon prevalence, while a steep
profile indicates reliance on the head of the distribution. Both are findings; neither is visible in a
single number. This is the long-tailed-recognition reporting protocol &mdash; many-shot / medium-shot /
few-shot bands reported separately from the aggregate [1, 2, 3] &mdash; adapted from training-set class
frequency to catalog abundance.

---

## Design Principles

**Difficulty is a property of the question, not the model.** Tiers are computed from the taxonomy and
the answer key alone. No prediction file influences any tier. This is what allows the same
stratification to be applied across an arbitrary number of evaluated systems, and what prevents the
circularity of calling an item "hard" because models found it hard.

**One computation, many files.** The corpus of unique `(group, uuid, question)` items is assembled
across *every* matched CSV before scoring. A single tier is computed per item and written to all files
that contain it. A model whose prediction file is short a few rows cannot shift a tertile boundary and
thereby receive a different stratification from its peers.

**Resolution is exhaustive and audited.** Every item resolves to a node in the taxonomy. Where the
answer string alone is insufficient, the image's own lineage &mdash; which the answer key fully
specifies &mdash; supplies the target. The mechanism used is recorded per item in a `resolution`
column, so the share of non-direct resolutions is reportable and the corresponding items can be
excluded in a robustness check [8].

**Tiers are taxon-deterministic.** Two items sharing a target taxon always receive the same tier. A
taxon never straddles a boundary, so per-tier accuracy can be read as accuracy over a *set of taxa*
rather than over an arbitrary partition of items.

**Minimal, idempotent output footprint.** Columns are added to the source files or written beside
them; re-running updates existing columns in place rather than appending duplicates.

---

## Pipeline

```
                       ┌──────────────────────────┐
                       │   catalog.parquet        │
                       │   (full image catalog)   │
                       └────────────┬─────────────┘
                                    │  DuckDB aggregation
                                    ▼
                       ┌──────────────────────────┐
                       │      Taxonomy tree       │
                       │  nodes + subtree counts  │──── cached (fingerprinted)
                       └────────────┬─────────────┘
                                    │
   ┌────────────────────────────────┼────────────────────────────────┐
   │                                │                                │
   ▼                                ▼                                ▼
┌──────────────┐          ┌───────────────────┐          ┌────────────────────┐
│  PASS 1      │          │  Image index      │          │  Vernacular index  │
│  collect     │────────► │  lineage + anchor │────────► │  name → LCA node   │
│  unique items│          │  per (group,uuid) │          │  (corpus-derived)  │
└──────────────┘          └─────────┬─────────┘          └──────────┬─────────┘
                                    │                               │
                                    └───────────┬───────────────────┘
                                                ▼
                                   ┌─────────────────────────┐
                                   │   Target resolution     │
                                   │  exact → vernacular →   │
                                   │  lineage → anchor       │
                                   └────────────┬────────────┘
                                                │
                         ┌──────────────────────┴──────────────────────┐
                         ▼                                             ▼
              ┌────────────────────┐                        ┌────────────────────┐
              │  Prevalence tier   │                        │ Confusability tier │
              │  (open recall)     │                        │  (4-way MCQ)       │
              │  log-exposure      │                        │  nearest-distractor│
              │  tertiles          │                        │  distance          │
              └──────────┬─────────┘                        └──────────┬─────────┘
                         └──────────────────┬──────────────────────────┘
                                            │
                                            ▼
                                 ┌─────────────────────┐
                                 │  PASS 2  write back │
                                 │  same │ separate    │
                                 └─────────────────────┘
```

---

## Taxonomic Backbone

The tree is built by aggregating the full image catalog over the Linnaean columns, producing one node
per distinct taxon and one integer per node: the number of catalog images in that node's subtree. This
count is the raw material for the prevalence construct.

**Ranks and depth.** Six ranks are modelled, with depth read from the rank *name* rather than from
position in the chain:

| Rank | phylum | class | order | family | genus | species |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| Depth | 1 | 2 | 3 | 4 | 5 | 6 |

**Incomplete lineages are first-class.** Contemporary taxonomic backbones do not guarantee a populated
value at every rank for every organism. Rank promotions are routine &mdash; Catalogue of Life 2022
elevates Squamata, Testudines, Crocodylia, Elasmobranchii and Teleostei to class rank, leaving no
intervening order beneath them. A backbone that required all six ranks would silently exclude entire
clades from the tree and, with them, every question about those clades.

The tree therefore requires only a populated **genus** and **species**; any intermediate rank may be
absent, in which case the node attaches to its nearest populated ancestor. A family whose order is null
attaches directly to its class. Because depth is a function of the rank label rather than of edge
count, the lowest-common-ancestor distances used by the confusability construct remain correct across
such a gap: the LCA of two species in different families is still reported at family or above, and the
numeric distance still counts *ranks* climbed, not *edges* traversed.

> **Exposure is global.** Subtree counts are read from the complete catalog, not from the sampled
> evaluation images. Prevalence therefore reflects the distribution an image-trained model plausibly
> encountered, rather than the composition of the benchmark sample &mdash; the appropriate analogue of
> training-set class frequency when the evaluated systems are pretrained on broad image collections
> rather than a fixed labelled split [2, 3].

---

## 5. Construct I — Prevalence Tier

Applies to all open-recall items: `phylum`, `class`, `order`, `family`, `genus`, `species` (epithet),
`scientific_name`, `common_name`.

### 5.1 Exposure

For a target taxon *t*, exposure is the number of catalog images beneath it:

```
c(t)  = |{ images whose lineage passes through t }|
x(t)  = log10( c(t) + 1 )
```

**Why the logarithm, and why base 10.** Catalog abundance is heavy-tailed: the most-photographed
species carry on the order of 10^5&ndash;10^6 images while the rarest carry a handful, a span of five to
six orders of magnitude. Difficulty does not scale with that count *linearly* &mdash; the step from 10
to 100 images represents the same multiplicative change in rarity as the step from 10,000 to 100,000,
and it is the multiplicative scale that governs how much visual evidence a model plausibly saw
(a Weber&ndash;Fechner-style relationship). A linear axis would collapse almost every taxon into an
indistinguishable low-count mass and let a few hyper-abundant taxa monopolise the top of the range. The
base-10 logarithm maps **each decade of abundance to one unit of the axis** (&Delta;x = 1 &hArr; a
ten-fold change in image count), which is both statistically well-behaved and directly interpretable.
The **unit pseudocount** `+1` keeps the transform defined at `c = 0` (a taxon present in the tree but
with an empty subtree) and prevents `log10(0) = -∞`, while leaving the ordering of all positive counts
unchanged.

> **Illustrative exposures** (numbers approximate, for intuition only):
>
> | taxon | ≈ catalog images `c(t)` | `x = log10(c+1)` |
> | :--- | ---: | ---: |
> | *Corvus corax* (common raven) | 45,000 | 4.65 |
> | *Danaus plexippus* (monarch) | 30,000 | 4.48 |
> | *Zootoca vivipara* (viviparous lizard) | 900 | 2.95 |
> | a sparsely-recorded sedge | 30 | 1.49 |
>
> The raven and the sedge differ by ~3 units of exposure &mdash; three decades of abundance &mdash;
> which the linear counts (45,000 vs 30) would exaggerate into visual noise.

### 5.2 Item-weighted, taxon-deterministic tertiles

Within each reference cell, distinct taxa are sorted by `(x, taxon name)` &mdash; the name breaking
ties deterministically. Each taxon is then assigned a **mid-rank percentile** weighted by the number of
items that target it:

```
              cum(t) + ½·n(t)
    p(t)  =  ──────────────────         cum(t) = Σ n(u) for u preceding t (rarer taxa)
                  N_cell                n(t)   = items targeting t
                                        N_cell = Σ n(u) over the cell
```

```
    p(t) <  ⅓   →  Hard          rarest third of the item mass
    p(t) <  ⅔   →  Medium
    otherwise   →  Easy          most abundant third
```

**Derivation &mdash; why mid-rank, and why weighted.** Two requirements pin the formula down:

1. *A tier must be a set of taxa, not a partition of items.* If a single high-count taxon were split
   across a boundary, two identical questions about the same organism could land in different strata,
   and per-tier accuracy would no longer be readable as accuracy over taxa. So each taxon must receive
   **exactly one** percentile.
2. *The three strata must be comparably sized in item mass,* to preserve statistical power in per-tier
   accuracy estimates.

Sorting taxa by exposure and walking the cumulative item mass, the block occupied by taxon *t* runs
from `cum(t)` to `cum(t) + n(t)`. Assigning *t* the **midpoint** of its own block,
`cum(t) + ½·n(t)`, and normalising by `N_cell`, gives one number per taxon that lands the whole taxon
on whichever side of ⅓ or ⅔ its centre of mass falls. This is the **weighted mid-rank (Hazen) plotting
position** [9]; the midpoint (rather than the leading or trailing edge of the block) is what removes
the boundary-splitting failure of a naive cut on the sorted item list, and the item-count weighting is
what keeps the three strata comparable in size.

> **Worked tier assignment** (illustrative, one reference cell):
>
> Suppose a `(Birds, species_level)` cell contains many taxa; after sorting rarest-first and
> accumulating item mass, three representative taxa fall at:
>
> | taxon | `x` | `p(t)` | tier | reading |
> | :--- | ---: | ---: | :--- | :--- |
> | sparsely-recorded warbler | 1.6 | 0.14 | **Hard** | in the rarest third &mdash; little visual exposure |
> | mid-abundance thrush | 3.1 | 0.52 | **Medium** | middle of the distribution |
> | *Corvus corax* | 4.65 | 0.91 | **Easy** | in the most-abundant third &mdash; heavily photographed |

### 5.3 Reference cells

Tertiles are cut within a **reference cell**, `(group, rank_class)`. The `rank_class` collapses the
three species-level phrasings so difficulty is a property of the organism, not of how the question was
worded:

| Detected rank | `rank_class` |
| :--- | :--- |
| `phylum` | `phylum` |
| `class` | `class` |
| `order` | `order` |
| `family` | `family` |
| `genus` | `genus` |
| `species` (epithet) | `species_level` |
| `scientific_name` | `species_level` |
| `common_name` | `species_level` |
| *any MCQ item* | `mcq` |

Cutting **within a rank** prevents the trivial confound that higher ranks always have larger subtree
counts (every class necessarily contains more images than any one of its families); cutting **within a
group** makes "rare" mean rare relative to the peers a model is being asked to distinguish.

---

## 6. Construct II — Confusability Tier

Applies to 4-way multiple-choice items, where prevalence is the wrong construct: recognising a *common*
organism among three congeners is harder than recognising a *rare* one among three distant relatives.
The difficulty of a choice item is governed by the semantic proximity of its negatives [4, 5, 6].

### 6.1 Nearest-Distractor Distance

For ground-truth node *g* and each distractor *j*:

```
    d_j  =  depth(g)  −  depth( LCA(g, j) )        ranks climbed from the answer
                                                    to a node that also contains j
    NDD  =  min_j  d_j
```

```
    NDD ≤ 1   →  Hard        distractor shares the genus (or is a congener)
    NDD = 2   →  Medium      distractor shares the family
    NDD ≥ 3   →  Easy        nearest distractor is at order rank or above
```

**Derivation &mdash; why LCA depth, and why the minimum.** The height of the lowest common ancestor in
a class hierarchy is the standard measure of how *severe* a confusion is [4, 5, 6, 7]: two taxa that
meet only at the root are maximally dissimilar, two that meet at the genus are nearly the same thing.
`d_j = depth(g) − depth(LCA(g, j))` is exactly the number of Linnaean ranks one must climb from the
answer before reaching a node that also contains distractor *j* &mdash; small `d_j` means a close,
confusable negative. Taking the **minimum** over the three distractors (`NDD`) encodes a *weakest-link*
model of a multiple-choice item: a single genus-level distractor makes the item hard *even if* the
other two options are from another phylum, because the model must still perform the fine discrimination
to exclude that one near option. The maximum or the mean would let two easy distractors mask one hard
one; the minimum reports the hardest discrimination the item actually demands.

### 6.2 Continuous companion

A graded score accompanies the tier, with one rank of separation **halving** credit:

```
    confusability  =  mean_j  exp( − ln2 · d_j )      note:  exp(−ln2·d) = 2^(−d)
```

**Why exponential, and why base-2 decay.** `exp(−ln2 · d) = 2^(−d)` yields `1.0, 0.5, 0.25, 0.125, …`
for `d = 0, 1, 2, 3, …`: **each additional rank of separation halves a distractor's contribution to
difficulty.** This matches the intuition that closeness in a hierarchy matters multiplicatively, not
additively &mdash; the gap between "same genus" and "same family" should weigh far more than the gap
between "same order" and "same class" &mdash; and it gives the continuous score the same
graded-hierarchical-cost interpretation used elsewhere in hierarchy-aware evaluation [5, 6]. Averaging
over the three distractors (rather than taking the minimum, as the tier does) makes `confusability` a
*profile* summary that complements the *floor* summary `NDD`: analyses needing a continuous covariate
&mdash; e.g. regressing accuracy on option proximity &mdash; use `confusability`, while the three-level
factor uses `NDD`.

### 6.3 Worked example

An item whose answer is *Zootoca vivipara* (depth 6) with distractors at another family, another class,
and another phylum:

| Distractor | LCA | LCA depth | `d_j` | `2^(−d_j)` |
| :--- | :--- | :-: | :-: | :-: |
| Same family, different genus | family | 4 | 2 | 0.250 |
| Same class, different family | class | 2 | 4 | 0.063 |
| Different phylum | root | 0 | 6 | 0.016 |

`NDD = min(2, 4, 6) = 2` &rarr; **Medium**;  `confusability = mean(0.250, 0.063, 0.016) = 0.110`.

> **Contrast &mdash; a Hard item.** Replace the first distractor with a *congener*
> (same genus, different species): `LCA = genus (depth 5)`, `d = 1`, contribution `0.500`. Now
> `NDD = 1` &rarr; **Hard**, and `confusability = mean(0.500, 0.063, 0.016) = 0.193`. The single close
> negative flips the tier, exactly as the weakest-link minimum intends.
>
> **Contrast &mdash; an Easy item.** If all three distractors are from other phyla,
> `d_j = 6` for each, `NDD = 6` &rarr; **Easy**, `confusability ≈ 0.016` &mdash; the model need only
> tell an arthropod from a chordate.

---

## Target Resolution

Every item is resolved to exactly one node. Because each image's answer key specifies its complete
lineage, resolution never depends on a single answer string being individually findable. A four-step
ladder is applied, and the step that succeeded is recorded per item.

```
  ┌──────────────┐   the answer names a node filed at the asked rank
  │  1. exact    │   e.g.  "Lacertidae" asked as family
  └──────┬───────┘
         │ miss
  ┌──────▼───────┐   a common name, via the vernacular index (§ Vernacular)
  │ 2. vernacular│   e.g.  "Viviparous Lizard" → Zootoca vivipara
  └──────┬───────┘
         │ miss
  ┌──────▼───────┐   the asked rank read off this image's resolved lineage
  │  3. lineage  │   absorbs rank-name disagreement between answer key
  └──────┬───────┘   and backbone (e.g. post-promotion class placement)
         │ miss
  ┌──────▼───────┐   lowest resolvable node in this image's lineage
  │  4. anchor   │   species → genus → family → order → class → phylum
  └──────────────┘
```

**The lineage step is what makes the ladder robust.** Since the image index resolves each
`(group, uuid)` once &mdash; from its scientific name, or from its genus and epithet &mdash; every rank
of that organism is available regardless of whether the individual answer string happens to match a
node label. A question asking for the class of an organism resolves *through the organism*, not through
the word.

**Worked resolution examples:**

| Item (question &rarr; answer) | Step | Why the earlier steps missed / this step fired |
| :--- | :--- | :--- |
| *"Which family?"* &rarr; `Lacertidae` | `exact` | `Lacertidae` is a node filed at family rank; the string matches directly. |
| *"Common name?"* &rarr; `Orb weavers` | `vernacular` | Not a Linnaean node label, so `exact` misses; the vernacular index maps it to **Araneidae** (family). |
| *"Species epithet?"* &rarr; `vivipara` | `lineage` | The bare epithet `vivipara` is ambiguous (many taxa share it), so no unique node matches; the species node is read off *this image's* resolved lineage, **Zootoca vivipara**. |
| *"Which order?"* for an organism whose backbone `order` is null | `anchor` | No node matches the asked rank and the lineage has no order to read; the item anchors to the lowest resolvable node in the image's lineage (e.g. its **genus**). |

The `resolution` column takes one of six values:

| Value | Meaning |
| :--- | :--- |
| `exact` | Answer string matched a node at the asked rank |
| `vernacular` | Common name resolved through the vernacular index |
| `lineage` | Rank read from the image's resolved lineage |
| `anchor` | Fell back to the lowest resolvable node for the image |
| `mcq` | Multiple-choice item scored against resolvable distractors |
| `mcq_no_distractor` | No option placeable in the tree; scored at root distance |

---

## Vernacular Names and Rank Heterogeneity

Common names are not a parallel nomenclature for species. In biodiversity corpora they denote taxa at
**whatever rank a naturalist finds useful**, and that rank varies by clade. Birds and mammals carry
species-level vernaculars almost universally; arachnids and bryophytes frequently do not, and their
familiar names refer to families or orders instead. *Orb weavers* is Araneidae. *Long-jawed orb
weavers* is Tetragnathidae. *Daddy longlegs* is an order. A resolution scheme that assumes one common
name maps to one species will mis-handle an entire phylum's worth of items.

**The vernacular index is derived from the corpus, not assumed from outside.** Each image contributes
the pair *(its common-name answer, its resolved species)*. A name observed against a single species
resolves to that species. A name observed against several species resolves to their **lowest common
ancestor**, which by construction is the taxon the name actually denotes:

```
   "Viviparous Lizard"       → { Zootoca vivipara }
                             → Zootoca vivipara               (species)

   "Long-jawed orb weavers"  → { Tetragnatha extensa,
                                 Tetragnatha montana, … }
                             → Tetragnathidae                 (family)

   "Orb weavers"             → { Araneus …, Argiope …,
                                 Neoscona …, … }
                             → Araneidae                      (family)
```

Three properties follow:

- **Rank is inferred, not asserted.** The index discovers that a name is family-level because the
  corpus uses it that way. No manual rank annotation and no external vernacular authority is required.
- **Exposure matches the question.** A family-level name resolves to a family node, so its prevalence
  is the family's &mdash; correctly reflecting that naming a spider *family* is an easier task than
  naming the species.
- **Distractors resolve.** MCQ distractors are drawn from the same pool of common names as the answers,
  so an index built from the answer keys covers them by construction. The catalog's own vernacular
  column is consulted as a secondary source, and lookups are tolerant of singular/plural variation.

> **Consequence for the `species_level` cell.** Because family-level vernaculars resolve to family
> nodes, the `species_level` reference cell contains a small number of targets at coarser rank,
> carrying correspondingly larger exposure. This is deliberate &mdash; it encodes a real difference in
> task difficulty &mdash; but it means the cell is not homogeneous in rank. The `target_taxon` and
> `resolution` columns make every such item identifiable, and `target_rank` preserves the phrasing that
> was asked.

---

## Tier Assignment, Degenerate Cells & Reference Populations

A tertile split is only meaningful when the reference cell contains enough distinct taxa to define one.
Some cells are structurally thin: within a single organismal group, every phylum answer may be
identical, and class answers may take two or three values.

**Thin cells escalate rather than fail.** A cell with fewer than `MIN_TAXA = 6` distinct taxa is re-cut
against that `rank_class` pooled across **all** groups. Since exposure is already a global quantity, the
pooled cut is a coherent reference population &mdash; it asks how prevalent a class is among all catalog
images rather than among one group's images. Only if the pooled cell is *also* too thin is it labelled
`Degenerate`.

```
   (group, rank_class)  ──  ≥ 6 taxa  ──►  tertiles, group-relative
            │
            └─ < 6 taxa ─►  (rank_class, all groups)  ── ≥ 6 taxa ──►  tertiles, global
                                      │
                                      └─ < 6 taxa ─►  Degenerate
```

**What `Degenerate` means, and why it is not a failure.** `Degenerate` marks items for which **no
prevalence contrast exists to measure** &mdash; the cell simply does not contain enough distinct taxa
to define a rarest/median/most-abundant split. It is a *substantive* label: the item still has a
resolved `target_taxon` and `exposure_count`; it is merely excluded from prevalence-stratified
analyses, because stratifying two or three taxa into three tiers is meaningless.

> **Why phylum questions degenerate but class questions usually do not.**
>
> - *Phylum, within Birds.* Every bird is phylum **Chordata** &mdash; the `(Birds, phylum)` cell holds
>   **one** distinct taxon. It escalates to `(phylum, all groups)`, but the whole benchmark spans only
>   a handful of phyla (Chordata, Arthropoda, and the plant/bryophyte phyla) &mdash; still **< 6**. So
>   the pooled cell is degenerate too, and every phylum item is labelled `Degenerate`. There is
>   genuinely no "rare phylum vs common phylum" contrast to report.
> - *Class, within Birds.* The `(Birds, class)` cell holds only **Aves** &mdash; one taxon, thin. But
>   escalating to `(class, all groups)` pools Amphibia, Aves, Mammalia, Insecta, Arachnida, Reptilia's
>   promoted classes, and more &mdash; comfortably **≥ 6**. So class questions are tiered against a
>   *global* class-abundance distribution rather than degenerating: a class represented by few catalog
>   images (e.g. a sparse bryophyte class) is Hard, a class with millions (Insecta, Aves) is Easy.
>
> This is the escalation ladder resolving one thin cell (class) while correctly declining to
> manufacture a contrast that does not exist (phylum).

Every escalation and every degenerate cell is announced in the run log, so the reference population
behind each tier is recoverable from the run record.

---

## Emitted Columns

Thirteen columns are attached to each processed CSV.

| Column | Type | Description |
| :--- | :--- | :--- |
| `target_taxon` | str | Display name of the resolved node |
| `target_rank` | str | Rank the question asked for, as detected from its text |
| `rank_class` | str | Reference cell for tiering (&sect;5.3); `mcq` for choice items |
| `resolution` | str | How the target was resolved (§ Target Resolution) |
| `exposure_count` | float | Catalog images in the target's subtree |
| `log_exposure` | float | `log10(exposure_count + 1)` |
| `prevalence_pct` | float | Item-weighted mid-rank percentile within the cell |
| `prevalence_tier` | str | `Hard` \| `Medium` \| `Easy` \| `Degenerate` |
| `mcq_ndd` | float | Nearest-distractor distance, in ranks |
| `mcq_nearest_lca_rank` | str | Rank at which the nearest distractor meets the answer |
| `mcq_confusability` | float | `mean_j exp(−ln2·d_j)` |
| `mcq_tier` | str | `Hard` \| `Medium` \| `Easy` |
| `difficulty` | str | Unified: `mcq_tier` for choice items, else `prevalence_tier` |

`difficulty` is the column intended for stratified reporting. The constituent columns are retained so
that the tier can be recomputed, audited, or replaced by the continuous covariates without re-running
the pipeline.

---

## Usage

### Discovery

Files are located by **substring match on the filename**, recursively beneath a root directory. Any CSV
whose name contains the key string is processed; previously written `_tiered.csv` outputs are excluded
so that repeated runs do not cascade.

```bash
python assign_difficulty.py \
    --base_dir  /path/to/hierarchical-taxonomic-reasoning \
    --keystring QA_finalized \
    --dry_run
```

`--dry_run` lists every matched file and its intended destination, then exits before the taxonomy is
built. Running it first is recommended, particularly before an in-place run.

### Scoring

```bash
python assign_difficulty.py \
    --base_dir   /path/to/hierarchical-taxonomic-reasoning \
    --parquet    /path/to/catalog.parquet \
    --tree_cache /path/to/tree.pkl \
    --keystring  QA_finalized \
    --mode       same
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--base_dir` | *required* | Root searched recursively for CSVs |
| `--keystring` | `QA_difficulty` | Filename substring selecting files to process |
| `--mode` | `separate` | `same` updates in place; `separate` writes `<name>_tiered.csv` |
| `--parquet` | — | Image catalog; required unless a valid cache exists |
| `--tree_cache` | — | Serialised taxonomy; built on first use, reused thereafter |
| `--dry_run` | off | List matched files and exit |

### Output modes

```
--mode same                          --mode separate
────────────────                     ────────────────
Group/model/data.csv                 Group/model/data.csv          (untouched)
        │ columns added/updated              │
        ▼                                    ▼
Group/model/data.csv                 Group/model/data_tiered.csv
```

**Both modes are idempotent.** If the difficulty columns are already present they are overwritten in
place, preserving column order; nothing is appended and no duplicate columns are created. A run may be
repeated safely after a change to the taxonomy or to the answer keys.

### Requirements

Input CSVs must carry a `uuid` column and a `qa_pairs` column holding a JSON object with keys `Q`
(question text) and `A` (answer). Files lacking these are reported and skipped rather than aborting the
run.

---

## Cluster Execution

The pipeline is single-node and memory-resident; the dominant cost is the one-time catalog aggregation.
A portable batch template:

```bash
#!/bin/bash
#SBATCH --job-name=assign_difficulty
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/assign_difficulty_%j.out

set -euo pipefail

BASE="/path/to/hierarchical-taxonomic-reasoning"
PARQ="/path/to/catalog.parquet"

python assign_difficulty.py \
    --base_dir   "${BASE}" \
    --parquet    "${PARQ}" \
    --tree_cache "${BASE}/tree.pkl" \
    --keystring  QA_finalized \
    --mode       same
```

Adjust `--cpus-per-task` to the parallelism available to the analytics engine and `--mem` to the
catalog size. Subsequent runs that reuse the cache complete substantially faster, as the aggregation is
skipped.

---

## Determinism and Reproducibility

**The pipeline contains no source of randomness.** There is no sampling, no shuffling and no seed.
Repeated execution over identical inputs produces byte-identical outputs.

| Mechanism | Guarantee |
| :--- | :--- |
| Ties in the exposure sort | Broken by taxon name, so ordering is total |
| Percentile assignment | Mid-rank formula; independent of iteration order |
| Item set construction | Deduplicated on `(group, uuid, question)` and sorted |
| File discovery | Sorted traversal |
| Taxonomy cache | Fingerprinted by catalog path, modification time and size |

The cache fingerprint is checked on every load. A catalog that has been replaced or modified
invalidates the cache automatically and the tree is rebuilt, so a stale backbone cannot silently
persist across runs.

**Scoring precedes writing.** The complete item set is assembled and scored before any file is written.
Consequently the tier assigned to a question does not depend on which subset of prediction files was
present, and a run over one group produces the same per-item tiers as a run over all groups, save for
the reference population of a pooled cell &mdash; which is itself logged.

---

## Audit Output

Each run closes with three summaries intended for the record:

```
[tiers]
Easy          ...
Medium        ...
Hard          ...
Degenerate    ...

[resolution]
exact              ...
vernacular         ...
lineage            ...
anchor             ...
mcq                ...
mcq_no_distractor  ...

[check] unresolved items: 0
```

The `[resolution]` block is the coverage statement for the component. The share resolved `exact`
quantifies how much of the stratification rests on direct name matching; the `lineage`, `vernacular`
and `anchor` counts quantify the remainder. Because the mechanism is recorded per item, any analysis
can be repeated on the `exact` subset alone as a robustness check [8], and the stability of model
rankings under that restriction can be reported.

The run log additionally records every reference-population escalation and every degenerate cell,
together with the number of distinct taxa that motivated it.

---

## Methodological Grounding

The two constructs are deliberately conventional. Each corresponds to an established practice in the
vision literature, applied here to a taxonomic setting.

**Frequency-stratified reporting.** Partitioning classes by their frequency in the underlying data
distribution and reporting accuracy per band is the standard protocol in long-tailed recognition, where
many-shot / medium-shot / few-shot splits are routine [1, 2, 3]. The prevalence tier is that protocol
with catalog subtree counts standing in for training-set class frequency &mdash; the appropriate
analogue when the evaluated systems are pretrained on broad image collections rather than on a fixed
labelled training split. Reporting the three tiers alongside the aggregate is what allows head
performance and tail performance to be distinguished.

**Hierarchy-aware distance.** Measuring the severity of a confusion by the height of the lowest common
ancestor in a class hierarchy is long-established in large-scale visual recognition [4, 5, 6], and the
underlying taxonomy-similarity measures (Wu&ndash;Palmer, Resnik, Lin) formalise "how close are two
nodes in a tree" precisely by reference to their common ancestor [7]. Nearest-distractor distance
applies this to the *option set* rather than to the prediction: it characterises how much hierarchical
discrimination an item demands before any model is run. The exponential (base-2) decay with rank
distance gives the continuous companion the same interpretation as the graded hierarchical costs used
elsewhere [5, 6].

**Distractor proximity in multiple choice.** That the difficulty of a multiple-choice item is governed
by the semantic proximity of its negatives is a standard assumption in benchmark construction, and
controlling or at least measuring that proximity is expected practice when choice items are used
alongside open generation.

**Provenance for derived annotations.** Recording how each derived label was obtained, and confirming
that headline conclusions survive restriction to the most conservatively derived subset, is ordinary
diligence for a dataset contribution [8]. The `resolution` column exists for that purpose.

None of these constructs is novel in isolation. Their combination &mdash; taxonomic prevalence for open
recall, hierarchical distractor distance for choice items, both cut within rank so that the two are
comparable across a six-level Linnaean hierarchy &mdash; is what makes per-tier accuracy on this
component interpretable.

---

## Limitations

**Exposure is a proxy for training exposure.** Catalog subtree counts describe one large biodiversity
image collection. They correlate with, but are not identical to, the distribution any particular
pretrained model observed. The prevalence tier should be read as a taxon-abundance stratification, not
as a measurement of a specific model's training data.

**Vernacular targets are heterogeneous in rank.** As described above, common names denote taxa at
different ranks depending on clade, and the resolution scheme reflects that faithfully rather than
forcing a species-level interpretation. The consequence is that the `species_level` reference cell
mixes granularities. Items are individually identifiable through `target_taxon` and `target_rank`, and
analyses requiring rank homogeneity should filter accordingly.

**Tertile boundaries are a convention.** Thirds are chosen for balance and convention, not derived from
any property of the distribution. `log_exposure` and `prevalence_pct` are retained precisely so that
alternative cuts &mdash; quartiles, or a continuous treatment &mdash; can be substituted without
recomputation.

**Nearest-distractor distance is a floor, not a profile.** Taking the minimum over distractors captures
the hardest single comparison an item demands and discards the rest of the option set. The
`mcq_confusability` mean is provided as the complementary summary; neither alone fully characterizes a
four-way item.

**Phylum-level items admit no prevalence contrast.** With only a handful of phyla represented, the
phylum cell remains degenerate even under global pooling. These items carry resolved targets and
exposure counts but no tier, and are excluded from prevalence-stratified analysis by construction.

**Rank distance assumes a uniform Linnaean depth.** Distances are counted in ranks, so two clades
separated by the same number of ranks are treated as equally distant regardless of their actual
phylogenetic divergence or of how finely their lineage is subdivided in the backbone.

---

## References

The two constructs and the diligence practices they rest on are drawn from the following. None is
reproduced or extended here; they are cited to locate this component's design within established
protocol.

1. Z. Liu, Z. Miao, X. Zhan, J. Wang, B. Gong, S. X. Yu. *Large-Scale Long-Tailed Recognition in an
   Open World.* CVPR, 2019. &mdash; many-shot / medium-shot / few-shot frequency bands.
2. G. Van Horn, P. Perona. *The Devil is in the Tails: Fine-grained Classification in the Wild.*
   arXiv:1709.01450, 2017. &mdash; long-tailed abundance in biodiversity imagery.
3. G. Van Horn, O. Mac Aodha, Y. Song, Y. Cui, C. Sun, A. Shepard, H. Adam, P. Perona, S. Belongie.
   *The iNaturalist Species Classification and Detection Dataset.* CVPR, 2018. &mdash; long-tailed,
   fine-grained species recognition.
4. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, L. Fei-Fei. *ImageNet: A Large-Scale Hierarchical
   Image Database.* CVPR, 2009. &mdash; a class hierarchy for large-scale recognition.
5. J. Deng, J. Krause, A. Berg, L. Fei-Fei. *Hedging Your Bets: Optimizing Accuracy-Specificity
   Trade-offs in Large Scale Visual Recognition.* CVPR, 2012. &mdash; hierarchy-aware cost of errors.
6. L. Bertinetto, R. Mueller, K. Tertikas, S. Samangooei, N. A. Lord. *Making Better Mistakes:
   Leveraging Class Hierarchies with Deep Networks.* CVPR, 2020. &mdash; LCA-height error severity and
   graded hierarchical cost.
7. Z. Wu, M. Palmer. *Verb Semantics and Lexical Selection.* ACL, 1994; P. Resnik. *Using Information
   Content to Evaluate Semantic Similarity in a Taxonomy.* IJCAI, 1995; D. Lin. *An Information-
   Theoretic Definition of Similarity.* ICML, 1998. &mdash; common-ancestor taxonomy-similarity
   measures.
8. T. Gebru, J. Morgenstern, B. Vecchione, J. W. Vaughan, H. Wallach, H. Daumé III, K. Crawford.
   *Datasheets for Datasets.* Communications of the ACM, 2021. &mdash; provenance and documentation of
   derived annotations.
9. A. Hazen. *Storage to be Provided in Impounding Reservoirs for Municipal Water Supply.* Transactions
   ASCE, 1914. &mdash; the mid-rank ("Hazen") plotting position used for the taxon percentile.

---

<p align="center">
  <sub>Part of the <b>Bio-VQA</b> benchmark &middot; Imageomics Institute</sub>
</p>
