<div align="center">

# Difficulty Assignment

### Taxonomy-Aware Difficulty Stratification for Hierarchical Taxonomic Reasoning

<p>
<img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
<img src="https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white" alt="pandas">
<img src="https://img.shields.io/badge/DuckDB-analytics-FFF000?logo=duckdb&logoColor=black" alt="DuckDB">
<img src="https://img.shields.io/badge/scheduler-SLURM-4B8BBE" alt="SLURM">
<img src="https://img.shields.io/badge/determinism-seed--free-success" alt="Deterministic">
<img src="https://img.shields.io/badge/coverage-100%25%20resolved-brightgreen" alt="Coverage">
</p>

<p><em>A component of the Hierarchical Taxonomic Reasoning benchmark</em></p>

</div>

---

> **What this does.** Every question in the Hierarchical Taxonomic Reasoning
> component is assigned a difficulty tier derived from the taxonomy itself —
> never from model behaviour. Difficulty is a property of the *question*, so it
> is computed once over the union of all answer keys and propagated identically
> to every model's prediction file. Two models are therefore always compared on
> the same stratification.

---

## Contents

1. [Motivation](#1-motivation)
2. [Design Principles](#2-design-principles)
3. [Pipeline](#3-pipeline)
4. [Taxonomic Backbone](#4-taxonomic-backbone)
5. [Construct I — Prevalence Tier](#5-construct-i--prevalence-tier)
6. [Construct II — Confusability Tier](#6-construct-ii--confusability-tier)
7. [Target Resolution](#7-target-resolution)
8. [Vernacular Names and Rank Heterogeneity](#8-vernacular-names-and-rank-heterogeneity)
9. [Tier Assignment and Reference Populations](#9-tier-assignment-and-reference-populations)
10. [Emitted Columns](#10-emitted-columns)
11. [Usage](#11-usage)
12. [Cluster Execution](#12-cluster-execution)
13. [Determinism and Reproducibility](#13-determinism-and-reproducibility)
14. [Audit Output](#14-audit-output)
15. [Methodological Grounding](#15-methodological-grounding)
16. [Limitations](#16-limitations)

---

## 1. Motivation

Aggregate accuracy on a taxonomic VQA benchmark conflates two very different
competencies. A model that names *Corvus corax* correctly and a model that names
a rarely photographed sedge correctly have not demonstrated the same thing, and
a four-way multiple-choice item whose distractors are congeners is not the same
task as one whose distractors come from another phylum. Reporting a single mean
over such a corpus hides exactly the structure a taxonomic benchmark exists to
expose.

**Difficulty stratification restores that structure.** By partitioning items
along axes that are defined *before* any model sees them, per-tier accuracy
becomes interpretable: a flat profile across tiers indicates a model whose
performance is driven by something other than taxon prevalence, while a steep
profile indicates reliance on the head of the distribution. Both are findings.
Neither is visible in a single number.

Two axes are used, because the two question formats measure different things:

| Format | Axis | Question it answers |
|:--|:--|:--|
| Open recall | **Prevalence** | How much of the visual world does this taxon occupy? |
| 4-way MCQ | **Confusability** | How close is the nearest wrong option in the tree? |

---

## 2. Design Principles

**Difficulty is a property of the question, not the model.** Tiers are computed
from the taxonomy and the answer key alone. No prediction file influences any
tier. This is what allows the same stratification to be applied across an
arbitrary number of evaluated systems, and what prevents the circularity of
calling an item "hard" because models found it hard.

**One computation, many files.** The corpus of unique
`(group, uuid, question)` items is assembled across *every* matched CSV before
scoring. A single tier is computed per item and written to all files that
contain it. A model whose prediction file is short a few rows cannot shift a
tertile boundary and thereby receive a different stratification from its peers.

**Resolution is exhaustive and audited.** Every item resolves to a node in the
taxonomy. Where the answer string alone is insufficient, the image's own
lineage — which the answer key fully specifies — supplies the target. The
mechanism used is recorded per item in a `resolution` column, so the share of
non-direct resolutions is reportable and the corresponding items can be excluded
in a robustness check.

**Tiers are taxon-deterministic.** Two items sharing a target taxon always
receive the same tier. A taxon never straddles a boundary, so per-tier accuracy
can be read as accuracy over a set of taxa rather than over an arbitrary
partition of items.

**Minimal, idempotent output footprint.** Columns are added to the source files
or written beside them; re-running updates existing columns in place rather than
appending duplicates. No intermediate artefacts are left behind.

---

## 3. Pipeline

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
                         └──────────────────┬───────────────────────────┘
                                            ▼
                                 ┌─────────────────────┐
                                 │  PASS 2  write back │
                                 │  same │ separate    │
                                 └─────────────────────┘
```

---

## 4. Taxonomic Backbone

The tree is built by aggregating the full image catalog over the Linnaean
columns, producing one node per distinct taxon and one integer per node: the
number of catalog images in that node's subtree. This count is the raw material
for the prevalence construct.

**Ranks and depth.** Six ranks are modelled, with depth read from the rank name
rather than from position in the chain:

| Rank | phylum | class | order | family | genus | species |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| Depth | 1 | 2 | 3 | 4 | 5 | 6 |

**Incomplete lineages are first-class.** Contemporary taxonomic backbones do
not guarantee a populated value at every rank for every organism. Rank
promotions are routine — Catalogue of Life 2022 elevates Squamata, Testudines,
Crocodylia, Elasmobranchii and Teleostei to class rank, leaving no intervening
order beneath them. A backbone that required all six ranks would silently
exclude entire clades from the tree and, with them, every question about those
clades.

The tree therefore requires only a populated **genus** and **species**; any
intermediate rank may be absent, in which case the node attaches to its nearest
populated ancestor. A family whose order is null attaches directly to its class.
Because depth is a function of the rank label rather than of edge count, the
lowest-common-ancestor distances used by the confusability construct remain
correct across such a gap: the LCA of two species in different families is still
reported at family or above, and the numeric distance still counts ranks
climbed, not edges traversed.

> **Exposure is global.** Subtree counts are read from the complete catalog, not
> from the sampled evaluation images. Prevalence therefore reflects the
> distribution an image-trained model plausibly encountered, rather than the
> composition of the benchmark sample.

---

## 5. Construct I — Prevalence Tier

Applies to all open-recall items: `phylum`, `class`, `order`, `family`,
`genus`, `species` (epithet), `scientific_name`, `common_name`.

### 5.1 Exposure

For a target taxon *t*, exposure is the number of catalog images beneath it:

```
c(t)  = |{ images whose lineage passes through t }|
x(t)  = log10( c(t) + 1 )
```

The logarithm is applied because catalog abundance spans several orders of
magnitude; the unit pseudocount keeps the transform defined at zero and
compresses the extreme tail without discarding it.

### 5.2 Item-weighted, taxon-deterministic tertiles

Within each reference cell, distinct taxa are sorted by `(x, taxon name)` — the
name breaking ties deterministically. Each taxon is then assigned a **mid-rank
percentile** weighted by the number of items that target it:

```
              cum(t) + ½·n(t)
    p(t)  =  ──────────────────         cum(t) = Σ n(u) for u preceding t
                  N_cell                n(t)   = items targeting t
                                        N_cell = Σ n(u) over the cell
```

```
    p(t) <  ⅓   →  Hard          rarest third of the item mass
    p(t) <  ⅔   →  Medium
    otherwise   →  Easy          most abundant third
```

**Why mid-rank rather than a cut on the sorted item list.** A naive tertile over
items would split a single high-count taxon across two tiers, so two identical
questions about the same organism could land in different strata. The mid-rank
formulation assigns one percentile per *taxon*, guaranteeing that a tier is a
set of taxa. Weighting by item count keeps the three strata comparably sized,
which preserves statistical power in per-tier accuracy estimates.

### 5.3 Reference cells

Tertiles are cut within a **reference cell**, defined as
`(group, rank_class)`. The `rank_class` collapses the three species-level
phrasings so that difficulty is a property of the organism rather than of how
the question was worded:

| Detected rank | `rank_class` |
|:--|:--|
| `phylum` | `phylum` |
| `class` | `class` |
| `order` | `order` |
| `family` | `family` |
| `genus` | `genus` |
| `species` (epithet) | `species_level` |
| `scientific_name` | `species_level` |
| `common_name` | `species_level` |
| *any MCQ item* | `mcq` |

Cutting within a rank prevents the trivial confound that higher ranks always
have larger subtree counts; cutting within a group makes "rare" mean rare
relative to peers a model is being asked to distinguish.

---

## 6. Construct II — Confusability Tier

Applies to 4-way multiple-choice items, where prevalence is the wrong construct:
recognising a common organism among three congeners is harder than recognising a
rare one among three distant relatives.

### 6.1 Nearest-Distractor Distance

For ground-truth node *g* and each distractor *j*:

```
    d_j  =  depth(g)  −  depth( LCA(g, j) )        ranks climbed from the
                                                    answer to meet distractor j
    NDD  =  min_j  d_j
```

```
    NDD ≤ 1   →  Hard        distractor shares the genus (or is a congener)
    NDD = 2   →  Medium      distractor shares the family
    NDD ≥ 3   →  Easy        nearest distractor is at order rank or above
```

### 6.2 Continuous companion

A graded score accompanies the tier, with one rank of separation halving credit:

```
    confusability  =  mean_j  exp( − ln2 · d_j )
```

This is reported alongside the tier so that analyses needing a continuous
covariate — regression of accuracy on option proximity, for example — are not
forced back onto the three-level factor.

### 6.3 Worked example

An item whose answer is *Zootoca vivipara* (depth 6) with distractors at
another family, another class, and another phylum:

| Distractor | LCA | LCA depth | `d_j` | `exp(−ln2·d_j)` |
|:--|:--|:-:|:-:|:-:|
| Same family, different genus | family | 4 | 2 | 0.250 |
| Same class, different family | class | 2 | 4 | 0.063 |
| Different phylum | root | 0 | 6 | 0.016 |

`NDD = 2` → **Medium**;  `confusability = 0.110`.

---

## 7. Target Resolution

Every item is resolved to exactly one node. Because each image's answer key
specifies its complete lineage, resolution never depends on a single answer
string being individually findable. A four-step ladder is applied, and the step
that succeeded is recorded per item.

```
  ┌──────────────┐   the answer names a node filed at the asked rank
  │  1. exact    │   e.g.  "Lacertidae" asked as family
  └──────┬───────┘
         │ miss
  ┌──────▼───────┐   a common name, via the vernacular index (§8)
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

**The lineage step is what makes the ladder robust.** Since the image index
resolves each `(group, uuid)` once — from its scientific name, or from its
genus and epithet — every rank of that organism is available regardless of
whether the individual answer string happens to match a node label. A question
asking for the class of an organism resolves through the organism, not through
the word.

The `resolution` column takes one of six values:

| Value | Meaning |
|:--|:--|
| `exact` | Answer string matched a node at the asked rank |
| `vernacular` | Common name resolved through the vernacular index |
| `lineage` | Rank read from the image's resolved lineage |
| `anchor` | Fell back to the lowest resolvable node for the image |
| `mcq` | Multiple-choice item scored against resolvable distractors |
| `mcq_no_distractor` | No option placeable in the tree; scored at root distance |

---

## 8. Vernacular Names and Rank Heterogeneity

Common names are not a parallel nomenclature for species. In biodiversity
corpora they denote taxa at **whatever rank a naturalist finds useful**, and
that rank varies by clade. Birds and mammals carry species-level vernaculars
almost universally; arachnids and bryophytes frequently do not, and their
familiar names refer to families or orders instead. *Orb weavers* is Araneidae.
*Long-jawed orb weavers* is Tetragnathidae. *Daddy longlegs* is an order. A
resolution scheme that assumes one common name maps to one species will
mis-handle an entire phylum's worth of items.

**The vernacular index is derived from the corpus, not assumed from outside.**
Each image contributes the pair *(its common-name answer, its resolved
species)*. A name observed against a single species resolves to that species. A
name observed against several species resolves to their **lowest common
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

- **Rank is inferred, not asserted.** The index discovers that a name is
  family-level because the corpus uses it that way. No manual rank annotation
  and no external vernacular authority is required.
- **Exposure matches the question.** A family-level name resolves to a family
  node, so its prevalence is the family's — correctly reflecting that naming a
  spider family is an easier task than naming the species.
- **Distractors resolve.** Multiple-choice distractors are drawn from the same
  pool of common names as the answers, so an index built from the answer keys
  covers them by construction. The catalog's own vernacular column is consulted
  as a secondary source, and lookups are tolerant of singular/plural variation.

> **Consequence for the `species_level` cell.** Because family-level vernaculars
> resolve to family nodes, the `species_level` reference cell contains a small
> number of targets at coarser rank, carrying correspondingly larger exposure.
> This is deliberate — it encodes a real difference in task difficulty — but it
> means the cell is not homogeneous in rank. The `target_taxon` and `resolution`
> columns make every such item identifiable, and the `target_rank` column
> preserves the phrasing that was asked.

---

## 9. Tier Assignment and Reference Populations

A tertile split is only meaningful when the reference cell contains enough
distinct taxa to define one. Some cells are structurally thin: within a single
organismal group, every phylum answer may be identical, and class answers may
take two or three values.

**Thin cells escalate rather than fail.** A cell with fewer than `MIN_TAXA = 6`
distinct taxa is re-cut against that `rank_class` pooled across **all** groups.
Since exposure is already a global quantity, the pooled cut is a coherent
reference population — it asks how prevalent a class is among all catalog
images rather than among one group's images. Only if the pooled cell is also too
thin is it labelled `Degenerate`.

```
   (group, rank_class)  ──  ≥ 6 taxa  ──►  tertiles, group-relative
            │
            └─ < 6 taxa ─►  (rank_class, all groups)  ── ≥ 6 taxa ──►  tertiles, global
                                      │
                                      └─ < 6 taxa ─►  Degenerate
```

`Degenerate` is a substantive label, not a failure state: it marks items for
which no prevalence contrast exists to measure. Phylum-level questions are the
expected case, there being only a handful of phyla across the benchmark. Such
items retain a resolved `target_taxon` and `exposure_count`, and are simply
excluded from prevalence-stratified analyses.

Every escalation and every degenerate cell is announced in the run log, so the
reference population behind each tier is recoverable from the run record.

---

## 10. Emitted Columns

Thirteen columns are attached to each processed CSV.

| Column | Type | Description |
|:--|:--|:--|
| `target_taxon` | str | Display name of the resolved node |
| `target_rank` | str | Rank the question asked for, as detected from its text |
| `rank_class` | str | Reference cell for tiering (§5.3); `mcq` for choice items |
| `resolution` | str | How the target was resolved (§7) |
| `exposure_count` | float | Catalog images in the target's subtree |
| `log_exposure` | float | `log10(exposure_count + 1)` |
| `prevalence_pct` | float | Item-weighted mid-rank percentile within the cell |
| `prevalence_tier` | str | `Hard` \| `Medium` \| `Easy` \| `Degenerate` |
| `mcq_ndd` | float | Nearest-distractor distance, in ranks |
| `mcq_nearest_lca_rank` | str | Rank at which the nearest distractor meets the answer |
| `mcq_confusability` | float | `mean_j exp(−ln2·d_j)` |
| `mcq_tier` | str | `Hard` \| `Medium` \| `Easy` |
| `difficulty` | str | Unified: `mcq_tier` for choice items, else `prevalence_tier` |

`difficulty` is the column intended for stratified reporting. The constituent
columns are retained so that the tier can be recomputed, audited, or replaced by
the continuous covariates without re-running the pipeline.

---

## 11. Usage

### 11.1 Discovery

Files are located by **substring match on the filename**, recursively beneath a
root directory. Any CSV whose name contains the key string is processed;
previously written `_tiered.csv` outputs are excluded so that repeated runs do
not cascade.

```bash
python assign_difficulty.py \
    --base_dir  /path/to/hierarchical-taxonomic-reasoning \
    --keystring QA_finalized \
    --dry_run
```

`--dry_run` lists every matched file and its intended destination, then exits
before the taxonomy is built. Running it first is recommended, particularly
before an in-place run.

### 11.2 Scoring

```bash
python assign_difficulty.py \
    --base_dir   /path/to/hierarchical-taxonomic-reasoning \
    --parquet    /path/to/catalog.parquet \
    --tree_cache /path/to/tree.pkl \
    --keystring  QA_finalized \
    --mode       same
```

| Flag | Default | Description |
|:--|:--|:--|
| `--base_dir` | *required* | Root searched recursively for CSVs |
| `--keystring` | `QA_difficulty` | Filename substring selecting files to process |
| `--mode` | `separate` | `same` updates in place; `separate` writes `<name>_tiered.csv` |
| `--parquet` | — | Image catalog; required unless a valid cache exists |
| `--tree_cache` | — | Serialised taxonomy; built on first use, reused thereafter |
| `--dry_run` | off | List matched files and exit |

### 11.3 Output modes

```
--mode same                          --mode separate
────────────────                     ────────────────
Group/model/data.csv                 Group/model/data.csv          (untouched)
        │ columns added/updated              │
        ▼                                    ▼
Group/model/data.csv                 Group/model/data_tiered.csv
```

**Both modes are idempotent.** If the difficulty columns are already present
they are overwritten in place, preserving column order; nothing is appended and
no duplicate columns are created. A run may be repeated safely after a change to
the taxonomy or to the answer keys.

### 11.4 Requirements

Input CSVs must carry a `uuid` column and a `qa_pairs` column holding a JSON
object with keys `Q` (question text) and `A` (answer). Files lacking these are
reported and skipped rather than aborting the run.

---

## 12. Cluster Execution

The pipeline is single-node and memory-resident; the dominant cost is the
one-time catalog aggregation. A portable batch template:

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

Adjust `--cpus-per-task` to the parallelism available to the analytics engine
and `--mem` to the catalog size. Subsequent runs that reuse the cache complete
substantially faster, as the aggregation is skipped.

---

## 13. Determinism and Reproducibility

**The pipeline contains no source of randomness.** There is no sampling, no
shuffling and no seed. Repeated execution over identical inputs produces
byte-identical outputs.

| Mechanism | Guarantee |
|:--|:--|
| Ties in the exposure sort | Broken by taxon name, so ordering is total |
| Percentile assignment | Mid-rank formula; independent of iteration order |
| Item set construction | Deduplicated on `(group, uuid, question)` and sorted |
| File discovery | Sorted traversal |
| Taxonomy cache | Fingerprinted by catalog path, modification time and size |

The cache fingerprint is checked on every load. A catalog that has been replaced
or modified invalidates the cache automatically and the tree is rebuilt, so a
stale backbone cannot silently persist across runs.

**Scoring precedes writing.** The complete item set is assembled and scored
before any file is written. Consequently the tier assigned to a question does
not depend on which subset of prediction files was present, and a run over one
group produces the same per-item tiers as a run over all groups, save for the
reference population of a pooled cell — which is itself logged.

---

## 14. Audit Output

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

The `[resolution]` block is the coverage statement for the component. The share
resolved `exact` quantifies how much of the stratification rests on direct name
matching; the `lineage`, `vernacular` and `anchor` counts quantify the
remainder. Because the mechanism is recorded per item, any analysis can be
repeated on the `exact` subset alone as a robustness check, and the stability of
model rankings under that restriction can be reported.

The run log additionally records every reference-population escalation and every
degenerate cell, together with the number of distinct taxa that motivated it.

---

## 15. Methodological Grounding

The two constructs are deliberately conventional. Each corresponds to an
established practice in the vision literature, applied here to a taxonomic
setting.

**Frequency-stratified reporting.** Partitioning classes by their frequency in
the underlying data distribution and reporting accuracy per band is the standard
protocol in long-tailed recognition, where many-shot / medium-shot / few-shot
splits are routine. The prevalence tier is that protocol with catalog subtree
counts standing in for training-set class frequency — the appropriate analogue
when the evaluated systems are pretrained on broad image collections rather than
on a fixed labelled training split. Reporting the three tiers alongside the
aggregate is what allows head performance and tail performance to be
distinguished.

**Hierarchy-aware distance.** Measuring the severity of a confusion by the
height of the lowest common ancestor in a class hierarchy is long-established in
large-scale visual recognition, where the semantic hierarchy is used precisely
because a flat error rate treats all mistakes as equivalent. Nearest-distractor
distance applies the same measure to the *option set* rather than to the
prediction: it characterises how much hierarchical discrimination an item
demands before any model is run. The exponential decay with rank distance gives
the continuous companion the same interpretation as graded hierarchical scores
used elsewhere.

**Distractor proximity in multiple choice.** That the difficulty of a
multiple-choice item is governed by the semantic proximity of its negatives is a
standard assumption in benchmark construction, and controlling or at least
measuring that proximity is expected practice when choice items are used
alongside open generation.

**Provenance for derived annotations.** Recording how each derived label was
obtained, and confirming that headline conclusions survive restriction to the
most conservatively derived subset, is ordinary diligence for a dataset
contribution. The `resolution` column exists for that purpose.

None of these constructs is novel in isolation. Their combination — taxonomic
prevalence for open recall, hierarchical distractor distance for choice items,
both cut within rank so that the two are comparable across a six-level Linnaean
hierarchy — is what makes per-tier accuracy on this component interpretable.

---

## 16. Limitations

**Exposure is a proxy for training exposure.** Catalog subtree counts describe
one large biodiversity image collection. They correlate with, but are not
identical to, the distribution any particular pretrained model observed. The
prevalence tier should be read as a taxon-abundance stratification, not as a
measurement of a specific model's training data.

**Vernacular targets are heterogeneous in rank.** As described in §8, common
names denote taxa at different ranks depending on clade, and the resolution
scheme reflects that faithfully rather than forcing a species-level
interpretation. The consequence is that the `species_level` reference cell
mixes granularities. Items are individually identifiable through `target_taxon`
and `target_rank`, and analyses requiring rank homogeneity should filter
accordingly.

**Tertile boundaries are a convention.** Thirds are chosen for balance and
convention, not derived from any property of the distribution. `log_exposure`
and `prevalence_pct` are retained precisely so that alternative cuts —
quartiles, or a continuous treatment — can be substituted without recomputation.

**Nearest-distractor distance is a floor, not a profile.** Taking the minimum
over distractors captures the hardest single comparison an item demands and
discards the rest of the option set. The `mcq_confusability` mean is provided as
the complementary summary; neither alone fully characterises a four-way item.

**Phylum-level items admit no prevalence contrast.** With only a handful of
phyla represented, the phylum cell remains degenerate even under global pooling.
These items carry resolved targets and exposure counts but no tier, and are
excluded from prevalence-stratified analysis by construction.

**Rank distance assumes a uniform Linnaean depth.** Distances are counted in
ranks, so two clades separated by the same number of ranks are treated as
equally distant regardless of their actual phylogenetic divergence or of how
finely their lineage is subdivided in the backbone.

---

<div align="center">

**Hierarchical Taxonomic Reasoning** · Difficulty Assignment

<sub>Developed under the <strong>Imageomics Institute</strong></sub>

</div>