<h1 align="center"> Bio-VQA </h1>

<p align="center">
  <b>A Biological Visual Question Answering Benchmark across Four Components</b>
</p>

<p align="center">
  <img alt="Questions" src="https://img.shields.io/badge/questions-1.87M-green">
  <img alt="Images" src="https://img.shields.io/badge/images-216K-blue">
  <img alt="Datasets" src="https://img.shields.io/badge/source%20datasets-11-yellow">
  <img alt="Groups" src="https://img.shields.io/badge/organismal%20groups-10-orange">
  <img alt="Models" src="https://img.shields.io/badge/models%20evaluated-18-purple">
  <img alt="Params" src="https://img.shields.io/badge/params-3.75B%E2%80%9338.4B-informational">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-blue">
</p>

---

## Overview

In biodiversity monitoring a single image raises several questions: which taxon,
how many, which trait, and whether the individual has been seen before. Each is
currently answered by its own specialist model, or by a field expert. Bio-VQA
asks whether a generalist vision-language model can answer all four.

The benchmark comprises **1.87M questions over 216K images drawn from 11 public
datasets**, organised into four components:

| | Component | What it asks | Questions | Source |
| :---: | :--- | :--- | ---: | :--- |
| **C1** | [Hierarchical taxonomic reasoning](C1_Hierarchical_Taxonomic_Reasoning/) | *Which taxon, at eight ranks?* | ~890K | TreeOfLife-200M |
| **C2** | [Fine-grained attribute grounding](C2_Fine_Grained_Attribute_Grounding/) | *Which trait, on which part, on which side?* | 824,104 | CUB-200-2011 |
| **C3** | [Quantitative spatial reasoning](C3_Quantitative_Spatial_Reasoning/) | *How many, how far, where, facing where?* | 141,516 | 5 ecological corpora |
| **C4** | [Individual re-identification](C4_Individual_Re_Identification/) | *Is this the same animal?* | 15,000 | 5 wildlife catalogues |

> **Design contract:** a question is generated **only where an existing
> annotation supplies the ground truth**, so every answer can be checked against
> its source. And every task is bounded from at least one side — by an explicit
> image-blind floor wherever one is definable, by a domain specialist wherever
> one exists.

---

## Table of Contents

- [What Bio-VQA Adds](#what-bio-vqa-adds)
- [The Four Components](#the-four-components)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Evaluation Protocol](#evaluation-protocol)
- [Metrics](#metrics)
- [Headline Results](#headline-results)
- [Cross-Component Structure](#cross-component-structure)
- [Reproducibility](#reproducibility)
- [References](#references)

---

## What Bio-VQA Adds

Existing biological VQA evaluation falls short in four ways, and each maps to a
design decision here.

| Gap in prior work | Bio-VQA |
| :--- | :--- |
| **Narrow coverage** — a few animal groups | **10 organismal groups**, including arachnids and bryophytes, absent from prior benchmarks |
| **Narrow scope** — mostly taxonomic naming | **Four components**: taxonomy, attributes, spatial quantification, individual identity |
| **Fixed scoring** — exact match penalises a near-miss like a total miss | **Graded metrics** throughout, each reported *alongside* the exact match it modifies |
| **Missing baselines** — no floors, no ceilings | **Image-blind floors** on every definable task; **specialist ceilings** where one exists |

Comparison with the closest prior benchmarks:

| Benchmark | Domain | Questions | Images | Groups | Taxon | Attr. | Spatial | Individual | Floor |
| :--- | :--- | ---: | ---: | ---: | :---: | :---: | :---: | :---: | :---: |
| VLM4Bio | biology | 469K | 30K | 3 | ✔ | ✔ | ✘ | ✘ | ✘ |
| PlantExpertVQA | biology | 765K | 151K | 1 | ~ | ✔ | ✘ | ✘ | ✘ |
| J-EDI QA | biology | 0.1K | 0.1K | 1 | ✔ | ✘ | ✘ | ✘ | ✘ |
| AVA-Bench | general | 218K | 218K | — | ✘ | ✔ | ✔ | ✘ | ~ |
| **Bio-VQA** | biology | **1.87M** | **216K** | **10** | ✔ | ✔ | ✔ | **✔** | **✔** |

---

## The Four Components

```
                             ┌───────────────────────┐
                             │        Bio-VQA        │
                             │    1.87M questions    │
                             │      216K images      │
                             └───────────┬───────────┘
                                         │
           ┌───────────────────┬─────────┴─────────┬───────────────────┐
           ▼                   ▼                   ▼                   ▼
  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
  │       C1       │  │       C2       │  │       C3       │  │       C4       │
  │    taxonomy    │  │   attribute    │  │    spatial     │  │   individual   │
  │                │  │   grounding    │  │   reasoning    │  │     re-ID      │
  ├────────────────┤  ├────────────────┤  ├────────────────┤  ├────────────────┤
  │    8 ranks     │  │   4 question   │  │    7 tasks     │  │  3 difficulty  │
  │   10 groups    │  │   categories   │  │  + 1 control   │  │     levels     │
  ├────────────────┤  ├────────────────┤  ├────────────────┤  ├────────────────┤
  │    ceiling:    │  │    ceiling:    │  │     floor:     │  │    ceiling:    │
  │   BioCLIP-2    │  │  3 annotators  │  │  image-blind   │  │ MegaDescriptor │
  └────────────────┘  └────────────────┘  └────────────────┘  └────────────────┘

```




Each component directory carries a full README with its construction,
metrics and results.

---

## Repository Layout

```
Bio-VQA/
├── README.md                                    # you are here
├── requirements.txt
├── C1_Hierarchical_Taxonomic_Reasoning/
│   ├── README.md
│   └── Tree-of-Life-200M/                       # download, verify, generateQA
├── C2_Fine_Grained_Attribute_Grounding/
│   ├── README.md
│   └── CUB-200-2011/
├── C3_Quantitative_Spatial_Reasoning/
│   ├── README.md
│   ├── Counting/          Viewpoint/            Distance_Absolute/
│   ├── Distance_Relative/ OCR/                  Counting_Relative/
│   └── Localization/      Relative_Position/
└── C4_Individual_Re_Identification/
    ├── README.md
    └── <five wildlife catalogues>
```

Every task directory follows the same four-stage structure:

```
<task>/
├── 1.Generate_QA/          question synthesis from annotations
├── 2.Experiment_Runs/      inference across 18 models
│   └── Preprocess/         manifest assembly and pre-run audit
└── 3.Evaluate/             scoring, metrics, and results
```

No shell scripts are committed. All launch commands live in the READMEs.

---

## Prerequisites

```bash
python -m venv myenv && source myenv/bin/activate
pip install -r requirements.txt
```

> ⚠️ **The 18 models do not share a `transformers` version.** Gemma-4 needs git
> HEAD, Qwen3-VL needs a current release, the other 16 are pinned to `4.57.1`.
> Each SLURM array task therefore builds a **disposable virtual environment in
> node-local `$TMPDIR`**, removed by a `trap` on exit. `requirements.txt` pins
> the majority case; see its section 3.

**Hardware.** Image download (C1) is network-bound — request CPU nodes.
Inference needs one H100-class GPU per array task; the 30B–38B models need two.

---

## Evaluation Protocol

Identical across all four components:

| Setting | Value |
| :--- | :--- |
| Decoding | **greedy** (`do_sample=False`) |
| System prompt | fixed per question family |
| Chain-of-thought | **none** |
| Few-shot examples | **none** |
| Attempts per item | **one** |
| Parser | a single parser reads all outputs |

Parse rates and letter–text mismatch rates are reported for every model, so a
formatting failure is never mistaken for a knowledge error.

**The 18 models** span 3.75B–38.4B parameters across eleven families: Qwen2.5-VL,
Qwen3-VL (dense and MoE), InternVL3, InternVL3.5 (dense and MoE), Gemma-3,
Gemma-4, Molmo 2, Pixtral, Phi-3.5-Vision, Llama-3.2-Vision, and two LLaVA-NeXT
variants. Two of the 30B models are mixture-of-experts activating 3B parameters
per token.

**Within each component, runner scripts are byte-identical outside a single
`MODEL BLOCK`.** This is verifiable:

```bash
for f in *.py; do sed -n '/^# HARNESS/,$p' "$f" | md5sum; done | sort -u | wc -l   # 1
```

---

## Metrics

Exact match is reported for every task, and every graded metric is reported
**alongside** the exact match it modifies, so no claim rests on the graded score
alone.

| Component | Graded metric | What it fixes |
| :--- | :--- | :--- |
| **C1** | Wu–Palmer, Lin similarity, soft EM, severity | A wrong family and a wrong kingdom are not the same error |
| **C2** | **CAGS** — chance-adjusted graded score | Answer spaces have structure *and* uneven cardinality |
| **C3** | **CAS** — bounded count agreement; lift over an image-blind floor | 14 beetles when the truth is 15 is not the same as 10 |
| **C4** | **d′** and criterion *c* from signal detection theory | A model answering *yes* to everything scores 50% |

### The floor, and lift over it

For C3, $\phi$ is the highest score an image-blind rule can reach:

$$
\phi = \max_{g_0 \in \mathcal{G}_0} \mathbb{E}\bigl[S(g_0, y)\bigr],
\qquad
\mathrm{lift} = \frac{S - \phi}{1 - \phi}
$$

A lift of 0 means the model does no better than a blind guess; **negative lift
means it does worse.** Two C3 tasks fall below their floor.

---

## Headline Results

Three things hold across the whole table:

- **No model comes close to the specialist on C1.**
- **No model sits far above the floor on C3.**
- **No model reaches $d' = 1$ on C4.**

Only C2 spans a wide range without saturating, from .378 to .593.

| Finding | Number |
| :--- | :--- |
| Taxonomic EM, phylum → scientific name | **.725 → .123** (5.9× fall) |
| Common name, open-ended vs 4-way MCQ | **.039 → .645** (16.5×) |
| Best model vs BioCLIP-2, at class / genus / sci. name | 1.20× / 1.56× / **2.02×** |
| InternVL3 8B→38B gain, class vs scientific name | +18.7 vs **+6.5** points |
| Counting exact match, best model | **≤ .242** (vs 55–70% on everyday objects) |
| Viewpoint: models below the majority-class floor | **17 of 18** |
| Absolute distance: models below a per-type constant | **15 of 18** |
| Laterality: models at or below chance on left/right | **17 of 18** |
| C4 best d′ under random pairing | **.696** (≈64% balanced accuracy) |
| C4 models with **negative** d′ under adversarial pairing | **17 of 18** |

> The pattern is a **domain gap, not a capacity ceiling.** Generalist models
> handle coarse categories well and fail on fine-grained ones. Scaling improves
> the coarse ranks a model already partly knew rather than resolving the fine
> distinctions it misses.

---

## Cross-Component Structure

A four-component benchmark has to show it is not one component measured four
times. **The answer is partial, and both halves are reported.**

**Much is shared.** Mean correlation across components is **.660** against
**.691** within them. A PCA on the 18 × 27 model-by-task matrix puts **62.2%** of
variance on the first component, with all 27 tasks loading positively. C2 and C3
correlate at .927, which disattenuates to 1.015 — at the reliability measurable
here, they are one construct. A single leaderboard is defensible for picking the
strongest model.

**What the components buy is everything below the top.** Tasks agree more within
a component than across one (**.749 vs .534**, $p = .0005$ by label permutation).
Three models appear in every component's top five, but **the median model moves
5.5 of 18 places between components**, and Molmo 2-8B moves **11** — from the 3rd
percentile on C1 to the 79th on C2. Anyone choosing a mid-field model cannot do
it from one number.

**The dissociations are where they were designed to be.** C4 loads lowest on the
shared factor across all five species (.091–.185). Within C2, laterality loads
lowest (.154); within C3, relative position (.168) and localization (.176);
within C1, gymnosperms (.183) — the group with the thinnest source coverage.

---

## Reproducibility

| Stage | Deterministic? | Mechanism |
| :--- | :---: | :--- |
| Image download (C1) | ❌ | Thread interleaving decides which downloads win the race to the budget. **The manifests are the artifact of record.** |
| QA generation | ✅ | Seeded per component |
| Manifest sharding | ✅ | `--seed 42`, applied after `row_id` assignment |
| Inference | ✅ | Greedy decoding, one attempt per item |
| Scoring | ✅ | Pure function of the merged CSVs; bootstraps seeded |

**Given a manifest, everything downstream is exactly reproducible.** Archive the
manifests rather than expecting to regenerate them byte-for-byte.


## References
 
Every source dataset, baseline, model, metric and benchmark referenced by
Bio-VQA or by any component README. The `key` column is the BibTeX key used in
the paper, so entries can be cross-referenced against the manuscript's `.bib`
directly.
 
**All source imagery retains its original license.** This repository
distributes code and metadata only. Verify licensing before redistributing any
downloaded imagery.
 
### Source datasets
 
Eleven public datasets supply every image and every ground-truth annotation. A question is generated only where an existing annotation supplies the answer.
 
| Dataset | Key | Used by |
|---|---|---|
| TreeOfLife-200M | `stevens2025bioclip2` | C1 — 100K image subset, 10 organismal groups |
| CUB-200-2011 | `wah2011caltech` | C2 — 11,788 images, 312 binary attributes |
| WAID | `waid2023` | C3 — aerial wildlife: camel, cattle, kiang, seal, sheep, zebra |
| Boxes on Bees and Pollen | `boxesonbees` | C3 — dense hive frames |
| 2018-NEON-beetles | `neonbeetles` | C3 — specimen trays with scalebars |
| DePondFi | `depondfi` | C3 — pond fish |
| GZCD | `gzcd` | C3 + C4 — Grevy's zebra |
| WhaleSharkID | `whalesharkid` | C4 — spot pattern |
| BelugaID | `belugaid` | C4 — scarring, dorsal ridge |
| HyenaID | `hyenaid` | C4 — spot pattern |
| LeopardID | `leopardid` | C4 — rosette pattern |
 
### Specialist baselines and encoders
 
BioCLIP-2 and MegaDescriptor are the measured ceilings for C1 and C4. DINOv2 is used for C4 pair mining, not as a baseline. The remaining entries are the re-identification literature the component situates itself against.
 
| Model | Key | Role |
|---|---|---|
| BioCLIP-2 | `stevens2025bioclip2` | C1 taxonomic ceiling |
| BioCLIP | `stevens2024bioclip` | predecessor; tree-of-life scale contrastive |
| BioTrove-CLIP | `biotrove2024` | alternative biological encoder |
| MegaDescriptor | `cermak2024wildlifedatasets` | C4 specialist baseline |
| DINOv2 | `oquab2024dinov2` | C4 pair mining by embedding similarity |
| MiewID | `otarashvili2024miewid` | multi-species re-ID descriptor |
| HotSpotter | `crall2013hotspotter` | hand-crafted local features for re-ID |
| PIE | `moskvyak2019pie` | learned re-ID embeddings |
| ALFReID | `nepovinnykh2024alfreid` | learned re-ID embeddings |
| TransReID | `he2021transreid` | transformer re-ID |
 
### Evaluated models
 
Eighteen open-weight models from 3.75B to 38.4B parameters. Two of the 30B models are mixture-of-experts activating 3B parameters per token.
 
| Family | Key | Sizes evaluated |
|---|---|---|
| Qwen2.5-VL | `qwen25vl` | 3.75B, 8.3B |
| Qwen3-VL | `qwen3vl` | 8.8B, 30.5B MoE, 32.8B |
| InternVL3 | `internvl3` | 8.1B, 38.4B |
| InternVL3.5 | `internvl35` | 8.1B, 30.5B MoE, 38.4B |
| Gemma-3 | `gemma3` | 12.2B |
| Gemma-4 | `gemma4` | 12.2B |
| Molmo 2 | `molmo2` | 8.0B, pointing-supervised |
| Pixtral | `pixtral` | 12.4B |
| Phi-3.5-Vision | `phi35` | 4.2B |
| Llama-3.2-Vision | `llama32` | 10.7B |
| LLaVA-NeXT | `llavanext` | 7.1B Vicuna, 7.6B Mistral |
 
### Metrics and statistical methods
 
Every graded metric is reported alongside the exact match it modifies.
 
| Method | Key | Used for |
|---|---|---|
| Wu–Palmer similarity | `wu1994verb` | C1 — depth of the lowest common ancestor |
| Lin similarity | `lin1998information` | C1 — information-content weighted |
| Open-ended taxonomic evaluation | `ging2024openended` | C1 — hierarchical scoring precedent |
| Cohen's kappa | `cohen1960coefficient` | C2/C3 — chance correction |
| Signal detection theory | `green1966signal` | C4 — d-prime and criterion |
| Detection theory | `macmillan2004detection` | C4 — d-prime and criterion |
| Log-linear correction | `hautus1995` | C4 — required when H or F reaches 0 or 1 |
| Labeled Faces in the Wild | `lfw2007` | C4 — the verification protocol |
 
### Evaluation protocol
 
| Element | Key | Setting |
|---|---|---|
| Greedy decoding | `holtzman2020` | no sampling |
| Prompt sensitivity | `sclar2024` | fixed system prompt per question family |
| Chain-of-thought | `wei2022cot` | explicitly NOT used |
| Few-shot prompting | `brown2020few` | explicitly NOT used |
 
### Related benchmarks
 
| Benchmark | Key | Relevance |
|---|---|---|
| VLM4Bio | `vlm4bio2024` | closest prior biological VQA |
| PlantExpertVQA | `plantexpertvqa2025` | crop diseases |
| J-EDI QA | `jediqa2024` | 100 deep-sea images |
| AgroCoT | `agrocot2025` | agricultural chain-of-thought |
| AVA-Bench | `avabench2026` | atomic visual skills — the decomposition precedent |
| MMBench | `liu2024mmbench` | general multi-task |
| SEED-Bench | `li2023seedbench` | general multi-task |
| BLINK | `fu2024blink` | general perception |
| MMVP | `tong2024eyes` | visual patterns |
| CV-Bench / Cambrian | `tong2024cambrian` | counting comparison |
| MM-Spatial | `mmspatial2025` | spatial reasoning; blind-baseline result |
| ScienceQA | `lu2022scienceqa` | expert-domain VQA |
| MMMU | `yue2024mmmu` | expert ceilings |
| MicroVQA | `burgess2025microvqa` | microscopy |
| TallyQA | `tallyqa2019` | counting |
| CountBench | `paiss2023countbench` | counting |
| VSR | `liu2023vsr` | spatial relations |
| SpatialVLM | `chen2024spatialvlm` | spatial reasoning |
| INQUIRE | `vendrow2024inquire` | biodiversity retrieval |
| IUCN LLM evaluation | `iucnllm2026` | biodiversity, no imagery |
| Animal re-ID survey | `animalreid2026` | no VLMs reported |
 
### Taxonomic infrastructure
 
| Resource | Key | Used for |
|---|---|---|
| GBIF Backbone | `gbif2024` | versioned snapshot; synonym map, accepted names |
| iNaturalist | `inaturalist2024` | common-name resolution to English |
| Wikidata | `wikidata` | common-name resolution to English |

 
---


<p align="center">
  <sub><b>Bio-VQA</b> &middot; code and metadata only &middot; images retain their original licenses</sub>
</p>
