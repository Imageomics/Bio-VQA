<h1 align="center"> Bio-VQA &rarr; Individual Re-Identification &rarr; Data Preparation </h1>

<p align="center">
  <b>Five wildlife re-identification corpora &mdash; Adversarial Pair Selection Pipeline</b>
</p>
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-blue">
  <img alt="Source datasets" src="https://img.shields.io/badge/sources-5%20species-green">
  <img alt="Pairs" src="https://img.shields.io/badge/pairs-15%2C000-orange">
  <img alt="Source images" src="https://img.shields.io/badge/source%20images-22%2C902-lightgrey">
  <img alt="Conditions" src="https://img.shields.io/badge/conditions-easy%20%7C%20random%20%7C%20hard-purple">
  <img alt="Release" src="https://img.shields.io/badge/release-v1.0-brightgreen">
</p>
<p align="center">
  We build a binary verification corpus in which appearance similarity is experimentally<br>
  rotated relative to individual identity &mdash; correlated, uncorrelated, or anti-correlated &mdash;<br>
  so that a model's reliance on visual resemblance becomes measurable.
</p>

---

## Table of Contents

1. [Overview](#1-overview)
2. [Why We Manipulate the Pairing](#2-why-we-manipulate-the-pairing)
3. [Pipeline Contents](#3-pipeline-contents)
4. [Source Corpora](#4-source-corpora)
5. [Dataset Layouts](#5-dataset-layouts)
6. [Environment](#6-environment)
7. [Stage 1. DINOv2 Embedding Extraction](#7-stage-1-dinov2-embedding-extraction)
8. [Stage 2. Adversarial Pair Selection](#8-stage-2-adversarial-pair-selection)
9. [The Three Conditions](#9-the-three-conditions)
10. [Selection Algorithm](#10-selection-algorithm)
11. [Controls We Hold Constant](#11-controls-we-hold-constant)
12. [Rendering](#12-rendering)
13. [Output Schemas](#13-output-schemas)
14. [Corpus Composition](#14-corpus-composition)
15. [Manipulation Check](#15-manipulation-check)
16. [Engineering Notes](#16-engineering-notes)
17. [Reproducibility](#17-reproducibility)
18. [Limitations](#18-limitations)
19. [Terms](#19-terms)

---

## 1. Overview

This component turns five wildlife re-identification corpora into a binary verification benchmark.
Each item shows two photographs side by side and asks one question:

> *This image shows two images placed side by side. Is the main subject in the left image the very
> same individual animal as the main subject in the right image?*

The form is elementary — a yes/no judgment over a balanced set. The substance is not. Individual
re-identification requires resolving individual-specific markings: stripe topology, scar patterns,
spot constellations, fluke notches. It is the finest granularity of visual discrimination in Bio-VQA.

We generate three matched conditions from the same images, identities, and viewpoint constraint. They
differ only in how we draw pairs from the appearance-similarity distribution:

| Condition | Same-individual pairs | Different-individual pairs | Similarity &harr; identity |
| --- | --- | --- | --- |
| **easy** | most visually **similar** | most visually **dissimilar** | correlated |
| **random** | uniformly sampled | uniformly sampled | uncorrelated |
| **hard** | most visually **dissimilar** | most visually **similar** | **anti-correlated** |

The `hard` condition is **adversarial pair selection**. We modify no pixels — every photograph is an
unaltered original. What is adversarial is the pairing: we choose pairs such that a decision rule of
the form *"answer yes if the two animals look alike"* is systematically wrong.

The release is **15,000 pairs**: 5 species × 3 conditions × 1,000 pairs, each condition balanced at
500 same-individual and 500 different-individual items.

```
Five re-identification corpora (COCO or flat metadata)
            │
            ▼
   [1] embed_dinov2.py    ──►  embeddings_dinov2/*.npy
            │
            ▼
   [2] generate_QA.py     ──►  instance_identification_1000_{easy,random,hard}/
                                  ├── QA.csv
                                  ├── pairs_metadata.csv
                                  ├── metadata.json
                                  └── concatenated_images/*.png
```

---

## 2. Why We Manipulate the Pairing

A single verification accuracy tells us how often a model is right. It does not tell us what the
model is doing. Appearance similarity and individual identity normally travel together — two photos
of the same zebra usually look alike — so a model that only measures resemblance and a model that
truly reads stripe patterns produce similar scores on ordinary data.

We separate them by rotating one variable against the other. Across three conditions the images, the
identities, and the viewpoint constraint stay fixed; only the relationship between similarity and
identity changes. A model reading identity is unaffected. A model reading resemblance moves from
above chance to below chance.

That gives us a per-model measurement of shortcut reliance rather than a single accuracy figure, and
it makes the benchmark an experiment rather than a leaderboard.

---

## 3. Pipeline Contents

| File | Role |
| --- | --- |
| `embed_dinov2.py` | Stage 1. One DINOv2 ViT-L/14 embedding per source photograph. |
| `generate_QA.py` | Stage 2. Pair selection, rendering, manifests. |
| `generate_QA.sh` | SLURM driver: one condition across all five species. |

---

## 4. Source Corpora

We use five publicly released wildlife re-identification corpora. All carry individual-level identity
annotations, which is what makes verification pairs constructible.

| Species | Corpus | Images | Identities | Identities &ge;2 images |
| --- | --- | ---: | ---: | ---: |
| Beluga whale | BelugaID 2022 | 5,902 | 788 | 633 |
| Spotted hyena | HyenaID 2022 | 3,104 | 256 | 256 |
| Leopard | LeopardID 2022 | 4,378 | 386 | 289 |
| Whale shark | WhalesharkID | 7,693 | 543 | 512 |
| Grévy's zebra | Grévy's zebra release | 1,825 | 361 | 281 |
| **Total** | | **22,902** | **2,334** | **1,971** |

These counts are what `generate_QA.py` reports at load time: images with both a readable file on disk
and an identity label. We log every exclusion by reason.

The corpora come from the Wild Me / LILA BC releases. We redistribute no imagery; the pipeline
produces derived pairings and manifests from data obtained separately.

---

## 5. Dataset Layouts

Two annotation layouts occur among the five corpora. We support both and auto-detect from directory
contents.

**COCO layout** — beluga, hyena, leopard, whale shark:

```
<species_dir>/
├── images/train<year>/*.jpg
├── annotations/instances_train<year>.json
└── embeddings_dinov2/*.npy
```

**Flat layout** — Grévy's zebra:

```
<species_dir>/
├── Images/*.jpg
├── metadata.csv
└── embeddings_dinov2/*.npy
```

We normalise column and field names to lowercase-underscore before matching, so `name uuid` and
`name_uuid` resolve identically, and we look up identity and viewpoint through priority lists:

```python
IDENTITY_KEYS  = ("name_uuid", "name", "annot_name_uuid", "individual_id", "identity")
VIEWPOINT_KEYS = ("viewpoint", "annot_viewpoint", "view")
IMAGE_KEYS     = ("image", "image_fname", "file_name", "filename", "image_path", "path")
```

One code path therefore handles both exports. Detection prefers an `annotations/` directory, then
`metadata.csv` / `.json` / `.tsv`. All paths can be overridden explicitly.

---

## 6. Environment

```bash
python -m venv myenv_py312 && source myenv_py312/bin/activate
pip install torch torchvision numpy pandas pillow faiss-cpu tqdm
```

We use FAISS `IndexFlatIP` — exact brute-force inner product. No approximate index, so neighbour
retrieval is deterministic and no recall parameter enters the construction.

---

## 7. Stage 1. DINOv2 Embedding Extraction

We embed every source image with **DINOv2 ViT-L/14** (`dinov2_vitl14`, via `torch.hub`), taking the
`x_norm_clstoken` output of `forward_features`. Preprocessing is uniform:

```python
transforms.Resize((518, 518))
transforms.ToTensor()
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

```bash
python embed_dinov2.py \
  --dataset_root <species_dir> \
  --output_dir   <species_dir>/embeddings_dinov2 \
  --model        dinov2_vitl14 \
  --year         2022
```

Extraction is idempotent — we skip images whose `.npy` exists — and `--start` / `--end` shard the
index range across array jobs.

We choose a self-supervised general-purpose encoder deliberately. DINOv2 is not trained on individual
identity, so its similarity structure reflects general visual resemblance: exactly the nuisance
variable we manipulate. An identity-supervised encoder would define the conditions in terms of the
quantity we are testing for.

We L2-normalise at load time, so all similarities downstream are cosine similarities in [-1, 1].

---

## 8. Stage 2. Adversarial Pair Selection

`generate_QA.py` builds one condition for one species per invocation.

```bash
python generate_QA.py \
  --base_dir   <species_dir> \
  --difficulty {easy|hard|random} \
  --num        500 \
  --seed       0
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--difficulty` | *required* | condition to generate |
| `--num` | 500 | pairs per class; total = 2 × num |
| `--seed` | 0 | seeds all sampling and shuffling |
| `--knn` | 50 | neighbours retrieved per anchor within its viewpoint group |
| `--max_uses_per_image` | 2 | cap on pairs any one image appears in |
| `--max_group_size` | 400 | cap on images per identity entering the similarity matrix |
| `--max_pairs_per_identity` | 64 | cap on candidate pairs kept per identity |
| `--ignore_viewpoint` | off | disable the viewpoint constraint |
| `--max_height` | 0 | cap composite height; 0 keeps full resolution |
| `--workers` | auto | parallel decode/encode processes |
| `--dry_run` | off | select and validate, write nothing |
| `--overwrite` | off | replace an existing output directory |

The SLURM driver runs one condition across all five species:

```bash
sbatch generate_QA.sh easy
sbatch generate_QA.sh random
sbatch generate_QA.sh hard
```

---

## 9. The Three Conditions

Let `s(i,j)` be the cosine similarity of DINOv2 embeddings, `y(i)` the identity, `v(i)` the
viewpoint. Every pair satisfies `v(i) = v(j)` in all conditions.

**Different-individual pairs** — for each anchor `i`, one partner `j` with `y(i) ≠ y(j)`:

| Condition | Partner |
| --- | --- |
| `hard` | `argmax_j s(i,j)` — nearest non-identity neighbour |
| `easy` | `argmin_j s(i,j)` — furthest non-identity neighbour |
| `random` | uniform within the viewpoint group |

**Same-individual pairs** — we enumerate all valid within-identity, within-viewpoint pairs and order
them:

| Condition | Ordering |
| --- | --- |
| `hard` | ascending `s` — furthest first |
| `easy` | descending `s` — closest first |
| `random` | shuffled |

We find furthest neighbours without materialising a similarity matrix. Since embeddings are
unit-normalised,

```
argmin_j ⟨e_i, e_j⟩  =  argmax_j ⟨-e_i, e_j⟩
```

so a single `IndexFlatIP.search(-E, k)` returns them in O(n·d) memory instead of O(n²).

---

## 10. Selection Algorithm

**Viewpoint partitioning.** We group indices by viewpoint and run the neighbour search *inside* each
group rather than filtering after retrieval. Viewpoint is the dominant axis of variation in DINOv2
space, so an anchor's most dissimilar neighbours are overwhelmingly of the opposite viewpoint.
Post-filtering yielded 0 valid candidates from 2,400 anchors on a viewpoint-dominated test set;
within-group search cannot fail this way.

**Same-individual candidates.** For each identity we split by viewpoint and enumerate *all* pairs
within each block, order by the condition's criterion, and truncate to `--max_pairs_per_identity`.
Enumerating all pairs matters on corpora with small identities: with a best-partner-only pool, mutual
best partners collapse under deduplication and an identity with *n* images yields about *n*/2 pairs
instead of C(*n*,2).

**Allocation order.** We select same-individual pairs first. That class is the scarce resource,
bounded by how many images each identity has; different-individual pairs are effectively unlimited.

**Round-robin.** We cycle across identities (same class) and viewpoint groups (different class)
rather than draining the top-scoring candidates globally, so no single individual or viewpoint
dominates and the viewpoint distribution stays comparable across conditions.

**Automatic relaxation.** If the target is unreachable at the configured per-image use cap, we
escalate 2 → 4 → 8 → unlimited, re-selecting at each level, and record the applied cap in
`metadata.json`. In the released corpora the cap held at 2 throughout.

**Validation before writing.** We open and decode every source image entering the selection, plus a
15% reserve, and drop pairs referencing an undecodable image. Only then do we create the output
directory.

---

## 11. Controls We Hold Constant

**Viewpoint.** Every pair, in every condition, is drawn within a single viewpoint group. Without
this, the `hard` condition would exploit viewpoint mismatch rather than appearance similarity, and
viewpoint would be perfectly confounded with condition.

**Image pool.** All three conditions draw from the same images and the same identities. Only the
selection criterion changes.

**Question text.** Identical across all conditions and species.

**Class balance.** 500 / 500 everywhere, so chance is 0.500 throughout.

**Reuse cap.** 1,000 pairs consume 2,000 image slots, and the smallest corpus has 1,825 images, so
some reuse is unavoidable. We bound it at 2 uses per image and record the applied cap.

---

## 12. Rendering

We render each pair as a single PNG with the two photographs side by side:

1. Open both, convert to RGB.
2. Target height is `min(h₁, h₂)`, further capped by `--max_height` when set.
3. Resize each to the target height, bilinear, aspect preserved.
4. Paste onto a canvas of width `w₁ + w₂`.
5. Write to `<name>.png.tmp`, re-open and decode to confirm integrity, then `os.replace`.

The atomic write means an interrupted job never leaves a truncated PNG visible to a later stage.
Left/right assignment follows selection order; we construct same- and different-individual pairs
identically, so position carries no cue to the label.

We preserve full source resolution by default. Some composites exceed 4,000 px, which affects
vision-token budgets downstream.

---

## 13. Output Schemas

Each `<species_dir>/instance_identification_1000_<condition>/` contains four artifacts.

**`QA.csv`** — the benchmark items, rows shuffled under `--seed`.

| Column | Description |
| --- | --- |
| `image_path` | composite PNG, relative to the task directory |
| `qa_pair` | JSON `{"Q": …, "A": "Yes"｜"No"}` |

**`pairs_metadata.csv`** — provenance, row-aligned with `QA.csv`.

| Column | Description |
| --- | --- |
| `image_path` | join key |
| `label` | `Yes` / `No` |
| `difficulty` | condition |
| `cosine_similarity` | DINOv2 similarity of the pair |
| `image_left`, `image_right` | source stems |
| `identity_left`, `identity_right` | identity labels |
| `viewpoint` | shared viewpoint |

This file makes the corpus auditable. The manipulation check, the specialist baseline, and any
post-hoc analysis of behaviour against similarity all run from it without regenerating anything.

**`metadata.json`** — configuration, counts, applied use cap, per-class viewpoint distribution, and
similarity margins.

**`concatenated_images/`** — 1,000 rendered PNGs.

---

## 14. Corpus Composition

| Species | Pairs / condition | Same | Different | Conditions | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Beluga whale | 1,000 | 500 | 500 | 3 | 3,000 |
| Spotted hyena | 1,000 | 500 | 500 | 3 | 3,000 |
| Leopard | 1,000 | 500 | 500 | 3 | 3,000 |
| Whale shark | 1,000 | 500 | 500 | 3 | 3,000 |
| Grévy's zebra | 1,000 | 500 | 500 | 3 | 3,000 |
| **Total** | | **7,500** | **7,500** | | **15,000** |

Viewpoint vocabularies come from the source annotations and differ by corpus: beluga is dominated by
`up` (aerial), hyena spans ten viewpoints including obliques, leopard has 98 images with no viewpoint
annotation. We treat the unannotated group as a viewpoint like any other — pairs form within it,
never across it.

---

## 15. Manipulation Check

We verify the construction from the generated corpus before running any model. Each `metadata.json`
records `margin_same_minus_different`: mean pair similarity of same-individual items minus that of
different-individual items.

Realised margins in the `hard` condition:

| Species | mean sim (same) | mean sim (different) | margin |
| --- | ---: | ---: | ---: |
| Beluga whale | 0.4495 | 0.9201 | **−0.4706** |
| Whale shark | 0.5866 | 0.9360 | **−0.3493** |
| Leopard | 0.6442 | 0.8907 | **−0.2464** |
| Grévy's zebra | 0.7722 | 0.9521 | **−0.1799** |
| Spotted hyena | 0.7404 | 0.9137 | **−0.1732** |

In every species, different-individual pairs are on average *more* visually similar than
same-individual pairs. The manipulation produced the intended anti-correlation, and we establish this
independently of any model.

One further observation: in the beluga corpus the `random` condition also has a negative margin
(**−0.0613**). Under uniform sampling, different-individual pairs are already more similar than
same-individual ones, which means DINOv2's embedding geometry for belugas is not organised by
individual identity. This predicts that beluga shows the weakest easy-condition separation of any
species, and the model results confirm it.

---

## 16. Engineering Notes

We record the failures that shaped the current implementation, since each produced a silent or
misleading result.

**Post-filtering starves the easy condition.** Retrieving neighbours globally then filtering for
viewpoint gives near-zero candidates in `easy`. We search within viewpoint groups instead.

**Mixed `None`/`str` viewpoint keys.** `sorted()` over a group dictionary mixing `None` with strings
raises `TypeError`. Leopard's 98 unannotated images triggered it. We use `key=str` at every grouping
site.

**Best-partner-only same-class pool.** One candidate per anchor collapses mutual best partners under
deduplication. Grévy's zebra fell short at 474 of 500. We enumerate all within-identity pairs.

**Allocation order.** Selecting the abundant class first exhausts the image budget on the wrong
class. Same-individual pairs now go first.

**Non-atomic writes.** A cancelled job left truncated PNGs that a later stage read as valid. We write
to `.tmp`, decode-verify, then `os.replace`.

**`np.save` filename mangling.** `np.save("x.npy.tmp", …)` writes `x.npy.tmp.npy`, breaking
tmp-then-rename. We save through an open file handle.

---

## 17. Reproducibility

| Aspect | Guarantee |
| --- | --- |
| Sampling | Seeded via `--seed`; identical arguments give byte-identical `QA.csv` (checksum-verified). |
| Neighbour search | Exact `IndexFlatIP`; no approximate index, no recall parameter. |
| Embeddings | Deterministic under fixed model and preprocessing; extraction idempotent and resumable. |
| Configuration | Every argument serialised into `metadata.json`. |
| Provenance | `pairs_metadata.csv` records source images, identities, viewpoint, similarity per pair. |
| Ordering | Row shuffle seeded; `pairs_metadata.csv` emitted in `QA.csv` order. |

Full regeneration from source imagery:

```bash
python embed_dinov2.py --dataset_root <species_dir> \
                       --output_dir <species_dir>/embeddings_dinov2
sbatch generate_QA.sh easy
sbatch generate_QA.sh random
sbatch generate_QA.sh hard
```

---

## 18. Limitations

**The conditions are defined by DINOv2 similarity.** Several models we evaluate share vision-encoder
lineage with DINOv2 or related self-supervised objectives, so the manipulation partly indexes shared
embedding geometry rather than a general property of visual identity representation. We do not
replicate the construction in a second embedding space.

**Composite format.** We present pairs as one side-by-side image rather than two separate images,
which adds a spatial-parsing burden to the identity judgment. The format is constant across models
and conditions, so it does not confound the difficulty comparison, but it affects absolute levels.

**Image reuse.** Some photographs appear in more than one pair, so items within a condition are not
fully independent and intervals computed under independence are mildly optimistic.

**Full frames.** We use whole photographs, not animal crops, even where the source corpora provide
bounding boxes. The subject sometimes occupies a small fraction of the frame.

---

## 19. Terms

Source imagery and identity annotations remain under the licences of their originating corpora
(BelugaID, HyenaID, LeopardID, WhalesharkID, and the Grévy's zebra release). We redistribute no
imagery.