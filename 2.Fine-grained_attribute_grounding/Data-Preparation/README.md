<h1 align="center">Bio-VQA &rarr; Fine-Grained Visual Attribute Grounding</h1>

<p align="center">
  <b>CUB-200-2011 &mdash; Data Acquisition &amp; QA Generation Pipeline</b>
</p>
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="Source dataset" src="https://img.shields.io/badge/source-CUB--200--2011-green">
  <img alt="Questions" src="https://img.shields.io/badge/questions-824%2C104-orange">
  <img alt="Images" src="https://img.shields.io/badge/images-23%2C576-lightgrey">
  <img alt="Release" src="https://img.shields.io/badge/release-v1.0-brightgreen">
</p>
<p align="center">
  A CUB-200-2011&ndash;derived question&ndash;answer corpus for evaluating fine-grained perceptual<br>
  and grounding ability in vision&ndash;language models.
</p>

---

## Table of Contents

1. [Overview](#1-overview)
2. [Design Principles](#2-design-principles)
3. [Pipeline Contents](#3-pipeline-contents)
4. [Obtaining the CUB-200-2011 Dataset](#4-obtaining-the-cub-200-2011-dataset)
5. [The Taxon Name Mapping File](#5-the-taxon-name-mapping-file)
6. [Environment and Dependencies](#6-environment-and-dependencies)
7. [Stage 1. Consolidated Attribute Table](#7-stage-1-consolidated-attribute-table)
8. [Stage 2. Question Synthesis and Visual Grounding](#8-stage-2-question-synthesis-and-visual-grounding)
9. [Stage 3. Corpus Statistics](#9-stage-3-corpus-statistics)
10. [Dataset Composition](#10-dataset-composition)
11. [Question Taxonomy](#11-question-taxonomy)
12. [Output Schemas](#12-output-schemas)
13. [Reproducibility](#13-reproducibility)
14. [Citation and Terms](#14-citation-and-terms)

---

## 1. Overview

This component of **Bio-VQA** transforms the *Caltech-UCSD Birds-200-2011* (CUB-200-2011) annotation corpus into a large-scale multiple-choice and open-ended visual question answering benchmark spanning four competencies:

| Competency | Probe |
| --- | --- |
| **Relative Salience** | Among several traits true of the taxon, which is *most prominently visible in this image*? |
| **Morphological Attributes** | Can the model read off discrete traits — bill shape, plumage colour, wing pattern, tail length? |
| **Anatomical Localization** | Can the model report part **visibility**, and resolve a **visual prompt** — a coloured marker overlaid on the image — to the anatomical part beneath it? |
| **Taxonomy & Naming** | Can the model recover common name, folk category, genus, and binomial nomenclature from pixels alone? |

The release comprises **824,104 questions over 11,788 photographs** drawn from **200 species**. Each photograph contributes one source image and one marker-annotated derivative, for 23,576 image files on disk. Full composition is given in Section 10.

The pipeline is **three-staged and file-mediated**: each stage writes a durable, inspectable artifact, so any stage may be re-run, audited, or replaced without recomputing the others.

```
CUB-200-2011 raw annotations
            │
            ▼
   [1] build_attribute_table.py  ──►  CUB200_attributes.csv     (one row per image)
            │
            ▼
   [2] generate_qa.py            ──►  QA.csv                    (one row per question)
                                 ──►  Images_Annotated/*.jpg    (visually-prompted images)
            │
            ▼
   [3] dataset_stats.py          ──►  data-stats.txt            (corpus report)
```

---

## 2. Design Principles

**Denormalize once, query forever.** CUB-200-2011 ships as a constellation of whitespace-delimited index files joined on integer keys. Stage 1 performs this join once and materializes a single wide table in which every image carries its class, taxon, bounding box, split assignment, full attribute dictionary (with annotator certainty *and* confidence score), and complete part-location record. Downstream consumers never touch the raw files again.

**Distractors are drawn from the empirical trait vocabulary, not invented.** Every multiple-choice item builds its distractor pool from the *observed* value set of the relevant attribute group — or, for salience items, from the global trait vocabulary — excluding all values true of the current image. Distractors are therefore plausible ornithological alternatives rather than nonsense strings, and are *genuinely* incorrect: a distractor is never a trait the annotators marked present.

**Grounding requires a controlled visual confound.** In the visually-prompted localization task, a marker of a given *(colour, shape)* is drawn on the target part and a battery of decoys is scattered across the image. The decoy set always contains **at least one marker sharing the target's colour but not its shape**, so colour alone is insufficient — conjunction binding is required.

**Answer spaces are narrow by construction, so evaluation is chance-adjusted.** With 15 anatomical parts and roughly 300 attribute values, per-item chance is non-trivial and varies across items. So for evaluation, **CAGS** (Chance-Adjusted Graded Score) is reported, along with raw accuracy.

---

## 3. Pipeline Contents

| Script | Role |
| --- | --- |
| `build_attribute_table.py` | Stage 1 — joins raw CUB annotations into `CUB200_attributes.csv` |
| `generate_qa.py` | Stage 2 — synthesizes `QA.csv` and renders `Images_Annotated/` |
| `dataset_stats.py` | Stage 3 — emits `data-stats.txt` |

Stages 1 and 2 each declare a module-level constant:

```python
BASE_DIR = "/path/to/your/dataset/directory"
```

`BASE_DIR` points at the **extracted `CUB_200_2011/` directory** (Section 4). It is the single source of truth for every path in the pipeline, and is set in both scripts to the same value.

---

## 4. Obtaining the CUB-200-2011 Dataset

The pipeline reads nothing from the network; the raw CUB annotations are its only input.

CUB-200-2011 (Wah, Branson, Welinder, Perona & Belongie, 2011) comprises **11,788 images** across **200 North American bird species**, annotated with **312 binary attributes**, **15 part locations**, and one bounding box per image.

```bash
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -xzvf CUB_200_2011.tgz
```

### 4.1 Post-extraction layout

The following resolve relative to `BASE_DIR`:

```
BASE_DIR/                                  # CUB_200_2011/
│
├── images/                                # 200 species subdirectories of JPEGs
│   ├── 001.Black_footed_Albatross/
│   ├── 002.Laysan_Albatross/
│   └── ...
│
├── attributes/
│   ├── image_attribute_labels.txt         # ATTR_FILE
│   ├── attributes.txt                     # ATTR_NAME_FILE
│   └── certainties.txt                    # CERTAINTY_FILE
│
├── parts/
│   ├── parts.txt                          # PARTS_FILE
│   └── part_locs.txt                      # PART_LOCS_FILE
│
├── classes.txt                            # CLASSES_FILE
├── images.txt                             # IMAGE_FILE
├── image_class_labels.txt                 # IMAGE_CLASS_FILE
├── bounding_boxes.txt                     # BBOX_FILE
├── train_test_split.txt                   # SPLIT_FILE
│
└── taxon_name_mapping.json                # SCI_MAP_FILE — see Section 5
```

These correspond exactly to the constant block at the head of Stage 1:

```python
ATTR_FILE        = os.path.join(BASE_DIR, "attributes/image_attribute_labels.txt")
ATTR_NAME_FILE   = os.path.join(BASE_DIR, "attributes/attributes.txt")
CERTAINTY_FILE   = os.path.join(BASE_DIR, "attributes/certainties.txt")
CLASSES_FILE     = os.path.join(BASE_DIR, "classes.txt")
IMAGE_FILE       = os.path.join(BASE_DIR, "images.txt")
IMAGE_CLASS_FILE = os.path.join(BASE_DIR, "image_class_labels.txt")
BBOX_FILE        = os.path.join(BASE_DIR, "bounding_boxes.txt")
PARTS_FILE       = os.path.join(BASE_DIR, "parts/parts.txt")
PART_LOCS_FILE   = os.path.join(BASE_DIR, "parts/part_locs.txt")
SPLIT_FILE       = os.path.join(BASE_DIR, "train_test_split.txt")
```

Some CUB releases place `attributes.txt` at the *archive root*, one level above `CUB_200_2011/`, rather than inside `attributes/`. The pipeline expects it inside `attributes/`:

```bash
mv attributes.txt "$BASE_DIR/attributes/attributes.txt"
```

### 4.2 Reference counts

The canonical release yields:

```bash
wc -l "$BASE_DIR/images.txt"                             # 11788
wc -l "$BASE_DIR/attributes/attributes.txt"              # 312
wc -l "$BASE_DIR/parts/parts.txt"                        # 15
wc -l "$BASE_DIR/attributes/image_attribute_labels.txt"  # ~3,677,856
find "$BASE_DIR/images" -name '*.jpg' | wc -l            # 11788
```

The attribute label file is the heavyweight: roughly **3.68 M rows** (11,788 images × 312 attributes). Stage 1 spends most of its wall-clock there.

---

## 5. The Taxon Name Mapping File

CUB-200-2011 supplies **common names only**, encoded in the directory prefixes (`001.Black_footed_Albatross`). Bio-VQA additionally asks for **scientific name** and **genus**, which requires an external mapping.

`BASE_DIR/taxon_name_mapping.json` is a flat object keyed on the *cleaned* common name — the class string with its numeric prefix stripped and underscores replaced by spaces:

```json
{
  "Black footed Albatross": "Phoebastria nigripes",
  "Laysan Albatross":       "Phoebastria immutabilis",
  "Sooty Albatross":        "Phoebetria fusca",
  "Groove billed Ani":      "Crotophaga sulcirostris",
  "...": "..."
}
```

**Semantics.**

- A **two-word** value (`"Phoebastria nigripes"`) yields *both* a scientific-name question and a genus question.
- A **one-word** value yields a genus question only; the first such occurrence is logged as a diagnostic.
- A missing key resolves to the sentinel `"Unknown"`, and **no** nomenclature question is emitted for that image.

**Coverage.** The shipped mapping resolves all 200 CUB classes; no class falls through to `"Unknown"`. Every entry was assigned manually by a human annotator and cross-referenced against **GBIF**. Of the 200, **199 resolve to species level** and carry a binomial, so they emit both a scientific-name and a genus question. The exception is `Sayornis`, a CUB class labelled at genus rank rather than species; it is mapped to the one-word value `"Sayornis"` and therefore emits a genus question only. This single class is the entirety of the scientific-name shortfall in the naming axis (Section 10.4).

---

## 6. Environment and Dependencies

```bash
source /path/to/myenv_py312/bin/activate
python -c "import pandas, numpy, cv2; print(pandas.__version__, cv2.__version__)"
```

| Package | Used by | Purpose |
| --- | --- | --- |
| `pandas` | 1, 2, 3 | Tabular join, explode, normalize |
| `numpy` | 1 | Numeric support |
| `opencv-python` | 2 | Image I/O and marker rasterization |

Standard library otherwise (`os`, `re`, `json`, `random`, `uuid`, `math`, `argparse`).

### 6.1 Resource envelope

| Stage | Wall-clock | Peak RSS | Disk written |
| --- | --- | --- | --- |
| 1 | ~10–25 min | ~4–8 GB | ~1–2 GB (`CUB200_attributes.csv`) |
| 2 | ~20–60 min | ~2 GB | ~1 GB images + ~200 MB CSV |
| 3 | seconds | ~1 GB | ~4 KB |

Stages 1 and 2 are single-threaded and CPU-bound. A standard (non-GPU) SLURM node suffices; memory rather than cores is the binding constraint.

---

## 7. Stage 1. Consolidated Attribute Table

```bash
python build_attribute_table.py
# → BASE_DIR/CUB200_attributes.csv
```

### 7.1 What it does

A sequence of defensive loaders, each tolerating malformed lines rather than aborting.

**`load_attributes()`** parses `image_attribute_labels.txt` into `(image_id, attr_id, present, certainty, score)`. Rows with fewer than five whitespace-delimited fields, or failing numeric coercion, are counted and skipped; the tally is printed. A handful of lines in the canonical release carry an extra trailing field, so `score` is read from `parts[-1]` rather than index 4 — anchoring on the *last* field makes the parse robust to that column.

**`load_simple_mapping()`** is a generic `int → value` reader over any two-column file (`classes.txt`, `images.txt`, `attributes.txt`, `certainties.txt`, `image_class_labels.txt`, `train_test_split.txt`), splitting on the *first* delimiter only so values containing spaces survive intact.

**`load_bboxes()`** returns `{image_id: [x, y, w, h]}` in absolute pixels (CUB convention: top-left origin).

**`load_parts()`** returns `{image_id: {part_name: {visible, x, y}}}`. Coordinates of **non-visible** parts are zeroed rather than retained — CUB stores undefined values there, and propagating them would corrupt downstream geometric reasoning.

**`build_trait_dict()`** inverts the long-format attribute frame into, per image:

```json
{ "has_bill_shape::dagger": { "present": 1, "certainty": "definitely", "score": 4.0 }, "...": "..." }
```

Attribute IDs are 1-indexed in CUB and decremented on lookup. Certainty integers resolve to their string labels (`not visible`, `guessing`, `probably`, `definitely`) via `certainties.txt`, preserving annotator confidence as a first-class field rather than discarding it.

### 7.2 Output columns

| Column | Type | Description |
| --- | --- | --- |
| `image_id` | int | CUB image index (1–11788) |
| `filename` | str | Path relative to `images/`, e.g. `001.Black_footed_Albatross/….jpg` |
| `class_id` | int | CUB class index (1–200) |
| `class_name` | str | Cleaned common name, e.g. `Black footed Albatross` |
| `taxon` | str | Scientific name from the mapping, or `Unknown` |
| `is_training_image` | int | `1` train / `0` test / `-1` unknown |
| `bbox` | list | `[x, y, w, h]` |
| `attributes` | **JSON string** | Full 312-key trait dictionary with certainty and score |
| `parts` | **JSON string** | 15-key part dictionary with visibility and coordinates |

`attributes` and `parts` are serialized as JSON *inside* CSV cells, keeping the artifact a single flat file at the cost of an escaping layer.

---

## 8. Stage 2. Question Synthesis and Visual Grounding

```bash
python generate_qa.py
# → BASE_DIR/QA.csv
# → BASE_DIR/Images_Annotated/*.jpg
```

`random.seed(42)` is set at import (Section 13).

### 8.1 Global vocabulary construction

Before any question is written, the corpus is swept twice:

- **`category_pool`** — the set of terminal words across all 200 common names (`Albatross`, `Warbler`, `Kingfisher`, …). The folk-taxonomic vocabulary from which *"What type of bird is this?"* distractors are drawn.
- **`build_attribute_pools()`** — yields `group_pool` (`{attribute group → observed values}`, e.g. `wing color → [blue, brown, …]`) and `full_trait_pool` (the flat `"group: value"` vocabulary across all groups).

The `has_size::` group is **excluded throughout**. Absolute size is not recoverable from a cropped photograph without a scale reference, so size questions would be unanswerable in principle and would depress scores for reasons orthogonal to perception.

### 8.2 Per-image question families

**(a) Nomenclature — open-ended and MCQ.**

- `"What is the common name of the bird?"` → free-form.
- `"What type of bird is this in the image?"` → 5-way MCQ over `category_pool`.
- `"What kind of {category} is the bird in this image?"` → free-form, but *scaffolded*: the folk category is given, so only the modifier must be produced. The contrast with the bare common-name question isolates retrieval failure from discrimination failure.
- `"What is the scientific name of the bird?"` → free-form binomial.
- `"What is the genus of the bird?"` → free-form.

**(b) Morphological attributes — grouped MCQ.** For every attribute group with at least one `present == 1` value, one 5-way MCQ: the key is sampled from the true values, up to four distractors from `group_pool[group]` *minus all true values*. When an image carries several values in a group — birds are genuinely multi-coloured — any one is an acceptable key, and the exclusion guarantees no alternative is secretly correct.

**(c) Relative salience — global MCQ.** For every trait clearing the confidence bar (`certainty == "definitely"` **or** `score ≥ 4.0`), an item asks which of five traits is *most prominently visible*. Distractors are drawn from `full_trait_pool` minus **all** traits present on the image — a stronger exclusion than (b), since a distractor from a different attribute group could otherwise be trivially true.

**(d) Part visibility — paired.** For `eye`, `leg`, and `wing`, one 4-way MCQ over `{Left, Right, Both, None}` derived from the left/right visibility flags. This probes lateral discrimination: a bird in profile has exactly one visible eye, and models routinely answer *"Both"*.

**(e) Part visibility — triadic.** Ten 5-way items per image of the form *"Which of the following parts are visible?"* with options `(A)(B)(C) {parts}, (D) None, (E) All three`. The generator schedules an **all-hidden** item (answer `D`) and an **all-visible** item (answer `E`) first, then fills the remainder with 1-visible / 2-hidden triads. Without this scheduling, `D` and `E` would be near-empty answer classes and the family would be gameable by never selecting them.

### 8.3 Visually-prompted localization

The most differentiated task family in C2, and the only one that mutates pixels.

For each image with at least one visible part:

1. **Target selection.** A visible part is sampled uniformly; a `(colour, shape)` from `{blue, green, red, white, yellow} × {dot, cross, plus, star}`.
2. **Target rendering.** `draw_mark()` rasterizes the marker at the part's annotated coordinate at `size=12` — larger than every decoy (`size=10`), so the target is not *hidden*, merely *confusable*.
3. **Decoy battery.** Decoys are placed at random positions under a **minimum-separation constraint** (`is_far_enough`, `min_dist=40 px`, up to 100 rejection-sampling attempts), preventing overlap that would render a marker unreadable.
4. **The colour confound.** The decoy list is *seeded* with a `(target_colour, non-target_shape)` pair, so the target colour recurs with a different shape and the conjunction of colour **and** shape is required to disambiguate. This is the central design decision of the task.
5. **Option construction.** Options follow a deliberate difficulty gradient: the target's **contralateral partner** (`left eye` ↔ `right eye`) is inserted first where applicable, then hidden parts, then other visible parts, until four distractors are held; the target is appended and the list shuffled.
6. **Persistence.** The composite is written to `Images_Annotated/{source_stem}_{uuid5}.jpg`.

A resulting item:

> *What is the body part of this bird marked with the yellow star?*
> *Options: (A) crown, (B) left eye, (C) right leg, (D) left leg, (E) belly*

### 8.4 Explosion and categorization

Nested `qa_pairs` lists are `explode()`-ed and `json_normalize()`-ed into one row per question. `filename` is rewritten to a `BASE_DIR`-relative path (`images/…` or `Images_Annotated/…`), and `categorize_qa_metadata()` assigns each question to one of the four axes by ordered substring matching on the lowercased stem (Section 11).

---

## 9. Stage 3. Corpus Statistics

Unlike Stages 1 and 2, this script takes its input on the command line:

```bash
python dataset_stats.py --csv_path "$BASE_DIR/QA.csv"
# → BASE_DIR/data-stats.txt   (written beside the CSV)

# Optional: keep sensitivity runs from overwriting one another
python dataset_stats.py --csv_path "$BASE_DIR/QA.csv" --out_name stats-v2.txt
```

The report comprises four blocks:

1. **High-level metrics** — total QA pairs; distinct photographs; source vs. annotated file counts; questions per photograph and per image.
2. **Species metrics** — unique species; photographs per species; questions per species (mean, min, max).
3. **Structural analysis** — MCQ vs. open-ended (detected by the literal `"Options:"` marker), binary Yes/No share, visually-prompted share, and answer-key letter balance.
4. **Per-category breakdown** — count, share, unique stems, unique answers, MCQ/open split, and photograph coverage per axis.

### 9.1 Photographs vs. image files

The corpus references two kinds of path, and the distinction governs every rate in the report:

| Row type | `filename` | Depth |
| --- | --- | --- |
| Source | `images/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg` | two-level, species in path |
| Annotated | `Images_Annotated/Black_Footed_Albatross_0046_18_63ac2.jpg` | flat, no species component |

Because `Images_Annotated/` is flat, species cannot be parsed from an annotated path. It is instead *recovered*: Stage 2 names its output `{source_stem}_{uuid5}.jpg`, so stripping the suffix and joining onto the source stems restores the species.

```python
UUID_SUFFIX = re.compile(r"_[0-9a-f]{5}$", re.IGNORECASE)

is_annot    = df.filename.str.startswith("Images_Annotated/")
src_species = df.loc[~is_annot, "filename"].str.split("/").str[1]
src_stems   = df.loc[~is_annot, "filename"].map(stem_of)
stem_to_species = dict(zip(src_stems, src_species))

raw         = df.loc[is_annot, "filename"].map(stem_of)
strip       = raw.str.replace(UUID_SUFFIX, "", regex=True)
annot_stems = strip.where(strip.isin(stem_to_species), raw)   # fall back if strip overshoots
```

This yields two quantities that must not be conflated:

| Concept | Value | Meaning |
| --- | ---: | --- |
| **Distinct photographs** | 11,788 | Underlying CUB images — the correct denominator for per-image and per-species rates |
| **Image files referenced** | 23,576 | Files on disk: photographs + marker renders |

Rows resolving to neither are labelled `Unresolved`, counted in the report, and echoed to stderr. The released corpus contains none.

---

## 10. Dataset Composition

### 10.1 Headline figures

| Quantity | Value |
| --- | ---: |
| Total QA pairs | **824,104** |
| Distinct photographs | **11,788** |
| Image files referenced | **23,576** |
| Source species | **200** |
| Questions per photograph | **69.91** |
| Photographs per species | **58.94** |
| Questions per species | **4,120.52** |
| Multiple-choice | 777,729 (94.4%) |
| Open-ended | 46,375 (5.6%) |
| Binary (Yes/No) | 0 (0.0%) |
| Visually-prompted (marker) items | 11,788 (1.4%) |

### 10.2 Axis composition

| Axis | Questions | Share | Unique stems | MCQ | Open |
| --- | ---: | ---: | ---: | ---: | ---: |
| Relative Salience | 321,073 | 39.0% | 321,073 | 321,073 | 0 |
| Morphological Attributes | 280,433 | 34.0% | 194,790 | 280,433 | 0 |
| Anatomical Localization | 165,032 | 20.0% | 14,462 | 165,032 | 0 |
| Taxonomy & Naming | 57,566 | 7.0% | 11,254 | 11,191 | 46,375 |
| **Total** | **824,104** | **100%** | — | **777,729** | **46,375** |

### 10.3 Structure of the corpus

**Every photograph yielded a grounding item.** `23,576 = 2 × 11,788`. Annotated images are emitted only when a source image has at least one visible part, so the visible-part precondition holds universally across CUB — no image was skipped.

**Anatomical Localization decomposes exactly.**

```
  35,364   paired visibility     (11,788 × 3: eye, leg, wing)
+117,880   triadic visibility    (11,788 × 10 — the full per-image budget)
+ 11,788   marker localization   (11,788 × 1)
─────────
 165,032
```

The triadic family reaches its ten-item budget on every image: it is budget-limited, not data-limited, and scales by a single constant.

**Annotation density per photograph.** Dividing by 11,788: **27.24** traits clear the salience confidence bar (`definitely` or `score ≥ 4.0`), and **23.79** attribute groups carry at least one present value — the effective count of CUB's attribute groups minus `has_size` (excluded) and those with no positive label.

**Stem uniqueness reflects vocabulary size, not duplication.** Relative Salience has one unique stem per question, because distractors are sampled from a 1,533-value global vocabulary. Anatomical Localization has 14,462 stems across 165,032 questions: the part vocabulary is 15 items, so option sets collide by combinatorial necessity. An identical stem paired with a different photograph is a different item; stem counts are read alongside photograph counts.

**Questions per photograph.** Three denominators are available; the paper reports the second:

| Denominator | Value | Questions each |
| --- | ---: | ---: |
| Image files on disk | 23,576 | 34.96 |
| Distinct photographs, all items | 11,788 | **69.91** |
| Distinct photographs, source items only | 11,788 | 68.91 |

The middle row counts every question against the photograph it depicts, without double-counting a photograph rendered twice.

### 10.4 The naming axis

The naming axis is smaller than the naive ceiling, and the gap is structural:

- **MCQ = 11,191**, not 11,788. The `"What type of bird is this?"` item requires a common name of at least two words to extract a folk category; **597 images** have single-word class names and are skipped.
- **Open-ended = 46,375**, against a ceiling of `11,788 × 4 = 47,152` (common name, *kind-of*, scientific name, genus). The **777-item** shortfall is the union of those 597 single-word names — which also suppress the *kind-of* item — and images whose `taxon` resolves to `Unknown` or to a single word.

Both quantities trace deterministically to `taxon_name_mapping.json` coverage and class-name morphology, and are reproduced exactly by any run against the released mapping.

---

## 11. Question Taxonomy

`categorize_qa_metadata()` is a first-match-wins cascade:

| Rank | Trigger substrings (lowercased) | Assigned category |
| --- | --- | --- |
| 1 | `genus`, `type of bird` | Taxonomy & Naming |
| 2 | `scientific name`, `common name`, `kind of` | Taxonomy & Naming |
| 3 | `most prominently visible` | Relative Salience |
| 4 | `are visible in this image` | Anatomical Localization |
| 5 | `color` | Morphological Attributes |
| 6 | `shape`, `length`, `pattern` | Morphological Attributes |
| 7 | `body part of this bird marked with` | Anatomical Localization |
| — | *fallthrough* | Morphological Attributes |

Rule 3 **must** precede rule 4, since salience stems contain the word *visible*; the ordering is load-bearing. The fallthrough absorbs residual attribute groups (`has_head_pattern`, `has_primary_color`, …) whose surface forms escape rules 5–6, so the Morphological bucket is defined *residually* rather than positively — any new question family added to `make_qa()` lands there unless a rule is added for it.

---

## 12. Output Schemas

### 12.1 `CUB200_attributes.csv` — one row per image (11,788)

See Section 7.2.

### 12.2 `QA.csv` — one row per question (824,104)

| Column | Type | Description |
| --- | --- | --- |
| `image_id` | int **or** str | CUB integer index, or a 5-hex UUID for annotated images |
| `filename` | str | `images/{class}/{file}.jpg` or `Images_Annotated/{file}.jpg`, relative to `BASE_DIR` |
| `question` | str | Question stem; MCQ items carry an inline `Options: (A) …, (B) …` suffix |
| `answer` | str | `"(C) crown"` for MCQ; bare string for open-ended |
| `category` | str | One of the four axes in Section 11 |

Because integer indices and UUID strings share `image_id`, the column is `object` dtype. Joins back to `CUB200_attributes.csv` go through the source stem (Section 9.1), not `image_id`.

### 12.3 `Images_Annotated/` — 11,788 JPEGs

Flat directory, one file per photograph, named `{source_stem}_{uuid5}.jpg`. These are **derived** artifacts, excluded from version control and regenerated by Stage 2 as needed.

---

## 13. Reproducibility

The corpus is deterministic given the CUB release, the taxon mapping, and the pipeline source:

- **Seeding.** `generate_qa.py` calls `random.seed(42)` at import, before any vocabulary construction or sampling. Every draw — attribute keys, distractor subsets, marker colour/shape, decoy positions, option shuffles — descends from that single seed.
- **Stable identifiers.** Annotated filenames use a `uuid5` derived from the source stem, not `uuid4`, so marker renders keep their names across regenerations.
- **Ordered traversal.** Images are processed in `image_id` order, so the consumption order of the random stream is fixed.
- **Fixed vocabularies.** `category_pool`, `group_pool`, and `full_trait_pool` are built by a full sweep of the corpus before generation, and do not depend on per-image state.

Re-running Stages 1–3 against the same inputs reproduces `QA.csv`, `Images_Annotated/`, and every figure in Section 10 exactly.

---

## 14. Citation and Terms

If you use this component, please cite both the underlying dataset and Bio-VQA:

```bibtex
@techreport{WahCUB_200_2011,
  title       = {{The Caltech-UCSD Birds-200-2011 Dataset}},
  author      = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
  institution = {California Institute of Technology},
  number      = {CNS-TR-2011-001},
  year        = {2011}
}
```

CUB-200-2011 is distributed for **non-commercial research and educational use**; derived corpora inherit those terms. Images retain the rights of their original photographers as documented in the CUB release.

---

<p align="center">
  <sub>Part of the <b>Bio-VQA</b> benchmark · Imageomics Institute</sub>
</p>
