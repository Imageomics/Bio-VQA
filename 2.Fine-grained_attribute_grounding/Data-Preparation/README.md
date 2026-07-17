<h1 align="center"> Bio-VQA &rarr; Fine-Grained Visual Attribute Grounding (CUB-200-2011 Dataset) </h1>

<p align="center">
  <b> Fine-Grained Visual Attribute Grounding &mdash; Data Acquisition &amp; QA Generation Pipeline </b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="Dataset" src="https://img.shields.io/badge/source-CUB-200-2011-green">
  <img alt="Status" src="https://img.shields.io/badge/status-active-brightgreen">
</p>


# BioVQA — Fine-Grained Visual Attribute Grounding

**A CUB-200-2011–derived question–answer corpus for evaluating fine-grained perceptual and grounding ability in vision–language models.**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Motivation and Design Philosophy](#2-motivation-and-design-philosophy)
3. [Repository Layout](#3-repository-layout)
4. [Step 0 — Obtaining the CUB-200-2011 Dataset](#4-step-0--obtaining-the-cub-200-2011-dataset)
5. [Step 0.5 — The Taxon Name Mapping File](#5-step-05--the-taxon-name-mapping-file)
6. [Environment and Dependencies](#6-environment-and-dependencies)
7. [Stage 1 — Building the Consolidated Attribute Table](#7-stage-1--building-the-consolidated-attribute-table)
8. [Stage 2 — Question Synthesis and Visual Grounding](#8-stage-2--question-synthesis-and-visual-grounding)
9. [Stage 3 — Corpus Statistics](#9-stage-3--corpus-statistics)
10. [Question Taxonomy](#10-question-taxonomy)
11. [Output Schemas](#11-output-schemas)
12. [Citation](#11-citation)

---

## 1. Overview

This component of **BioVQA** converts the *Caltech-UCSD Birds-200-2011* (CUB-200-2011) annotation corpus into a large-scale multiple-choice and open-ended visual question answering benchmark that probes four distinct competencies:

| Competency | Probe |
| --- | --- |
| **Taxonomy & Naming** | Can the model recover common name, folk category, genus, and binomial nomenclature from pixels alone? |
| **Morphological Attributes** | Can the model read off discrete traits — bill shape, plumage colour, wing pattern, tail length? |
| **Relative Salience** | Among several traits true of the taxon, can the model identify which is *most prominently visible in this image*? |
| **Anatomical Localization** | Can the model report part **visibility**, and resolve a **visual prompt** — a coloured marker overlaid on the image — to the anatomical part beneath it? |

The pipeline is deliberately **three-staged and file-mediated**: each stage writes a durable, inspectable artifact to disk, so any stage may be re-run, audited, or replaced without recomputing the others.

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

## 2. Motivation and Design Philosophy

Three principles govern the construction of this corpus.

**(i) Denormalize once, query forever.** CUB-200-2011 ships as a constellation of whitespace-delimited index files that must be joined on integer keys. Stage 1 performs this join exactly once and materializes a single wide table in which every image carries its class, taxon, bounding box, split assignment, full attribute dictionary (with annotator certainty *and* confidence score), and complete part-location record. Downstream consumers never touch the raw files again.

**(ii) Distractors are drawn from the empirical trait vocabulary, not invented.** Every multiple-choice item constructs its distractor pool from the *observed* value set of the relevant attribute group — or, for salience items, from the global trait vocabulary — excluding all values true of the current image. This guarantees distractors are (a) plausible ornithological alternatives rather than nonsense strings, and (b) *genuinely* incorrect: a distractor is never a trait the annotators marked present.

**(iii) Grounding requires a controlled visual confound.** For the visually-prompted localization task a marker of a given *(colour, shape)* is drawn on the target part, and a battery of decoys is scattered across the image. Critically, the decoy set always contains **at least one marker sharing the target's colour but not its shape**, so a model cannot succeed by attending to colour alone. Conjunction binding is required.

---

## 3. Repository Layout

The three scripts are referred to throughout by the following canonical names. Rename your local copies to match, or substitute freely.

| Script | Role |
| --- | --- |
| `build_attribute_table.py` | Stage 1 — joins raw CUB annotations into `CUB200_attributes.csv` |
| `generate_qa.py` | Stage 2 — synthesizes `QA.csv` and renders `Images_Annotated/` |
| `dataset_stats.py` | Stage 3 — emits `data-stats.txt` |

Stages 1 and 2 each declare a module-level constant:

```python
BASE_DIR = "/path/to/your/dataset/directory"   # ← EDIT THIS
```

`BASE_DIR` must point at the **extracted `CUB_200_2011/` directory** (see §4). It is the single source of truth for every path in the pipeline; set it in both scripts before running anything.

---

## 4. Step 0 — Obtaining the CUB-200-2011 Dataset

**Download the CUB dataset first.** The pipeline reads nothing from the network and will fail immediately if the raw annotations are absent.

CUB-200-2011 (Wah, Branson, Welinder, Perona & Belongie, 2011) comprises **11,788 images** spanning **200 North American bird species**, annotated with **312 binary attributes**, **15 part locations**, and one bounding box per image.

```bash
# Canonical distribution (Caltech Data / CVL):
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -xzvf CUB_200_2011.tgz
```

### Expected post-extraction layout

Upon extraction you must be able to resolve the following, **relative to `BASE_DIR`**:

```
BASE_DIR/                                  # ← CUB_200_2011/
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
└── taxon_name_mapping.json                # SCI_MAP_FILE — user-supplied, see §5
```

which corresponds exactly to the constant block at the head of Stage 1:

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

> **Note on `attributes.txt`.** Some CUB releases place `attributes.txt` at the *archive root*, one level **above** `CUB_200_2011/`, rather than inside `attributes/`. If Stage 1 reports an empty attribute-name mapping, move or symlink it into place:
> ```bash
> mv attributes.txt "$BASE_DIR/attributes/attributes.txt"
> ```

### Sanity check

```bash
wc -l "$BASE_DIR/images.txt"                             # 11788
wc -l "$BASE_DIR/attributes/attributes.txt"              # 312
wc -l "$BASE_DIR/parts/parts.txt"                        # 15
wc -l "$BASE_DIR/attributes/image_attribute_labels.txt"  # ~3,677,856
find "$BASE_DIR/images" -name '*.jpg' | wc -l            # 11788
```

The attribute label file is the heavyweight: roughly **3.68 M rows** (11,788 images × 312 attributes). Expect Stage 1 to spend most of its wall-clock there.

---

## 5. Step 0.5 — The Taxon Name Mapping File

CUB-200-2011 supplies **common names only**, encoded in the directory prefixes (`001.Black_footed_Albatross`). BioVQA additionally asks for **scientific name** and **genus**, which requires an external mapping.

Create `BASE_DIR/taxon_name_mapping.json` as a flat object keyed on the *cleaned* common name — the class string with its numeric prefix stripped and underscores replaced by spaces:

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
- A **one-word** value yields a genus question only; the script logs the first such occurrence as a diagnostic.
- A missing key resolves to the sentinel `"Unknown"`, and **no** nomenclature question is emitted for that image.

If the file is absent entirely, Stage 1 prints a warning and proceeds — every taxon becomes `"Unknown"`, silently suppressing the entire Taxonomy & Naming axis. **Verify the warning does not appear** in your run.

---

## 6. Environment and Dependencies

Consistent with the wider BioVQA HPC workflow, use the existing `myenv_py312` environment; do not create a new one unless strictly necessary.

```bash
source /path/to/myenv_py312/bin/activate
python -c "import pandas, numpy, cv2; print(pandas.__version__, cv2.__version__)"
```

| Package | Used by | Purpose |
| --- | --- | --- |
| `pandas` | 1, 2, 3 | Tabular join, explode, normalize |
| `numpy` | 1 | Transitive numeric support |
| `opencv-python` | 2 | Image I/O and marker rasterization |

Standard library only otherwise (`os`, `json`, `random`, `uuid`, `math`, `argparse`).

### Resource envelope

| Stage | Wall-clock (typical) | Peak RSS | Disk written |
| --- | --- | --- | --- |
| 1 | ~10–25 min (dominated by `iterrows` over 3.68 M attribute rows) | ~4–8 GB | ~1–2 GB (`CUB200_attributes.csv`) |
| 2 | ~20–60 min (dominated by per-image decode/encode) | ~2 GB | ~1 GB images + ~200 MB CSV |
| 3 | seconds | ~1 GB | ~4 KB |

Stages 1 and 2 are single-threaded and CPU-bound. On the SLURM cluster a standard (non-GPU) node suffices; request generous **memory** rather than generous cores.

---

## 7. Stage 1 — Building the Consolidated Attribute Table

```bash
python build_attribute_table.py
# → BASE_DIR/CUB200_attributes.csv
```

### What it does

The script is a sequence of small, defensive loaders, each of which tolerates malformed lines rather than aborting.

**`load_attributes()`** parses `image_attribute_labels.txt` into a five-column frame `(image_id, attr_id, present, certainty, score)`. Rows with fewer than five whitespace-delimited fields, or which fail integer/float coercion, are counted and skipped; the tally is printed. This matters: a handful of lines in the canonical release carry an extra trailing field, and naive split-and-unpack parsers crash on them. Note the deliberate use of `parts[-1]` for `score` — anchoring on the *last* field rather than index 4 makes the parse robust to exactly that extra column.

**`load_simple_mapping()`** is a generic `int → value` reader over any two-column file (`classes.txt`, `images.txt`, `attributes.txt`, `certainties.txt`, `image_class_labels.txt`, `train_test_split.txt`), splitting on the *first* delimiter only so values containing spaces survive intact.

**`load_bboxes()`** returns `{image_id: [x, y, w, h]}` in absolute pixel coordinates (CUB convention: top-left origin, width/height in pixels).

**`load_parts()`** returns a nested dict `{image_id: {part_name: {visible, x, y}}}`. Coordinates of **non-visible** parts are zeroed rather than retained — CUB stores undefined values there, and propagating them would poison any downstream geometric reasoning.

**`build_trait_dict()`** inverts the long-format attribute frame into, per image, a dictionary:

```json
{ "has_bill_shape::dagger": { "present": 1, "certainty": "definitely", "score": 4.0 }, ... }
```

Attribute IDs are 1-indexed in CUB and are decremented on lookup. Certainty integers are resolved to their string labels (`not visible`, `guessing`, `probably`, `definitely`) via `certainties.txt`, preserving annotator confidence as a first-class field rather than discarding it.

### Output columns

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

`attributes` and `parts` are serialized as JSON *inside* CSV cells. This keeps the artifact a single flat file at the cost of an escaping layer — see §13.

---

## 8. Stage 2 — Question Synthesis and Visual Grounding

```bash
python generate_qa.py
# → BASE_DIR/QA.csv
# → BASE_DIR/Images_Annotated/*.jpg
```

`random.seed(42)` is set at import. See §12 for what this does and does not guarantee.

### 8.1 Global vocabulary construction

Before any question is written, the script sweeps the entire corpus twice:

- **`category_pool`** — the set of terminal words across all 200 common names (`Albatross`, `Warbler`, `Kingfisher`, …). This is the folk-taxonomic vocabulary from which *"What type of bird is this?"* distractors are drawn.
- **`build_attribute_pools()`** — yields `group_pool` (`{attribute group → observed values}`, e.g. `wing color → [blue, brown, …]`) and `full_trait_pool` (the flat `"group: value"` vocabulary spanning all groups).

The `has_size::` group is **excluded throughout**. Absolute size is not recoverable from a cropped photograph without a scale reference, so size questions would be unanswerable in principle and would depress scores for reasons orthogonal to perception.

### 8.2 Per-image question families

**(a) Nomenclature — open-ended and MCQ.**

- `"What is the common name of the bird?"` → free-form.
- `"What type of bird is this in the image?"` → 5-way MCQ over `category_pool`.
- `"What kind of {category} is the bird in this image?"` → free-form, but *scaffolded*: the folk category is given, so only the modifier must be produced. The contrast between this and the bare common-name question isolates how much of the recognition failure is retrieval versus discrimination.
- `"What is the scientific name of the bird?"` → free-form binomial.
- `"What is the genus of the bird?"` → free-form.

**(b) Morphological attributes — grouped MCQ.** For every attribute group with at least one `present == 1` value, one 5-way MCQ is emitted: the key is sampled from the true values, and up to four distractors are sampled from `group_pool[group]` *minus all true values*. When an image is annotated with several values in a group (birds are genuinely multi-coloured), any one is an acceptable key — the distractor exclusion guarantees no alternative is secretly correct.

**(c) Relative salience — global MCQ.** For every trait clearing the confidence bar (`certainty == "definitely"` **or** `score ≥ 4.0`), an item asks which of five traits is *most prominently visible*. Distractors are drawn from `full_trait_pool` minus **all** traits present on the image — a much stronger exclusion than (b), since a distractor from a different attribute group could otherwise be trivially true.

**(d) Part visibility — paired.** For `eye`, `leg`, and `wing`, one 4-way MCQ over `{Left, Right, Both, None}`, derived from the left/right visibility flags. This probes lateral discrimination, a known failure mode: a bird in profile has exactly one visible eye, and models routinely answer *"Both"*.

**(e) Part visibility — triadic.** Up to ten 5-way items of the form *"Which of the following parts are visible?"* with options `(A)(B)(C) {parts}, (D) None, (E) All three`. The generator deliberately schedules an **all-hidden** item (answer `D`) and an **all-visible** item (answer `E`) first, then fills the remainder with 1-visible / 2-hidden triads. Without this scheduling `D` and `E` would be near-empty answer classes and the family would be gameable by never selecting them.

### 8.3 Visually-prompted localization

This is the most differentiated task family in C2, and the only one that mutates pixels.

For each image with at least one visible part:

1. **Target selection.** A visible part is sampled uniformly; a `(colour, shape)` is sampled from `{blue, green, red, white, yellow} × {dot, cross, plus, star}`.
2. **Target rendering.** `draw_mark()` rasterizes the marker at the part's annotated coordinate at `size=12` — deliberately larger than every decoy (`size=10`), so the target is not *hidden*, merely *confusable*.
3. **Decoy battery.** Decoys are placed at random positions subject to a **minimum-separation constraint** (`is_far_enough`, `min_dist=40 px`, up to 100 rejection-sampling attempts), preventing overlap that would render a marker unreadable.
4. **The colour confound.** The decoy list is *seeded* with a `(target_colour, non-target_shape)` pair. Consequently the target colour appears at least twice per image with different shapes, and the conjunction of colour **and** shape — not either feature alone — is required to disambiguate. This is the single most important design decision in the task.
5. **Option construction.** Options are assembled with a deliberate difficulty gradient: the target's **contralateral partner** (`left eye` ↔ `right eye`) is inserted first when applicable, then hidden parts, then other visible parts, until four distractors are held; the target is appended and the list shuffled.
6. **Persistence.** The composite is written to `Images_Annotated/{basename}_{uuid5}.jpg`, and a row is emitted with `image_id` set to the 5-hex UUID — a *string*, distinct from the integer IDs of unmodified images (see §13).

The resulting item reads:

> *What is the body part of this bird marked with the red star?*
> *Options: (A) left leg, (B) throat, (C) nape, (D) right eye, (E) belly*

### 8.4 Explosion and categorization

The nested `qa_pairs` lists are `explode()`-ed and `json_normalize()`-ed into one row per question. `filename` is rewritten to a `BASE_DIR`-relative path (`images/…` or `Images_Annotated/…`), and `categorize_qa_metadata()` assigns each question to one of the four axes by ordered substring matching on the lowercased question text — order matters, as `"most prominently visible"` must be tested before the more general `"are visible in this image"`.

---

## 9. Stage 3 — Corpus Statistics

Unlike Stages 1 and 2, this script takes its input path on the command line:

```bash
python dataset_stats.py --csv_path "$BASE_DIR/QA.csv"
# → BASE_DIR/data-stats.txt   (written beside the CSV)
```

The report comprises four blocks:

1. **High-level metrics** — total QA pairs, unique images, mean/max/min questions per image.
2. **Species metrics** — species recovered by splitting `filename` on `/` and taking the second component (`images/001.Black_footed_Albatross/x.jpg → 001.Black_footed_Albatross`); images per species, questions per species.
3. **Structural analysis** — MCQ vs. open-ended (detected by the literal `"Options:"` marker) and the binary Yes/No share.
4. **Per-category breakdown** — count, share, unique questions, unique answers, and MCQ/open-ended split for each of the four axes.

> **Caveat on the species block.** Rows from `Images_Annotated/` have paths of the form `Images_Annotated/Black_Footed_Albatross_0046_18_a3f21.jpg` — a **flat** directory with no species component. `x.split('/')[1]` therefore returns the bare filename, and every annotated image is counted as its own "species". Treat block [2] as valid for the unmodified-image subset only, or filter first:
> ```python
> df = df[~df.filename.str.startswith("Images_Annotated/")]
> ```

---

## 10. Question Taxonomy

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

Two structural remarks. First, rule 3 **must** precede rule 4, since salience stems contain the word *visible*; the ordering is load-bearing. Second, the fallthrough absorbs residual attribute groups (`has_head_pattern`, `has_primary_color`, …) whose surface forms escape rules 5–6 — correct by construction, but it means the Morphological bucket is defined *residually* rather than positively. Any future question family added to `make_qa()` will silently land there unless a rule is added for it.

---

## 11. Output Schemas

### `CUB200_attributes.csv` — one row per image (11,788)

See Section §7.

### `QA.csv` — one row per question

| Column | Type | Description |
| --- | --- | --- |
| `image_id` | int **or** str | CUB integer index, or a 5-hex UUID for annotated images |
| `filename` | str | `images/{class}/{file}.jpg` or `Images_Annotated/{file}.jpg`, relative to `BASE_DIR` |
| `question` | str | Question stem; MCQ items carry an inline `Options: (A) …, (B) …` suffix |
| `answer` | str | `"(C) dagger"` for MCQ; bare string for open-ended |
| `category` | str | One of the four axes in §10 |

### `Images_Annotated/`

Flat directory of JPEGs, one per source image that had ≥1 visible part. Filenames are `{source_basename}_{uuid5}.jpg`. These are **derived** artifacts and can be regenerated at any time; exclude them from version control.

---

## 12. Citation

If you use this component, please cite both the underlying dataset and BioVQA:

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
