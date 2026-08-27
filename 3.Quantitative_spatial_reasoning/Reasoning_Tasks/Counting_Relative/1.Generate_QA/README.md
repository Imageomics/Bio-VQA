<h1 align="center"> Counting-Relative &rarr; Stage 1 &rarr; Grid Synthesis &amp; QA Generation </h1>

<p align="center">
  <b>2&times;2 composite construction, unique-extremum enforcement, question templating</b>
</p>

<p align="center">
  <img alt="Stage" src="https://img.shields.io/badge/stage-1%20of%204-blue">
  <img alt="Compute" src="https://img.shields.io/badge/compute-CPU%20only-lightgrey">
  <img alt="Output" src="https://img.shields.io/badge/output-20%2C000%20grids-green">
  <img alt="Resample" src="https://img.shields.io/badge/resampling-LANCZOS-orange">
</p>

---

## Overview

| Script | Condition | Output |
| :--- | :--- | :--- |
| `generateQA_same.py` | four panels from **one** dataset | `Same/Images/`, `Same/qa.csv` |
| `generateQA_diff.py` | four panels from **four** datasets | `Different/Images/`, `Different/qa.csv` |

Together they emit **20,000 composites** and two questions per composite (`most`, `least`), giving
**42,412 rows** once GrevysZebra's three animal terms are counted.

> **Design contract:** a quadruple is composited **if and only if** its maximum and minimum counts are
> each attained exactly once. Rejected quadruples are discarded, never emitted degraded.

---

## Table of Contents

- [Canvas Construction](#canvas-construction)
- [Selection Rule](#selection-rule)
- [Question Templating](#question-templating)
- [Output Schema](#output-schema)
- [The Parsing Hazard](#the-parsing-hazard)
- [The `different`-Condition Prior](#the-different-condition-prior)
- [Usage](#usage)
- [FAQ](#faq)

---

## Canvas Construction

Four source photographs of differing native resolution and aspect ratio become one 2&times;2 grid in
three stages:

```
 ── STEP 1 ──────────────────────────────────────────────────────────
   resize img2 to img1's HEIGHT, concatenate horizontally
        ┌──────────────┬────────────────┐
        │   A (img1)   │   B (img2')    │   row1_w = w1 + w2'
        └──────────────┴────────────────┘

 ── STEP 2 ──────────────────────────────────────────────────────────
   resize img4 to img3's HEIGHT, concatenate horizontally
        ┌────────────┬──────────────────┐
        │  C (img3)  │   D (img4')      │   row2_w = w3 + w4'
        └────────────┴──────────────────┘

 ── STEP 3 ──────────────────────────────────────────────────────────
   rescale row 2 so its width matches row 1 exactly, stack vertically
        ┌──────────────┬────────────────┐
        │   A          │   B            │
        ├──────────────┴────────────────┤
        │   C          │   D            │
        └──────────────┴────────────────┘
```

```python
# row 1: match heights, then concatenate
h1     = i1.height
w2_new = int(i2.width * (h1 / i2.height))
i2_res = i2.resize((w2_new, h1), Image.Resampling.LANCZOS)
row1_w = i1.width + w2_new

# row 2: same, then rescale the whole row to row 1's width
scale      = row1_w / row2_w_temp
row2_final = row2_temp.resize((row1_w, int(h3 * scale)),
                              Image.Resampling.LANCZOS)
```

**LANCZOS throughout**, to preserve the small high-frequency detail that counting depends on. Bilinear
or nearest-neighbour resampling visibly merges adjacent animals in dense aerial frames.

### Quadrant &rarr; letter mapping

| Quadrant | Letter |
| :--- | :---: |
| top-left | **A** |
| top-right | **B** |
| bottom-left | **C** |
| bottom-right | **D** |

Fixed, never permuted, and restated in the runner's prompt frame so a model never has to infer it.

### Borders &mdash; `different` condition only

`generateQA_diff.py` adds a 3-pixel black border per panel via `ImageOps.expand` before compositing:

```python
i1_b = ImageOps.expand(i1,     border=border_width, fill='black')
i2_b = ImageOps.expand(i2_res, border=border_width, fill='black')
```

Panels from different datasets can share background tone &mdash; two aerial grassland crops, for
instance &mdash; and without a separator the panel boundary is not always visible. The `same` condition
does not need this because all four panels share an acquisition style.

> ⚠️ **This asymmetry should be stated in any write-up.** The borders are uniform and content-free, so
> they carry no count information, but they remain a visible difference between the two conditions
> beyond the intended one.

---

## Selection Rule

```python
counts = [i1['animal_count'], i2['animal_count'],
          i3['animal_count'], i4['animal_count']]
max_count, min_count = max(counts), min(counts)

# both extrema MUST occur exactly once among the four
if counts.count(max_count) == 1 and counts.count(min_count) == 1:
    max_index = counts.index(max_count)
    min_index = counts.index(min_count)
    ...emit both questions
```

Three consequences:

| Consequence | Effect |
| :--- | :--- |
| Both polarities have a unique answer | Otherwise "which has the most" is ambiguous and the item unscoreable. |
| `most` and `least` never coincide | This is what licenses the ground-truth-free `incoherence` metric. |
| Near-ties are excluded | **Selection effect** &mdash; the corpus is the easier tail of the quadruple distribution. |

Duplicate grids are prevented by hashing the sorted panel signature before acceptance:

```python
sig = hashlib.md5("|".join(sorted(paths)).encode()).hexdigest()
if sig in seen: continue
```

---

## Question Templating

Generated per row, carrying the animal term and the fixed option list:

```
Which of these 4 images contains most number of bees: Image A (top-left),
Image B (top-right), Image C (bottom-left), Image D (bottom-right)?
```

| Condition | Animal term | Reason |
| :--- | :--- | :--- |
| `same` | the species (`bees`, `beetles`, …) | all four panels share it |
| `different` | generic `animals` | panels hold different species |

### Pluralisation

```python
if cls_name == 'kiang':                return "kiangs (wild asses)"
elif cls_name in ('cattle', 'sheep'):  return cls_name        # already plural
else:                                  return f"{cls_name}s"
```

The kiang gloss is the **only** explanatory parenthetical in the corpus. It is retained for fidelity to
the generator, and `Preprocess/split_qa.py` records the original question separately so its effect
could be isolated if needed.

---

## Output Schema

`qa.csv`, one row per question:

| Column | Example | Notes |
| :--- | :--- | :--- |
| `image` | `quad_Bees_17c4ef41.jpg` | basename |
| `image_path` | `Images/quad_Bees_17c4ef41.jpg` | relative to the condition directory |
| `question` | `Which of these 4 images contains most number of bees: …` | full text |
| `answer` | `Image A (top-left)` | verbatim, with the quadrant name |

`same` filenames embed the dataset (`quad_<Dataset>_<hash>.jpg`), which `split_qa.py` uses to recover
`source_dataset`. `different` filenames use `mixed_quad_<hash>.jpg` and carry no dataset, by design.

---

## The Parsing Hazard

The gold answer is the string `"Image A (top-left)"`. A case-insensitive `[ABCD]` search on it matches
the **lower-case `a` inside the word *Image***:

```python
re.search(r"([ABCD])", "Image B (top-right)", re.I).group(1)   # → 'A'   ✗ WRONG
```

Every gold answer would silently collapse to `A`. Extraction is therefore anchored on the word:

```python
GOLD_RE = re.compile(r"[Ii]mage\s*([ABCD])\b")                 # ✓
```

> A regression here raises no error &mdash; the key simply comes out 100% `A`. `Preprocess/stats.py`
> prints the letter distribution as a standing regression check.

---

## The `different`-Condition Prior

Because panels come from four datasets and count magnitude varies by dataset, domain recognition alone
answers correctly at well above chance. A Monte-Carlo over the empirical count distributions, applying
the generator's own unique-extremum rule, gives:

| Polarity | Prior baseline | vs chance |
| :--- | ---: | ---: |
| `most` | **0.530** | +0.280 |
| `least` | **0.548** | +0.298 |

Approximate per-dataset means driving the effect:

| Band | Datasets | Mean count |
| :--- | :--- | ---: |
| high | Bees | ~37 |
| mid | Beetles-500, Fish, Sheep-500, Seal-500, Cattle-500 | ~15&ndash;22 |
| low | Zebra-500, GrevysZebra, Camel-500, Kiang-500 | ~8&ndash;9 |

> These are **corpus properties, not constants**. Regenerating with a different dataset mix changes
> them. `3.Evaluate/evaluate.py` takes them as command-line arguments for exactly this reason, and they
> must be quoted alongside any `different`-condition number.

---

## Usage

```bash
python generateQA_same.py --base_dir   /path/to/datasets \
                          --output_dir /path/to/Counting-Relative/Same

python generateQA_diff.py --base_dir   /path/to/datasets \
                          --output_dir /path/to/Counting-Relative/Different
```

CPU-only. Runtime is dominated by image I/O and LANCZOS resampling &mdash; expect a few hours for
20,000 composites on a single node.

---

