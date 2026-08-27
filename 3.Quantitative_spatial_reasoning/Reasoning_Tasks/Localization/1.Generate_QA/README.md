<h1 align="center"> Localization &rarr; Stage 1 &rarr; Candidate Selection &amp; QA Generation </h1>

<p align="center">
  <b>Two loaders, single-annotation filtering, resolution-proxy ranking</b>
</p>

<p align="center">
  <img alt="Stage" src="https://img.shields.io/badge/stage-1%20of%204-blue">
  <img alt="Compute" src="https://img.shields.io/badge/compute-CPU%20seconds-lightgrey">
  <img alt="Output" src="https://img.shields.io/badge/output-4%2C000%20rows-green">
  <img alt="Catalogues" src="https://img.shields.io/badge/catalogues-4-orange">
</p>

---

## Overview

| Script | Output |
| :--- | :--- |
| `generateQA.py` | `qa.csv` &mdash; 4,000 rows across four catalogues |

> **Design contract:** an image enters the corpus **if and only if** it carries exactly one annotation
> **and** the file exists on disk. Everything else is discarded silently at generation rather than
> surfacing as an error during inference.

---

## Table of Contents

- [Source Catalogues](#source-catalogues)
- [Selection Rule](#selection-rule)
- [Question Templating](#question-templating)
- [Output Schema](#output-schema)
- [The `image_path` Quirk](#the-image_path-quirk)
- [Corpus Properties](#corpus-properties)
- [Usage](#usage)
- [FAQ](#faq)

---

## Source Catalogues

Images are drawn from **Component 4's** re-identification datasets, not from C3-specific imagery. This
is deliberate: it makes image-matched correlation between localization and re-identification possible
for the same photograph.

| Dataset | Format | Loader |
| :--- | :--- | :--- |
| GrevysZebra | flat CSV, columns `bbox x/y/w/h` | `process_csv()` |
| hyena | COCO JSON | `process_coco()` |
| leopard | COCO JSON | `process_coco()` |
| whale shark | COCO JSON | `process_coco()` |

Both loaders return the same tuple, so downstream handling is uniform:

```python
candidates.append((size, image_path, bbox, image_name))
```

Both formats store boxes as **`[x, y, width, height]`, `x, y` at the top-left corner, absolute
pixels** &mdash; COCO's convention. **No conversion is applied**, so the gold boxes already share the
frame the prompt asks for.

---

## Selection Rule

```
     all annotations
           │
           ├── len(anns) != 1 ────────────────────► DISCARD  (ambiguous referent)
           │
           ├── file missing on disk ──────────────► DISCARD
           │
           ▼
     candidates  ──►  sort by os.path.getsize DESC  ──►  take top N per catalogue
```

### 1 &mdash; Exactly one annotation

```python
valid = [g for _, g in grouped if len(g) == 1]      # CSV path
...
if len(anns) != 1:                                   # COCO path
    continue
```

An image with two annotated animals makes *"the subject animal (zebra)"* referentially ambiguous
&mdash; the definite article does not denote uniquely, and the item becomes unscoreable.

> **Consequence:** `n_targets = 1` for every row. The multi-target confusion diagnostic in Stage 4 is
> therefore undefined and reports `NaN`. This is expected behaviour, not a failure.

### 2 &mdash; Ranked by file size, descending

```python
size = os.path.getsize(image_path)
candidates.append((size, image_path, bbox, image_name))
candidates.sort(key=lambda x: (-x[0], x[1]))         # ties break on path
return candidates[:n_samples]
```

File size proxies for image resolution and detail. Ties break on path, so the selection is
**deterministic without needing a seed** &mdash; the only C3 generator for which this holds.

> ⚠️ **This is a selection effect and should be stated.** The corpus is biased toward the
> highest-quality photographs in each catalogue, not a random sample. Localization accuracy measured
> here is an optimistic estimate relative to the full catalogue.

---

## Question Templating

One fixed template per dataset, differing only in the species name:

```
Provide bounding box coordinates for the subject animal (zebra) in this image.
```

The species is named because the catalogues are single-species and the name disambiguates the subject
from any incidental animals in frame. **It does not make this a recognition task** &mdash; the model is
told what to look for.

> No coordinate format is specified at generation time. The format instruction is appended by the
> runner (Stage 3), so it can be held identical across tasks and revised without regenerating the
> corpus.

---

## Output Schema

| Column | Example | Notes |
| :--- | :--- | :--- |
| `image` | `0b49122e-39c5-401b-aaa3-7f61a9b780fe.png` | basename |
| `image_path` | `../4-Instance-Level-Identification/GrevysZebra/Images/….png` | relative to `--base_dir` |
| `question` | `Provide bounding box coordinates for the subject animal (zebra) in this image.` | |
| `answer` | `[1760.0, 1762.0, 1083.0, 785.0]` | `[x, y, w, h]`, absolute pixels |
| `dataset` | `zebra` | grouping key |

---

## The `image_path` Quirk

Paths are written with `os.path.relpath(img_path, args.base_dir)` and therefore contain
`../4-Instance-Level-Identification/…`. **They resolve against the parent of the Localization
directory, not against it.**

`Preprocess/split_qa.py` accepts an `--id_root` argument for exactly this reason and tries several
candidate resolutions in order:

```python
cands = [os.path.normpath(os.path.join(d, p)),
         os.path.normpath(os.path.join(task_root, p)),
         os.path.join(d, "Images", os.path.basename(p))]
if id_root:
    tail = p.split("4-Instance-Level-Identification", 1)
    if len(tail) == 2:
        cands.insert(0, os.path.join(id_root, tail[1].lstrip("/")))
```

Passing the wrong root produces a clean *"images missing"* report at split time rather than a failed
SLURM job.

---

## Corpus Properties

Measured after generation:

| Property | Value |
| :--- | ---: |
| Rows | 4,000 |
| Distinct images | 4,000 |
| Median canvas | 2400 &times; 1920 px |
| Gold box area, mean | **0.373** |
| Gold box area, median | 0.332 |
| Gold box area, min / max | 0.0046 / 0.9977 |
| Targets under 1% of canvas | 0.4% |
| Gold boxes outside their own canvas | 12 |

### The mean target occupies 37% of the frame

This is the single most important property of the corpus, and it follows from the source material:
wildlife re-identification photographs are framed on the subject, because that is what they are
collected for.

Its evaluation consequence &mdash; high trivial baselines &mdash; is documented in `3.Evaluate` and in
the task README.

### Out-of-bounds gold

Twelve boxes do not fit inside their own canvas, with small negative offsets such as `x = −2` or
`y = −7`, consistent with annotation rounding rather than corruption. `split_qa.py` flags them with
`gt_out_of_bounds`; `evaluate.py` excludes them; the count belongs in any limitations section.

---

## Usage

```bash
python generateQA.py \
    --base_dir          /path/to/Bio-VQA \
    --grevys_csv        /path/to/GrevysZebra/metadata.csv \
    --grevys_images     /path/to/GrevysZebra/Images \
    --hyena_json        /path/to/hyena/annotations.json \
    --hyena_images      /path/to/hyena/Images \
    --leopard_json      /path/to/leopard/annotations.json \
    --leopard_images    /path/to/leopard/Images \
    --whaleshark_json   /path/to/whaleshark/annotations.json \
    --whaleshark_images /path/to/whaleshark/Images \
    --output_dir        /path/to/Localization \
    --num_samples       1000
```

`--num_samples` is **per catalogue**; four catalogues give 4,000 rows.

CPU-only and fast &mdash; the work is a directory scan plus a `getsize` call per candidate. **No images
are opened or written**, which is why generation takes seconds rather than the hours the composite-image
tasks require.
