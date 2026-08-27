<h1 align="center"> Relative-Position &rarr; Stage 1 &rarr; Box Solving &amp; QA Generation </h1>

<p align="center">
  <b>Six-box composition, solved decoy placement, octant labelling, rendering</b>
</p>

<p align="center">
  <img alt="Stage" src="https://img.shields.io/badge/stage-1%20of%204-blue">
  <img alt="Compute" src="https://img.shields.io/badge/compute-CPU%20only-lightgrey">
  <img alt="Output" src="https://img.shields.io/badge/output-1%2C000%20pairs-green">
  <img alt="Solver" src="https://img.shields.io/badge/solver-600%20attempts%2Fpair-orange">
  <img alt="Seed" src="https://img.shields.io/badge/seeded-fully-brightgreen">
</p>

---

## Overview

| Script | Output |
| :--- | :--- |
| `generateQA.py` | `Images/` + `qa.csv` &mdash; 1,000 pairs from 208 photographs |

> **Design contract:** every emitted item carries **exactly six boxes**, size-matched within a bounded
> ratio, with the two empty decoys **solved** so all three incorrect pairings yield distinct
> directions. Items that cannot satisfy the full composition are **rejected rather than emitted
> degraded** &mdash; there is no partial-distractor fallback.

---

## Table of Contents

- [What Is Drawn](#what-is-drawn)
- [Size Matching](#size-matching)
- [Solved Decoy Placement](#solved-decoy-placement)
- [Direction Labelling](#direction-labelling)
- [Format Calibration](#format-calibration)
- [Budget and Fill Passes](#budget-and-fill-passes)
- [Output Schema](#output-schema)
- [Usage](#usage)
- [FAQ](#faq)

---

## What Is Drawn

```
   ┌──────────────────────────────────────────────────────────────────┐
   │                                                                  │
   │    ┌────┐ red_empty              ┌────┐ cyan                     │
   │    │    │  (empty tray)          │ 🪲 │  (non-target specimen)   │
   │    └────┘                        └────┘                          │
   │                                                                  │
   │           ┏━━━━┓ red_target    ┏━━━━┓ blue_target                │
   │           ┃ 🪲 ┃  ← THE ONE    ┃ 🪲 ┃  ← THE REFERENCE          │
   │           ┗━━━━┛                ┗━━━━┛                           │
   │                                                                  │
   │    ┌────┐ green                        ┌────┐ blue_empty         │
   │    │ 🪲 │  (non-target specimen)       │    │  (empty tray)      │
   │    └────┘                              └────┘                    │
   └──────────────────────────────────────────────────────────────────┘
```

| Colour | Count | Placement |
| :--- | :---: | :--- |
| red | 2 | one on the **target** specimen, one on **empty space** |
| blue | 2 | one on the **target** specimen, one on **empty space** |
| green | 1 | on a **non-target** specimen |
| cyan | 1 | on a **non-target** specimen |

### Why empty boxes rather than only extra specimens

If both red boxes contained specimens, *"the specimen inside the red box"* would be ambiguous and the
item unanswerable. If neither decoy existed, *"find the red rectangle"* would suffice and the model
would never check contents. **One empty red and one empty blue is the minimal construction** that
forces colour-plus-content discrimination while keeping the referent unique.

### Why green and cyan sit on real specimens

If every decoy were empty and every target occupied, *"which box contains an animal"* would identify
the targets directly &mdash; a new shortcut replacing the old one. Two neutral boxes on genuine
non-target specimens close it: **box occupancy alone is uninformative**.

---

## Size Matching

| Flag | Default | Effect |
| :--- | ---: | :--- |
| `--max_size_ratio` | 2.0 | Bound on largest &divide; smallest area across the six boxes |
| `--size_jitter` | 0.10 | Random size variation applied to empty boxes |
| `--min_box_frac` | 0.0005 | Floor on box area as a fraction of the canvas |

Empty boxes are sized from the **median annotated box in that image**, not from a global constant, so
they match the local specimen scale.

> Without this, an empty box could be identified by size alone and the content check bypassed.

---

## Solved Decoy Placement

This is the technical core of the generator.

```python
def build_free_pool(W, H, real_boxes, ref_wh, rng, margin, jitter, n=120):
    """Empty regions sized from the median annotated box (so distractors match
    the local specimen scale) ... an undetected specimen is less likely to fall
    inside a supposedly empty box."""

def solve_distractors(pool, rt, bt, ans, margin_deg, rng, tries=600):
    """Pick an empty red box and an empty blue box so that all three incorrect
    referent pairings produce distinct wrong directions."""
```

The solver searches for a `(red_empty, blue_empty)` assignment satisfying:

| Pairing | Must yield |
| :--- | :--- |
| `red_empty → blue_target` | a direction &ne; gold, and &ne; the other two |
| `red_target → blue_empty` | a direction &ne; gold, and &ne; the other two |
| `red_empty → blue_empty` | a direction &ne; gold, and &ne; the other two |

Up to **600 attempts per pair**. Candidate empty regions come from a pre-built free-space pool that
avoids all annotated boxes with a `--free_margin` of 0.015 of the canvas, reducing the chance that an
unannotated specimen falls inside a supposedly empty box.

> **Every wrong answer therefore identifies a specific confusion**, rather than being generic error.
> That is what makes the error attribution in Stage 4 possible, and it costs nothing at annotation time
> &mdash; only search time during generation.

---

## Direction Labelling

```python
def direction_of(c1, c2, margin_deg):
    """None if within margin_deg of an octant boundary."""
```

$\theta = \operatorname{atan2}(y_r - y_b,\; x_r - x_b)$ in image coordinates ($y$ increasing
downward), binned into eight 45&deg; octants.

### Three quality filters

| Filter | Default | Rejects |
| :--- | ---: | :--- |
| `--margin_deg` | 8.0 | Pairs within 8&deg; of an octant boundary. At 22.4&deg; the label is `right`, at 22.6&deg; `bottom-right`; the visual difference is imperceptible and the label arbitrary. |
| `--min_sep` | 0.60 | Centre distance below 0.6&times; the summed half-diagonals. Adjacent specimens have unstable centre-difference directions. |
| `--max_iou` | 0.05 | Target boxes that meaningfully overlap. |

> These mean the corpus is drawn from the **easier tail** of the pair distribution. A deliberate trade:
> an ambiguous ground truth is worse than a restricted sample, but the restriction must be stated.

---

## Format Calibration

Before any pair is emitted, the generator runs a calibration pass over a sample of images:

```python
def calibrate(ds_path, df, rng, sample_imgs=120, sample_rows=1500):
```

It **detects** the dataset's bounding-box convention and normalisation empirically rather than assuming
a format from column names. Annotation formats vary &mdash; `x_center/y_center/width/height`
normalised, `x/y/w/h` absolute, `x_min/y_min/x_max/y_max` &mdash; and a wrong assumption produces boxes
that are silently displaced rather than obviously broken.

---

## Budget and Fill Passes

| Flag | Default |
| :--- | ---: |
| `--target_pairs` | 1000 |
| `--pairs_per_image` | 2 |
| `--pairs_per_image_cap` | 5 |

Two pairs per image nominally; if some images yield fewer after the rejection filters, later passes draw
up to five from images that can supply them.

The released corpus has **1,000 pairs from 208 photographs**, averaging **4.81** per image &mdash; so
the fill passes were heavily used and most plates contributed near the cap.

> ⚠️ **Consequence for statistics:** effective sample size is the **208 photographs**, not the 1,000
> rows. Pairs from one plate share specimens, lighting and scale. Stage 4 bootstraps over
> `source_image`.

---

## Output Schema

| Column | Example | Notes |
| :--- | :--- | :--- |
| `image` | `Beetles-500_A00000046073_pair0_dist.jpg` | rendered file |
| `image_path` | `Images/…` | relative to the task directory |
| `dataset` | `Beetles-500` | |
| `condition` | `distractor` | |
| `pair_id` | `Beetles-500_A00000046073_pair0` | groups conditions if both generated |
| `source_image` | `A00000046073.jpg` | **bootstrap cluster key** |
| `answer` | `left` | gold direction |
| `options` | `left\|bottom-right\|top-right\|right` | retained for evaluation, **no longer shown to the model** |
| `n_boxes` | `6` | |
| `size_ratio` | `1.591` | largest/smallest box area |
| `boxes` | JSON | **all six boxes plus `wrong_pairings`** |

### The `boxes` field

```json
{"red_target":  [807.5, 626.7, 958.8, 946.3],
 "red_empty":   [1605.7, 13.9, 1819.1, 269.5],
 "blue_target": [1054.4, 659.6, 1305.7, 943.8],
 "blue_empty":  [43.5, 124.8, 224.2, 376.6],
 "green":       [525.8, 329.2, 730.4, 584.2],
 "cyan":        [1548.4, 307.0, 1740.6, 540.4],
 "wrong_pairings": ["top-right", "bottom-right", "right"]}
```

Boxes are `[x1, y1, x2, y2]`. `wrong_pairings` lists the three mis-pairing directions in a **fixed
order**:

1. `red_empty → blue_target`
2. `red_target → blue_empty`
3. `red_empty → blue_empty`

`Preprocess/split_qa.py` recomputes all three from the coordinates and verifies they reproduce this
field exactly &mdash; a standing check that the direction convention in the analysis matches the one
used at generation.

---

## Usage

```bash
python generateQA.py \
    --base_dir   /path/to/datasets \
    --output_dir /path/to/Relative_Position \
    --dataset    Beetles-500 \
    --seed       1234 \
    --target_pairs 1000 \
    --pairs_per_image 2 --pairs_per_image_cap 5 \
    --max_size_ratio 2.0 --size_jitter 0.10 \
    --min_sep 0.60 --margin_deg 8.0 \
    --max_iou 0.05 --free_margin 0.015
```

CPU-only. `--seed` is honoured throughout &mdash; box selection, free-pool sampling and the distractor
solver all draw from a seeded RNG, so **the corpus is fully reproducible**.
