<h1 align="center"> Relative-Position &rarr; Stage 2 &rarr; Manifest Assembly &amp; Geometry Verification </h1>

<p align="center">
  <b>Mis-pairing recomputation, direction-convention check, key-balance audit</b>
</p>

<p align="center">
  <img alt="Stage" src="https://img.shields.io/badge/stage-2%20of%204-blue">
  <img alt="Compute" src="https://img.shields.io/badge/compute-login%20node-lightgrey">
  <img alt="Check" src="https://img.shields.io/badge/geometry%20check-0%2F1000%20mismatches-brightgreen">
  <img alt="Key" src="https://img.shields.io/badge/key-125%2Fdirection-success">
</p>

---

## Overview

| Script | Role |
| :--- | :--- |
| `split_qa.py` | Flatten, recompute the mis-pairings, **verify the direction convention**, shard |
| `stats.py` | Read-only audit: key balance, option order, clustering, runtime |

> **Design contract:** the direction convention used in the analysis is **verified against the
> generator's**, not assumed. A silent 180&deg; inversion would produce entirely plausible-looking
> tables downstream, and this is the only place it can be caught.

---

## Table of Contents

- [The Geometry Check](#the-geometry-check)
- [Mis-Pairing Recomputation](#mis-pairing-recomputation)
- [Columns Added](#columns-added)
- [The Audit](#the-audit)
- [Usage](#usage)
- [FAQ](#faq)

---

## The Geometry Check

**Run this before anything else.** The generator stored a gold direction; this script **recomputes it
from the box coordinates** and compares.

```python
_BINS = [("right", -22.5, 22.5), ("bottom-right", 22.5, 67.5),
         ("bottom", 67.5, 112.5), ("bottom-left", 112.5, 157.5),
         ("left", 157.5, 202.5), ("top-left", -157.5, -112.5),
         ("top", -112.5, -67.5), ("top-right", -67.5, -22.5)]

def direction(a, b):
    (x1, y1), (x2, y2) = centre(a), centre(b)
    ang = math.degrees(math.atan2(y1 - y2, x1 - x2))
    ca  = ang if ang >= -157.5 else ang + 360.0
    for name, lo, hi in _BINS:
        if lo <= ca < hi:
            return name
```

Output on the released corpus:

```
geometry check: 1000 of 1000 rows recomputed from the stored boxes
  all recomputed directions match the stored answers.
```

> ⚠️ **A mismatch means the direction convention in the analysis does not match the generator's**, and
> the most likely form of that error is a **180&deg; inversion** &mdash; which would look entirely
> plausible in every table downstream. The script reports it loudly and instructs that the run not
> proceed.

---

## Mis-Pairing Recomputation

Three columns derived from the `boxes` field:

| Column | Pairing |
| :--- | :--- |
| `mis_red_empty` | `red_empty → blue_target` |
| `mis_blue_empty` | `red_target → blue_empty` |
| `mis_both_empty` | `red_empty → blue_empty` |

These reproduce the generator's own `wrong_pairings` field **exactly, in order** &mdash; verified on
all 1,000 rows.

They are what makes the error attribution in Stage 4 possible: **every wrong answer maps to a specific
box confusion**.

---

## Columns Added

| Column | Purpose |
| :--- | :--- |
| `row_id` | assigned **before** shuffling; restores original order at merge |
| `answer_dir` | gold direction, canonicalised |
| `gt_recomputed` | recomputed gold &mdash; the geometry check |
| `mis_red_empty`, `mis_blue_empty`, `mis_both_empty` | error attribution |
| `image_abspath` | resolved once, absolutely |
| `n_options`, `answer_in_options`, `answer_pos` | option-order audit |
| `source_image` | **bootstrap cluster key** |

### Deterministic shuffle before interleaving

Rows arrive grouped by dataset and condition, so a plain `i::6` stride would hand each chunk a
systematic slice. `row_id` is assigned first, so the original order remains recoverable at merge.

---

## The Audit

### 1 &mdash; The answer key is perfectly uniform

| Direction | n | Share |
| :--- | ---: | ---: |
| `top-right` | 125 | 0.125 |
| `bottom` | 125 | 0.125 |
| `left` | 125 | 0.125 |
| `top-left` | 125 | 0.125 |
| `top` | 125 | 0.125 |
| `bottom-right` | 125 | 0.125 |
| `bottom-left` | 125 | 0.125 |
| `right` | 125 | 0.125 |

**Exactly 125 per direction.** The generator balances the key by construction, so **chance and the
majority baseline coincide at 0.125** and `lift = acc − 0.125` needs no skew correction.

> Worth stating in any write-up: unlike the viewpoint task, where `left` and `right` account for ~83%
> of gold labels, this key carries no directional bias.

### 2 &mdash; Option-order audit

```
options offered per item: {4: 1000}
rows whose gold answer is NOT offered: 0

position of the gold answer among the options:
  position 1   0.230
  position 2   0.257
  position 3   0.237
  position 4   0.276
```

Near-uniform, so a model preferring a fixed option *position* would not gain. The columns are retained
even though options are **no longer shown to the model** &mdash; Stage 4 uses them to recover which
direction each mis-pairing corresponds to.

Because options are not shown, `optPosBias` and `errOffOption` are undefined and Stage 4 drops those
columns rather than printing NaN.

### 3 &mdash; Error attribution availability

```
1000 of 1000 rows carry the three recomputed mis-pairings.
geometry check: 0 mismatches between the recomputed and stored directions.
```

### 4 &mdash; Clustering

```
rows per source photograph: mean 4.81  max 5
```

1,000 rows from **208 photographs** &mdash; the generator's fill passes pushed most plates to the
`--pairs_per_image_cap` of 5.

> **Effective sample size is 208**, not 1,000. Stage 4 bootstraps over `source_image`. At $n = 208$
> the intervals separate *"reads the layout"* from *"does not"*, but cannot rank adjacent models
> against each other.

---

## Usage

```bash
python split_qa.py --task_root /path/to/Relative_Position
python split_qa.py ... --n_chunks 8 --seed 1234 --force

python stats.py
python stats.py --run_dir /path/to/8.Relative_Position
```

### Reading order

```
split_qa.py  →  [geometry check MUST show 0 mismatches]  →  stats.py  →  run_all.sh
```

If the geometry check fails, **nothing downstream is valid**.

---

## FAQ

**Why recompute the gold direction when the generator already stored it?**
Because the analysis code and the generator are separate implementations of the same convention, written
at different times. If they disagree — a sign flip on `dy`, a different octant boundary, a
non-image-coordinate `atan2` — every result would be systematically wrong while looking perfectly
reasonable. Recomputing costs milliseconds and converts an invisible failure into a loud one.

**Why does the check also validate `wrong_pairings`?**
Because the error attribution in Stage 4 depends on the three mis-pairing directions being correct and
in a known order. Verifying that the recomputed triple reproduces the generator's stored list, in
order, confirms both the convention and the ordering convention simultaneously. If only the gold
direction were checked, an ordering error in `wrong_pairings` would still corrupt the attribution.

**Why keep the option columns if options are no longer shown?**
Two reasons. Stage 4 uses them to map mis-pairings back to directions. And the option-position audit
documents that when options *were* displayed, the gold answer's position was near-uniform — which
forecloses the objection that earlier results were confounded by position bias. Retaining the columns
costs nothing and preserves the ability to run the option-shown variant without regenerating.

**What would a non-uniform key mean for the metrics?**
`lift` would need the majority-direction floor rather than 0.125, and `balAcc` would diverge from
`acc`. Both are computed correctly either way — `evaluate.py` derives the majority floor from the data
rather than hardcoding it — but the uniform key means the two coincide, which simplifies reporting. If
the corpus is ever regenerated with a different filter set, this table is where the change would first
be visible.