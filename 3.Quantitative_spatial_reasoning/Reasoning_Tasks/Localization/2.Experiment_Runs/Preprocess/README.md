<h1 align="center"> Localization &rarr; Stage 2 &rarr; Manifest Assembly &amp; Pre-Run Audit </h1>

<p align="center">
  <b>Cross-boundary path resolution, header-only dimension reads, baseline computation</b>
</p>

<p align="center">
  <img alt="Stage" src="https://img.shields.io/badge/stage-2%20of%204-blue">
  <img alt="Compute" src="https://img.shields.io/badge/compute-login%20node-lightgrey">
  <img alt="Chunks" src="https://img.shields.io/badge/chunks-6-green">
  <img alt="Rows" src="https://img.shields.io/badge/rows-4%2C000-informational">
</p>

---

## Overview

| Script | Role |
| :--- | :--- |
| `split_qa.py` | Resolve paths, read image dimensions, parse gold boxes, flag out-of-bounds, shard |
| `stats.py` | Read-only audit: **trivial baselines**, target sizes, clustering, runtime |

> **Design contract:** the baseline table printed by `stats.py` determines which metric column the
> results can legitimately be quoted from. Getting it wrong produces a paper claim that a trivial
> predictor would satisfy. **It is not optional reading.**

---

## Table of Contents

- [Path Resolution](#path-resolution)
- [Dimension Reads](#dimension-reads)
- [Derived Columns](#derived-columns)
- [The Out-of-Bounds Check](#the-out-of-bounds-check)
- [Keyed on the File, Not the Basename](#keyed-on-the-file-not-the-basename)
- [The Audit](#the-audit)
- [Usage](#usage)
- [FAQ](#faq)

---

## Path Resolution

The generator writes `image_path` relative to `--base_dir`, so entries point **outside** the
Localization directory:

```
../4-Instance-Level-Identification/GrevysZebra/Images/0b49122e-….png
```

Four candidate resolutions are tried in order:

```python
cands = [os.path.normpath(os.path.join(d, p)),
         os.path.normpath(os.path.join(task_root, p)),
         os.path.join(d, "Images", os.path.basename(p))]
if id_root:
    tail = p.split("4-Instance-Level-Identification", 1)
    if len(tail) == 2:
        cands.insert(0, os.path.join(id_root, tail[1].lstrip("/")))
```

`--id_root` is the explicit escape hatch. Passing the wrong root yields a clean *"images missing"*
report at split time, on a login node, rather than a failed SLURM job.

---

## Dimension Reads

```python
for p in df["image_abspath"].unique():
    with Image.open(p) as im:
        dims[p] = im.size          # header read only, no decode
```

`img_w` and `img_h` are needed **twice** downstream:

1. **In the prompt** &mdash; the model is told the canvas size, without which "in pixels" has no
   referent.
2. **In the resolver** &mdash; to convert whatever convention a model replies in into absolute pixels.

Reading them once per unique image at split time costs seconds. Reading them per row during inference
would repeat the work 4,000 times per model, eighteen times over.

---

## Derived Columns

| Column | Purpose |
| :--- | :--- |
| `gt_x`, `gt_y`, `gt_w`, `gt_h` | the gold box as four floats |
| `gt_area_frac` | box area &divide; canvas area &mdash; **the stratification key** |
| `gt_out_of_bounds` | gold box does not fit inside its own canvas |
| `n_targets` | annotations on this image (always 1 in this tier) |
| `img_w`, `img_h` | canvas dimensions |
| `row_id` | assigned **before** shuffling; restores original order |

$$
\text{gt\_area\_frac} \;=\; \frac{w \cdot h}{W \cdot H}
$$

---

## The Out-of-Bounds Check

```python
oob = ((df.gt_x < -1) | (df.gt_y < -1)
       | (df.gt_x + df.gt_w > df.img_w * 1.02)
       | (df.gt_y + df.gt_h > df.img_h * 1.02))
```

A gold box that does not fit its own image means the stored coordinates and the stored file disagree
&mdash; **a benchmark defect, not a model failure**.

Twelve rows are affected, with small negative offsets such as `x = −2`, consistent with annotation
rounding. They are flagged here, excluded by `evaluate.py`, and the count belongs in any limitations
section.

---

## Keyed on the File, Not the Basename

```python
df["n_targets"] = df.groupby("image_abspath")["image_abspath"].transform("size")
```

In the full three-tier corpus, **455 basenames are shared by different canvases across tiers**
(7,000 distinct files, 6,545 distinct basenames). Grouping on the basename would tell an Easy row it
has three targets because a Medium canvas happens to carry the same filename &mdash; and then feed it
into a confusion diagnostic it has no business in.

`split_qa.py` prints a note whenever basenames are not unique.

---

## The Audit

### 1 &mdash; What do the trivial baselines score?

**The most consequential output of the script.**

```
baseline        meanIoU  acc@.25  acc@.50  acc@.75  acc@.90
  wholeImage      0.373    0.602    0.295    0.114    0.044
  centre50        0.370    0.782    0.249    0.014    0.000
  meanBox         0.415    0.784    0.369    0.025    0.001
```

None requires looking at the image. `meanBox` &mdash; predict the average gold box every time &mdash;
reaches **IoU 0.415** and clears IoU 0.5 on **37%** of items.

| Consequence | Action |
| :--- | :--- |
| Mean IoU barely discriminates | Report `acc@.75` and `acc@.90`, where floors drop to 0.114 and 0.044 |
| Baselines **swap rank** between thresholds | Use the **maximum of the three** at whichever threshold is quoted |

### 2 &mdash; How large are the targets?

```
gold box area as a fraction of the canvas:
        mean     min     25%     50%     75%     max
easy  0.3734  0.0046  0.1469  0.3316  0.5495  0.9977

targets under 1% of the canvas: 0.4% of rows
```

The subject averages **37% of the frame** and reaches 99.8% &mdash; a property of the source material,
and what drives the high baselines above.

### 3 &mdash; How badly are rows clustered?

```
rows per image: mean 1.00  max 1
```

One annotation per image, so there is **no clustering** and the bootstrap resamples 4,000 independent
units. Unusually clean for Component 3, where every other task has 2&ndash;20 dependent rows per image.

It also means `panelConfusion` is undefined and reports `NaN` &mdash; expected, since there is no second
target to be confused with.

---

## Usage

```bash
python split_qa.py --task_root /path/to/Localization \
                   --id_root   /path/to/4-Instance-Level-Identification
python split_qa.py ... --tiers easy,medium,hard --n_chunks 8 --seed 1234 --force

python stats.py
python stats.py --run_dir /path/to/7.Localization
```

`--tiers` defaults to `easy`. `--force` is required to overwrite `chunks/`, because re-splitting would
orphan model outputs already written there.

### Reading order

```
split_qa.py  →  stats.py  →  [read the baseline table]  →  run_all.sh
```
