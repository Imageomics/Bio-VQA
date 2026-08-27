<h1 align="center"> Bio-VQA &rarr; Quantitative spatial reasoning &rarr; Relative-Position </h1>

<p align="center">
  <b>Directional Judgement &mdash; Six-Box Construction, Inference &amp; Evaluation Pipeline</b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-blue">
  <img alt="Task" src="https://img.shields.io/badge/task-8--way%20direction-orange">
  <img alt="Items" src="https://img.shields.io/badge/questions-1%2C000-green">
  <img alt="Models" src="https://img.shields.io/badge/models-18-purple">
  <img alt="Key" src="https://img.shields.io/badge/key-uniform%20125%2Fclass-brightgreen">
  <img alt="Chance" src="https://img.shields.io/badge/chance-0.125-red">
</p>

---

## Overview

This module builds the **directional judgement** task of **Bio-VQA Component 3**. A model is shown a
beetle plate on which two specimens are outlined &mdash; one red, one blue &mdash; and must name the
direction of the red specimen relative to the blue one on an eight-point compass.

| Stage | Script | Responsibility |
| :---: | :--- | :--- |
| 1 | `1.Generate_QA/generateQA.py` | Solve the six-box composition, label the direction, render, emit QA. |
| 2 | `2.Experiment_Runs/Preprocess/split_qa.py` | Recompute the mis-pairings, **verify the direction convention**, shard. |
| 3 | `2.Experiment_Runs/<model>.py` | Inference across 18 VLMs under a byte-identical harness. |
| 4 | `3.Evaluate/evaluate.py` | Lift, **error attribution**, angular error &mdash; **results reported there**. |

> **Design contract:** every emitted item carries **exactly six boxes**, and the two empty decoys are
> *solved* so that each of the three incorrect referent pairings yields a **distinct** direction. Pairs
> that cannot satisfy this are rejected rather than emitted degraded.

---

## Table of Contents

- [Task Definition](#task-definition)
- [Why Six Boxes](#why-six-boxes)
- [The Mis-Pairing Structure](#the-mis-pairing-structure)
- [Repository Layout](#repository-layout)
- [Output Layout](#output-layout)
- [Prerequisites](#prerequisites)
- [Data Source](#data-source)
- [Corpus Statistics](#corpus-statistics)
- [Metric Definitions](#metric-definitions)
- [Running on a SLURM Cluster](#running-on-a-slurm-cluster)
- [Reproducibility](#reproducibility)
- [FAQ](#faq)
- [Citation](#citation)

---

## Task Definition

Given two box centres $c_r$ (red target) and $c_b$ (blue target) in image coordinates with $y$
increasing downward:

$$
\theta \;=\; \operatorname{atan2}\bigl(y_r - y_b,\; x_r - x_b\bigr)
$$

$\theta$ is binned into eight 45&deg; octants:

```
                        top
                         │
          top-left   ╲   │   ╱   top-right
                      ╲  │  ╱
                       ╲ │ ╱
        left ───────────( ● )─────────── right      ● = blue specimen
                       ╱ │ ╲                            (the reference)
                      ╱  │  ╲
       bottom-left   ╱   │   ╲   bottom-right
                         │
                       bottom
```

| Direction | Angle range |
| :--- | :--- |
| `right` | &minus;22.5&deg; … 22.5&deg; |
| `bottom-right` | 22.5&deg; … 67.5&deg; |
| `bottom` | 67.5&deg; … 112.5&deg; |
| `bottom-left` | 112.5&deg; … 157.5&deg; |
| `left` | 157.5&deg; … 202.5&deg; |
| `top-left` | &minus;157.5&deg; … &minus;112.5&deg; |
| `top` | &minus;112.5&deg; … &minus;67.5&deg; |
| `top-right` | &minus;67.5&deg; … &minus;22.5&deg; |

### Quality filters

| Filter | Default | Purpose |
| :--- | ---: | :--- |
| `--margin_deg` | 8.0 | Reject pairs within 8&deg; of an octant boundary &mdash; an item at 22.4&deg; would flip between `right` and `bottom-right` on a hair. |
| `--min_sep` | 0.60 | Centre distance must exceed 0.6&times; the summed half-diagonals, so specimens are clearly apart. |
| `--max_iou` | 0.05 | Target boxes may not meaningfully overlap. |

> These filters draw the corpus from the **easier tail** of the pair distribution. That is a deliberate
> trade: an ambiguous ground truth is worse than a restricted sample, but the restriction is stated.

---

## Why Six Boxes

The obvious version of this task is trivially shortcut-able. *"Where is the specimen in the red box
relative to the specimen in the blue box?"* sounds like a question about two beetles. **With only two
boxes drawn, it is not** &mdash; a model can find two coloured rectangles, compare their centres, and
answer correctly without ever determining whether either box contains an animal.

```
   ┌────────────────────────────────────────────────────────────────────┐
   │   red   ×2     one on the TARGET specimen, one on EMPTY SPACE      │
   │   blue  ×2     one on the TARGET specimen, one on EMPTY SPACE      │
   │   green ×1     on a NON-TARGET specimen                            │
   │   cyan  ×1     on a NON-TARGET specimen                            │
   └────────────────────────────────────────────────────────────────────┘
```

| Decoy | Shortcut it closes |
| :--- | :--- |
| **empty red + empty blue** | *"find the red rectangle"* becomes insufficient &mdash; it must be the red rectangle **with a specimen in it**. |
| **green + cyan on real specimens** | *"which box contains an animal"* becomes uninformative &mdash; box occupancy alone does not identify the target. |

All six boxes are **size-matched**: the largest-to-smallest area ratio is bounded by
`--max_size_ratio` (default 2.0), and empty boxes are sized from the median annotated box in that
image. Without this, an empty box would be identifiable by its size alone.

---

## The Mis-Pairing Structure

Because the decoys are **placed deliberately rather than sampled**, each wrong box pairing produces a
specific, distinct direction:

| Pairing | Meaning if a model answers it |
| :--- | :--- |
| `red_target → blue_target` | **correct** |
| `red_empty → blue_target` | did not check whether the red box held a specimen |
| `red_target → blue_empty` | did not check the blue box |
| `red_empty → blue_empty` | matched colour only, ignored content |

```python
def solve_distractors(pool, rt, bt, ans, margin_deg, rng, tries=600):
    """Pick an empty red box and an empty blue box so that all three incorrect
    referent pairings produce distinct wrong directions."""
```

> **Consequence: every error is attributable to a specific confusion**, rather than being generic. This
> is what converts wrong answers into diagnostic information, at no additional annotation cost &mdash;
> only search time during generation.

`Preprocess/split_qa.py` recomputes all three from the stored box coordinates and verifies they
reproduce the generator's own `wrong_pairings` field **exactly, in order**. It also recomputes the gold
direction and **aborts loudly** if it disagrees &mdash; the one way the direction convention could
silently be 180&deg; wrong.

---

## Repository Layout

```
Relative_Position/
├── README.md                                # you are here
├── 1.Generate_QA/
│   ├── README.md
│   └── generateQA.py                        # box solver, direction labeller, renderer
├── 2.Experiment_Runs/
│   ├── README.md
│   ├── Preprocess/
│   │   ├── README.md
│   │   ├── split_qa.py                      # mis-pairings + geometry check
│   │   └── stats.py                         # key balance, clustering
│   └── <model>.py                           # ×18, identical outside MODEL BLOCK
└── 3.Evaluate/
    ├── README.md                            # ← metrics AND results
    ├── merge_results.py
    └── evaluate.py
```

---

## Output Layout

```
<TASK_DIR>/
├── Images/
│   └── Beetles-500_<plate>_pair<N>_dist.jpg   # rendered, six boxes drawn
├── qa.csv                                     # 1,000 rows
├── chunks/
│   └── chunk00 … chunk05/
│       ├── qa.csv                             # ~167 rows each
│       └── qa_<MODEL_KEY>.csv
└── <RESULTS_DIR>/                             # merged + scored, not committed
```

**QA schema** &mdash; the `boxes` field is what makes error attribution possible:

| Column | Example | Notes |
| :--- | :--- | :--- |
| `image` | `Beetles-500_A00000046073_pair0_dist.jpg` | rendered file |
| `pair_id` | `Beetles-500_A00000046073_pair0` | groups conditions |
| `source_image` | `A00000046073.jpg` | **bootstrap cluster key** |
| `answer` | `left` | gold direction |
| `n_boxes` | `6` | |
| `size_ratio` | `1.591` | largest/smallest box area |
| `boxes` | JSON | all six boxes **plus `wrong_pairings`** |

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
order**: `red_empty→blue_target`, `red_target→blue_empty`, `red_empty→blue_empty`.

---

## Prerequisites

```bash
python -m venv myenv && source myenv/bin/activate
pip install "torch>=2.6" torchvision accelerate transformers==4.57.1 \
            pandas pillow sentencepiece protobuf einops
```

**Hardware guidance.** Stage 1 is CPU-only (box solving and rendering). Stage 3 needs one H100-class
GPU per array task; 30B&ndash;38B models need two. The corpus is 1,000 items, so this is the **cheapest
sweep in Component 3** at ~3.2 GPU-hours.

---

## Data Source

**Beetles-500 only**, deliberately. Quoted from the generator's docstring:

> Controlled specimen-tray setting: constant working distance and near-uniform specimen scale, so that
> referent identification and direction labelling are both well posed. Restricted to a single dataset
> deliberately &mdash; in field imagery (drone/telephoto wildlife) targets are frequently a few dozen
> pixels across and pairs mix a foreground subject with a background speck, which makes both the "which
> box holds an animal" step and the centre-to-centre direction label indefensible.

On an aerial zebra frame a target may be 30 px across. Asking whether a 30-px box "contains an animal"
is not meaningful at the resolutions models receive, and the centre of a barely-resolved blob is not a
stable reference point. The specimen tray removes both problems.

> ⚠️ **Scope limitation.** The task measures directional judgement on museum specimen imagery.
> Generalisation to field photography is untested.

---

## Corpus Statistics

| Property | Value |
| :--- | ---: |
| Questions | **1,000** |
| Source photographs | **208** |
| Pairs per photograph, mean / max | 4.81 / 5 |
| Dataset | Beetles-500 |
| Condition | `distractor` |
| Boxes per image | 6 |
| Answer key | **125 per direction, exactly** |
| Chance = majority floor | **0.125** |

### The key is uniform by construction

Exactly 125 items per direction. **Chance and the majority baseline coincide at 0.125**, so
`lift = acc − 0.125` needs no skew correction &mdash; unusual and convenient. Unlike the viewpoint
task, where `left` and `right` account for ~83% of gold labels, this key carries no directional bias.

### Effective sample size is 208, not 1,000

The generator's fill passes pushed most plates to the `--pairs_per_image_cap` of 5. Pairs from one
plate share specimens, lighting and scale, so `evaluate.py` bootstraps over `source_image`. At
$n = 208$ the intervals separate *"reads the layout"* from *"does not"*, but cannot rank adjacent
models against each other.

---

## Metric Definitions

| Metric | Definition |
| :--- | :--- |
| **`lift`** | **headline.** Accuracy minus the majority-direction floor (0.125) |
| `balAcc` | Recall averaged over the gold directions |
| `kappa` | Accuracy corrected for both marginals |
| `angErr`, `within45` | Mean angular error on the 45&deg; ring, share within one step |
| **`errRedEmptyX`** etc. | **Chance-corrected** error attribution |
| `posBias`, `predTopShare` | Answer-distribution diagnostics |

Angular error treats a one-octant miss as different from an opposite answer:

$$
\Delta_i = \min\bigl(|\alpha(y_i) - \alpha(\hat{y}_i)|,\; 360 - |\alpha(y_i) - \alpha(\hat{y}_i)|\bigr),
\qquad
\text{within45} = \frac{1}{N}\sum_i \mathbb{1}[\Delta_i \le 45^\circ]
$$

Error attribution is chance-corrected using the Cohen's-$\kappa$ agreement term, because a random
answer lands on a mis-pairing direction about one time in eight:

$$
\text{errX} \;=\; \underbrace{\Pr[\hat{y} = m]}_{\text{observed}}
\;-\; \underbrace{\sum_{d} \Pr[\hat{y} = d]\,\Pr[m = d]}_{\text{expected}}
$$

Full metric table and all results: **[`3.Evaluate/README.md`](3.Evaluate/README.md)**.

---

## Running on a SLURM Cluster

### Stage 1 &mdash; generation (CPU)

```bash
python 1.Generate_QA/generateQA.py \
    --base_dir <DATASETS> --output_dir <TASK_DIR> \
    --dataset Beetles-500 --seed 1234 \
    --target_pairs 1000 --pairs_per_image 2 --pairs_per_image_cap 5 \
    --max_size_ratio 2.0 --size_jitter 0.10 \
    --min_sep 0.60 --margin_deg 8.0 \
    --max_iou 0.05 --free_margin 0.015
```

### Stage 2 &mdash; manifest

```bash
python 2.Experiment_Runs/Preprocess/split_qa.py --task_root <TASK_DIR>
python 2.Experiment_Runs/Preprocess/stats.py
```

> ⚠️ **The geometry check must report 0 mismatches.** If it does not, the direction convention in the
> analysis does not match the generator's, and every table downstream would be wrong in a way that
> looks entirely plausible.

### Stage 3 &mdash; inference (GPU array)

```bash
bash 2.Experiment_Runs/run_all.sh --dry --lanes 3
bash 2.Experiment_Runs/run_all.sh --lanes 3
```

| Resource | Value |
| :--- | ---: |
| Serial GPU-hours | **~3.2** |
| GPUs per task | 1 (2 for &ge;30B) |

**Cheapest sweep in Component 3.**

---

## Reproducibility

| Component | Deterministic? | Mechanism |
| :--- | :---: | :--- |
| Box solving (Stage 1) | ✅ | `--seed 1234` honoured throughout &mdash; selection, free-pool sampling, distractor solver. |
| Format calibration (Stage 1) | ✅ | `calibrate()` detects the bounding-box convention from a sample rather than assuming column names. |
| Manifest sharding (Stage 2) | ✅ | `--seed 42`, applied after `row_id` assignment. |
| Inference (Stage 3) | ✅ | `do_sample=False` (greedy). |
| Scoring (Stage 4) | ✅ | Pure function of the merged CSVs; bootstrap seeded. |

**Fully reproducible end to end.**

---

<p align="center">
  <sub>Part of the <b>Bio-VQA</b> benchmark &middot; Component 3 &mdash; Quantitative Spatial Reasoning</sub>
</p>