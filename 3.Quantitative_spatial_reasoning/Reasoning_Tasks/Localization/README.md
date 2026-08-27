<h1 align="center"> Bio-VQA &rarr; Quantitative spatial reasoning &rarr; Localization </h1>

<p align="center">
  <b>Bounding-Box Prediction &mdash; Candidate Selection, Inference &amp; Evaluation Pipeline</b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-blue">
  <img alt="Task" src="https://img.shields.io/badge/task-bounding%20box-orange">
  <img alt="Items" src="https://img.shields.io/badge/questions-4%2C000-green">
  <img alt="Models" src="https://img.shields.io/badge/models-18-purple">
  <img alt="Scheduler" src="https://img.shields.io/badge/scheduler-SLURM-red">
  <img alt="GPU" src="https://img.shields.io/badge/GPU--hours-~11.5-brightgreen">
</p>

---

## Overview

This module builds the **bounding-box prediction** task of **Bio-VQA Component 3**. Given a photograph
containing one annotated animal, a model returns four numbers delimiting the animal's extent. This is
the only C3 task with a **continuous, high-dimensional response** &mdash; every other task asks for a
letter, a word, or a scalar.

| Stage | Script | Responsibility |
| :---: | :--- | :--- |
| 1 | `1.Generate_QA/generateQA.py` | Select single-annotation candidates across four catalogues, emit boxes in pixels. |
| 2 | `2.Experiment_Runs/Preprocess/split_qa.py` | Read image dimensions, flag out-of-bounds gold, shard for SLURM. |
| 3 | `2.Experiment_Runs/<model>.py` | Inference; record raw string, resolved box **and** parse mode. |
| 4 | `3.Evaluate/evaluate.py` | IoU tiers, three baselines, model-level convention resolution &mdash; **results reported there**. |

> **Design contract:** a photograph enters the corpus **if and only if** it carries exactly one
> annotation. This makes *"the subject animal (zebra)"* a referring expression with a unique referent,
> which is a precondition for the item being scoreable at all.

---

## Table of Contents

- [Task Definition](#task-definition)
- [The Coordinate-Frame Problem](#the-coordinate-frame-problem)
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

Given image $I$ of dimensions $W \times H$ and a gold box
$b^{\ast} = (x, y, w, h)$ in pixels, the model returns $\hat{b}$. Agreement is
intersection-over-union:

$$
\mathrm{IoU}(b^{\ast}, \hat{b}) \;=\;
\frac{\bigl|\, b^{\ast} \cap \hat{b} \,\bigr|}{\bigl|\, b^{\ast} \cup \hat{b} \,\bigr|}
\;\in\; [0, 1]
$$

Because a predicted box never matches exactly, agreement is reported at several tolerances rather than
one threshold:

$$
\mathrm{acc}@\tau \;=\; \frac{1}{N}\sum_{i=1}^{N}
\mathbb{1}\bigl[\mathrm{IoU}(b_i^{\ast}, \hat{b}_i) \ge \tau\bigr],
\qquad \tau \in \{0.25,\, 0.50,\, 0.75,\, 0.90\}
$$

### Why the task exists

Every other C3 task can in principle be answered by a model that perceives the image only coarsely
&mdash; an ordinal comparison, a compass direction, a count. A bounding box demands that the model
commit to **where the object is and how far it extends**, in a coordinate frame it must infer from the
image itself. It is the most direct probe of spatial grounding in the component.

It also shares imagery with **Component 4 (individual re-identification)**: the four source catalogues
are the same, so image-matched correlation between localization and re-identification is available for
the same photograph.

---

## The Coordinate-Frame Problem

Bounding-box prediction has a failure mode absent from every other task: **models disagree about the
coordinate system**.

```
   ┌──────────────────────────────────────────────────────────────────┐
   │  Same animal. Same correct localisation. Five different strings. │
   └──────────────────────────────────────────────────────────────────┘

     abs_xywh        [1240,  88,  540,  410]     pixels, top-left + size
     abs_xyxy        [1240,  88, 1780,  498]     pixels, two corners
     norm1000_xywh   [ 517,  46,  225,  214]     0–1000 normalised
     norm1000_xyxy   [ 517,  46,  742,  260]     0–1000, corners
     norm01_xyxy     [0.517, 0.046, 0.742, 0.260]  0–1 fractions
```

A naive evaluation scores a perfectly localised box as **zero** because of its units, and thereby
measures format compliance rather than grounding.

The resolution has two levels:

| Level | Where | Signal |
| :--- | :--- | :--- |
| per reply | `2.Experiment_Runs` | ordered first-valid-wins, by magnitude |
| **per model** | `3.Evaluate` | **median box centre** across all replies |

Ground truth is **never** consulted at either level. Full treatment in
[`3.Evaluate/README.md`](3.Evaluate/README.md); the consequence of getting it wrong cost two models
0.6 IoU in an earlier version of the evaluation.

---

## Repository Layout

```
Localization/
├── README.md                                # you are here
├── 1.Generate_QA/
│   ├── README.md
│   └── generateQA.py                        # four catalogues, two loaders
├── 2.Experiment_Runs/
│   ├── README.md
│   ├── Preprocess/
│   │   ├── README.md
│   │   ├── split_qa.py                      # dimensions, sharding
│   │   └── stats.py                         # baselines, sizes, OOB audit
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
├── qa.csv                                   # 4,000 rows
├── chunks/
│   └── chunk00 … chunk05/
│       ├── qa.csv                           # ~667 rows each
│       └── qa_<MODEL_KEY>.csv
└── <RESULTS_DIR>/                           # merged + scored, not committed
```

**QA schema** (`qa.csv`):

| Column | Example | Notes |
| :--- | :--- | :--- |
| `image` | `0b49122e-….png` | basename |
| `image_path` | `../4-Instance-Level-Identification/GrevysZebra/Images/….png` | **relative to `--base_dir`**, points *outside* the task directory |
| `question` | `Provide bounding box coordinates for the subject animal (zebra) in this image.` | |
| `answer` | `[1760.0, 1762.0, 1083.0, 785.0]` | `[x, y, w, h]`, absolute pixels |
| `dataset` | `zebra` | grouping key |

> The `../` prefix is why `split_qa.py` accepts `--id_root`. Several candidate resolutions are tried
> and whatever is missing is reported at split time, on a login node.

---

## Prerequisites

```bash
python -m venv myenv && source myenv/bin/activate
pip install "torch>=2.6" torchvision accelerate transformers==4.57.1 \
            pandas pillow sentencepiece protobuf einops
```

| Package | Why |
| :--- | :--- |
| `torch>=2.6` | Gemma-3 raises `Using or_mask_function … require torch>=2.6` below this. |
| `sentencepiece` | LLaVA-NeXT Vicuna slow-tokenizer instantiation. |
| `protobuf` | The follow-on `ImportError` from the same path. |
| `einops` | InternVL remote code imports `rearrange` at module load. |
| `pillow` | Header-only dimension reads in Stage 2. |

**Hardware guidance.** Stage 1 is a directory scan &mdash; seconds, no images opened. Stage 3 needs one
H100-class GPU per array task; 30B&ndash;38B models need two.

---

## Data Source

Four wildlife re-identification catalogues, shared with Component 4:

| Dataset | Annotation format | Question phrasing |
| :--- | :--- | :--- |
| GrevysZebra | CSV, columns `bbox x/y/w/h` | `…the subject animal (zebra)…` |
| hyena | COCO JSON, `annotations[].bbox` | `…the subject animal (hyena)…` |
| leopard | COCO JSON | `…the subject animal (leopard)…` |
| whale shark | COCO JSON | `…the subject animal (whale shark)…` |

Two loaders are required because one catalogue ships a flat CSV and three ship COCO. Both produce the
same `(size, image_path, bbox, image_name)` tuple, so downstream handling is uniform.

Both formats store boxes as **`[x, y, width, height]` with `x, y` at the top-left corner, in absolute
pixels** &mdash; COCO's convention. No conversion is applied, so gold boxes share the frame the prompt
asks for.

---

## Corpus Statistics

| Property | Value |
| :--- | ---: |
| Questions | **4,000** |
| Distinct images | **4,000** (one annotation each) |
| Median canvas | 2400 &times; 1920 px |
| Gold box area, mean | **0.373** of the frame |
| Gold box area, median | 0.332 |
| Gold box area, min / max | 0.0046 / 0.9977 |
| Targets under 1% of canvas | 0.4% of rows |
| Out-of-bounds gold boxes | 12 |
| Rows per image | 1.00 (no clustering) |

### The property that shapes everything downstream

**The mean target occupies 37% of the frame.** This follows from the source material: wildlife
re-identification photographs are framed on the subject, because that is what they are collected for.

The direct consequence is that trivial baselines are high:

| Baseline | meanIoU | acc@.50 | acc@.75 | acc@.90 |
| :--- | ---: | ---: | ---: | ---: |
| whole image | 0.373 | 0.295 | **0.114** | **0.044** |
| centre 50% | 0.370 | 0.249 | 0.014 | 0.000 |
| **mean gold box** | **0.415** | **0.369** | 0.025 | 0.001 |

> ⚠️ **Mean IoU barely discriminates on this corpus.** Predicting the average box with no vision at all
> reaches 0.415. Report `acc@.75` and `acc@.90`, and note that the baselines **swap rank** between
> thresholds &mdash; use the maximum of the three at whichever threshold is quoted.

---

## Metric Definitions

| Metric | Definition |
| :--- | :--- |
| **`meanIoU_all`** | **headline.** Mean IoU with unparsed replies charged as zero |
| `meanIoU` | Mean IoU over parsed replies only |
| `acc25`/`acc50`/`acc75`/`acc90` | Share at or above that IoU |
| **`centreIn`** | Prediction **centre** inside the gold box |
| `gtCentreIn` | Gold centre inside the prediction |
| `parse_rate`, `mode` | Fraction resolvable, and the chosen convention |

$$
\text{meanIoU\_all} \;=\; \text{meanIoU} \times \text{parse\_rate}
$$

A model returning nothing parseable on half its items has not localised those items. `meanIoU` alone
would score it on the easier half &mdash; a factor-of-two difference for the worst-affected model here.

`centreIn` separates *knows where the animal is but cannot size the box* from *has no idea*, and is the
only column on which a pointing-trained model can be compared fairly against a box-trained one.

Full metric table, baselines and all results: **[`3.Evaluate/README.md`](3.Evaluate/README.md)**.

---

## Running on a SLURM Cluster

### Stage 1 &mdash; generation (login node, seconds)

```bash
python 1.Generate_QA/generateQA.py \
    --base_dir       <ROOT> \
    --grevys_csv     <...>  --grevys_images     <...> \
    --hyena_json     <...>  --hyena_images      <...> \
    --leopard_json   <...>  --leopard_images    <...> \
    --whaleshark_json <...> --whaleshark_images <...> \
    --output_dir     <TASK_DIR> \
    --num_samples    1000
```

`--num_samples` is **per catalogue**; four catalogues give 4,000 rows.

### Stage 2 &mdash; manifest

```bash
python 2.Experiment_Runs/Preprocess/split_qa.py \
       --task_root <TASK_DIR> --id_root <ID_ROOT>
python 2.Experiment_Runs/Preprocess/stats.py
```

### Stage 3 &mdash; inference (GPU array)

```bash
bash 2.Experiment_Runs/run_all.sh --dry --lanes 3
bash 2.Experiment_Runs/run_all.sh --lanes 3
```

| Resource | Value |
| :--- | ---: |
| Serial GPU-hours | **~11.5** |
| Wall clock, 6 chunks &times; 3 lanes | ~36 min |
| GPUs per task | 1 (2 for &ge;30B) |

**Cheapest task in Component 3.**

### Scheduler notes

- **`--nodes=1`, always.** Each runner is a single process.
- Disposable venv per array task in node-local `$TMPDIR`, removed by a `trap` on exit &mdash; including
  on failure, timeout and `scancel`.
- Relative `#SBATCH --output` paths require `cd` into the run directory before submission.

---

## Reproducibility

| Component | Deterministic? | Mechanism |
| :--- | :---: | :--- |
| Candidate selection (Stage 1) | ✅ | Sorted by file size descending, ties broken on path &mdash; **no seed required**. |
| Manifest sharding (Stage 2) | ✅ | `--seed 42`, applied after `row_id` assignment. |
| Inference (Stage 3) | ✅ | `do_sample=False` (greedy). |
| Convention resolution (Stage 4) | ✅ | Deterministic function of predictions + image dimensions. |
| Scoring (Stage 4) | ✅ | Pure function of the merged CSVs; bootstrap seeded. |

**Fully reproducible end to end** &mdash; the only C3 task for which this holds, because generation
involves no sampling.


---

<p align="center">
  <sub>Part of the <b>Bio-VQA</b> benchmark &middot; Component 3 &mdash; Quantitative Spatial Reasoning</sub>
</p>