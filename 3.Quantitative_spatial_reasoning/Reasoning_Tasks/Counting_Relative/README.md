<h1 align="center"> Bio-VQA &rarr; Quantitative spatial reasoning &rarr; Counting-Relative </h1>

<p align="center">
  <b>Ordinal Quantity Comparison &mdash; Grid Synthesis, Inference &amp; Evaluation Pipeline</b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-blue">
  <img alt="Task" src="https://img.shields.io/badge/task-ordinal%20comparison-orange">
  <img alt="Items" src="https://img.shields.io/badge/questions-42%2C412-green">
  <img alt="Models" src="https://img.shields.io/badge/models-18-purple">
  <img alt="Scheduler" src="https://img.shields.io/badge/scheduler-SLURM-red">
  <img alt="Parse" src="https://img.shields.io/badge/parse%20rate-1.000-brightgreen">
</p>

---

## Overview

This module builds the **ordinal quantity comparison** task of **Bio-VQA Component 3**. A model is
shown a 2&times;2 grid of four photographs and asked which quadrant holds the **most**, or the
**fewest**, animals. It never states a number: the task isolates *comparison* of quantity from
*estimation* of quantity, which the absolute Counting task measures separately.

| Stage | Script | Responsibility |
| :---: | :--- | :--- |
| 1 | `1.Generate_QA/generateQA_{same,diff}.py` | Sample four source photographs, enforce a unique extremum, composite the grid, emit both polarities. |
| 2 | `2.Experiment_Runs/Preprocess/split_qa.py` | Parse condition / polarity / gold letter, resolve paths, shard for SLURM. |
| 3 | `2.Experiment_Runs/<model>.py` | Inference across 18 VLMs under a byte-identical harness. |
| 4 | `3.Evaluate/evaluate.py` | Per-condition lift, coherence, position bias &mdash; **results reported there**. |

> **Design contract:** a grid exists **if and only if** its maximum and minimum counts are each
> attained by exactly one panel. This guarantees both polarities have a unique answer, and is what
> makes the ground-truth-free coherence check in Stage 4 valid.

---

## Table of Contents

- [Task Definition](#task-definition)
- [Why Two Conditions](#why-two-conditions)
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

Let a grid $G = \{P_A, P_B, P_C, P_D\}$ hold four panels with animal counts $n_A, n_B, n_C, n_D$.
Two questions are asked of every grid:

$$
y_{\text{most}}(G) = \arg\max_{q} n_q,
\qquad
y_{\text{least}}(G) = \arg\min_{q} n_q
$$

A grid is admitted to the corpus only when both extrema are unique:

$$
\bigl|\{q : n_q = \max_r n_r\}\bigr| = 1
\quad\wedge\quad
\bigl|\{q : n_q = \min_r n_r\}\bigr| = 1
$$

Because $y_{\text{most}}(G) \neq y_{\text{least}}(G)$ by construction, a model answering the same
letter to both is **provably wrong on at least one**, with no reference to the key. That is the basis
of the `incoherence` metric.

### Quadrant encoding

```
        ┌─────────────────┬─────────────────┐
        │                 │                 │
        │       A         │       B         │
        │   top-left      │   top-right     │
        │                 │                 │
        ├─────────────────┼─────────────────┤
        │                 │                 │
        │       C         │       D         │
        │  bottom-left    │  bottom-right   │
        │                 │                 │
        └─────────────────┴─────────────────┘
```

Fixed and never permuted. Restated in the prompt frame, so the mapping is never inferred.

---

## Why Two Conditions

A naive version of this task is trivially shortcut-able. If the four panels come from four different
datasets, then *which dataset a panel came from* predicts its count &mdash; aerial bee frames average
~37 animals, camel and kiang frames ~8. A model recognising "this is an insect plate" answers
correctly without comparing anything.

```
┌────────────────────────────────────────────────────────────────────────────┐
│  SAME  condition                                       22,412 items        │
│  ───────────────                                                           │
│    all four panels drawn from ONE dataset                                  │
│    question names the animal:  "…most number of bees…"                     │
│    ⇒ dataset identity carries NO information                               │
│    ⇒ floor = 0.250  (chance)                            THE MEASUREMENT    │
├────────────────────────────────────────────────────────────────────────────┤
│  DIFFERENT  condition                                  20,000 items        │
│  ────────────────────                                                      │
│    four panels drawn from FOUR datasets                                    │
│    question says "animals"                                                 │
│    ⇒ dataset identity PREDICTS count                                       │
│    ⇒ floor = 0.530 (most) / 0.548 (least)               THE PROBE          │
└────────────────────────────────────────────────────────────────────────────┘
```

`same` measures ordinal comparison. `different` asks whether a model *falls back* on domain
recognition when that shortcut exists. **Neither condition should be reported alone**; the contrast
between them is the result.

---

## Repository Layout

```
Counting_Relative/
├── README.md                                # you are here
├── 1.Generate_QA/
│   ├── README.md
│   ├── generateQA_same.py                   # four panels, one dataset
│   └── generateQA_diff.py                   # four panels, four datasets
├── 2.Experiment_Runs/
│   ├── README.md
│   ├── Preprocess/
│   │   ├── README.md
│   │   ├── split_qa.py                      # manifest assembly, sharding
│   │   └── stats.py                         # pre-run audit
│   └── <model>.py                           # ×18, identical outside MODEL BLOCK
└── 3.Evaluate/
    ├── README.md                            # ← metrics AND results
    ├── merge_results.py
    └── evaluate.py
```

No shell scripts are committed. All launch commands live in these READMEs.

---

## Output Layout

```
<TASK_DIR>/
├── Same/
│   ├── Images/
│   │   └── quad_<Dataset>_<hash>.jpg        # 2×2 composite, one dataset
│   └── qa.csv
├── Different/
│   ├── Images/
│   │   └── mixed_quad_<hash>.jpg            # 2×2 composite, four datasets
│   └── qa.csv
├── chunks/
│   └── chunk00 … chunk05/
│       ├── qa.csv                           # ~7,069 rows each
│       └── qa_<MODEL_KEY>.csv               # written by inference
└── <RESULTS_DIR>/                           # merged + scored, not committed
```

**QA schema** (`qa.csv`):

| Column | Description |
| :--- | :--- |
| `image` | Composite filename. |
| `image_path` | `Images/<file>` &mdash; relative to the condition directory. |
| `question` | Full question text including the option list. |
| `answer` | Verbatim gold, e.g. `Image C (bottom-left)`. |

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
| `sentencepiece` | LLaVA-NeXT Vicuna: `Cannot instantiate this tokenizer from a slow version`. |
| `protobuf` | The follow-on `ImportError` from the same tokenizer path. |
| `einops` | InternVL remote code imports `rearrange` at module load. |
| `pillow` | LANCZOS resampling during grid composition. |

**Hardware guidance.** Stage 1 is CPU- and I/O-bound (image compositing). Stages 3&ndash;4 need one
H100-class GPU per array task; the 30B&ndash;38B models need two.

---

## Data Source

Ten annotated wildlife and specimen corpora, each supplying per-image animal counts:

| Dataset | Modality | Mean count / frame |
| :--- | :--- | :---: |
| Bees | aerial hive frame | ~37 |
| Beetles-500 | specimen tray | ~15&ndash;22 |
| Fish | underwater / tank | ~15&ndash;22 |
| Sheep-500, Seal-500, Cattle-500 | aerial | ~15&ndash;22 |
| Zebra-500, GrevysZebra, Camel-500, Kiang-500 | aerial | ~8&ndash;9 |

Counts come from the same per-dataset annotations that drive the absolute Counting task, so the two
are commensurable by construction.

> The count spread across datasets is **herd size, not taxonomy** &mdash; it tracks how many
> individuals a frame typically contains, not what they are. This is precisely the confound the
> `different` condition is built to expose.

---

## Corpus Statistics

| Property | Value |
| :--- | ---: |
| Questions | **42,412** |
| Composite images | **20,000** |
| `same` items | 22,412 |
| `different` items | 20,000 |
| Grids with both polarities | 20,000 (100%) |
| Rows per image, mean / max | 2.12 / 6 |
| Answer-key balance | 0.245 &ndash; 0.255 across A/B/C/D |

### Why some grids carry six rows

GrevysZebra grids are asked about three animal terms &mdash; `animals`, `grevys zebras`, `zebras`
&mdash; giving $3 \times 2 = 6$ rows per image. Consequences:

- **Effective sample size is 20,000 images**, not 42,412 rows. Intervals bootstrap over images.
- The coherence check pairs on `(image, animal_term)`, not on `image` alone.
- GrevysZebra contributes 4,412 of the 22,412 `same` rows against 2,000 for every other dataset
  &mdash; hence `same_macroAcc`.

---

## Metric Definitions

Let $\hat{y}_i$ be the model's answer and $y_i$ the gold letter over $N$ items.

**Accuracy and lift.** Reported *per condition*, because the floors differ:

$$
\text{lift}_c = \frac{1}{N_c}\sum_{i \in c} \mathbb{1}[\hat{y}_i = y_i] - \phi_c,
\qquad
\phi_{\text{same}} = 0.250,\quad
\phi_{\text{diff}} \approx 0.539
$$

**Balanced accuracy.** Recall averaged over the four gold letters; a constant predictor scores exactly
$1/4$ whatever the key looks like:

$$
\text{balAcc} = \frac{1}{4}\sum_{\ell \in \{A,B,C,D\}}
\frac{\bigl|\{i : y_i = \ell \wedge \hat{y}_i = \ell\}\bigr|}{\bigl|\{i : y_i = \ell\}\bigr|}
$$

**Incoherence.** Requires no ground truth:

$$
\text{incoh} = \frac{1}{|\mathcal{P}|}
\sum_{(g,t) \in \mathcal{P}} \mathbb{1}\bigl[\hat{y}_{\text{most}}(g,t) = \hat{y}_{\text{least}}(g,t)\bigr]
$$

where $\mathcal{P}$ ranges over `(image, animal_term)` pairs.

**Position bias.** Total-variation distance between the model's answer distribution and the key's:

$$
\text{posBias} = \tfrac{1}{2}\sum_{\ell} \bigl| \Pr[\hat{y} = \ell] - \Pr[y = \ell] \bigr|
$$

Full metric table and all results: **[`3.Evaluate/README.md`](3.Evaluate/README.md)**.

---

## Running on a SLURM Cluster

### Stage 1 &mdash; generation (CPU)

```bash
#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=cr-generate
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/cr_gen_%j.out

PYTHON_BIN="<PATH>/myenv/bin/python"
BASE_DIR="<DATASETS>"
TASK_DIR="<TASK_DIR>"

$PYTHON_BIN 1.Generate_QA/generateQA_same.py \
    --base_dir "$BASE_DIR" --output_dir "$TASK_DIR/Same"
$PYTHON_BIN 1.Generate_QA/generateQA_diff.py \
    --base_dir "$BASE_DIR" --output_dir "$TASK_DIR/Different"
```

### Stage 2 &mdash; manifest (login node, seconds)

```bash
python 2.Experiment_Runs/Preprocess/split_qa.py --task_root "$TASK_DIR"
python 2.Experiment_Runs/Preprocess/stats.py
```

### Stage 3 &mdash; inference (GPU array)

```bash
bash 2.Experiment_Runs/run_all.sh --dry --lanes 3    # print, submit nothing
bash 2.Experiment_Runs/run_all.sh --lanes 3
```

| Resource | Value |
| :--- | ---: |
| Serial GPU-hours | **~189** |
| Wall clock, 6 chunks &times; 3 lanes | ~10.5 h |
| GPUs per task | 1 (2 for &ge;30B) |
| Array shape | `0-5` per model |

Largest sweep in Component 3.

### Scheduler notes

- **`--nodes=1`, always.** Each runner is a single process; a second node sits idle.
- Each array task builds a **disposable venv in node-local `$TMPDIR`**, removed by a `trap` on exit
  &mdash; including on failure, timeout and `scancel`. Package pins cannot collide between models.
- Relative `#SBATCH --output` paths require `cd` into the run directory before submission; the
  launcher does this and exports `C3_RUN_DIR`.


---

<p align="center">
  <sub>Part of the <b>Bio-VQA</b> benchmark &middot; Component 3 &mdash; Quantitative Spatial Reasoning</sub>
</p>