<h1 align="center">Bio-VQA &rarr; Spatial Visual Reasoning &rarr; Optical Character Recognition</h1>
<p align="center">
  <b>Accession-Code Transcription from Specimen Barcode Labels</b>
</p>
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="Models" src="https://img.shields.io/badge/models-18%20open--weight%20VLMs-orange">
  <img alt="Instances" src="https://img.shields.io/badge/instances-522%20per%20model-green">
  <img alt="Corpus" src="https://img.shields.io/badge/corpus-2018--NEON--Beetles-lightgrey">
  <img alt="Headline" src="https://img.shields.io/badge/headline-EM%20%2B%20suffixCharAcc-yellow">
  <img alt="Floor" src="https://img.shields.io/badge/constant%20charAcc-0.675-red">
  <img alt="Resolution" src="https://img.shields.io/badge/vision%20budget-near--native-purple">
  <img alt="Status" src="https://img.shields.io/badge/status-complete-brightgreen">
</p>

> An evaluation of small-text transcription in vision–language models. Given a specimen tray
> photograph with the barcode region outlined, the model transcribes the twelve-character
> accession code printed beneath it. This document describes the task construction, why a
> constant prefix makes character accuracy misleading, the per-position diagnostic that
> exposes it, the empirical findings, and the exact commands to reproduce them.

---

## Table of contents

1. [Overview](#1-overview)
2. [Motivation](#2-motivation)
   - 2.1 [Transcription as a spatial-acuity task](#21-transcription-as-a-spatial-acuity-task)
   - 2.2 [The constant prefix and why character accuracy misleads](#22-the-constant-prefix-and-why-character-accuracy-misleads)
   - 2.3 [Exact match as the operational metric](#23-exact-match-as-the-operational-metric)
3. [System architecture](#3-system-architecture)
   - 3.1 [Pipeline stages and dependency order](#31-pipeline-stages-and-dependency-order)
   - 3.2 [Repository layout](#32-repository-layout)
4. [The evaluation set](#4-the-evaluation-set)
   - 4.1 [Task definition](#41-task-definition)
   - 4.2 [Ground truth is not a model read](#42-ground-truth-is-not-a-model-read)
   - 4.3 [Scale and code structure](#43-scale-and-code-structure)
5. [The constant-predictor floor](#5-the-constant-predictor-floor)
   - 5.1 [The modal-string predictor](#51-the-modal-string-predictor)
   - 5.2 [The floors it establishes](#52-the-floors-it-establishes)
6. [Results](#6-results)
   - 6.1 [The leaderboard](#61-the-leaderboard)
   - 6.2 [Seven models fall below the constant predictor](#62-seven-models-fall-below-the-constant-predictor)
   - 6.3 [The prefix–suffix cliff](#63-the-prefixsuffix-cliff)
   - 6.4 [Format compliance dissociates from reading](#64-format-compliance-dissociates-from-reading)
   - 6.5 [Scale is irrelevant; family is decisive](#65-scale-is-irrelevant-family-is-decisive)
   - 6.6 [Metric agreement](#66-metric-agreement)
7. [Scope of the released evaluation](#7-scope-of-the-released-evaluation)
8. [Reproduction](#8-reproduction)
9. [Outputs](#9-outputs)
10. [References](#10-references)

---

## 1. Overview

We evaluate whether vision–language models can transcribe small printed text from museum
specimen imagery. Each item presents a NEON beetle tray photograph in which the barcode
region is outlined in red; the model transcribes the twelve-character accession code printed
beneath the barcode — the letter `A` followed by eleven digits. We pose 522 items and
evaluate **eighteen open-weight vision–language models**.

The task is the acuity floor of Component C3. It requires no counting, no comparison, no
orientation reasoning, and no measurement: the answer is printed in the image and the
question is whether the model can resolve the glyphs. Every other task in the component
depends on the model resolving fine detail; this one isolates that capability with nothing
else attached.

One property of the corpus determines the whole scoring design. **Every code begins with the
same fixed prefix**, six of twelve characters, and a seventh position is near-constant. A
model that emits one fixed string and never looks at an image therefore reaches a character
accuracy of **0.675** while reading nothing
([Section 5](#5-the-constant-predictor-floor)). Character accuracy — the standard OCR metric
— is therefore inflated by construction on this corpus, and we lead with exact match and
with per-character accuracy restricted to the variable suffix.

Three findings organise the component. **Seven of eighteen models score below the
constant-string predictor** on character accuracy, and five fall below it even on the
variable suffix ([Section 6.2](#62-seven-models-fall-below-the-constant-predictor)). Every
model shows a sharp discontinuity exactly at the prefix boundary — mean accuracy **0.992
across the constant positions and 0.568 across the variable ones**, with no left-to-right
decay inside the suffix, so this is a prefix-reproduction effect rather than sequence-length
degradation ([Section 6.3](#63-the-prefixsuffix-cliff)). And performance is essentially
**uncorrelated with parameter count** ($\rho = 0.045$) while being almost perfectly separated
by model family ([Section 6.5](#65-scale-is-irrelevant-family-is-decisive)).

---

## 2. Motivation

### 2.1 Transcription as a spatial-acuity task

Counting, morphometry, and viewpoint each combine acuity with a second operation —
enumeration, ratio formation, orientation inference. Transcription combines it with nothing.
The characters are present in the pixels at a known location, the format is stated, and the
answer requires no inference beyond glyph identification.

That makes the task a **lower bound on fine detail resolution** for the whole component. A
model that cannot read twelve printed characters from a marked region cannot be relied upon
to resolve a 5% difference in elytra length
([Relative Size](../Distance_Relative/README.md)) or a 12-pixel line against a 41-pixel scale
bar ([Absolute Distance](../Distance_Absolute/README.md)). Conversely, a model that fails
those tasks while succeeding here has failed at the reasoning operation rather than at
perception — a dissociation this task makes available.

It is also the task where the vision-token budget matters most directly. Downsampling a
label crop is the one operation guaranteed to convert a reading task into a guessing task, so
the budget is raised to approximately native resolution for every model
([Section 6.4 of the inference document](2.Experiment_Runs/README.md#64-vision-token-budget)).

### 2.2 The constant prefix and why character accuracy misleads

The NEON accession scheme assigns codes of the form `A00000NNNNNN`, so within this corpus a
long leading substring is identical on every label. Character accuracy — the standard OCR
metric — awards credit for every position matched, including the six that never vary.

The consequence is concrete and large. A predictor that emits the per-position modal
character and never looks at an image reaches

$$
\mathrm{charAcc}_{\text{const}} = 0.675, \qquad \mathrm{CER}_{\text{const}} = 0.322 ,
$$

or **8.10 of 12 characters correct**, without reading. Any evaluation reporting character
accuracy against zero rather than against this floor would credit prefix reproduction as
transcription. We therefore compute the floor from the data
([Section 5](#5-the-constant-predictor-floor)), report it in every results row, and report
per-character accuracy restricted to the variable suffix as the continuous headline.

### 2.3 Exact match as the operational metric

For most tasks in this benchmark exact match is too blunt to be a headline. Here it is
exactly right, for a domain reason: **a transcription with one wrong digit is the wrong
specimen.** An accession code is a database key. A collection manager who reads
`A00000032929` as `A00000032939` has not made a small error; they have retrieved a different
beetle. There is no partial credit in the operational use of this output.

Exact match is therefore the primary metric, with `suffixCharAcc` as the continuous companion
that says *how close* a failing model came, and `CER` as the edit-distance measure that —
unlike positional character accuracy — tolerates insertions and deletions. The three agree
closely in ranking ([Section 6.6](#66-metric-agreement)), which means the choice of headline
is not load-bearing for the conclusions.

---

## 3. System architecture

### 3.1 Pipeline stages and dependency order

```
        ┌──────────────────────────────────────────────┐
        │  2018-NEON-Beetles tray photographs          │
        │  filename stem = NEON specimen accession ID  │
        └──────────────────┬───────────────────────────┘
                           │
                           ▼
   (1a) 1.Generate_QA/generate_ocr.py
                           │   Moondream detect("Barcode") — LOCALISATION ONLY
                           │   draw red box, save <stem>.jpg
                           │   sharded across array tasks, resumable
                           ▼
                       Barcodes/  +  Barcodes_task_<id>.csv
                           │   merged to Barcodes.csv
                           ▼
   (1b) 1.Generate_QA/generate_QA.py
                           │   ground truth = filename stem, NOT a model read
                           ▼
                       qa.csv
                           │
                           ▼
   (2)  2.Experiment_Runs/Preprocess/split_qa.py
                           │   parse dict-repr cells, normalise codes,
                           │   compute the common prefix, resolve paths, shard
                           ▼
                       chunks/chunk00..05/qa.csv
                           │
                           ├──►  stats.py   informative-character count + floors
                           ▼
   (3)  2.Experiment_Runs/<model>.py  x 18
                           │   near-native vision budget, invariant harness
                           ▼
                       chunks/chunkNN/qa_<KEY>.csv  (+ summary.json)
                           │
                           ▼  merge_results.py  ->  results/qa_<KEY>.csv
                           ▼
   (4)  3.Evaluate/evaluate.py  ──►  ocr_overall.csv
                                     ocr_by_position.csv
                                     ocr_report.md
```

Stage 1 splits into two scripts because localisation and QA construction have different
failure modes and different costs. `generate_ocr.py` calls an external detection API and is
network-bound, rate-limited, and resumable; `generate_QA.py` is a pure local transformation
over the resulting table. Separating them means a rate-limit failure never corrupts the QA
manifest, and the manifest can be rebuilt without re-detecting.

### 3.2 Repository layout

```
OCR/
├── README.md                      this document
│
├── 1.Generate_QA/
│   ├── README.md
│   ├── generate_ocr.py            barcode localisation + red-box rendering
│   └── generate_QA.py             QA manifest from the detection table
│
├── 2.Experiment_Runs/
│   ├── README.md
│   ├── Preprocess/
│   │   ├── split_qa.py            parse, normalise, compute prefix, shard
│   │   └── stats.py               informative-character audit + floors
│   └── <model>.py  × 18           inference runners, invariant harness
│
└── 3.Evaluate/
    ├── README.md
    ├── merge_results.py           reassemble shards into per-model CSVs
    └── evaluate.py                metrics, constant floor, per-position table
```

| # | Stage | Document |
|---|---|---|
| 1 | Localise barcodes and build the QA manifest | [`1.Generate_QA/README.md`](1.Generate_QA/README.md) |
| 2–3 | Parse, shard, and run inference across 18 models | [`2.Experiment_Runs/README.md`](2.Experiment_Runs/README.md) |
| 4 | Merge, score against the constant floor, and report | [`3.Evaluate/README.md`](3.Evaluate/README.md) |

---

## 4. The evaluation set

### 4.1 Task definition

Each item is a NEON beetle tray photograph on which the barcode region has been outlined with
a **red rectangle**. The question is fixed:

> *What is the text under the barcode (starting with A0...) shown in this image?*

and the answer is the twelve-character accession code printed beneath the barcode.

The red box is a **localisation aid, not the answer**. A tray photograph contains dozens of
pinned specimens, printed labels, a scale bar, and often several small text regions; without
the box the model would have to first decide which text to read, which is a different task.
Outlining the barcode region isolates transcription from region selection.

Note that the box is drawn around the **bars**, not the printed characters — the detector
targets the barcode symbol itself, and the code is printed beneath it. The generator exposes
a `--box_pad_frac` argument that would enlarge the box to enclose the text, left at zero for
the released set ([Section 4 of the generation document](1.Generate_QA/README.md#4-barcode-localisation)).
The box therefore points at the region without circling the answer, which is the intended
degree of assistance.

### 4.2 Ground truth is not a model read

This point is essential to the task's validity, because an external vision model appears in
the pipeline.

**Moondream is used for localisation only.** It is asked to detect a barcode and its output
is a bounding box that gets drawn onto the image. It is never asked to read anything, and its
output never enters the answer.

**The gold answer is the NEON specimen accession ID**, which is the source photograph's
filename stem — a curatorial record assigned by the collection, not a transcription produced
by any model. Ground truth is therefore exact by construction and independent of every model
in the pipeline, including the detector.

One consequence follows directly and is recorded in the code as a warning to future work:
because the answer equals the filename stem, **any extension of this pipeline must never
place the filename in the prompt.** Doing so would leak the answer completely. The released
runners pass only the resolved image path to the loader and the question text to the model,
never the basename.

### 4.3 Scale and code structure

| Quantity | Value |
|---|---:|
| Scored items per model | 522 |
| Code length | 12 characters |
| Format | `A` + 11 digits |
| Common prefix (all codes agree) | **6 characters** |
| Effectively constant positions | **7** (position 6 is near-constant) |
| Variable suffix | 5–6 characters |
| Clustering | **none** — one label per specimen |

This is the smallest evaluation set in Component C3 and the only one with **no cluster
structure**. Every label belongs to a distinct specimen with its own accession number, so
rows are independent observations and the bootstrap resamples rows rather than plates or
canvases ([Section 7 of the evaluation document](3.Evaluate/README.md#7-uncertainty)). The
plate-based tasks elsewhere in the component all require clustered resampling; this one does
not, and stating that explicitly prevents an unnecessary correction being applied by analogy.

---

## 5. The constant-predictor floor

### 5.1 The modal-string predictor

The floor is not the common prefix. It is the **per-position modal character** over the
corpus:

$$
\hat{c}_j = \mathrm{arg\,max}_{c} \; \bigl|\{ i : y_i[j] = c \}\bigr|,
\qquad
\hat{y}_{\text{const}} = \hat{c}_0 \hat{c}_1 \cdots \hat{c}_{11} ,
$$

which is the constant string minimising character error over this corpus — the strongest
predictor available to a model that never looks at an image.

**This is strictly stronger than the common-prefix predictor, and the difference is
measurable.** `os.path.commonprefix` requires *every* code to agree and reports 6 characters.
The modal string additionally captures position 6, where one character dominates the corpus
without being universal. The arithmetic confirms it:

$$
\mathrm{charAcc}_{\text{const}} \times 12 = 0.675 \times 12 = 8.10 \text{ characters},
$$

of which 6.00 come from the universal prefix and **2.10 from the variable region** — most of
that from position 6 alone. Using the common prefix as the floor would understate it by
roughly one full character, and would place two additional models above the line.

### 5.2 The floors it establishes

Computed on the identical rows as each model:

| Metric | Constant predictor | What it means |
|---|---:|---|
| `constEM` | **0.002** | Exact match is essentially free of the exploit — one code in 522 happens to equal the modal string |
| `constCharAcc` | **0.675** | 8.1 of 12 characters, without reading |
| `constSuffixCharAcc` | **0.349** | 2.1 of 6 suffix characters, without reading |
| `constCER` | **0.322** | Edit-distance floor |

The contrast between `constEM` = 0.002 and `constCharAcc` = 0.675 is the entire argument for
the metric choice in [Section 2.3](#23-exact-match-as-the-operational-metric). Exact match is
**immune** to the prefix exploit because a single wrong character costs the whole item;
character accuracy is heavily exposed to it. This is the rare case where the blunt metric is
also the robust one, and where the domain requirement and the statistical requirement point
the same way.

---

## 6. Results

Full tables are written to `3.Evaluate/results/ocr_report.md`.

### 6.1 The leaderboard

Sorted by exact match. `constCharAcc` = 0.675 and `constSuffixCharAcc` = 0.349 for every row.

| Model | EM | suffixEM | suffixCharAcc | charAcc | CER | wellformed | parse |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3-VL-30B-A3B** | **0.979** | 0.979 | **0.987** | **0.993** | **0.002** | 0.985 | 1.000 |
| **Qwen2.5-VL-7B** | 0.916 | 0.918 | 0.940 | 0.970 | 0.008 | 0.923 | 1.000 |
| **Qwen2.5-VL-3B** | 0.904 | 0.908 | 0.940 | 0.969 | 0.009 | 0.929 | 1.000 |
| Qwen3-VL-32B | 0.900 | 0.900 | 0.927 | 0.964 | 0.009 | 0.918 | 1.000 |
| Llama-3.2-11B-Vision | 0.722 | 0.722 | 0.798 | 0.898 | 0.029 | 0.789 | 1.000 |
| Qwen3-VL-8B | 0.711 | 0.711 | 0.735 | 0.867 | 0.025 | 0.715 | 1.000 |
| Molmo2-8B | 0.692 | 0.692 | 0.809 | 0.905 | 0.040 | 0.772 | 1.000 |
| InternVL3.5-8B-Instruct | 0.460 | 0.462 | 0.655 | 0.825 | 0.092 | 0.636 | 1.000 |
| Gemma-3-12B | 0.352 | 0.352 | 0.430 | 0.714 | 0.062 | 0.393 | 1.000 |
| Gemma-4-12B | 0.345 | 0.347 | 0.648 | 0.821 | 0.112 | 0.676 | 1.000 |
| InternVL3-8B-Instruct | 0.314 | 0.314 | 0.574 | 0.784 | 0.092 | 0.584 | 1.000 |
| InternVL3.5-38B-Instruct | 0.192 | 0.192 | 0.360 | **0.670** | 0.111 | 0.414 | 1.000 |
| InternVL3.5-30B-A3B | 0.188 | 0.190 | 0.366 | **0.656** | 0.133 | 0.400 | 1.000 |
| InternVL3-38B-Instruct | 0.105 | 0.105 | **0.246** | **0.620** | 0.103 | 0.278 | 1.000 |
| Pixtral-12B | 0.065 | 0.065 | **0.199** | **0.593** | 0.144 | 0.217 | 0.996 |
| Phi-3.5-Vision | 0.044 | 0.044 | **0.161** | **0.568** | 0.147 | 0.201 | 1.000 |
| LLaVA-NeXT-Vicuna-7B | **0.000** | 0.000 | **0.214** | **0.607** | **0.383** | **1.000** | 1.000 |
| LLaVA-NeXT-Mistral-7B | **0.000** | 0.000 | **0.228** | **0.614** | **0.466** | 0.010 | 1.000 |

Bold values in the accuracy columns sit **below the constant-predictor floor**. Parse rate is
1.000 for seventeen of eighteen models, so every model is scored on essentially the same rows.

The span is the widest of any task in Component C3: from 0.979 to 0.000 on exact match. Two
models transcribe almost perfectly; two never produce a single correct code across 522
attempts.

### 6.2 Seven models fall below the constant predictor

$$
\mathrm{charAcc} < 0.675 \text{ for } \mathbf{7/18} \text{ models};
\qquad
\mathrm{suffixCharAcc} < 0.349 \text{ for } \mathbf{5/18};
\qquad
\mathrm{CER} > 0.322 \text{ for } 2/18 .
$$

Seven models — including both 38B InternVL variants and the 30B InternVL3.5-A3B — achieve a
character accuracy **lower than a fixed string emitted without looking at any image**. On the
variable suffix, where the constant predictor's advantage is smallest, five models remain
below it.

This is not a marginal shortfall. InternVL3-38B reaches `suffixCharAcc` = 0.246 against a
floor of 0.349, and Phi-3.5-Vision 0.161 — less than half the floor. A model scoring below a
constant string on the informative characters is producing output that is **actively
anti-correlated** with the truth on those positions, which the per-position table explains
directly ([Section 6.3](#63-the-prefixsuffix-cliff)): these models reproduce the prefix and
then fabricate digits.

Two models exceed even the edit-distance floor of `constCER` = 0.322: LLaVA-NeXT-Mistral-7B
at 0.466 and LLaVA-NeXT-Vicuna-7B at 0.383. Both post exact match of exactly 0.000.

### 6.3 The prefix–suffix cliff

The per-position table is the component's clearest single diagnostic. Averaging accuracy over
the six constant positions and the six variable ones:

$$
\overline{\mathrm{acc}}_{\text{pos }0\text{--}5} = \mathbf{0.992},
\qquad
\overline{\mathrm{acc}}_{\text{pos }6\text{--}11} = \mathbf{0.568},
\qquad
\overline{\text{cliff}} = 0.424 .
$$

**Fifteen of eighteen models exceed 0.98 on the constant positions**, including every model
that fails the task outright. The single-step drop from position 5 to position 6 — the first
character that varies — averages **0.248** and reaches **0.744** for InternVL3-38B.

| Model | Prefix (0–5) | Suffix (6–11) | Cliff |
|---|---:|---:|---:|
| Phi-3.5-Vision | 0.974 | 0.161 | **0.813** |
| Pixtral-12B | 0.988 | 0.199 | 0.789 |
| LLaVA-NeXT-Vicuna-7B | **1.000** | 0.214 | 0.786 |
| LLaVA-NeXT-Mistral-7B | **1.000** | 0.228 | 0.772 |
| InternVL3-38B-Instruct | 0.994 | 0.246 | 0.748 |
| … | | | |
| Qwen2.5-VL-7B | 1.000 | 0.939 | 0.060 |
| Qwen2.5-VL-3B | 0.998 | 0.940 | 0.058 |
| Qwen3-VL-30B-A3B | 1.000 | 0.987 | **0.013** |

**The discontinuity is not sequence-length degradation.** Within the suffix there is
essentially no left-to-right decay: the mean difference between position 7 and position 11 is
**0.033** across all eighteen models. If the effect were a decoder losing coherence over a
long digit run, accuracy would fall monotonically along the sequence. Instead it is flat
across the variable region and drops sharply exactly at the boundary where the corpus stops
being predictable.

That localises the mechanism precisely: **these models are reproducing a memorised or
inferred prefix and fabricating the remainder.** The prefix is recoverable from the format
statement in the prompt, from the `A0...` hint in the question, and from the corpus's own
regularity; the suffix requires reading. Accuracy tracks that division exactly.

The cliff also grades the leaderboard continuously. Qwen3-VL-30B-A3B has a cliff of 0.013 —
its suffix accuracy is nearly its prefix accuracy, meaning it is reading rather than
reconstructing. Phi-3.5-Vision's cliff of 0.813 means essentially all of its character
accuracy comes from the part that requires no reading.

### 6.4 Format compliance dissociates from reading

`wellformed` — the share of answers matching `A` followed by eleven digits — separates
instruction-following from transcription, and the two LLaVA models make the dissociation
maximally sharp:

| Model | wellformed | EM | suffixCharAcc | Reading |
|---|---:|---:|---:|---|
| LLaVA-NeXT-Vicuna-7B | **1.000** | **0.000** | 0.214 | none |
| LLaVA-NeXT-Mistral-7B | **0.010** | **0.000** | 0.228 | none |

**LLaVA-NeXT-Vicuna-7B produces a perfectly formatted twelve-character accession code on
every one of 522 items and gets zero of them right.** Its per-position profile is the
explanation: positions 0–6 at 0.990–1.000, position 7 at **0.025**. It emits the prefix
flawlessly and then invents six digits. This is the constant-string exploit realised in full
by an actual model, and it is visible only because the evaluation reports format compliance,
per-position accuracy, and the constant floor together.

Its sibling fails the opposite way. LLaVA-NeXT-Mistral-7B is well-formed on 1.0% of items and
carries the highest CER in the set at 0.466 — it neither reads nor formats. The two models
have near-identical suffix accuracy (0.214 and 0.228) and near-identical exact match (0.000),
yet `wellformed` differs by a factor of a hundred. Reporting format compliance separately is
what keeps these from being described as the same failure.

Across the leaderboard, $\rho(\mathrm{EM}, \mathrm{wellformed}) = 0.673$ — substantially
below the near-unity agreement among the reading metrics
([Section 6.6](#66-metric-agreement)) — confirming that format compliance is a genuinely
separate axis.

### 6.5 Scale is irrelevant; family is decisive

$$
\rho\!\left(\text{parameters},\, \mathrm{EM}\right) = \mathbf{0.045} .
$$

Essentially zero — the weakest scale relationship anywhere in Bio-VQA. A **3B Qwen2.5-VL
scores 0.904 while a 38B InternVL3 scores 0.105**, a factor of 8.6 across a twelve-fold
parameter difference in the opposite direction.

Family, by contrast, separates the leaderboard almost completely:

| Family | Models | Mean EM | Range |
|---|---:|---:|---|
| **Qwen** | 5 | **0.882** | 0.711 – 0.979 |
| Gemma | 2 | 0.349 | 0.345 – 0.352 |
| InternVL | 5 | 0.252 | 0.105 – 0.460 |
| LLaVA-NeXT | 2 | 0.000 | 0.000 – 0.000 |

**The worst Qwen model (0.711) outperforms the best InternVL model (0.460).** The separation
is complete: no InternVL checkpoint at any size reaches any Qwen checkpoint at any size, and
the Qwen family occupies four of the top five positions with its 3B member in third place.

Within families, size does almost nothing. Qwen runs 3B → 0.904, 7B → 0.916, 8B → 0.711,
30B → 0.979, 32B → 0.900 — non-monotone, with the 3B outperforming both 32B and 8B. InternVL
runs 8B → 0.314/0.460, 30B → 0.188, 38B → 0.105/0.192 — **decreasing with size**, with both
38B variants below both 8B variants.

Because the vision budget is fixed identically across all sizes within each family
([Section 6.4 of the inference document](2.Experiment_Runs/README.md#64-vision-token-budget)),
this is not a resolution artefact. Small-text transcription is a capability determined by the
vision encoder's design and its pretraining data, and scaling the language model does not
supply it. This is the sharpest instance in Bio-VQA of the **granularity collapse** thesis: a
fine-grained perceptual capability on which parameter count is uninformative.

### 6.6 Metric agreement

The three reading metrics rank models almost identically:

$$
\rho\!\left(\mathrm{EM},\, \mathrm{suffixCharAcc}\right) = 0.966,
\qquad
\rho\!\left(\mathrm{EM},\, \mathrm{charAcc}\right) = 0.971,
\qquad
\rho\!\left(\mathrm{EM},\, \mathrm{CER}\right) = -0.973 .
$$

The choice of headline is therefore **not load-bearing for the ranking** — a reader who
prefers character accuracy or edit distance obtains essentially the same order. What the
choice does affect is the *level*, and there the difference is decisive: on `charAcc` the
weakest model reads 0.568, which sounds like partial competence, while on `suffixCharAcc` it
reads 0.161 against a floor of 0.349. Mean inflation of `charAcc` over `suffixCharAcc` across
the eighteen models is **0.212**, and it is largest exactly where it matters most — 0.407 for
Phi-3.5-Vision, 0.007 for Qwen3-VL-30B-A3B.

The metrics agree on who is better; only the suffix-restricted metric and the constant floor
say whether anyone is reading at all.

---

## 7. Scope of the released evaluation

The evaluation establishes that seven of eighteen models fall below a no-look constant
predictor and localises the mechanism to prefix reproduction via the per-position table. Two
extensions would sharpen it further.

A **prefix-randomised control** would remove the exploit at the source rather than correcting
for it in the metric. Re-rendering the same specimen labels with synthetic accession codes
drawn from a uniform character distribution would make the constant-predictor floor equal
chance, converting `charAcc` from an inflated metric into a directly interpretable one and
turning the cliff in [Section 6.3](#63-the-prefixsuffix-cliff) from an inference into a
measurement.

A **box-padding ablation** is available from a generator argument already exposed.
`--box_pad_frac` enlarges the detection box to enclose the printed text rather than only the
bars ([Section 4 of the generation document](1.Generate_QA/README.md#4-barcode-localisation));
rendering the corpus at two padding values and comparing would separate the cost of locating
the text from the cost of reading it. The detection results are cached in `Barcodes.csv`, so
the re-render requires no further API calls.

---

## 8. Reproduction

```bash
# 1a. Localise barcodes and render red boxes. Requires MOONDREAM_API_KEY.
export MOONDREAM_API_KEY=...
python 1.Generate_QA/generate_ocr.py \
    --base_dir   $BIOVQA_DATA/eda/2018-NEON-Beetles \
    --output_dir "$BIOVQA_DATA/tasks/Optical Character Recognition - OCR" \
    --task_id $SLURM_ARRAY_TASK_ID --num_tasks 6 --resume

# merge the per-task CSVs into Barcodes.csv, then:

# 1b. Build the QA manifest. Ground truth is the filename stem.
python 1.Generate_QA/generate_QA.py \
    --base_dir $BIOVQA_DATA/tasks \
    --barcode_dir_basename "Optical Character Recognition - OCR"

# 2. Parse, normalise, compute the prefix, shard.
python 2.Experiment_Runs/Preprocess/split_qa.py \
    --task_root "$BIOVQA_DATA/tasks/Optical Character Recognition - OCR" \
    --run_dir 2.Experiment_Runs --n_chunks 6 --seed 42

# 3. Informative-character audit and the constant-predictor floors.
python 2.Experiment_Runs/Preprocess/stats.py --run_dir 2.Experiment_Runs

# 4. Inference — 18 models x 6 shards.
python 2.Experiment_Runs/qwenVL2_5_7B.py --qa_csv 2.Experiment_Runs/chunks/chunk00/qa.csv

# 5. Merge, then score.
python 3.Evaluate/merge_results.py --run_dir 2.Experiment_Runs --out_dir 3.Evaluate/results
python 3.Evaluate/evaluate.py \
    --results_dir 3.Evaluate/results --out_dir 3.Evaluate/results --bootstrap 2000
```

Quote the task root — it contains spaces and a dash.

---

## 9. Outputs

| File | Contents |
|---|---|
| `3.Evaluate/results/ocr_overall.csv` | One row per model: `n, EM, suffixEM, suffixCharAcc, charAcc, CER, wellformed, parse_rate`, plus all four constant-predictor floors and bootstrap bounds. |
| `3.Evaluate/results/ocr_by_position.csv` | Per-character-position accuracy, twelve columns per model — the diagnostic of [Section 6.3](#63-the-prefixsuffix-cliff). |
| `3.Evaluate/results/ocr_report.md` | Leaderboard, per-position table, and interpretation. |
| `chunks/chunkNN/qa_<KEY>.summary.json` | Per-shard run record including the verbatim prompt and vision budget. |

Every results row carries its own `const*` columns, so a score can never be quoted without
the floor it must clear.

---

## 10. References

Edit distance follows Levenshtein [1]. Character error rate is the standard OCR evaluation
measure [2]. The barcode localisation step uses the Moondream vision–language model [3]; its
role is confined to detection, as set out in
[Section 4.2](#42-ground-truth-is-not-a-model-read). The source corpus is the NEON
invertebrate specimen collection [4].

[1] V. I. Levenshtein. **Binary Codes Capable of Correcting Deletions, Insertions, and
Reversals.** *Soviet Physics Doklady*, 10(8):707–710, 1966.

[2] S. V. Rice, F. R. Jenkins, and T. A. Nartker. **The Fifth Annual Test of OCR Accuracy.**
Information Science Research Institute, University of Nevada, Las Vegas, TR-96-01, 1996.

[3] V. Korrapati. **Moondream: A Small Vision Language Model.** https://moondream.ai

[4] National Ecological Observatory Network (NEON). **Ground beetles sampled from pitfall
traps (DP1.10022.001).** NEON Data Portal.