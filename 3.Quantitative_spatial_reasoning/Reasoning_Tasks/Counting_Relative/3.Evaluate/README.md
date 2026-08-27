<h1 align="center"> Counting-Relative &rarr; Stage 4 &rarr; Evaluation &amp; Results </h1>

<p align="center">
  <b>Per-condition lift, ground-truth-free coherence, position bias &mdash; and the full result set</b>
</p>

<p align="center">
  <img alt="Stage" src="https://img.shields.io/badge/stage-4%20of%204-blue">
  <img alt="Items" src="https://img.shields.io/badge/items-42%2C412-green">
  <img alt="Models" src="https://img.shields.io/badge/models-18-purple">
  <img alt="Parse" src="https://img.shields.io/badge/parse%20rate-1.000-brightgreen">
  <img alt="Bootstrap" src="https://img.shields.io/badge/bootstrap-over%20images-informational">
</p>

---

## Overview

| Script | Role |
| :--- | :--- |
| `merge_results.py` | Stitch six chunks per model, write a completeness manifest, strip path columns |
| `evaluate.py` | Score; per-condition lift, coherence, position bias |

> **Design contract:** scoring lives in exactly one place, so every number below comes from one code
> path. No metric is computed ad hoc.

---

## Table of Contents

- [Metric Definitions](#metric-definitions)
- [Why the Pooled Lift Is Not Comparable](#why-the-pooled-lift-is-not-comparable)
- [Results &mdash; Overall](#results--overall)
- [Results &mdash; By Condition and Polarity](#results--by-condition-and-polarity)
- [Results &mdash; Answer Distribution](#results--answer-distribution)
- [Findings](#findings)
- [What the Results Imply](#what-the-results-imply)
- [Limitations](#limitations)
- [Usage](#usage)
- [FAQ](#faq)

---

## Metric Definitions

| Metric | Definition | Floor |
| :--- | :--- | :---: |
| **`lift_same`** | **headline.** Accuracy on `same` minus chance | 0 |
| `lift_different` | Accuracy on `different` minus the dataset prior | 0 |
| `same_macroAcc` | `same` accuracy averaged with equal weight over the 10 datasets | 0.250 |
| `balAcc` | Recall averaged over the four gold letters | 0.250 |
| `kappa` | Accuracy corrected for both marginal distributions | 0 |
| `incoherence` | Share of pairs where `most` and `least` get the same letter | &mdash; |
| `posBias` | Total-variation distance, model answers vs key | 0 |
| `predTop`, `predTopShare` | Most frequent answer and its share | 0.250 |

### Formal statements

Per-condition lift, where $\phi_c$ is that condition's floor:

$$
\text{lift}_c = \frac{1}{N_c}\sum_{i \in c} \mathbb{1}[\hat{y}_i = y_i] - \phi_c,
\qquad
\phi_{\text{same}} = 0.250, \quad \phi_{\text{diff}} \in \{0.530, 0.548\}
$$

Balanced accuracy &mdash; a constant predictor scores exactly $1/4$ regardless of the key:

$$
\text{balAcc} = \frac{1}{4}\sum_{\ell \in \{A,B,C,D\}}
\frac{\bigl|\{i : y_i = \ell \wedge \hat{y}_i = \ell\}\bigr|}{\bigl|\{i : y_i = \ell\}\bigr|}
$$

Cohen's $\kappa$, correcting for chance agreement between both marginals:

$$
\kappa = \frac{p_o - p_e}{1 - p_e},
\qquad
p_e = \sum_{\ell} \Pr[y = \ell]\,\Pr[\hat{y} = \ell]
$$

Incoherence, over `(image, animal_term)` pairs &mdash; **requires no ground truth**:

$$
\text{incoh} = \frac{1}{|\mathcal{P}|}\sum_{(g,t) \in \mathcal{P}}
\mathbb{1}\bigl[\hat{y}_{\text{most}}(g,t) = \hat{y}_{\text{least}}(g,t)\bigr]
$$

```python
piv = d.pivot_table(index=["image", "animal_term"],   # NOT image alone
                    columns="polarity", values="_p", aggfunc="first")
return float((piv["most"] == piv["least"]).mean())
```

Validated against simulated archetypes:

| Archetype | balAcc | lift | incoherence | predTopShare |
| :--- | ---: | ---: | ---: | ---: |
| real model | 0.723 | +0.329 | **0.149** | 0.255 |
| coin-flip | 0.254 | &minus;0.141 | **0.248** | 0.257 |
| always answers `A` | 0.250 | &minus;0.149 | **1.000** | 1.000 |

---

## Why the Pooled Lift Is Not Comparable

`relcount_overall.csv` retains a pooled `lift` for continuity. **It must not be quoted.** It subtracts
a weighted average of two floors from a weighted average of two accuracies:

| | `same` | `different` | pooled lift |
| :--- | :--- | :--- | ---: |
| model A | acc 0.560, lift **+0.310** | acc 0.253, lift **&minus;0.286** | +0.029 |
| model B | acc 0.420, lift **+0.170** | acc 0.409, lift **&minus;0.130** | +0.029 |

Identical headline, opposite conclusions. `evaluate.py` therefore hoists `acc_same`, `lift_same`,
`acc_different` and `lift_different` into the overall table, prints them as a dedicated block, and
sorts by `lift_same`.

---

## Results &mdash; Overall

18 models &middot; 42,412 items &middot; 20,000 images &middot; `parse_rate = 1.000` throughout.
Sorted by `lift_same`.

| # | Model | `lift_same` | acc `same` | macroAcc | `lift_diff` | acc `diff` | balAcc | &kappa; | incoh | posBias | predTop |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | qwenVL3_32B | **+0.403** | 0.653 | 0.672 | &minus;0.094 | 0.445 | 0.555 | 0.407 | 0.052 | 0.039 | C 0.28 |
| 2 | gemma4_12B | +0.388 | 0.638 | 0.658 | &minus;0.060 | 0.479 | 0.563 | 0.417 | 0.043 | 0.027 | C 0.27 |
| 3 | internvl3_38B-Instruct | +0.366 | 0.616 | 0.634 | **&minus;0.026** | **0.513** | **0.567** | **0.423** | **0.008** | 0.072 | C 0.31 |
| 4 | internvl3_5_38B-Instruct | +0.338 | 0.588 | 0.599 | &minus;0.130 | 0.409 | 0.504 | 0.338 | 0.037 | 0.051 | C 0.29 |
| 5 | molmo2_8B | +0.329 | 0.579 | 0.594 | &minus;0.304 | 0.235 | 0.416 | 0.222 | 0.055 | 0.065 | C 0.31 |
| 6 | qwen3vl_8B | +0.320 | 0.570 | 0.587 | &minus;0.300 | 0.239 | 0.414 | 0.219 | 0.047 | 0.045 | C 0.28 |
| 7 | qwenVL3_30B | +0.312 | 0.562 | 0.576 | &minus;0.145 | 0.394 | 0.483 | 0.310 | 0.032 | 0.043 | C 0.27 |
| 8 | internvl3_5_30B_A3B | +0.305 | 0.555 | 0.569 | &minus;0.123 | 0.416 | 0.489 | 0.319 | 0.029 | 0.044 | A 0.29 |
| 9 | internvl3_5_8B-Instruct | +0.277 | 0.527 | 0.540 | &minus;0.264 | 0.275 | 0.408 | 0.211 | 0.017 | 0.076 | C 0.29 |
| 10 | qwenVL2_5_7B | +0.273 | 0.523 | 0.536 | &minus;0.256 | 0.283 | 0.410 | 0.213 | 0.129 | 0.120 | D 0.37 |
| 11 | pixtral_12B | +0.233 | 0.483 | 0.494 | &minus;0.334 | 0.205 | 0.352 | 0.136 | 0.030 | 0.067 | A 0.29 |
| 12 | gemma3_12B | +0.228 | 0.478 | 0.486 | &minus;0.194 | 0.345 | 0.415 | 0.220 | 0.047 | 0.084 | A 0.33 |
| 13 | qwenVL2_5_3B | +0.184 | 0.434 | 0.441 | &minus;0.346 | 0.193 | 0.321 | 0.095 | 0.104 | 0.247 | D 0.38 |
| 14 | phi35_vision | +0.140 | 0.390 | 0.397 | &minus;0.307 | 0.232 | 0.316 | 0.088 | 0.085 | 0.210 | A 0.41 |
| 15 | internvl3_8B-Instruct | +0.098 | 0.348 | 0.357 | &minus;0.250 | 0.289 | 0.320 | 0.093 | **0.750** | 0.172 | D 0.36 |
| 16 | llama3_2_11B | +0.056 | 0.306 | 0.311 | &minus;0.277 | 0.262 | 0.286 | 0.048 | **0.590** | 0.456 | A 0.55 |
| 17 | llava_next_mistral_7B | +0.005 | 0.255 | 0.256 | &minus;0.282 | 0.257 | 0.259 | 0.012 | **0.833** | 0.662 | A 0.91 |
| 18 | llava_next_vicuna_7B | **&minus;0.005** | 0.245 | 0.245 | &minus;0.290 | 0.249 | 0.250 | 0.000 | **1.000** | 0.753 | A **1.00** |

Floors: `same` **0.250** &middot; `different` **0.530** (most) / **0.548** (least).

---

## Results &mdash; By Condition and Polarity

Lift over the polarity-specific floor. Four cells per model.

| Model | diff / least | diff / most | same / least | same / most |
| :--- | ---: | ---: | ---: | ---: |
| qwenVL3_32B | &minus;0.253 | **+0.066** | +0.373 | **+0.433** |
| gemma4_12B | &minus;0.183 | **+0.063** | +0.374 | +0.401 |
| internvl3_38B-Instruct | &minus;0.064 | **+0.013** | +0.356 | +0.376 |
| internvl3_5_38B-Instruct | &minus;0.258 | &minus;0.001 | +0.302 | +0.374 |
| qwenVL3_30B | &minus;0.288 | &minus;0.002 | +0.276 | +0.347 |
| internvl3_5_30B_A3B | &minus;0.225 | &minus;0.021 | +0.283 | +0.326 |
| gemma3_12B | &minus;0.276 | &minus;0.112 | +0.169 | +0.286 |
| qwenVL2_5_7B | &minus;0.352 | &minus;0.160 | +0.233 | +0.312 |
| internvl3_8B-Instruct | &minus;0.332 | &minus;0.169 | **&minus;0.082** | +0.278 |
| qwen3vl_8B | &minus;0.401 | &minus;0.198 | +0.279 | +0.361 |
| molmo2_8B | &minus;0.407 | &minus;0.202 | +0.271 | +0.388 |
| internvl3_5_8B-Instruct | &minus;0.322 | &minus;0.207 | +0.261 | +0.293 |
| llava_next_mistral_7B | &minus;0.304 | &minus;0.261 | &minus;0.003 | +0.013 |
| phi35_vision | &minus;0.343 | &minus;0.271 | +0.113 | +0.167 |
| llama3_2_11B | &minus;0.274 | &minus;0.280 | +0.002 | +0.109 |
| llava_next_vicuna_7B | &minus;0.299 | &minus;0.282 | &minus;0.005 | &minus;0.005 |
| pixtral_12B | &minus;0.381 | &minus;0.286 | +0.186 | +0.280 |
| qwenVL2_5_3B | &minus;0.403 | &minus;0.289 | +0.150 | +0.217 |

### Two structural observations

**`most` is easier than `least` for every model without exception.** The mean same-condition gap is
**+0.084**. Identifying the densest panel is a peak-detection problem; identifying the sparsest
requires ruling out three alternatives, and sparse panels are visually less distinctive.

**Only three models exceed the prior on `different`/`most`** &mdash; qwenVL3_32B (+0.066), gemma4_12B
(+0.063), internvl3_38B (+0.013). No model exceeds it on `different`/`least`, where the floor is
higher (0.548) and the discrimination harder.

---

## Results &mdash; Answer Distribution

Gold shares are 0.247&ndash;0.252 across the four options. Predicted shares for five representative
models:

| Model | pred A | pred B | pred C | pred D | posBias |
| :--- | ---: | ---: | ---: | ---: | ---: |
| gemma4_12B | 0.244 | 0.265 | 0.266 | 0.225 | **0.027** |
| qwenVL3_32B | 0.216 | 0.268 | 0.279 | 0.236 | 0.039 |
| internvl3_38B-Instruct | 0.197 | 0.266 | 0.310 | 0.227 | 0.072 |
| gemma3_12B | 0.331 | 0.244 | 0.238 | 0.187 | 0.084 |
| llava_next_vicuna_7B | **1.000** | 0.000 | 0.000 | 0.000 | **0.753** |

`acc ≈ balAcc` to three decimals for every model, confirming that no model is winning through position
preference &mdash; the top performers' advantage is real rather than a lucky alignment with the key.

---

## Findings

### 1 &mdash; No model beats the dataset prior on `different`

All eighteen `lift_different` values are negative, from **&minus;0.026** to **&minus;0.346**. The best
absolute accuracy is 0.513 against a 0.539 floor.

> The correct reading is **not** "models fail." It is that the `different` condition **cannot
> demonstrate counting**, because performance is bounded above by what domain recognition alone
> achieves. `internvl3_38B` at &minus;0.026 sits essentially *at* the oracle-prior ceiling &mdash;
> matching a strategy that requires no comparison at all.

The `same` condition, which has no such shortcut, shows clear positive lift up to **+0.403**. That
contrast is the methodological result: a shortcut-available condition and a shortcut-free one, on
identical machinery, produce opposite verdicts.

### 2 &mdash; Six models score *below chance* on `different`

| Model | acc `different` | rank on `same` |
| :--- | ---: | ---: |
| **molmo2_8B** | **0.235** | **5 of 18** |
| **qwen3vl_8B** | **0.239** | **6 of 18** |
| phi35_vision | 0.232 | 14 |
| pixtral_12B | 0.205 | 11 |
| qwenVL2_5_3B | 0.193 | 13 |
| llava_next_vicuna_7B | 0.249 | 18 |

Below 0.250 on a four-way task is systematic anti-correlation, not noise. And molmo2 and qwen3vl are
**top-six on `same`**, so this is not incompetence.

The likely mechanism is the one the prompt explicitly warns against: **the four panels are pasted at
different magnifications**, so a model comparing apparent animal size rather than count is reliably
wrong. Panels with large-looking animals (camels, cattle) hold *fewer* of them than panels with
small-looking ones (bees, fish). A size-comparison heuristic inverts the answer.

### 3 &mdash; The polarity failure

`incoherence` separates two failure modes that accuracy conflates:

| Model | incoherence | predTopShare | Interpretation |
| :--- | ---: | ---: | :--- |
| llava_next_vicuna_7B | 1.000 | 1.00 | answers `A` always &mdash; degenerate |
| llava_next_mistral_7B | 0.833 | 0.91 | near-degenerate |
| **internvl3_8B-Instruct** | **0.750** | **0.36** | **not degenerate &mdash; ignores polarity** |
| llama3_2_11B | 0.590 | 0.55 | mixed |
| qwenVL2_5_7B | 0.129 | 0.37 | occasional |
| top six | 0.008 &ndash; 0.055 | 0.27 &ndash; 0.31 | polarity processed correctly |

> `internvl3_8B` is the informative case. At `predTopShare 0.36` it varies its answer per image &mdash;
> it is **not** a constant predictor &mdash; yet it gives the same letter to `most` and `least` on
> three-quarters of grids. **It is not reading the polarity word.** That is a specific
> instruction-following failure, invisible in accuracy alone.

The clean separation between the top six (&le;0.055) and the bottom four (&ge;0.590) confirms the
metric is measuring something real rather than reflecting noise.

---

## What the Results Imply

**Ordinal comparison is easier than absolute counting, but not solved.** The best `same` accuracy is
0.653 against a 0.250 floor &mdash; higher than the same models achieve on exact-match absolute
counting (0.285 for the best model). Removing the requirement to produce a number helps substantially.
It does not make the task easy.

**Shortcut availability changes model behaviour, not just model scores.** The `different` condition is
not merely harder; several models perform *worse than chance* on it while performing well on `same`.
That is only possible if they use a different strategy when domain cues are present. Benchmarks that
mix panel sources without measuring the prior report a number that mostly reflects domain recognition.

**Instruction-following failures are model-specific and metric-invisible.** Four models fail the
polarity check at rates from 0.590 to 1.000. Without `incoherence`, `internvl3_8B` would appear as a
merely weak model rather than one that ignores a word in the question.

**Scale helps but does not determine.** The top three span 12.2B&ndash;38.4B, but `molmo2_8B` (8.0B)
ranks 5th on `same` while `internvl3_8B` (8.1B) ranks 15th. Architecture and instruction tuning matter
more than parameter count within this range.

---

## Limitations

| Limitation | Effect |
| :--- | :--- |
| **Selection effect in the key** | Grids admitted only with unique extrema; near-ties excluded. Measured accuracy is on an easier-than-representative sample. |
| **`different` floors are corpus properties** | 0.530 / 0.548 come from a Monte-Carlo over *these* datasets. Regenerating changes them; they must be stated wherever a `different` number appears. |
| **Dataset imbalance in `same`** | GrevysZebra contributes 4,412 of 22,412 rows. Quote `same_macroAcc`. |
| **Border asymmetry** | The `different` condition draws 3px panel borders; `same` does not. Content-free, but a visible difference beyond the intended one. |
| **Clustering** | 2&ndash;6 questions per grid. Intervals bootstrap over the 20,000 images. |

---

## Usage

```bash
python merge_results.py --out_dir <RESULTS_DIR>

python evaluate.py --results_dir <RESULTS_DIR> \
                   --out_dir     <RESULTS_DIR> \
                   --bootstrap 2000 \
                   --prior_different_most 0.530 \
                   --prior_different_least 0.548
```

### Output files

| File | Contents |
| :--- | :--- |
| `relcount_overall.csv` | One row per model, all metrics including per-condition lift |
| `relcount_by_group.csv` | One row per model &times; condition &times; polarity |
| `relcount_confusion.csv` | `gold_X` vs `pred_X` letter distribution |
| `relcount_report.md` | Generated markdown summary |
| `merge_manifest.json` | Completeness record &mdash; every model must be `complete: true` |

### Reading order

1. **`predTopShare`** &mdash; near 1.00 means a constant predictor; `balAcc` will sit at 0.250 and
   $\kappa$ at 0 however respectable raw accuracy looks.
2. **`incoherence`** &mdash; a model can have moderate accuracy and still be ignoring the polarity word.
3. **`lift_same`** &mdash; the shortcut-free measurement, and the headline.
4. **`lift_different`, in context** &mdash; all eighteen are negative; report the condition as bounded
   by domain recognition rather than as a model failure.

---
