<h1 align="center"> Relative-Position &rarr; Stage 4 &rarr; Evaluation &amp; Results </h1>

<p align="center">
  <b>Lift over a uniform floor, chance-corrected error attribution, angular error &mdash; and the full result set</b>
</p>

<p align="center">
  <img alt="Stage" src="https://img.shields.io/badge/stage-4%20of%204-blue">
  <img alt="Items" src="https://img.shields.io/badge/items-1%2C000-green">
  <img alt="Photos" src="https://img.shields.io/badge/effective%20n-208-yellow">
  <img alt="Models" src="https://img.shields.io/badge/models-18-purple">
  <img alt="Best" src="https://img.shields.io/badge/best%20lift-%2B0.425-brightgreen">
  <img alt="Chance" src="https://img.shields.io/badge/chance-0.125-red">
</p>

---

## Overview

| Script | Role |
| :--- | :--- |
| `merge_results.py` | Stitch chunks, completeness manifest, strip path and dead columns |
| `evaluate.py` | Lift, **chance-corrected error attribution**, angular error |

> **Design contract:** the three mis-pairing columns say *which* box a model grabbed. Because a random
> answer lands on a mis-pairing direction about one time in eight, **only the chance-corrected `X`
> columns are evidence.** The raw rates are retained for audit and must not be read directly.

---

## Table of Contents

- [Metric Definitions](#metric-definitions)
- [Error Attribution](#error-attribution)
- [Chance Correction](#chance-correction)
- [Results &mdash; Overall](#results--overall)
- [Results &mdash; Error Attribution](#results--error-attribution)
- [Findings](#findings)
- [What the Results Imply](#what-the-results-imply)
- [Limitations](#limitations)
- [Usage](#usage)
- [FAQ](#faq)

---

## Metric Definitions

| Metric | Definition | Floor |
| :--- | :--- | :---: |
| **`lift`** | **headline.** Accuracy minus the majority-direction floor | 0 |
| `acc`, `majority` | Raw accuracy and the floor it is measured against | 0.125 |
| `balAcc` | Recall averaged over the gold directions | 0.125 |
| `kappa` | Accuracy corrected for both marginals | 0 |
| `angErr` | Mean angular error on the 45&deg; ring | 90&deg; |
| `within45` | Share of answers within one octant of correct | 0.375 |
| **`errRedEmptyX`** etc. | Chance-corrected error attribution | 0 |
| `errOther` | Wrong, matching **no** box pairing | &mdash; |
| `posBias`, `predTopShare` | Answer-distribution diagnostics | &mdash; |

Because the key is **uniform by construction** (125 per direction), the majority floor coincides with
chance:

$$
\text{lift} = \frac{1}{N}\sum_i \mathbb{1}[\hat{y}_i = y_i] \;-\; 0.125
$$

`majority` is still computed on the identical rows as each model rather than assumed, so the metric
stays correct if the corpus is regenerated with an unbalanced key.

Angular error, with $\alpha(\cdot)$ mapping a direction to its octant angle:

$$
\Delta_i = \min\bigl(|\alpha(y_i) - \alpha(\hat{y}_i)|,\; 360^\circ - |\alpha(y_i) - \alpha(\hat{y}_i)|\bigr)
$$

> `angErr` matters here: answering `bottom-left` when the truth is `left` is a **45&deg; miss**, not the
> same error as answering `right`, and exact match cannot tell those apart. A model with low accuracy
> but low `angErr` is perceiving the layout and losing on granularity &mdash; a different finding from
> one that is guessing.

---

## Error Attribution

Because Stage 1 **solves** the decoy placement so each incorrect box pairing yields a distinct
direction, every wrong answer is attributable:

| Column | The model did this |
| :--- | :--- |
| `errRedEmpty` | took the **empty red** box with the correct blue &mdash; did not check whether the red box held a specimen |
| `errBlueEmpty` | correct red, took the **empty blue** box |
| `errBothEmpty` | took **both empty** boxes &mdash; matched colour only |
| `errOther` | wrong, but matching **no** box pairing &mdash; not a box confusion at all |

`errOther` is informative *because* the options are no longer shown. When four options were listed they
were exactly gold plus the three mis-pairings, so the attribution was exhaustive and `errOther` was zero
by construction. With eight directions open, a model can be wrong **without** confusing boxes.

---

## Chance Correction

A model answering at random still lands on a mis-pairing direction about **one time in eight**. Raw
rates are therefore not evidence: `errRedEmpty = 0.16` is barely above guessing.

$$
\text{errX} \;=\; \underbrace{\Pr[\hat{y} = m]}_{\text{observed}}
\;-\; \underbrace{\sum_{d} \Pr[\hat{y} = d]\,\Pr[m = d]}_{\text{expected, Cohen's } \kappa \text{ term}}
$$

```python
pm  = d["_p"].value_counts(normalize=True)      # the model's own answers
mm  = d[col].value_counts(normalize=True)       # that mis-pairing's directions
exp = sum(pm.get(k, 0) * mm.get(k, 0) for k in set(pm.index) | set(mm.index))
out[key + "X"] = observed - exp
```

| Sign | Meaning |
| :---: | :--- |
| **positive** | genuine box confusion |
| **negative** | the model *avoids* that pairing more than its own answer habits predict &mdash; **what content-checking looks like** |

### What the correction changes

| Model | raw `errRedEmpty` | `errRedEmptyX` | Reading |
| :--- | ---: | ---: | :--- |
| qwenVL3_32B | 0.010 | **&minus;0.128** | strongly avoids |
| internvl3_38B | 0.202 | **+0.066** | genuinely confuses |
| llava_next_vicuna_7B | 0.136 | **0.000** | chance hits only |

Without the correction, `internvl3_5_8B` at raw 0.164 and `llava_vicuna` at raw 0.136 look similar.
Corrected they are **+0.037** and **0.000** &mdash; one is confusing boxes, the other is guessing.

> **Built-in sanity check:** `llava_next_vicuna_7B` answers one word on every item, so its answer
> distribution is a point mass and observed **must** equal expected. It scores **exactly 0.000** on all
> three columns.

---

## Results &mdash; Overall

18 models &middot; 1,000 items &middot; 208 photographs &middot; `parse_rate = 1.000` throughout.
Floor **0.125**.

| # | Model | acc | **lift** | balAcc | &kappa; | angErr | within45 | posBias | predTop | share |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| 1 | qwenVL3_32B | **0.550** | **+0.425** | 0.550 | 0.486 | **27.0** | **0.895** | 0.341 | top-right | 0.29 |
| 2 | gemma4_12B | 0.495 | +0.370 | 0.495 | 0.423 | 35.4 | 0.815 | 0.365 | bottom-right | 0.28 |
| 3 | molmo2_8B | 0.457 | +0.332 | 0.457 | 0.379 | 32.9 | 0.864 | 0.297 | top-left | 0.22 |
| 4 | qwen3vl_8B | 0.421 | +0.296 | 0.421 | 0.338 | 40.1 | 0.803 | 0.435 | top-right | 0.30 |
| 5 | qwenVL3_30B | 0.355 | +0.230 | 0.355 | 0.263 | 55.0 | 0.673 | 0.276 | top-left | 0.24 |
| 6 | pixtral_12B | 0.266 | +0.141 | 0.266 | 0.161 | 64.4 | 0.576 | 0.514 | right | 0.42 |
| 7 | internvl3_5_30B_A3B | 0.252 | +0.127 | 0.252 | 0.145 | 69.8 | 0.561 | 0.435 | bottom-right | 0.51 |
| 8 | internvl3_5_8B-Instruct | 0.224 | +0.099 | 0.224 | 0.113 | 71.5 | 0.541 | 0.531 | bottom-right | 0.50 |
| 9 | internvl3_5_38B-Instruct | 0.219 | +0.094 | 0.219 | 0.107 | 73.1 | 0.520 | 0.502 | bottom-left | 0.32 |
| 10 | qwenVL2_5_7B | 0.213 | +0.088 | 0.213 | 0.101 | 69.2 | 0.545 | 0.487 | top-right | 0.35 |
| 11 | internvl3_38B-Instruct | 0.205 | +0.080 | 0.205 | 0.091 | 80.5 | 0.471 | 0.471 | top-right | 0.35 |
| 12 | internvl3_8B-Instruct | 0.203 | +0.078 | 0.203 | 0.089 | 72.9 | 0.515 | 0.483 | top-right | 0.41 |
| 13 | phi35_vision | 0.160 | +0.035 | 0.160 | 0.040 | 80.7 | 0.440 | 0.567 | left | 0.44 |
| 14 | gemma3_12B | 0.145 | +0.020 | 0.145 | 0.023 | 86.9 | 0.395 | 0.605 | bottom-right | 0.51 |
| 15 | llava_next_mistral_7B | 0.131 | +0.006 | 0.131 | 0.007 | 89.7 | 0.395 | 0.683 | bottom-right | 0.81 |
| 16 | llama3_2_11B | 0.125 | 0.000 | 0.125 | 0.000 | 89.7 | 0.382 | 0.726 | bottom-right | 0.68 |
| 17 | llava_next_vicuna_7B | 0.125 | 0.000 | 0.125 | 0.000 | 90.0 | 0.375 | 0.875 | bottom-left | **1.00** |
| 18 | qwenVL2_5_3B | 0.110 | **&minus;0.015** | 0.110 | &minus;0.017 | 92.3 | 0.352 | 0.750 | bottom-right | 0.56 |

`acc = balAcc` exactly for every model &mdash; a direct consequence of the uniform key.

---

## Results &mdash; Error Attribution

Raw rate, expected rate, and the chance-corrected excess. **Read the `X` columns.**

| Model | redRaw | redExp | **redX** | blueRaw | blueExp | **blueX** | bothRaw | bothExp | **bothX** | errOther |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwenVL3_32B | 0.010 | 0.138 | **&minus;0.128** | 0.062 | 0.140 | &minus;0.078 | 0.006 | 0.115 | &minus;0.109 | 0.372 |
| gemma4_12B | 0.039 | 0.141 | **&minus;0.102** | 0.078 | 0.141 | &minus;0.063 | 0.021 | 0.113 | &minus;0.092 | 0.367 |
| molmo2_8B | 0.022 | 0.139 | **&minus;0.117** | 0.059 | 0.138 | &minus;0.079 | 0.008 | 0.134 | &minus;0.126 | 0.454 |
| qwen3vl_8B | 0.043 | 0.141 | **&minus;0.098** | 0.076 | 0.138 | &minus;0.062 | 0.022 | 0.102 | &minus;0.080 | 0.438 |
| qwenVL3_30B | 0.110 | 0.140 | &minus;0.030 | 0.066 | 0.140 | &minus;0.074 | 0.071 | 0.130 | &minus;0.059 | 0.398 |
| pixtral_12B | 0.073 | 0.128 | &minus;0.055 | 0.146 | 0.143 | +0.003 | 0.109 | 0.186 | &minus;0.077 | 0.406 |
| **internvl3_5_30B_A3B** | 0.160 | 0.126 | **+0.034** | 0.073 | 0.156 | &minus;0.083 | 0.098 | 0.119 | &minus;0.021 | 0.417 |
| **internvl3_5_8B** | 0.164 | 0.127 | **+0.037** | 0.092 | 0.149 | &minus;0.057 | 0.078 | 0.101 | &minus;0.023 | 0.442 |
| **internvl3_5_38B** | 0.188 | 0.135 | **+0.053** | 0.093 | 0.141 | &minus;0.048 | 0.098 | 0.095 | +0.003 | 0.402 |
| qwenVL2_5_7B | 0.143 | 0.139 | +0.004 | 0.112 | 0.138 | &minus;0.026 | 0.074 | 0.096 | &minus;0.022 | 0.458 |
| **internvl3_38B** | 0.202 | 0.136 | **+0.066** | 0.087 | 0.141 | &minus;0.054 | 0.119 | 0.098 | +0.021 | 0.387 |
| internvl3_8B | 0.118 | 0.135 | &minus;0.017 | 0.144 | 0.140 | +0.004 | 0.099 | 0.100 | &minus;0.001 | 0.436 |
| phi35_vision | 0.137 | 0.156 | &minus;0.019 | 0.138 | 0.122 | +0.016 | 0.147 | 0.185 | &minus;0.038 | 0.418 |
| gemma3_12B | 0.161 | 0.126 | +0.035 | 0.145 | 0.149 | &minus;0.004 | 0.117 | 0.099 | +0.018 | 0.432 |
| llava_next_mistral_7B | 0.147 | 0.123 | +0.024 | 0.136 | 0.162 | &minus;0.026 | 0.112 | 0.104 | +0.008 | 0.474 |
| llama3_2_11B | 0.120 | 0.105 | +0.015 | 0.153 | 0.148 | +0.005 | 0.091 | 0.099 | &minus;0.008 | 0.511 |
| **llava_next_vicuna_7B** | 0.136 | 0.136 | **0.000** | 0.132 | 0.132 | **0.000** | 0.090 | 0.091 | &minus;0.001 | 0.517 |
| qwenVL2_5_3B | 0.144 | 0.127 | +0.017 | 0.150 | 0.153 | &minus;0.003 | 0.106 | 0.100 | +0.006 | 0.490 |

---

## Findings

### 1 &mdash; Failure is granularity, not perception

`qwenVL3_32B` reaches accuracy 0.550 but **`within45 = 0.895`** &mdash; 89.5% of its answers are within
one 45&deg; step of correct. Mean angular error 27.0&deg; against ~90&deg; for random. The pattern holds
for gemma4 (0.815) and molmo2 (0.864).

And **`errOther` sits at 0.37&ndash;0.52 for every model including the best.** So the working models are
not confusing boxes and not guessing &mdash; they see the layout and miss the exact octant.

$$
\mathrm{corr}(\text{acc},\, \text{within45}) = +0.988
$$

> This is a substantially more interesting claim than "models are bad at relative position." It says the
> representation is approximately correct and the **discretisation** is not.

### 2 &mdash; A family-level signature in InternVL

Four of five InternVL variants show **positive `redX` with negative `blueX`**:

| Model | redX | blueX | redX &minus; blueX |
| :--- | ---: | ---: | ---: |
| internvl3_38B | **+0.066** | &minus;0.054 | **+0.120** |
| internvl3_5_38B | **+0.053** | &minus;0.048 | +0.101 |
| internvl3_5_8B | **+0.037** | &minus;0.057 | +0.094 |
| internvl3_5_30B_A3B | **+0.034** | &minus;0.083 | +0.117 |
| internvl3_8B | &minus;0.017 | +0.004 | &minus;0.021 |

Mean `redX − blueX` is **+0.082 for InternVL** versus **&minus;0.024 for the top five**. InternVL
specifically fails on the *queried* box &mdash; the one the question asks about first &mdash; while
handling the blue reference box normally.

Consistency across 8B, 30B-A3B, 38B and InternVL3-38B makes this **architectural rather than noise**.
It is visible only because of the mis-pairing design combined with the chance correction.

### 3 &mdash; A three-way behavioural split, not a single ranking

| Group | redX | blueX | bothX | Reading |
| :--- | :--- | :--- | :--- | :--- |
| **top five** | &minus;0.098 avg | &minus;0.068 avg | &minus;0.093 avg | avoids **every** mis-pairing &mdash; genuinely checking box contents |
| **InternVL (4 of 5)** | +0.034 … +0.066 | &minus;0.048 … &minus;0.083 | &approx; 0 | grabs the empty **red** box |
| **bottom five** | within &plusmn;0.04 of zero | | | not box-confused &mdash; guessing |

Accuracy alone would present this as one ordered list. The attribution columns show **three distinct
behaviours**.

### 4 &mdash; Seven of eighteen are near-constant predictors

`predTopShare > 0.45` for seven models. `llava_next_vicuna_7B` answers a single word &mdash;
`bottom-left` &mdash; on **every** item, and its `X` columns are exactly 0.000, which is the
correction's built-in sanity check.

$$
\mathrm{corr}(\text{acc},\, \text{predTopShare}) = -0.692
$$

Those should be **flagged rather than ranked**.

---

## What the Results Imply

**Directional judgement between two marked objects is not solved, but the errors are structured.** The
best model reaches 0.550 against a 0.125 floor. Its errors are overwhelmingly one octant away, not
random &mdash; pointing at the discretisation of a continuous percept rather than at a failure to
perceive.

**Content-checking is a separable ability, and models differ on it systematically.** The distractor
construction turns *"did the model verify the box contents"* from an assumption into a measurement. The
top five verify; InternVL does not verify the queried box; the bottom five are doing neither.

**Instruction-following failures dominate the bottom of the table.** Seven models are near-constant
predictors. Any benchmark reporting a single accuracy per model would present them as ordered rather
than degenerate.

**A benchmark that draws boxes must place decoys deliberately.** Random decoys would produce random
wrong answers and no attribution. Solving the placement so each incorrect pairing yields a distinct
direction converts every error into diagnostic information at no additional annotation cost.

---

## Limitations

| Limitation | Effect |
| :--- | :--- |
| **Single dataset** | Beetles-500 only. The finding is about specimen-tray imagery, not field photography. |
| **No clean condition** | Only `distractor` was run, so the clean&rarr;distractor drop quantifying the distractors' cost is **not available**. |
| **n = 208 photographs** | Intervals are set by 208, not 1,000. They separate *reads the layout* from *does not*; they cannot rank adjacent models. |
| **Geometric, not biological** | Distractors force colour-plus-content discrimination; they do **not** make the task require species recognition. |
| **Selection filters** | `--margin_deg 8.0` and `--min_sep 0.60` exclude near-boundary angles and near-adjacent pairs &mdash; the easier tail. |

---

## Usage

```bash
python merge_results.py --out_dir <RESULTS_DIR>

python evaluate.py --results_dir <RESULTS_DIR> \
                   --out_dir     <RESULTS_DIR> \
                   --bootstrap 2000
```

### Output files

| File | Contents |
| :--- | :--- |
| `relpos_overall.csv` | One row per model, all metrics |
| `relpos_by_group.csv` | Per condition &times; dataset |
| `relpos_errors.csv` | Attribution: raw rates, expected rates, and the excess |
| `relpos_report.md` | Generated markdown summary |
| `merge_manifest.json` | Completeness record |

### Columns dropped rather than printed as NaN

`optPosBias` and `errOffOption` are defined only when the four options are listed in the prompt. They
are not, so both are `NaN` for every model. A column of NaNs in a published table reads as a bug rather
than as *not applicable*, so `evaluate.py` detects all-NaN columns, drops them, and prints:

```
not applicable for this prompt, omitted: optPosBias, errOffOption
```

They return automatically if the options are ever restored.

### Reading order

1. **`predTopShare`** &mdash; seven of eighteen exceed 0.45; one answers a single word on every item.
2. **`lift`** &mdash; accuracy above the 0.125 floor.
3. **The `X` columns, never the raw rates** &mdash; they separate three distinct behaviours.
4. **`within45` against `acc`** &mdash; a large gap means the failure is granularity, not perception.
