<h1 align="center"> Localization &rarr; Stage 4 &rarr; Evaluation &amp; Results </h1>

<p align="center">
  <b>IoU tiers, three trivial baselines, model-level convention resolution &mdash; and the full result set</b>
</p>

<p align="center">
  <img alt="Stage" src="https://img.shields.io/badge/stage-4%20of%204-blue">
  <img alt="Items" src="https://img.shields.io/badge/items-4%2C000-green">
  <img alt="Models" src="https://img.shields.io/badge/models-18-purple">
  <img alt="Best" src="https://img.shields.io/badge/best%20meanIoU-0.866-brightgreen">
  <img alt="Floor" src="https://img.shields.io/badge/strongest%20baseline-0.415-red">
</p>

---

## Overview

| Script | Role |
| :--- | :--- |
| `merge_results.py` | Stitch chunks, write a completeness manifest, strip absolute path columns |
| `evaluate.py` | IoU tiers, three baselines, **model-level convention resolution** |

> **Design contract:** the coordinate convention is chosen from a model's own predictions and the image
> dimensions. **The gold boxes are never consulted**, so the choice cannot tune the metric it precedes.

---

## Table of Contents

- [Convention Resolution](#convention-resolution)
- [Why the Median, Not a High Percentile](#why-the-median-not-a-high-percentile)
- [Metric Definitions](#metric-definitions)
- [Three Trivial Baselines](#three-trivial-baselines)
- [Results &mdash; Overall](#results--overall)
- [Results &mdash; Coordinate Conventions](#results--coordinate-conventions)
- [Results &mdash; By Target Size](#results--by-target-size)
- [Findings](#findings)
- [What the Results Imply](#what-the-results-imply)
- [Limitations](#limitations)
- [Usage](#usage)
- [FAQ](#faq)

---

## Convention Resolution

The runner resolves each reply independently and records the mode. **That is not sufficient**: on a
2400-pixel canvas, `600` could be a pixel coordinate or a 0–1000 normalised one, and no single reply
can separate them.

A model's whole set of replies can, and the signal is *where the boxes sit*:

$$
c_x = \operatorname{median}_i \frac{\hat{x}_i + \hat{w}_i/2}{W_i},
\qquad
c_y = \operatorname{median}_i \frac{\hat{y}_i + \hat{h}_i/2}{H_i}
$$

$$
\text{score} \;=\; \bigl(1 - \min(|c_x - 0.5|\cdot 2,\, 1)\bigr)\,
                   \bigl(1 - \min(|c_y - 0.5|\cdot 2,\, 1)\bigr)\,
                   \cdot \text{parse\_fraction}
$$

```
   ┌───────────────────────────────┐      ┌───────────────────────────────┐
   │  ●  ●    ●                    │      │        ●    ●       ●         │
   │      ●        ●     ●         │      │    ●        ●   ●        ●    │
   │  ●      ●                     │      │       ●   ●        ●     ●    │
   │                               │      │   ●      ●    ●        ●      │
   └───────────────────────────────┘      └───────────────────────────────┘
     WRONG convention                       CORRECT convention
     0–1000 read as pixels on a              boxes spread across the frame
     2400px canvas: nothing can              median centre ≈ (0.5, 0.5)
     be placed past 42% of width
     median centre ≈ (0.21, 0.26)
```

Only the predictions and the image dimensions enter.

---

## Why the Median, Not a High Percentile

An earlier version scored conventions by **99th-percentile coverage** &mdash; how far right and down
the boxes reached. It fails on models that **mix** conventions.

`gemma4_12B` emits:

| Value range | Share of replies | Reading |
| :--- | ---: | :--- |
| &le; 1 (0–1 fractions) | 0.0% | &mdash; |
| 1 &ndash; 1000 | **60.3%** | 0–1000 normalised |
| 1000 &ndash; 2400 | 18.4% | could be pixels |
| **&gt; 2400** | **21.2%** | pixels for an *imagined larger canvas*, e.g. `[0, 0, 4608, 3456]` |

Under the absolute reading the oversized 21% are rejected as out-of-bounds, while the surviving
minority *does* reach the edges &mdash; so a biased subset made the wrong convention look well-spread.
**The median is immune to that.**

| Model | Percentile rule chose | Its IoU | Correct convention | IoU |
| :--- | :--- | ---: | :--- | ---: |
| internvl3_38B | `abs_xywh` | 0.147 | `norm1000_xyxy` | **0.829** |
| gemma4_12B | `abs_xywh` | 0.150 | `norm1000_xywh` | **0.774** |
| internvl3_8B | `abs_xywh` | 0.254 | `norm1000_xyxy` | 0.712 |

> **Agreement with the ground-truth-optimal convention: 18/18 for the median rule, 16/18 for the
> percentile rule.** Two models moved nine places in the ranking, and Localization's correlation with
> the other C3 tasks doubled (mean &rho; 0.222 &rarr; 0.465). A parsing decision changed a scientific
> conclusion, which is why `loc_parse_modes.csv` records the chosen convention, the row-level majority,
> and the coverage score for every model.

---

## Metric Definitions

| Metric | Definition |
| :--- | :--- |
| **`meanIoU_all`** | **headline.** `meanIoU × parse_rate` &mdash; unparsed replies charged as zero |
| `meanIoU` | Over parsed replies only |
| `acc25`/`acc50`/`acc75`/`acc90` | Share at or above that IoU |
| **`centreIn`** | Prediction centre inside the gold box |
| `gtCentreIn` | Gold centre inside the prediction |
| `panelConfusion` | Multi-target only; `NaN` in this tier |

$$
\mathrm{IoU}(b^{\ast}, \hat{b}) = \frac{|b^{\ast} \cap \hat{b}|}{|b^{\ast} \cup \hat{b}|},
\qquad
\text{meanIoU\_all} = \frac{1}{N}\sum_{i=1}^{N}
\begin{cases}
\mathrm{IoU}(b_i^{\ast}, \hat{b}_i) & \hat{b}_i \text{ parsed} \\
0 & \text{otherwise}
\end{cases}
$$

**Use `meanIoU_all`.** `meanIoU` averages only over replies that produced a usable box, so a model
returning nothing parseable on half its items is scored on the easier half. `llava_next_mistral_7B`
scores **0.170** on the 52% it parses and **0.089** charged properly.

`acc50/75/90` are reported **together** because a model can be broadly right and never precise, or
occasionally exact and usually lost, and one threshold cannot tell those apart.

`centreIn` separates *knows where the animal is but cannot size the box* from *has no idea*, and is the
only column on which a pointing-trained model can be compared fairly against a box-trained one.

---

## Three Trivial Baselines

Computed **per model, on exactly the rows that model was scored on**.

| Baseline | meanIoU | acc@.25 | acc@.50 | acc@.75 | acc@.90 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| whole image | 0.373 | 0.602 | 0.295 | **0.114** | **0.044** |
| centre 50% | 0.370 | 0.782 | 0.249 | 0.014 | 0.000 |
| **mean gold box** | **0.415** | **0.784** | **0.369** | 0.025 | 0.001 |

> ⚠️ **The baselines swap rank between thresholds.** `meanBox` wins at the loose end, `wholeImage` at
> the tight end. Quoting one alone flatters the models either way. **Use the maximum of the three at
> whichever threshold is reported.**

A reported IoU that does not clear all three is not a localisation.

---

## Results &mdash; Overall

18 models &middot; ~4,000 items &middot; sorted by `meanIoU_all`.

| # | Model | `meanIoU_all` | meanIoU | acc25 | acc50 | acc75 | acc90 | centreIn | gtCentreIn | parse | convention |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | qwenVL3_32B | **0.866** | 0.866 | 0.995 | 0.984 | **0.877** | **0.460** | 0.994 | 0.993 | 1.000 | `norm1000_xyxy` |
| 2 | qwen3vl_8B | 0.862 | 0.862 | 0.995 | 0.985 | 0.868 | 0.449 | 0.994 | 0.994 | 1.000 | `norm1000_xyxy` |
| 3 | qwenVL3_30B | 0.861 | 0.861 | 0.995 | 0.984 | 0.867 | 0.444 | 0.995 | 0.995 | 1.000 | `norm1000_xyxy` |
| 4 | internvl3_38B-Instruct | 0.744 | 0.745 | 0.941 | 0.832 | 0.675 | 0.306 | 0.964 | 0.879 | 0.999 | `norm1000_xyxy` |
| 5 | llava_next_vicuna_7B | 0.701 | 0.701 | 0.977 | 0.885 | 0.444 | 0.095 | 0.983 | 0.986 | 1.000 | `norm01_xyxy` |
| 6 | gemma4_12B | 0.589 | 0.590 | 0.749 | 0.631 | 0.475 | 0.196 | 0.827 | 0.700 | 0.999 | `norm1000_xywh` |
| &mdash; | *mean gold box baseline* | *0.415* | | *0.784* | *0.369* | *0.025* | *0.001* | | | | |
| 7 | internvl3_5_38B-Instruct | 0.412 | 0.413 | 0.605 | 0.329 | 0.102 | 0.047 | 0.852 | 0.707 | 0.997 | `abs_xywh` |
| &mdash; | *whole-image baseline* | *0.373* | | *0.602* | *0.295* | *0.114* | *0.044* | | | | |
| 8 | gemma3_12B | 0.329 | 0.329 | 0.525 | 0.263 | 0.054 | 0.016 | 0.738 | 0.576 | 1.000 | `abs_xywh` |
| 9 | qwenVL2_5_7B | 0.308 | 0.309 | 0.529 | 0.210 | 0.003 | 0.000 | 0.805 | 0.687 | 0.998 | `abs_xywh` |
| 10 | qwenVL2_5_3B | 0.307 | 0.307 | 0.527 | 0.210 | 0.003 | 0.000 | 0.806 | 0.686 | 1.000 | `abs_xywh` |
| 11 | internvl3_5_30B_A3B | 0.289 | 0.290 | 0.446 | 0.174 | 0.053 | 0.033 | 0.734 | 0.440 | 0.995 | `abs_xywh` |
| 12 | internvl3_8B-Instruct | 0.246 | 0.262 | 0.399 | 0.200 | 0.086 | 0.031 | 0.601 | 0.405 | 0.940 | `abs_xywh` |
| 13 | internvl3_5_8B-Instruct | 0.236 | 0.244 | 0.376 | 0.156 | 0.032 | 0.013 | 0.627 | 0.375 | 0.965 | `abs_xywh` |
| 14 | pixtral_12B | 0.229 | 0.229 | 0.331 | 0.202 | 0.077 | 0.025 | 0.469 | 0.340 | 0.999 | `abs_xywh` |
| 15 | llava_next_mistral_7B | 0.089 | 0.170 | 0.239 | 0.171 | 0.141 | 0.053 | 0.411 | 0.211 | **0.523** | `abs_xywh` |
| 16 | llama3_2_11B | 0.061 | 0.061 | 0.017 | 0.001 | 0.001 | 0.000 | 0.482 | 0.059 | 0.996 | `abs_xywh` |
| 17 | molmo2_8B | 0.055 | 0.055 | 0.070 | 0.027 | 0.018 | 0.012 | 0.276 | 0.048 | 0.998 | `abs_xywh` |
| 18 | phi35_vision | 0.030 | 0.030 | 0.026 | 0.008 | 0.001 | 0.000 | 0.257 | 0.021 | 1.000 | `abs_xywh` |

**Six models clear the strongest baseline (0.415); twelve do not.**

---

## Results &mdash; Coordinate Conventions

`chosen` is the model-level convention; `row_majority` is what the per-row resolver concluded. **They
disagree for four models**, and in each case the model-level choice is correct.

| Model | chosen | row_majority | coverage | abs_xywh | n1000_xywh | n1000_xyxy | abs_xyxy | n01_xyxy | n01_xywh |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwenVL3_32B | **`norm1000_xyxy`** | `abs_xywh` | 0.948 | 2906 | 508 | 583 | 2 | 0 | 0 |
| qwenVL3_30B | **`norm1000_xyxy`** | `abs_xywh` | 0.951 | 2894 | 515 | 588 | 2 | 0 | 0 |
| qwen3vl_8B | **`norm1000_xyxy`** | `abs_xywh` | 0.949 | 2901 | 509 | 588 | 2 | 0 | 0 |
| internvl3_38B | **`norm1000_xyxy`** | `abs_xywh` | 0.728 | 2751 | 685 | 492 | 66 | 0 | 0 |
| gemma4_12B | **`norm1000_xywh`** | `abs_xywh` | 0.540 | 2378 | 1614 | 0 | 2 | 0 | 0 |
| llava_next_vicuna_7B | `norm01_xyxy` | `norm01_xyxy` | 0.985 | 12 | 0 | 0 | 0 | 2614 | 1374 |
| llava_next_mistral_7B | `abs_xywh` | `abs_xywh` | **0.131** | 1639 | 28 | 1 | 4 | 309 | 100 |
| gemma3_12B | `abs_xywh` | `abs_xywh` | 0.711 | 3908 | 86 | 1 | 3 | 0 | 0 |
| internvl3_5_38B | `abs_xywh` | `abs_xywh` | 0.841 | 3786 | 92 | 3 | 108 | 0 | 0 |
| molmo2_8B | `abs_xywh` | `abs_xywh` | 0.324 | 3405 | 567 | 11 | 10 | 0 | 0 |
| phi35_vision | `abs_xywh` | `abs_xywh` | 0.305 | 3666 | 152 | 51 | 82 | 21 | 26 |

> The three Qwen3-VL models show an almost identical mode histogram (2894–2906 `abs_xywh`,
> 508–515 `n1000_xywh`, 583–588 `n1000_xyxy`) &mdash; a family fingerprint. Their high coverage scores
> (0.948–0.951) are what make the model-level correction unambiguous for them.

`llava_next_mistral_7B`'s coverage of **0.131** reflects its 0.523 parse rate: fewer than half its
replies resolve under any convention.

---

## Results &mdash; By Target Size

`meanIoU` by quartile of `gt_area_frac`. Q1 median area 0.075; Q4 median 0.735.

| Model | Q1 smallest | Q2 | Q3 | Q4 largest | Q4 &minus; Q1 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| qwenVL3_32B | 0.812 | 0.873 | 0.888 | 0.890 | **+0.078** |
| qwenVL3_30B | 0.807 | 0.867 | 0.884 | 0.888 | +0.081 |
| qwen3vl_8B | 0.805 | 0.868 | 0.884 | 0.891 | +0.086 |
| internvl3_38B-Instruct | 0.660 | 0.812 | 0.778 | 0.730 | +0.070 |
| gemma4_12B | 0.571 | 0.645 | 0.553 | 0.589 | +0.018 |
| llava_next_vicuna_7B | 0.544 | 0.686 | 0.744 | 0.828 | +0.284 |
| internvl3_5_38B-Instruct | 0.258 | 0.387 | 0.422 | 0.585 | +0.327 |
| qwenVL2_5_7B | 0.259 | 0.345 | 0.306 | 0.324 | +0.065 |
| qwenVL2_5_3B | 0.259 | 0.345 | 0.305 | 0.319 | +0.060 |
| gemma3_12B | 0.120 | 0.261 | 0.393 | 0.543 | **+0.423** |
| internvl3_5_30B_A3B | 0.120 | 0.262 | 0.325 | 0.453 | +0.333 |
| internvl3_8B-Instruct | 0.087 | 0.176 | 0.249 | 0.534 | +0.447 |
| internvl3_5_8B-Instruct | 0.123 | 0.222 | 0.252 | 0.380 | +0.257 |
| **pixtral_12B** | **0.006** | 0.056 | 0.267 | 0.589 | **+0.583** |
| llava_next_mistral_7B | 0.178 | 0.140 | 0.068 | 0.293 | +0.115 |
| molmo2_8B | 0.028 | 0.032 | 0.046 | 0.113 | +0.085 |
| llama3_2_11B | 0.055 | 0.085 | 0.057 | 0.049 | &minus;0.006 |
| phi35_vision | 0.017 | 0.031 | 0.028 | 0.042 | +0.025 |

### Two size regimes

**The top three are nearly flat** (+0.078 to +0.086). They localise small subjects almost as well as
large ones &mdash; genuine scale invariance.

**`pixtral_12B` is the extreme opposite**: 0.006 on the smallest quartile, 0.589 on the largest, a
**+0.583** swing. It is competitive with `gemma4_12B` on large subjects and worse than a random box on
small ones. A pooled number would report it as uniformly mediocre and hide both facts.

Averaged across models, `centreIn` rises from **0.506** (Q1) to **0.930** (Q4) while `meanIoU` rises
only 58%: models find large subjects reliably and size them imperfectly; on small subjects they often
cannot find them at all.

---

## Findings

### 1 &mdash; A sharp divide, not a gradient

Six models exceed the strongest baseline; twelve fall below it. The largest gap in the table sits
between `gemma4_12B` (0.589) and `internvl3_5_38B` (0.412) &mdash; exactly at the baseline. The models
below cluster in the 0.23&ndash;0.33 band a mean-box predictor already achieves.

The top three reach **IoU 0.86** with `acc50` at 0.984 and `centreIn` at 0.994 &mdash; the only C3 task
where leading models approach saturation on the loose metric while leaving room at the tight end
(`acc90` peaks at 0.460).

`acc75` is where the task discriminates: it separates the top three (0.867&ndash;0.877) from
`internvl3_38B` (0.675) from `gemma4_12B` (0.475), a gradient invisible in `acc50` where the top four
all exceed 0.83.

### 2 &mdash; Two distinct failure modes at the bottom

| Model | `meanIoU_all` | parse | Cause |
| :--- | ---: | ---: | :--- |
| llava_next_mistral_7B | 0.089 | **0.523** | **parse failure** &mdash; nearly half its replies contain no resolvable box. On what it does parse it reaches 0.170. |
| molmo2_8B | 0.055 | 0.998 | **prompt echo** &mdash; repeats the worked example on **61.9%** of items |
| phi35_vision | 0.030 | 1.000 | **prompt echo** &mdash; **75.8%** |

Both parse essentially everything and still score near zero. The raw outputs explain why:

| Model | Replies exactly equal to `[1240, 88, 540, 410]` |
| :--- | ---: |
| phi35_vision | **75.8%** |
| molmo2_8B | **61.9%** |
| llava_next_mistral_7B | 37.2% |
| llama3_2_11B | 29.7% |
| internvl3_5_8B-Instruct | 13.0% |

The gold base rate of that exact box is **0.0%**.

> `molmo2_8B` is explicitly **pointing-trained** and should be strong at grounding. Its last-place
> finish is an *instruction-following* failure, not a grounding failure, and reporting it as poor
> localisation would be a misattribution.

### 3 &mdash; `centreIn` and `acc50` diverge

| Model | centreIn | acc50 | gap |
| :--- | ---: | ---: | ---: |
| qwenVL2_5_7B | 0.805 | 0.210 | **+0.595** |
| qwenVL2_5_3B | 0.806 | 0.210 | +0.596 |
| internvl3_5_38B | 0.852 | 0.329 | +0.523 |
| llama3_2_11B | 0.482 | 0.001 | +0.481 |
| qwenVL3_32B | 0.994 | 0.984 | +0.010 |

The Qwen2.5 pair place the box centre inside the animal **80%** of the time yet clear IoU 0.5 only 21%
of the time. They know where the animal is and cannot size the box. That is a different deficit from
`molmo2_8B` (centreIn 0.276), which does not find it at all.

---

## What the Results Imply

**Spatial grounding is present in some models and absent in others, with an unusually sharp divide.**
There is no gradual gradient across the eighteen &mdash; either a model clears the mean-box baseline
comfortably or it sits in the band that baseline already occupies.

**Coordinate convention is a first-class experimental variable, not a parsing detail.** Three models
were assigned the wrong frame by a plausible heuristic, and the resulting ranking would have supported
a materially different claim about C3's internal structure. Any benchmark reporting bounding boxes
across model families must state how it resolves conventions and demonstrate the resolution is correct.

**Worked examples in prompts can dominate a model's output.** Five of eighteen echoed the example on
&ge;13% of items, two on more than 60%. The affected scores are not ability estimates. This is a
general caution for prompt design, and it is measurable only because the raw-output column was retained.

**High baselines constrain what this corpus can measure.** With the subject filling 37% of the frame on
average, single-animal wildlife photography is close to a solved framing problem, and mean IoU barely
discriminates. A corpus with smaller or off-centre subjects would be a stronger test.

---

## Limitations

| Limitation | Effect |
| :--- | :--- |
| **Selection toward large files** | Candidates ranked by file size descending; not a random sample. |
| **Single-annotation images only** | Removes referential ambiguity but also removes the multi-target confusion diagnostic. |
| **High trivial baselines** | Mean IoU is weakly discriminative; report `acc@.75` / `acc@.90`. |
| **12 out-of-bounds gold boxes** | Small negative offsets from annotation rounding; excluded and counted. |
| **Prompt echo** | Contaminates at least two models' scores; footnote rather than read as grounding ability. |

---

## Usage

```bash
python merge_results.py --out_dir <RESULTS_DIR>

python evaluate.py --results_dir <RESULTS_DIR> \
                   --out_dir     <RESULTS_DIR> \
                   --bootstrap 2000
# --no_consistent_mode  falls back to per-row convention scoring
```

### Output files

| File | Contents |
| :--- | :--- |
| `loc_overall.csv` | One row per model, all metrics and all three baselines |
| `loc_by_group.csv` | Per tier |
| `loc_by_size.csv` | Per model &times; target-size quartile |
| `loc_parse_modes.csv` | Chosen convention, row majority, coverage score, mode histogram |
| `loc_report.md` | Generated markdown summary |
| `merge_manifest.json` | Completeness record |

### Reading order

1. **`loc_parse_modes.csv` first, before any score.** Any model where `chosen ≠ row_majority` had its
   convention corrected at the model level &mdash; the difference between IoU 0.15 and 0.83 for two
   models here.
2. **`parse_rate`.** Below ~0.95, `meanIoU` is computed on a favourable subset; use `meanIoU_all`.
3. **`acc75` / `acc90` against the baseline maximum.** Not mean IoU.
4. **`centreIn` versus `acc50`.** A large gap is a sizing failure, not a grounding failure.
