<h1 align="center">Bio-VQA &rarr; Fine-Grained Visual Attribute Grounding</h1>
<p align="center">
  <b>CUB-200-2011 &mdash; Graded &amp; Exact-Match Evaluation of Vision&ndash;Language Models</b>
</p>
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="Source dataset" src="https://img.shields.io/badge/source-CUB--200--2011-green">
  <img alt="Models" src="https://img.shields.io/badge/models-18-orange">
  <img alt="Questions" src="https://img.shields.io/badge/questions-824K-purple">
  <img alt="Metric" src="https://img.shields.io/badge/metric-CAGS%20%2B%20EM-red">
  <img alt="Release" src="https://img.shields.io/badge/release-v1.0-brightgreen">
</p>
<p align="center">
  A closeness-aware, chance-corrected evaluation harness for the <b>fine-grained visual<br>
  attribute grounding axis</b> of BioVQA &mdash; it grades <i>how</i> a model is wrong, not just<br>
  how often, and proves the ranking is a fact about models, not about the metric.
</p>

---

## Table of Contents

1. [Overview](#1-overview)
2. [Design Principles](#2-design-principles)
3. [The Corpus: Categories and Question Families](#3-the-corpus-categories-and-question-families)
4. [Two Metrics, and Why: Exact Match vs. CAGS](#4-two-metrics-and-why-exact-match-vs-cags)
5. [The Scoring Pipeline](#5-the-scoring-pipeline)
6. [The Similarity Kernels — Equations and Worked Examples](#6-the-similarity-kernels--equations-and-worked-examples)
7. [Chance Adjustment: from Raw Similarity to CAGS](#7-chance-adjustment-from-raw-similarity-to-cags)
8. [Uncertainty, Severity, and Format Diagnostics](#8-uncertainty-severity-and-format-diagnostics)
9. [A Full End-to-End Worked Example](#9-a-full-end-to-end-worked-example)
10. [Metric Reference Card](#10-metric-reference-card)
11. [Experimental Setup](#11-experimental-setup)
12. [Results](#12-results)
13. [Robustness: the Sensitivity Sweep](#13-robustness-the-sensitivity-sweep)
14. [Discussion and Threats to Validity](#14-discussion-and-threats-to-validity)
15. [Reproducibility](#15-reproducibility)
16. [Appendix A — Full Per-Model × Per-Family Tables](#appendix-a--full-per-model--per-family-tables)
17. [Appendix B — Constants and Lexicons](#appendix-b--constants-and-lexicons)
18. [Appendix C — Glossary and Per-Family Example Catalogue](#appendix-c--glossary-and-per-family-example-catalogue)

---
## 1. Overview

This is the **evaluation pipeline** for BioVQA's fine-grained visual attribute grounding axis (built on CUB-200-2011). The corpus and inference runners are documented separately — this document covers only **how model answers are scored**.

This axis grades 18 candidate **vision–language models (VLMs)** on its **824,104 questions per model** and turns raw answers into a leaderboard. It is evaluated over **18 open-weight VLMs** spanning roughly 3B–38B parameters, and it contributes one of BioVQA's most differentiated tasks — a **novel visually-prompted part localization** family in which a coloured symbolic marker, placed amid adversarial distractors, must be bound to a body part by colour and shape.

The axis spans five competencies mapped onto four scoring categories — **nomenclature** (*Taxonomy & Naming*), **morphological attribute recognition** (*Morphological Attributes*), **relative trait salience** (*Relative Salience*), and **anatomical part visibility + visually-prompted localization** (*Anatomical Localization*) — mixing multiple-choice items with open-ended naming. The evaluation reports **two metrics side by side**: a strict **Exact Match (EM)** that anchors the benchmark, and a graded **CAGS (Chance-Adjusted Graded Score)** that awards partial credit for being *close* and then subtracts the credit a confused guesser would have earned by chance. Where the two disagree is where the science lives.


```
                   ┌──────────────────────────────────────────────────────────────────────────────┐
                   │   QA_<model>.csv        raw predictions, one <tag>_answer column per model   │
                   └──────────────────────────────────────────────────────────────────────────────┘
                                                        │
                              ┌─────────────────────────┴─────────────────────────┐
                              ▼                                                   ▼
                   ┌───────────────────────────────┐               ┌───────────────────────────────┐
                   │   evaluate_soft.py            │               │   evaluate.py                 │
                   │   graded / CAGS scoring       │               │   exact-match + diagnostics   │
                   └───────────────────────────────┘               └───────────────────────────────┘
                              │                                                   │
                              ▼                                                   ▼
                   ┌───────────────────────────────┐               ┌───────────────────────────────┐
                   │  <model>/Evaluation/          │               │  <model>/Evaluation/          │
                   │  eval_soft_*  per-item CAGS,  │               │  eval_*  exact match, format  │
                   │  family / category tables     │               │  & position-bias diagnostics  │
                   └───────────────────────────────┘               └───────────────────────────────┘
                              │                                                   │
                              ▼                                                   ▼
                   ┌───────────────────────────────┐               ┌───────────────────────────────┐
                   │  Results/Eval - CAGS/         │               │  Results/Eval - Exact Match/  │
                   │  leaderboard · by-family ·    │               │  leaderboard · by-family ·    │
                   │  by-category · stats          │               │  by-category · format         │
                   └───────────────────────────────┘               └───────────────────────────────┘
                              │
                              ▼
                   ┌───────────────────────────────────────────────────────────────────────────────┐
                   │   sensitivity_sweep.py   ──►   Results/Eval - CAGS/sweep_*                    │
                   │   ranking stability vs. the metric's free constants (τ, α, kernel, policy)    │
                   └───────────────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
                   ┌───────────────────────────────────────────────────────────────────────────────┐
                   │   setup_manual_review_*   ──►   Results/Manual Review/                        │
                   │   blind, stratified human validation of the CAGS metric                       │
                   └───────────────────────────────────────────────────────────────────────────────┘

```

> **The measurement problem this axis solves.** Fine-grained attribute questions are *near-miss-rich*: "wing is **blue**" vs. "wing is **purple**" is a small perceptual error; "marked part is the **nape**" vs. "**crown**" is a small spatial error; "**song sparrow**" vs. "**fox sparrow**" shares a genus and a head-noun. A metric that scores all of these as a flat **0** discards most of the signal that separates a good model from a bad one, and conflates *being slightly wrong* with *being completely lost*. CAGS is designed to recover exactly that signal.

---

## 2. Design Principles

**Two metrics, always reported together.** EM is the conservative anchor everyone already trusts; CAGS is the diagnostic instrument. Neither is presented alone, because their *agreement* validates the ranking and their *disagreement* localizes model behaviour (Section 12.7).

**Closeness is family-specific.** A near miss on colour is a perceptual-distance question; on a body part, a 3-D-layout question; on size, an ordinal-scale question; on a species name, a taxonomic-decomposition question. CAGS therefore dispatches a **different kernel per question family** (Section 6), rather than pretending one notion of "close" fits all.

**Difficulty-awareness by construction.** Every graded score is corrected against the *question's own distractors*. A question whose wrong options are all almost-correct is intrinsically easier to look good on; CAGS subtracts that advantage per item, so the metric measures competence *above chance for that specific question*, not raw string overlap.

**Determinism and auditability.** Model discovery, option parsing, gold parsing, prediction resolution, and the family router are shared verbatim between the CAGS and EM evaluators, so both metrics partition the 824,104 items identically. Every per-item quantity (`s_raw`, `chance_b`, `cags`, `severity`, `resolve_status`) is written to disk and auditable against the raw prediction.

**Robustness is proven, not asserted.** The metric's free constants ($\tau$, $\alpha$, kernel shape, two policy toggles) are swept, and the ranking's invariance is quantified with rank correlations, displacement, and top-set stability (Section 13). A benchmark ranking that moved with its own hyper-parameters would be a fact about the metric, not the models.

---

## 3. The Corpus: Categories and Question Families

This axis's 824,104 questions decompose into **four categories** (coarse axes of visual skill) and **ten question families** (concrete templates, each with its own answer space and its own natural notion of a near miss). Counts are exact, per model.

| Category | Family | # Questions | Answer space | Format |
| --- | --- | ---: | --- | --- |
| **Morphological Attributes** | `attr_bare_value` | 280,433 | attribute value (colour / pattern / size / bill length) | MCQ |
| **Relative Salience** | `attr_group_value` | 321,073 | `region: value` trait | MCQ |
| **Anatomical Localization** | `marker_localization` | 11,788 | a named body part | MCQ |
| | `visibility_partset` | 117,880 | a set of visible parts | MCQ |
| | `visibility_side` | 35,364 | Left / Right / Both / None | MCQ |
| **Taxonomy & Naming** | `category_mcq` | 11,191 | coarse bird type | MCQ |
| | `open_common_name` | 11,788 | common (English) species name | open-ended |
| | `open_scientific_name` | 11,668 | Latin binomial | open-ended |
| | `open_genus` | 11,728 | genus | open-ended |
| | `open_kind_of` | 11,191 | folk category | open-ended |
| **Total** | | **824,104** | | |

> **Two structural facts drive the entire design.** (1) ~94% of items are MCQ (2–5 distractors); ~6% are open-ended naming that must be *generated*. The gap between a model's MCQ and open-ended competence is one of the most informative quantities in the benchmark (Section 12.5). (2) "Close" means something different in every family — which is precisely why one flat exact-match number is inadequate, and why CAGS uses a per-family kernel.

**Representative rows** (from the fine-grained axis corpus, verbatim):

```csv
category,question,answer
Taxonomy & Naming,What is the scientific name of the bird?,Phoebastria nigripes
Morphological Attributes,"What is the eye color of this bird? Options: (A) purple, (B) orange, (C) blue, (D) white, (E) brown",(E) brown
Relative Salience,"…most prominently visible physical trait? Options: (A) throat color: buff, (B) eye color: red, (C) leg color: olive, …",(A) throat color: buff
Anatomical Localization,"What is the body part of this bird marked with the yellow star? Options: (A) crown, (B) left eye, (C) right leg, (D) left leg, (E) belly",(A) crown
```

The last row is a **visually-prompted localization** item — a star-annotated image derivative — and is the axis's **novel visually-prompted part localization** task — one of BioVQA's most differentiated contributions relative to prior fine-grained benchmarks: a coloured symbolic marker, placed amid adversarial distractors, must be bound to a *named body part* by colour and shape.

### 3.1 The four categories in plain terms

- **Morphological Attributes** (`attr_bare_value`) — "What is the ⟨region⟩ ⟨property⟩ of this bird?" Bare-value options; property ∈ {colour, pattern, size, bill length}. Answering **purple** for **blue** is a perceptual near miss; **yellow** is far.
- **Relative Salience** (`attr_group_value`) — "What is the most prominently visible physical trait?" Options are full `region: value` traits, so the model must get the *region*, the *value*, **and** the *salience judgment* right. `breast color: blue` for gold `wing color: blue` is a region miss; `wing color: purple` is a value miss — different errors, scored differently.
- **Anatomical Localization** — three spatial families: `marker_localization` (name the part under a marker; crown-for-nape is a small error, tail-for-nape a large one), `visibility_partset` (the answer is a *set*; partial overlap earns partial credit), and `visibility_side` (Left/Right/Both/None; "Both" overlaps "Left" but not "None").
- **Taxonomy & Naming** — from an easy coarse MCQ (`category_mcq`) to hard open generation (`open_common_name` → *song sparrow*, `open_scientific_name` → *Melospiza melodia*, `open_genus` → *Melospiza*, `open_kind_of` → folk category). The naming kernels reward a shared genus, head-noun, or epithet, so *fox sparrow* for *song sparrow* is a near miss while *American crow* is not.

---

## 4. Two Metrics, and Why: Exact Match vs. CAGS

### 4.1 Why keep a hard Exact-Match metric

A prediction is EM-correct only if, after light normalization, it equals the gold answer (for MCQ, if the chosen **letter** equals the gold letter). EM is retained because it is (i) the metric the field already trusts and the one that makes this axis comparable to prior work; (ii) an unambiguous *lower bound* on competence — no partial-credit machinery to dispute; and (iii) the natural home for **format diagnostics** (parse rate, letter/text mismatch, position bias) that reveal when an accuracy number is inflated or deflated by *formatting* rather than *seeing* (Section 8.3).

### 4.2 Why EM alone is insufficient — and what CAGS adds

EM has two well-known failure modes on fine-grained data:

- **It cannot separate a near miss from a total miss.** `purple`-for-`blue` scores the same 0 as `yellow`-for-`blue`. On a 200-way naming task nearly every wrong answer is a flat 0, so a model that reliably names the right *genus* looks identical to one that names random birds.
- **It ignores question difficulty.** A 2-option and a 5-option question both contribute a binary outcome, though random guessing yields 50% vs. 20%. A question whose distractors are all almost the correct colour is far easier than one whose distractors are wildly different — EM cannot tell.

CAGS repairs both. For each item it computes a graded raw similarity $s \in [0,1]$ with the family-appropriate kernel, then subtracts the **chance baseline** $b$ (the mean similarity the item's own distractors would earn) and renormalizes. Reported **unclipped**, its anchors are $\mathrm{CAGS}=1$ (perfect), $\mathrm{CAGS}=0$ (no better than the distractor pool), and $\mathrm{CAGS}<0$ (*worse than chance* — confidently, specifically wrong).

> **In one line.** EM says how often a model is exactly right; CAGS says how good it is when graded for closeness and corrected for chance. Reporting both is what converts a table of accuracies into a characterization of *how each model fails*.

---

## 5. The Scoring Pipeline

Every question flows through one deterministic pipeline (`evaluate_soft.py`); the kernels of Section 6 plug into step (5).

```
                           ┌──────────────────────────────────────────────────────────┐
         raw CSV row  ──►  │  q = question,  a = gold answer,  p = model prediction   │
                           └──────────────────────────────────────────────────────────┘
                                                  │
                      (1) parse_options(q)                ──►  options = {A: "...", B: "...", ...}  or  {} (open-ended)
                                                  │
                      (2) question_family(q, options)     ──►  fam ∈ {attr_bare_value, ..., open_scientific_name}
                                                  │
                      (3) parse_gold(a, options)          ──►  (gold_letter, gold_text)
                                                  │
                      (4) resolve_prediction(p, options)  ──►  (pred_letter, pred_text, status)
                                                  │
                      (5) s_raw = score_pair(fam, pred_text, gold_text, options, anat, cfg)   ∈ [0,1]
                                                  │
                      (6) b = chance_baseline(fam, gold_text, options / open_pool, ...)       ∈ [0,1] or None
                                                  │
                      (7) cags = (s_raw - b) / (1 - b)         [unclipped]
                                                  │
                      (8) exact = 1[ norm(pred_text) == norm(gold_text) ]
                                                  │
      ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
      │         record { family, category, gold/pred, status, exact, s_raw, b, cags, severity }      │
      └──────────────────────────────────────────────────────────────────────────────────────────────┘
 
```

**(1) Normalization** — `_norm` lower-cases, maps `_→space` and `gray→grey`, strips punctuation except `:-()`, and collapses whitespace, so `Blue_Jay`, `blue jay`, and `blue jay.` are identical.

**(2) Option parsing** — the presence of the literal token `Options:` routes an item into the MCQ branch (an ordered `{letter: text}` map) or the open-ended branch (empty map).

**(3) Gold parsing** — handles gold answers stored either as `(A) text` or as a bare value.

**(4) Prediction resolution** — the most consequential step. `resolve_prediction` is deliberately permissive so that *visual* competence is never punished for *formatting* quirks. In order: strip conversational scaffolding and keep the first line → detect an explicit `(A)`/`A.`/`A)` letter → detect verbatim option text → **reconcile** (agree ⇒ `letter`; disagree ⇒ `letter_text_mismatch`, broken by `--prefer`, default letter) → **fuzzy** `difflib` fallback (≥ 0.86) → else `no_parse` (scored 0). The diagnostic rates of Section 8.3 all fall out of this step; in our runs every model parses at ≥ 98.9%.

**(5) Family router** — matches template phrases in priority order (marker phrase → `marker_localization`; "most prominently visible physical trait" → `attr_group_value`; "are visible in this image" with `{left,right,both,none}` → `visibility_side` else `visibility_partset`; "what type of bird" → `category_mcq`; "common/scientific/genus name" → the naming families; "what kind of" → `open_kind_of`; bare "what is the … of this bird?" with options → `attr_bare_value`). Used identically by both evaluators.

---

## 6. The Similarity Kernels — Equations and Worked Examples

The graded similarity $s$, computed by `score_pair(family, pred, gold, …)`, dispatches on the question family. Every value in the "worked examples" is the **actual output of the code** for that pair under the default config (exponential colour kernel, $\tau=30$, $\alpha=0.5$, schematic anatomy for illustration).


### 6.1 Colour kernel (perceptual, CIEDE2000)

Colour terms are compared **perceptually**, not lexically. Each of the 15 CUB colour words is anchored to an sRGB triple and pushed through the standard colour pipeline — **sRGB → (inverse gamma) linear RGB → (matrix M) XYZ → (D65 white point) CIE-Lab** — then compared with **CIEDE2000** $\Delta E_{00}$ (Sharma, Wu & Dalal, 2005). The gamma expansion is $c_{\mathrm{lin}} = c/12.92$ if $c \le 0.04045$ else $((c+0.055)/1.055)^{2.4}$; the Lab nonlinearity is $f(t)=t^{1/3}$ if $t>0.008856$ else $7.787\,t + 16/116$. The distance maps to a similarity by one of two kernels:

$$
\mathrm{color}(c_1,c_2) =
\begin{cases}
1 & c_1 = c_2 \\
0 & c_1 \text{ or } c_2 = \mathrm{iridescent} \\
\exp(-\Delta E_{00}/\tau) & \text{exponential (default, } \tau=30) \\
\max(0,\ 1 - \Delta E_{00}/\Delta E_{\max}) & \text{linear}
\end{cases}
$$

where $\Delta E_{\max}$ is the largest pairwise anchor distance; `iridescent` has no fixed hue and is exact-match-only.

| pair | exp ($\tau=30$) | linear |
| --- | ---: | ---: |
| blue vs **purple** | **0.498** | 0.791 |
| blue vs green | 0.134 | 0.396 |
| blue vs yellow | 0.053 | 0.120 |
| brown vs **rufous** | **0.760** | 0.918 |
| yellow vs orange | 0.507 | 0.796 |
| black vs white | 0.036 | 0.000 |

> **Implication.** The exponential kernel decays faster with distance than the linear one; both are swept in Section 13 to show the ranking is invariant to that choice. Perceptual scoring is what lets `attr_bare_value` distinguish "purple-for-blue" (near) from "yellow-for-blue" (far) — an EM impossibility.

### 6.2 Ordinal kernel (bill length, size)

Ordered scales are scored by normalized rank distance:

$$\mathrm{ord}(v_1,v_2) = 1 - \frac{|\,\mathrm{rank}(v_1) - \mathrm{rank}(v_2)\,|}{\mathrm{rank}_{\max}}.$$

*Examples.* bill length `shorter` vs `about the same` $= 0.500$; `shorter` vs `longer` $= 0.000$. size `small` vs `medium` $= 0.750$; `very small` vs `large` $= 0.250$. An off-by-one size error keeps 75% of its credit; a maximal error keeps none.

### 6.3 Pattern kernel (nominal confusion)

Patterns are unordered; the confusion matrix encodes one qualitative fact — the structured patterns are mutually confusable, but confusing any of them with **solid** ("has markings" vs. "has none") is categorical:

$$
\mathrm{pattern}(v_1,v_2)=
\begin{cases}
0.5 & \{v_1,v_2\}\subseteq\{\mathrm{spotted,\ striped,\ multicolored}\} \\
0.0 & \mathrm{solid} \in \{v_1,v_2\}
\end{cases}
$$

*Examples.* spotted vs striped $= 0.500$; solid vs spotted $= 0.000$.

### 6.4 Anatomy kernel (3-D part layout)

Body parts live at coordinates in a 3-D layout (facing right, $y$ down, $z$ a small lateral offset that keeps left/right distinct). Part similarity is normalized proximity, and region similarity compares part-set centroids:

$$\mathrm{part}(p_1,p_2)=\max\!\Big(0,\ 1-\frac{|\,x_{p_1}-x_{p_2}\,|}{d_{\max}}\Big),\qquad d_{\max}=\text{layout diameter}.$$

The layout is **schematic** (fixed template) or **empirical** — per part, the median bounding-box-normalized $(x,y)$ over all images where the part is visible in CUB's `part_locs.txt`. *The reported runs use the empirical layout.*

| pair | part_sim |
| --- | ---: |
| nape vs **crown** (adjacent) | **0.922** |
| beak vs forehead | 0.857 |
| left wing vs right wing | 0.883 |
| beak vs **tail** (opposite ends) | **0.000** |
| crown vs belly | 0.430 |

> **Implication.** This kernel is what makes `marker_localization` reward "crown for nape" (0.922) far above "tail for nape" (0.000). Under EM both are a flat 0, erasing the difference between a model that *almost* localized the marker and one that was lost.

### 6.5 Paired-attribute kernel (Relative Salience)

`attr_group_value` options are full `region: value` traits, so a near miss can be in the region, the value, or both. The kernel factorizes region agreement $g$ and value similarity $v$:

$$\mathrm{pair} = \alpha\, g(\mathrm{region}_1,\mathrm{region}_2) + (1-\alpha)\, v(\mathrm{value}_1,\mathrm{value}_2),\qquad \alpha=0.5,$$

where $g$ blends the anatomy region-similarity with a property-match indicator, and $v$ dispatches to the colour / ordinal / pattern kernels. *Examples.* `[wing color: blue]` vs `[wing color: purple]` $= 0.749$ (same region, near colour); vs `[breast color: blue]` $= 0.952$ (same value, near region); vs `[tail color: yellow]` $= 0.395$ (both off). $\alpha$ trades region credit against value credit and is swept in Section 13.

### 6.6 Set-overlap kernel (visibility)

`visibility_side` and `visibility_partset` are set-valued, scored by a **soft Jaccard**:

$$\mathrm{sJac}(A,B)=\frac{\sum_{a\in A}\max_{b\in B}\ \kappa(a,b)}{|A|+|B|-\sum_{a\in A}\max_{b\in B}\ \kappa(a,b)},$$

with element kernel $\kappa$ = exact match (for sides, and for `visibility_partset` by default) or the anatomy part-similarity (the `anat_in_vis` toggle, Section 13). *Examples (side).* both vs left $= 0.500$; both vs none $= 0.000$; left vs right $= 0.000$; left vs left $= 1.000$.

### 6.7 Naming kernels (common / scientific / genus)

Open-ended naming decomposes the name and rewards taxonomic overlap. With token-$F_1$ defined as the harmonic mean of token precision and recall,

$$
\begin{aligned}
\mathrm{common}(p,g) &= 0.6\cdot F_1(p,g) + 0.4\cdot \mathbf{1}[\,\mathrm{head}(p)=\mathrm{head}(g)\,],\ \text{lifted to } \ge 0.75 \text{ if same genus} \\
\mathrm{sci}(p,g) &= 0.5\cdot \mathrm{genusAgree} + 0.5\cdot \mathrm{epithetAgree} \\
\mathrm{genus}(p,g) &= \mathbf{1}[\,\mathrm{genus}(p)=\mathrm{genus}(g)\,]\ \text{else a down-weighted string ratio}
\end{aligned}
$$

*Examples.* common `song sparrow` vs `fox sparrow` $= 0.700$ (shared head-noun + token overlap); vs `american crow` $= 0.000$. scientific `Melospiza melodia` vs `Melospiza georgiana` $= 0.575$ (same genus, different epithet); vs `Corvus brachyrhynchos` $= 0.054$. genus `Melospiza` vs `Melospiza` $= 1.000$, vs `Passerella` $= 0.000$.

> **Implication.** A model that reliably names the right genus scores far above one that names random species — even when neither gets the exact binomial. This is the mechanism behind the recognition–generation analysis (Section 12.5), and it is invisible to EM.

---

## 7. Chance Adjustment: from Raw Similarity to CAGS

### 7.1 The chance baseline $b$

For each item, the chance baseline is the mean raw similarity of the item's own distractors to the gold answer:

$$b = \frac{1}{|D|}\sum_{d\in D}\ \mathrm{score\_pair}(\mathrm{fam},\, d,\, g,\, \dots),$$

where for MCQ, $D$ = the non-gold options; for open-ended items, $D$ = a fixed-seed random sample (default 40) of other gold answers seen for that family. $b$ is the *difficulty-aware chance level* of that specific question.

### 7.2 The CAGS transform

Computed **per item**, then averaged:

$$
\mathrm{CAGS}_i = \frac{s_i - b_i}{1 - b_i}, \qquad
\overline{\mathrm{CAGS}} = \frac{1}{N}\sum_{i=1}^{N}\mathrm{CAGS}_i,
$$

with the convention $\mathrm{CAGS}_i = s_i$ when $b_i$ is undefined or $1-b_i<\varepsilon$. Reported **unclipped**, so $\mathrm{CAGS}\in(-\infty,1]$ with anchors:

- $\mathrm{CAGS}=1$ — perfect answer.
- $\mathrm{CAGS}=0$ — exactly at the distractor-pool chance level.
- $\mathrm{CAGS}<0$ — *worse than chance*: farther from the gold than the average distractor.

Because the correction is per item, a question whose distractors are all near the gold (high $b$) yields little CAGS even for a decent answer, while a question with far distractors (low $b$) rewards a good answer fully. This is the exact mechanism by which CAGS is difficulty-aware where EM is not.

### 7.3 Reading $\mathrm{GS}_{\mathrm{raw}}$ and CAGS together

The mean raw score $\mathrm{GS}_{\mathrm{raw}} = \frac1N\sum_i s_i$ is reported alongside. The gap $\mathrm{GS}_{\mathrm{raw}} - \overline{\mathrm{CAGS}}$ is large exactly when a family's questions are "easy to look good on" (high $b$) — itself a diagnostic of question design, not just of the model.

---

## 8. Uncertainty, Severity, and Format Diagnostics

### 8.1 Bootstrap confidence intervals

Each mean CAGS carries a 95% bootstrap interval. Resampling the per-item CAGS with replacement $B=1000$ times and recomputing the mean:

$$[\mathrm{CAGS}_{\mathrm{lo}},\ \mathrm{CAGS}_{\mathrm{hi}}] = \Big[\,Q_{0.025}\big(\{\bar c^{(b)}\}\big),\ Q_{0.975}\big(\{\bar c^{(b)}\}\big)\,\Big].$$

> **Interpretation rule used throughout.** Two models whose CAGS intervals **overlap** are statistically tied on that slice; a bare ranking difference between them is not, on its own, meaningful. This is the standard we apply when deciding whether a sweep rank-swap is "real" (Section 13).

### 8.2 Mistake severity

Averaged over **wrong items only**:

$$\mathrm{Severity} = \frac{1}{N_{\text{wrong}}}\sum_{i:\ \mathrm{EM}_i=0}\big(1 - \mathrm{CAGS}_i\big),$$

where $N_{\text{wrong}}$ is the number of items the model got exactly wrong ($\mathrm{EM}_i=0$). $\mathrm{Severity}\le 1$ ⇒ the average miss still beats chance (near misses). $\mathrm{Severity}>1$ ⇒ **the average miss is worse than a random distractor** — a confidently, specifically wrong model. Severity $>1$ is not a defect of the metric; it is one of its most telling outputs (Section 12.6).

### 8.3 Format diagnostics — and what `chose_A` means

The exact-match evaluator reports three MCQ format diagnostics. They exist because an accuracy number can be moved by *how a model formats answers* rather than *what it sees*, and a benchmark should surface that rather than bake it in.

| Diagnostic | Definition | Why it matters |
| --- | --- | --- |
| `parse_rate` | fraction of MCQ predictions with `status ≠ no_parse` | A low parse rate means accuracy is depressed by *unreadable* output, not wrong reasoning. Ours are ≥ 98.9%, so accuracy reflects vision, not parsing. |
| `letter_text_mismatch` | fraction where the chosen letter and the spelled-out option disagree | Flags models that emit "(A) … blue" while meaning a different option — an instruction-following failure that would silently corrupt letter-based scoring. |
| **`chose_A`** | **fraction of MCQ answers equal to option (A)** | **Detects position bias — a prior over answer *positions* rather than *content*.** |

> **Why `chose_A` matters, in detail.** In a well-behaved model on this corpus (mostly 5-way MCQ), the answer distribution over letters should be roughly uniform, i.e. each letter near $1/k \approx 20\%$, because the gold letter is balanced across positions by construction. A model that answers **(A)** far above $20\%$ is exhibiting **letter-position bias**: it is partly guessing the *first option* regardless of the image. This inflates or deflates accuracy for reasons unrelated to visual competence, and it is a known failure mode of MLLMs on multiple choice. Reporting `chose_A` makes this legible on the leaderboard — and it is a direct argument for a graded, chance-corrected score that is *harder to game* by position priors. The extreme case here is **Llama-3.2-Vision at 63.26% (≈ 3× uniform)**; see Section 12.5.

---

## 9. A Full End-to-End Worked Example

**Q:** "What is the wing color of this bird? Options: (A) blue, (B) purple, (C) yellow, (D) brown." · **Gold:** (A) blue · **Prediction:** "purple" · family `attr_bare_value`.

1. `parse_options` → `{A: blue, B: purple, C: yellow, D: brown}`; `question_family` → `attr_bare_value` (implicit group "wing color").
2. `resolve_prediction("purple")` → text matches option **B** → `(B, purple, text)`.
3. **Raw similarity:** $s = \mathrm{color}(\text{purple},\text{blue}) = \mathbf{0.498}$.
4. **Chance baseline:** distractor similarities to blue are purple $=0.498$, yellow $=0.053$, brown $=0.196$, so
   $$b = \tfrac{1}{3}(0.498 + 0.053 + 0.196) = \mathbf{0.249}.$$
5. **CAGS:**
   $$\mathrm{CAGS} = \frac{0.498 - 0.249}{1 - 0.249} = \mathbf{0.331}.$$
6. **Exact match:** $\mathrm{norm(purple)} \ne \mathrm{norm(blue)} \Rightarrow \mathrm{EM}=0$. **Severity** $= 1 - 0.331 = 0.669$ (a mild miss).

> **The distinction, multiplied 824,104 times.** EM records this as a flat failure; CAGS records it as a *decent* answer that beats the question's chance level; severity confirms the miss was mild. Aggregated across 18 models, this is why the CAGS leaderboard resolves models that EM ties (Section 12.7).

---

## 10. Metric Reference Card

| Symbol | Name | Definition |
| --- | --- | --- |
| $\mathrm{EM}$ | Exact match | $\mathbf{1}[\ \mathrm{norm(pred)} = \mathrm{norm(gold)}\ ]$ (MCQ: chosen letter = gold letter) |
| $s$ | Raw graded similarity | family kernel output in $[0,1]$ |
| $\mathrm{GS}_{\mathrm{raw}}$ | Mean raw score | $\tfrac1N\sum_i s_i$ |
| $b$ | Chance baseline | mean $s$ of the item's distractors to the gold |
| $\mathrm{CAGS}$ | Chance-adjusted graded score | $\tfrac1N\sum_i (s_i-b_i)/(1-b_i)$, unclipped |
| $\mathrm{CAGS}_{\mathrm{lo/hi}}$ | 95% bootstrap CI | percentiles of the resampled mean |
| $\mathrm{Severity}$ | Miss badness | $\frac{1}{N_{\text{wrong}}}\sum_{i:\,\mathrm{EM}_i=0}(1-\mathrm{CAGS}_i)$; $>1$ ⇒ worse than chance |
| `parse_rate` | Parse success | fraction with `status ≠ no_parse` |
| `letter_text_mismatch` | Format inconsistency | fraction with `status = letter_text_mismatch` |
| `chose_A` | Position bias | fraction of MCQ answers equal to option (A) |

---

## 11. Experimental Setup

- **Data.** CUB-200-2011, 824,104 questions per model. **Empirical** anatomy from `parts/part_locs.txt` + `bounding_boxes.txt`; a **200-entry** `taxon_name_mapping.json` supplies genus lookups for the naming kernels (both loaded successfully in the reported run).
- **Models (18).** Qwen3-VL {32B, 30B, 8B}; Qwen2.5-VL {7B, 3B}; InternVL3 {38B, 8B}; InternVL3.5 {38B, 30B-A3B, 8B}; Gemma {4-12B, 3-12B}; Molmo2-8B; Pixtral-12B; Phi-3.5-Vision; Llama-3.2-Vision-11B; LLaVA-NeXT {Mistral-7B, Vicuna-7B}.
- **Default metric config.** exponential colour kernel, $\tau=30$, $\alpha=0.5$, letter-preferring resolution, anatomy *not* folded into visibility, CAGS unclipped, $B=1000$, open-pool 40, seed 42.
- **Parse quality.** All 18 models parse at **≥ 98.9%**, with letter/text mismatch **≤ 1.17%**, so the numbers reflect visual reasoning, not parsing luck.

---

## 12. Results

### 12.1 CAGS leaderboard (primary)

| Rank | Model | EM | $\mathrm{GS}_{\mathrm{raw}}$ | **CAGS** | CI (lo–hi) | Severity |
| ---: | --- | ---: | ---: | ---: | :---: | ---: |
| 1 | Qwen3VL_32B | 0.5696 | 0.6867 | **0.5929** | 0.5918–0.5940 | 0.9458 |
| 2 | Qwen3VL_30B | 0.5380 | 0.6654 | **0.5689** | 0.5679–0.5700 | 0.9331 |
| 3 | InternVL3_38B | 0.5351 | 0.6648 | **0.5680** | 0.5670–0.5691 | 0.9292 |
| 4 | Gemma4-12B | 0.5259 | 0.6513 | **0.5540** | 0.5528–0.5551 | 0.9409 |
| 5 | Qwen3VL_8B | 0.5275 | 0.6543 | **0.5526** | 0.5513–0.5536 | 0.9470 |
| 6 | InternVL3.5_38B | 0.5282 | 0.6482 | **0.5476** | 0.5464–0.5487 | 0.9591 |
| 7 | Molmo2_8B | 0.5200 | 0.6341 | **0.5327** | 0.5315–0.5338 | 0.9736 |
| 8 | Qwen2.5VL_7B | 0.4966 | 0.6346 | **0.5291** | 0.5280–0.5302 | 0.9355 |
| 9 | InternVL3.5_30B | 0.5011 | 0.6306 | **0.5242** | 0.5231–0.5254 | 0.9536 |
| 10 | InternVL3_8B | 0.4858 | 0.6191 | **0.5067** | 0.5055–0.5079 | 0.9592 |
| 11 | InternVL3.5_8B | 0.4607 | 0.6123 | **0.5046** | 0.5035–0.5056 | 0.9186 |
| 12 | Gemma3-12B | 0.4681 | 0.6047 | **0.4867** | 0.4855–0.4877 | 0.9652 |
| 13 | Qwen2.5VL_3B | 0.4409 | 0.5856 | **0.4609** | 0.4598–0.4620 | 0.9644 |
| 14 | Pixtral_12B | 0.4135 | 0.5652 | **0.4392** | 0.4381–0.4403 | 0.9561 |
| 15 | LLaVa-NeXT_Vicuna_7B | 0.3669 | 0.5410 | **0.4191** | 0.4181–0.4202 | 0.9175 |
| 16 | Phi-3.5_Vision | 0.4147 | 0.5353 | **0.4045** | 0.4033–0.4056 | **1.0175** |
| 17 | Llama3.2_Vision | 0.3755 | 0.5266 | **0.3815** | 0.3803–0.3826 | 0.9904 |
| 18 | LLaVa-NeXT_Mistral_7B | 0.3762 | 0.5171 | **0.3783** | 0.3771–0.3794 | 0.9967 |

> **Findings.** The ordering is *sensible and discriminative*: the newest large Qwen3-VL and InternVL3 models occupy the podium, the 2023-era 7B LLaVA-NeXT variants sit at the floor, and the field spans a wide **0.378–0.593** CAGS range with no ceiling saturation — exactly the profile a benchmark needs. Note three anomalies the graded metric surfaces immediately, each expanded below: **Phi-3.5 is the only model with Severity > 1** and the only one whose CAGS falls *below* its EM; **Molmo2-8B (7th) punches far above its parameter count** on the spatial families; and **the CI columns are tight** (~0.001 wide) so adjacent ranks separated by < 0.001 CAGS (e.g. ranks 4/5, 17/18) are statistical ties, not orderings to over-read.

### 12.2 Exact-match leaderboard (anchor) — with format diagnostics

| Rank | Model | Acc % | MCQ % | Open % | parse % | mismatch % | `chose_A` % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Qwen3VL_32B | 57.31 | 57.63 | 51.96 | 100.0 | 0.00 | 19.15 |
| 2 | Qwen3VL_30B | 54.08 | 54.82 | 41.53 | 100.0 | 0.03 | 24.30 |
| 3 | InternVL3_38B | 53.70 | 54.42 | 41.72 | 100.0 | 0.02 | 17.08 |
| 4 | Qwen3VL_8B | 52.98 | 53.92 | 37.11 | 100.0 | 0.04 | 14.56 |
| 5 | InternVL3.5_38B | 52.88 | 54.39 | 27.57 | 100.0 | 0.02 | 22.85 |
| 6 | Gemma4-12B | 52.71 | 55.06 | 13.33 | 98.90 | 0.01 | 17.00 |
| 7 | Molmo2_8B | 52.09 | 54.54 | 10.94 | 100.0 | 0.02 | 21.41 |
| 8 | InternVL3.5_30B | 50.22 | 51.46 | 29.49 | 100.0 | 0.02 | 27.41 |
| 9 | Qwen2.5VL_7B | 49.99 | 50.04 | 49.05 | 99.93 | 1.17 | 13.98 |
| 10 | InternVL3_8B | 48.69 | 49.68 | 32.14 | 99.85 | 0.16 | 17.87 |
| 11 | Gemma3-12B | 46.96 | 48.46 | 21.79 | 100.0 | 0.02 | 14.63 |
| 12 | InternVL3.5_8B | 46.15 | 47.67 | 20.68 | 100.0 | 0.01 | 15.35 |
| 13 | Qwen2.5VL_3B | 44.44 | 44.43 | 44.48 | 100.0 | 0.05 | 10.77 |
| 14 | Pixtral_12B | 41.51 | 42.72 | 21.21 | 100.0 | 0.10 | 20.83 |
| 15 | Phi-3.5_Vision | 41.48 | 43.73 | 3.81 | 100.0 | 0.06 | 17.50 |
| 16 | LLaVa-NeXT_Mistral_7B | 37.65 | 39.69 | 3.36 | 100.0 | 0.21 | 33.96 |
| 17 | Llama3.2_Vision | 37.55 | 38.04 | 29.33 | 99.96 | 0.02 | **63.26** |
| 18 | LLaVa-NeXT_Vicuna_7B | 36.71 | 38.73 | 2.83 | 100.0 | 0.19 | 17.98 |

> **Findings.** Every model's `MCQ %` exceeds its `Open %`, often by tens of points (e.g. Gemma4-12B 55.06 vs 13.33; Molmo2-8B 54.54 vs 10.94) — the recognition–generation gap, quantified in 12.5. Parse rates are uniformly ≥ 98.9%, so no model is penalized for unreadable output. Two `chose_A` values stand out as position-bias red flags against the ≈ 20% uniform expectation: **Llama-3.2 at 63.26%** and **LLaVA-Mistral at 33.96%** (12.5). Qwen2.5VL_7B's 1.17% `letter_text_mismatch` is the field's highest — small in absolute terms, but the one model where letter-vs-text policy has any measurable effect (consistent with its behaviour under `prefer=text` in Section 13).

### 12.3 Per-category CAGS (all 18 models)

| Model | Anatomical Loc. | Morphological | Rel. Salience | Taxonomy & Naming |
| --- | ---: | ---: | ---: | ---: |
| Qwen3VL_32B | 0.4754 | 0.6441 | 0.5839 | **0.7310** |
| Qwen3VL_30B | 0.4525 | 0.6313 | 0.5582 | 0.6584 |
| InternVL3_38B | 0.4132 | 0.6451 | 0.5621 | 0.6691 |
| Gemma4-12B | 0.4306 | 0.6205 | **0.5915** | 0.3742 |
| Qwen3VL_8B | 0.4411 | 0.6244 | 0.5309 | 0.6429 |
| InternVL3.5_38B | 0.3747 | **0.6379** | 0.5595 | 0.5363 |
| Molmo2_8B | **0.4741** | 0.6429 | 0.5027 | 0.3301 |
| Qwen2.5VL_7B | 0.3556 | 0.5800 | 0.5409 | 0.7127 |
| InternVL3.5_30B | 0.3504 | 0.6093 | 0.5367 | 0.5391 |
| InternVL3_8B | 0.3462 | 0.5844 | 0.5081 | 0.5813 |
| InternVL3.5_8B | 0.3722 | 0.6028 | 0.4953 | 0.4572 |
| Gemma3-12B | 0.4103 | 0.5506 | 0.4657 | 0.5113 |
| Qwen2.5VL_3B | 0.3890 | 0.5005 | 0.4272 | 0.6615 |
| Pixtral_12B | 0.3199 | 0.4950 | 0.4471 | 0.4654 |
| LLaVa-NeXT_Vicuna_7B | 0.3640 | 0.4535 | 0.4571 | 0.1977 |
| Phi-3.5_Vision | **0.1541** | 0.5335 | 0.4540 | 0.2175 |
| Llama3.2_Vision | 0.2862 | 0.4831 | 0.3168 | 0.5203 |
| LLaVa-NeXT_Mistral_7B | 0.2587 | 0.4843 | 0.3722 | 0.2383 |

> **Findings.** The naming axis has by far the widest spread (**0.198 → 0.731**), i.e. taxonomic naming is where models most differentiate; Morphological Attributes is the most compressed (0.454 → 0.645), i.e. basic attribute reading is a broadly-shared competence. The category leaders are not all the same model — **Gemma4-12B leads Relative Salience (0.5915)**, **Molmo2-8B leads Anatomical Localization (0.4741)**, and **Qwen3VL-32B leads Naming (0.7310)** — which is direct evidence that a single scalar hides real, category-localized strengths. **Phi-3.5's 0.1541 on Anatomical Localization** is the single lowest category score in the table and previews its Severity > 1 pathology (12.6).

### 12.4 Family difficulty (mean CAGS across all 18 models)

| Family | mean CAGS | min | max | reading |
| --- | ---: | ---: | ---: | --- |
| `visibility_partset` | **0.341** | 0.108 | 0.435 | hardest — anatomical *set* reasoning is the frontier |
| `open_genus` | 0.353 | 0.023 | 0.696 | huge spread — genus naming separates models sharply |
| `open_scientific_name` | 0.359 | 0.049 | 0.629 | binomial generation is broadly hard |
| `open_common_name` | 0.406 | 0.141 | 0.627 | free species naming |
| `visibility_side` | 0.411 | 0.225 | 0.746 | laterality; Molmo dominates the top |
| `attr_group_value` | 0.495 | 0.317 | 0.592 | salience: region+value+salience is moderately hard |
| `marker_localization` | 0.540 | 0.254 | 0.743 | marker→part binding |
| `attr_bare_value` | 0.574 | 0.454 | 0.645 | single-attribute reading; the most *compressed* family |
| `open_kind_of` | 0.580 | 0.096 | 0.763 | folk category — easier than formal naming |
| `category_mcq` | **0.833** | 0.532 | 0.962 | nearly solved — coarse recognition among options |

> **Findings.** The benchmark's **headroom is concentrated in two places**: open-ended taxonomic naming (genus/scientific/common, mean ≤ 0.41) and anatomical *set/side* reasoning (`visibility_partset` mean 0.341). Coarse recognition (`category_mcq`, mean 0.833) is close to saturated. This is the ideal difficulty profile for a fine-grained benchmark: an easy family that confirms basic competence, and several hard families with clear, non-saturated room to improve. The single most compressed family — `attr_bare_value` (min 0.454) — shows that *reading one attribute* is a floor competence every modern VLM has, which makes the harder families the discriminating ones.

### 12.5 The recognition–generation gap (headline finding #1)

The most informative contrast in this axis is `category_mcq` (recognize among options) vs. `open_common_name` (generate unaided) — both probe taxonomic knowledge, differing only in *recognition vs. generation*.

| Model | `category_mcq` CAGS | `open_common_name` CAGS | **gap** |
| --- | ---: | ---: | ---: |
| InternVL3.5_30B | 0.869 | 0.301 | **0.568** |
| Phi-3.5_Vision | 0.716 | 0.177 | 0.539 |
| InternVL3.5_38B | 0.920 | 0.389 | 0.531 |
| InternVL3_8B | 0.884 | 0.374 | 0.510 |
| InternVL3.5_8B | 0.783 | 0.277 | 0.506 |
| Gemma4-12B | 0.772 | 0.285 | 0.487 |
| InternVL3_38B | 0.948 | 0.495 | 0.453 |
| Qwen3VL_32B | 0.959 | 0.619 | 0.340 |
| Qwen2.5VL_7B | 0.962 | 0.627 | 0.335 |
| **Llama3.2_Vision** | 0.859 | **0.624** | **0.235** |

> **Findings.** Every model can pick the right bird type among options far better than it can name it unaided; on **exact match** the effect is even starker — `category_mcq` peaks at **96.16%** while the best `open_common_name` EM is only **35.49%**. This is a direct, quantitative measurement of the recognition-vs-generation divide in current MLLMs, and it is *invisible to any single averaged accuracy*. Two second-order readings: the **InternVL3.5 family shows the largest gaps** (0.51–0.57) — strong recognizers that generate poorly — whereas the **Qwen3-VL / Qwen2.5-VL family shows the smallest gaps among strong models**, i.e. their generation keeps pace with their recognition. And **Llama-3.2 is the standout exception (gap 0.235)** for an unusual reason: its `open_common_name` CAGS (0.624) is second-best *in the entire field*, even though its overall rank is 17th. Llama can genuinely *name* birds; what sinks it is MCQ position bias (12.6), not naming ability — a conclusion only reachable because CAGS and the format diagnostics are read jointly.

### 12.6 Position bias, "confidently wrong," and the grounding outlier (headline findings #2–#4)

**Position bias (finding #2).** `chose_A`, sorted:

| Model | `chose_A` % | vs. ~20% uniform |
| --- | ---: | --- |
| **Llama3.2_Vision** | **63.26** | ≈ 3.2× — severe |
| LLaVa-NeXT_Mistral_7B | 33.96 | ≈ 1.7× — strong |
| InternVL3.5_30B | 27.41 | mild |
| Qwen3VL_30B | 24.30 | mild |
| … (remaining 14 models) | 10.8–22.9 | near-uniform |

> Llama-3.2 answers **(A)** almost two-thirds of the time — a prior over answer *position*, not *content*. Its MCQ accuracy is therefore partly an artifact of where the gold answer happens to sit, not of what it sees; combined with 12.5, the picture is a model that names well but cannot follow the MCQ format. This is the concrete case for reporting graded, chance-corrected scores that are harder to game by position — and for reporting `chose_A` openly so the inflation is visible rather than hidden.

**"Confidently wrong" (finding #3).** **Phi-3.5-Vision is the only model with overall Severity > 1 (1.0175), and the only one whose CAGS (0.4045) falls below its EM (0.4147).** Every other model has CAGS ≥ EM because partial credit for near misses lifts it. Phi is the exception: its misses land *farther from the gold than a random distractor* (e.g. `visibility_partset` CAGS 0.108 with Severity 1.088). Severity > 1 is the cleanest single illustration of what CAGS measures that EM cannot.

**Grounding outlier (finding #4).** **Molmo2-8B is the field's best spatial grounder**, posting the top `visibility_side` CAGS (**0.7463**) and top `marker_localization` CAGS (**0.7429**) despite being an 8B model against 30–38B competitors, consistent with its pointing/grounding-oriented pretraining. It is simultaneously weak on naming (Taxonomy 0.3301), so its 7th-place *overall* rank understates a genuine, localized strength — exactly the nuance the per-family matrix (Appendix A) is designed to expose.

### 12.7 CAGS re-ranks where it matters (headline finding #5)

CAGS and EM agree on the broad ordering (reassuring — CAGS is not capricious) but disagree in the lower cluster, and the disagreements are informative rather than arbitrary:

| Model | CAGS rank | EM rank | Δ (EM→CAGS) | why |
| --- | ---: | ---: | ---: | --- |
| Gemma4-12B | 4 | 6 | **+2** | near-misses earn credit (high $\mathrm{GS}_{\mathrm{raw}}$ 0.651) |
| LLaVa-NeXT_Vicuna_7B | 15 | 18 | **+3** | *less often exact*, but misses land close |
| Qwen2.5VL_7B | 8 | 9 | +1 | |
| InternVL3.5_8B | 11 | 12 | +1 | |
| Phi-3.5_Vision | 16 | 15 | **−1** | confidently wrong (Severity > 1) |
| LLaVa-NeXT_Mistral_7B | 18 | 16 | **−2** | misses land far |
| *(11 models unchanged)* | — | — | 0 | CAGS and EM concur |

> **Findings.** The signature swap is **LLaVA-Vicuna (EM #18) rising above Phi-3.5 (EM #15)** under CAGS: Vicuna is *less often exactly right* than Phi, but its misses are *closer*, so a closeness-aware, chance-corrected metric correctly ranks it higher — EM alone gets this backwards. Symmetrically, **Gemma4-12B rises EM #6 → CAGS #4** because its near-misses carry real partial credit. Crucially, no swap exceeds 3 positions and none touches the podium, so CAGS *refines* the EM ranking rather than *contradicting* it — which is the property that makes the graded metric trustworthy rather than merely different.

---

## 13. Robustness: the Sensitivity Sweep

A graded metric has free constants; before its ranking can be reported as a fact about models, we must show it is not a fact about those constants. `sensitivity_sweep.py` re-scores every model under a grid of configurations on a fixed, family-stratified 60,000-row/model subsample (seed 42, identical rows across configs) and measures how far the leaderboard moves from the default.

### 13.1 Stability statistics

Let $r^{(0)}_i$ and $r^{(c)}_i$ be model $i$'s rank under the default and under config $c$.

$$
\rho = 1 - \frac{6\sum_i d_i^2}{n(n^2-1)}, \quad d_i = r^{(0)}_i - r^{(c)}_i
\qquad
\tau = \frac{C - D}{\,n(n-1)/2\,}
$$

$$
\mathrm{displacement} = \max_i\, |\, r^{(0)}_i - r^{(c)}_i\, |
\qquad
\mathrm{mean\,|\Delta CAGS|} = \frac{1}{n}\sum_i |\, s^{(c)}_i - s^{(0)}_i\, |
$$

Here $C$ and $D$ are the numbers of concordant and discordant model pairs. Spearman $\rho$ and Kendall $\tau$ are rank correlations with the default ordering (1 = identical; $\tau$ is stricter); displacement is the worst single-model rank move; **top-3 stable** is whether the podium *set* is unchanged; the mean absolute $\Delta$CAGS is the size of the score shift independent of ranking.

### 13.2 Base grid — $\tau$, $\alpha$, colour kernel (9 configs)

| config | Spearman $\rho$ | Kendall $\tau$ | max disp. | top-3 | mean &#124;&Delta;CAGS&#124; |
| --- | ---: | ---: | ---: | :---: | ---: |
| default | 1.0000 | 1.0000 | 0 | ✓ | 0.0000 |
| tau=20 | 0.9979 | 0.9869 | 1 | ✓ | 0.0008 |
| tau=40 | 0.9979 | 0.9869 | 1 | ✓ | 0.0008 |
| tau=50 | 0.9979 | 0.9869 | 1 | ✓ | 0.0015 |
| alpha=0.3 | 1.0000 | 1.0000 | 0 | ✓ | 0.0011 |
| alpha=0.4 | 1.0000 | 1.0000 | 0 | ✓ | 0.0006 |
| alpha=0.6 | 1.0000 | 1.0000 | 0 | ✓ | 0.0007 |
| alpha=0.7 | 0.9979 | 0.9869 | 1 | ✓ | 0.0015 |
| kernel=linear | 0.9959 | 0.9739 | 1 | ✓ | 0.0027 |

> **Headline (base grid):** min $\rho = 0.9959$, min $\tau = 0.9739$, **max displacement = 1 rank**, top-3 stable in **8/8**, worst mean |&Delta;CAGS| = 0.0027 (`kernel=linear`). The ranking is **near-invariant** to the metric's numeric constants, and the only rank exchanges are single-position swaps between models whose bootstrap CIs already overlap (12.1) — statistical ties, not instability.

### 13.3 Policy grid — adding `prefer=text` and `anat_in_vis=1` (11 configs)

| config | Spearman $\rho$ | Kendall $\tau$ | max disp. | top-3 | mean &#124;&Delta;CAGS&#124; |
| --- | ---: | ---: | ---: | :---: | ---: |
| *(9 base configs, as above)* | ≥ 0.9959 | ≥ 0.9739 | ≤ 1 | ✓ | ≤ 0.0027 |
| prefer=text | 0.9979 | 0.9869 | 1 | ✓ | 0.0004 |
| **anat_in_vis=1** | **0.9587** | **0.8562** | **3** | ✓ | **0.0200** |

**Which models move under `anat_in_vis=1`** (the only ≥ 2-rank perturbation in the entire study):

| Model | default | anat_in_vis=1 | Δ |
| --- | ---: | ---: | ---: |
| Gemma3-12B | 12 | 9 | **+3** |
| Llama3.2_Vision | 17 | 15 | +2 |
| LLaVa-NeXT_Vicuna_7B | 15 | 18 | **−3** |
| InternVL3.5_8B | 11 | 13 | −2 |
| Qwen2.5VL_7B | 8 | 10 | −2 |
| **Top-3** (Qwen3VL_32B/30B, InternVL3_38B) | 1 / 2 / 3 | 1 / 2 / 3 | **0** |

> **Findings.** Under all numeric constants *and* the letter-vs-text policy, the ranking is near-invariant ($\rho \ge 0.996$; `prefer=text` is negligible at $\rho = 0.9979$). The single configuration that moves mid-table ranks is `anat_in_vis=1`, which folds anatomy part-similarity into the `visibility_partset` set kernel so that "named the left wing when the right wing was visible" earns partial credit. This is a **substantive modelling decision, not an arbitrary constant** — it legitimately changes the visibility-family scores — yet even here the **top-3 set is unchanged** and the movement is confined to families where crediting near-miss parts is a defensible design choice. We adopt the exact-match default, and disclose that the alternative would reshuffle the middle of the table by up to three positions while leaving the podium intact. Reporting this openly is stronger than hiding it: it shows the ranking's one sensitivity is a *known semantic knob*, not a fragile numeric one.

---

## 14. Discussion and Threats to Validity

- **Why the ranking is trustworthy.** It tracks the field's expectations (large/new models on top, old 7B models at the floor), *survives* the hyper-parameter sweep, and *still* exposes non-obvious structure (Molmo's grounding, Phi's over-confidence, Llama's position bias, the recognition–generation gap). A benchmark that only reproduced known orderings would be redundant; one whose surprising orderings collapsed under perturbation would be unreliable. This axis is neither.
- **The graded layer sits on a stable backbone.** CAGS perturbs a partial-credit signal on top of an exact-match / chance-adjustment core that does not depend on $\tau$, $\alpha$, or kernel shape — which is *part of why* the base-grid ranking barely moves. We state this so the stability does not read as trivial.
- **Naming kernels depend on a taxonomy map.** The common-name and genus kernels use a 200-entry `taxon_name_mapping.json`; coverage gaps degrade gracefully to token/head-noun overlap rather than failing, but naming-family CAGS is only as complete as that map.
- **The open-ended chance baseline is sampled** (40-answer pool, fixed seed). The sweep's stability under seed-fixed resampling indicates this is not a sensitive choice, but it is an approximation to the full answer distribution.
- **`visibility_partset` is the one genuinely policy-sensitive family** (Section 13.3). We recommend reporting the default and, in an appendix, the `anat_in_vis=1` variant, so readers see the podium is invariant either way.
- **Macro-averaged label metrics are intentionally omitted.** For open-vocabulary naming with hundreds of classes, macro P/R/F1 collapse to near-noise; this axis reports accuracy, CAGS, and the per-family breakdown, which carry the real signal.

---

## 15. Reproducibility

### 15.1 Directory layout

```
<base_dir>/                          # holds one sub-directory per model, and Results/
  <Model>/QA_*.csv                   #   consolidated predictions (question, answer, <model>_answer)
  <Model>/*chunk*.csv                #   inference shards (mark the folder as a model dir)
  <Model>/Evaluation/
      eval_soft_scored.csv           #   per-item: family, gold/pred, status, s, b, cags, severity
      eval_soft_summary.csv          #   tidy overall/family/category CAGS tables
      eval_soft_stats.txt            #   per-model CAGS report (metric glossary embedded)
      eval_stats.txt / eval_rows.csv / eval_summary.csv     # exact-match per-model outputs
  Results/
    Eval - CAGS/         eval_soft_leaderboard.csv · eval_soft_by_{category,family}_ALL.csv ·
                         eval_soft_stats_ALL.txt · sweep_{stability,cags_matrix,rank_matrix}[_with_policy].csv ·
                         sweep_report[_with_policy].txt
    Eval - Exact Match/  eval_leaderboard.csv · eval_by_{category,family}_ALL.csv ·
                         eval_format_ALL.csv · eval_stats_ALL.txt
    Manual Review/       annotation_sheet.csv · annotation_key.csv · INSTRUCTIONS.txt · cards/ (+ .pdf)

<data_dir>/                          # resource side-files (may differ from base_dir)
  taxon_name_mapping.json · parts/part_locs.txt · parts/parts.txt · bounding_boxes.txt
```

### 15.2 Commands

`--base_dir` = the folder holding the model sub-dirs (and `Results/`); `--data_dir` = the folder holding the resource side-files (defaults to `base_dir`). Only the two graded steps need `--data_dir`.

```bash
python evaluate_soft.py                     --base_dir "$BASE_DIR" --data_dir "$DATA_DIR"   # 1. CAGS
python evaluate.py                          --base_dir "$BASE_DIR"                          # 2. Exact match
python sensitivity_sweep.py                 --base_dir "$BASE_DIR" --data_dir "$DATA_DIR"   # 3. Sweep
python sensitivity_sweep.py --with_policy    --base_dir "$BASE_DIR" --data_dir "$DATA_DIR"  # 4. Sweep + policy
python setup_manual_review_images.py        --base_dir "$BASE_DIR"                          # 5. Blind sample
python setup_manual_review_pdf_annotate.py  --base_dir "$BASE_DIR" --pdf                    # 6. Cards + PDF
```

`run_all_eval.sh` chains all six in dependency order. Metric flags: `--tau`, `--alpha`, `--color_kernel {exp,linear}`, `--prefer {letter,text}`, `--anatomy_in_visibility`, `--clip`, `--bootstrap`, `--open_pool_sample`, `--seed`.

> **A note on running under SLURM.** The scripts print a flushed `[start] … | base_dir=… | data_dir=…` heartbeat *before* touching the filesystem; set `PYTHONUNBUFFERED=1` so the log shows progress rather than appearing to hang, and confirm the compute node can see `--data_dir` (a stale mount stalls the first stat silently).

### 15.3 Human validation

CAGS is validated against human judgment by a **blind, stratified manual-review sample**: 90 items across a (category × CAGS-band) grid (low/mid/high CAGS), rendered as cards showing the image, question, gold answer, and model answer — with model identity, CAGS, and band hidden. Three raters score 0/1/2 (wrong / partially right / right); their ratings are correlated against the hidden CAGS to confirm the metric tracks human perception of closeness.

---

## Appendix A — Full Per-Model × Per-Family Tables

### A.1 CAGS by question family (rows = model)

| Model | attr_bare | attr_group | cat_mcq | marker | common | genus | kind_of | sci | vis_set | vis_side |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma3-12B | 0.551 | 0.466 | 0.915 | 0.572 | 0.438 | 0.299 | 0.688 | 0.243 | 0.416 | 0.339 |
| Gemma4-12B | 0.621 | 0.592 | 0.772 | 0.720 | 0.286 | 0.121 | 0.615 | 0.105 | 0.385 | 0.488 |
| InternVL3.5_30B | 0.609 | 0.537 | 0.869 | 0.574 | 0.301 | 0.505 | 0.606 | 0.433 | 0.328 | 0.350 |
| InternVL3.5_38B | 0.638 | 0.560 | 0.920 | 0.656 | 0.389 | 0.417 | 0.518 | 0.454 | 0.310 | 0.498 |
| InternVL3.5_8B | 0.603 | 0.495 | 0.783 | 0.499 | 0.277 | 0.347 | 0.557 | 0.342 | 0.348 | 0.409 |
| InternVL3_38B | 0.645 | 0.562 | 0.948 | 0.603 | 0.495 | 0.636 | 0.665 | 0.614 | 0.373 | 0.484 |
| InternVL3_8B | 0.584 | 0.508 | 0.884 | 0.468 | 0.374 | 0.523 | 0.644 | 0.499 | 0.332 | 0.353 |
| LLaVa-NeXT_Mistral_7B | 0.484 | 0.372 | 0.636 | 0.392 | 0.154 | 0.023 | 0.338 | 0.063 | 0.210 | 0.377 |
| LLaVa-NeXT_Vicuna_7B | 0.454 | 0.457 | 0.550 | 0.254 | 0.142 | 0.027 | 0.231 | 0.056 | 0.358 | 0.420 |
| Llama3.2_Vision | 0.483 | 0.317 | 0.859 | 0.488 | **0.624** | 0.186 | 0.704 | 0.252 | 0.260 | 0.307 |
| Molmo2_8B | 0.643 | 0.503 | 0.532 | **0.743** | 0.284 | 0.147 | 0.579 | 0.129 | 0.366 | **0.746** |
| Phi-3.5_Vision | 0.534 | 0.454 | 0.716 | 0.337 | 0.177 | 0.066 | 0.096 | 0.049 | **0.108** | 0.245 |
| Pixtral_12B | 0.495 | 0.447 | 0.847 | 0.529 | 0.396 | 0.259 | 0.613 | 0.235 | 0.328 | 0.225 |
| Qwen2.5VL_3B | 0.501 | 0.427 | 0.926 | 0.334 | 0.566 | 0.565 | 0.665 | 0.597 | 0.399 | 0.373 |
| Qwen2.5VL_7B | 0.580 | 0.541 | **0.962** | 0.563 | 0.627 | 0.636 | 0.736 | 0.616 | 0.329 | 0.377 |
| Qwen3VL_30B | 0.631 | 0.558 | 0.951 | 0.660 | 0.608 | 0.423 | 0.733 | 0.594 | 0.434 | 0.445 |
| Qwen3VL_32B | 0.644 | **0.584** | 0.959 | 0.727 | 0.619 | **0.696** | **0.763** | **0.629** | **0.435** | 0.526 |
| Qwen3VL_8B | 0.624 | 0.531 | 0.958 | 0.608 | 0.543 | 0.486 | 0.691 | 0.553 | 0.426 | 0.435 |

### A.2 Exact-match accuracy (%) by question family (rows = model)

| Model | attr_bare | attr_group | cat_mcq | marker | common | genus | kind_of | sci | vis_set | vis_side |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma3-12B | 54.73 | 46.37 | 91.45 | 42.54 | 17.27 | 29.92 | 26.80 | 13.40 | 41.48 | 29.41 |
| Gemma4-12B | 63.17 | 60.40 | 77.20 | 59.93 | 12.73 | 12.53 | 23.78 | 4.73 | 19.30 | 52.84 |
| InternVL3.5_30B | 60.50 | 54.45 | 86.86 | 43.16 | 11.88 | 50.15 | 22.43 | 33.28 | 23.50 | 37.24 |
| InternVL3.5_38B | 64.46 | 56.46 | 92.00 | 54.01 | 13.18 | 40.89 | 21.68 | 34.38 | 21.63 | 53.08 |
| InternVL3.5_8B | 59.29 | 49.64 | 78.31 | 37.57 | 8.75 | 34.98 | 14.76 | 24.03 | 13.85 | 43.99 |
| InternVL3_38B | 64.44 | 56.54 | 94.81 | 47.25 | 18.94 | 63.22 | 34.14 | 50.38 | 23.02 | 49.94 |
| InternVL3_8B | 58.19 | 52.84 | 88.42 | 41.35 | 13.30 | 51.65 | 25.16 | 38.23 | 22.14 | 35.82 |
| LLaVa-NeXT_Mistral_7B | 46.20 | 38.84 | 63.64 | 35.34 | 1.53 | 2.96 | 7.03 | 2.08 | 24.88 | 39.08 |
| LLaVa-NeXT_Vicuna_7B | 45.30 | 43.56 | 55.02 | 25.96 | 1.14 | 3.01 | 4.72 | 2.56 | 15.46 | 19.44 |
| Llama3.2_Vision | 46.75 | 32.74 | 85.86 | 37.72 | 34.54 | 18.87 | 50.79 | 14.00 | 29.40 | 30.87 |
| Molmo2_8B | 64.56 | 51.03 | 53.18 | 65.41 | 7.86 | 14.97 | 15.61 | 5.50 | 34.37 | **71.11** |
| Phi-3.5_Vision | 53.54 | 46.11 | 71.60 | 28.30 | 3.16 | 7.26 | 1.94 | 2.80 | 18.04 | 26.28 |
| Pixtral_12B | 49.20 | 46.01 | 84.68 | 37.22 | 8.92 | 25.32 | 36.10 | 15.23 | 17.93 | 32.60 |
| Qwen2.5VL_3B | 48.37 | 41.86 | 92.64 | 28.61 | 33.19 | 56.52 | 39.67 | 48.40 | 41.22 | 37.29 |
| Qwen2.5VL_7B | 58.24 | 54.86 | **96.16** | 43.18 | 34.00 | 63.28 | 46.75 | **52.17** | 17.05 | 38.96 |
| Qwen3VL_30B | 63.16 | 55.53 | 95.11 | 51.92 | 33.57 | 40.96 | 44.17 | 47.61 | 32.07 | 46.34 |
| Qwen3VL_32B | 64.51 | 59.01 | 95.90 | **61.15** | **35.49** | **69.46** | **51.15** | 51.80 | 34.12 | 55.65 |
| Qwen3VL_8B | 62.81 | 53.35 | 95.77 | 47.12 | 25.13 | 47.83 | 33.85 | 41.56 | 33.40 | 46.10 |

---

## Appendix B — Constants and Lexicons

**Colour anchors (sRGB).** black (0,0,0); white (255,255,255); grey/gray (128,128,128); brown (139,69,19); buff (240,220,180); rufous (168,84,45); red (220,20,20); orange (255,165,0); yellow (255,225,30); olive (128,128,0); green (30,150,50); blue (30,60,220); purple (128,0,128); pink (255,170,190). Non-chromatic (exact-match-only): **iridescent**.

**Ordinal scales.** *bill length:* shorter than head (0) < about the same (1) < longer than head (2). *size:* very small 3–5 in (0) < small 5–9 in (1) < medium 9–16 in (2) < large 16–32 in (3) < very large 32–72 in (4).

**Pattern confusion.** {spotted, striped, multi-colored} pairwise $= 0.5$; any pairing with **solid** $= 0.0$.

**Schematic parts** $(x, y, z)$: beak (0.95, 0.35, 0), forehead (0.85, 0.28, 0), crown (0.78, 0.22, 0), left/right eye (0.83, 0.32, ∓0.03), nape (0.72, 0.25, 0), throat (0.82, 0.45, 0), breast (0.68, 0.55, 0), belly (0.50, 0.62, 0), back (0.55, 0.30, 0), left/right wing (0.55, 0.45, ∓0.05), left/right leg (0.45, 0.85, ∓0.04), tail (0.10, 0.45, 0). `primary`, `shape`, `size` map to GLOBAL (non-localizable).

---

## Appendix C — Glossary and Per-Family Example Catalogue

**Glossary.** *EM* exact-match accuracy · *$s$* raw graded similarity · *$\mathrm{GS}_{\mathrm{raw}}$* mean raw similarity before chance adjustment · *CAGS* chance-adjusted graded score $(s-b)/(1-b)$, per item, unclipped · *$b$* chance baseline (mean distractor similarity to gold) · *Severity* mean $(1-\mathrm{CAGS})$ over wrong items, $>1$ ⇒ worse-than-chance misses · *parse_rate / letter_text_mismatch / chose_A* MCQ format and position-bias diagnostics.

**Per-family example catalogue** (every similarity value is a real kernel output):

| Family | Example question | Gold | Near miss (score) | Far miss (score) |
| --- | --- | --- | --- | --- |
| `attr_bare_value` | "…wing color?" | blue | purple (0.498) | yellow (0.053) |
| `attr_group_value` | "…most prominent trait?" | wing color: blue | breast color: blue (0.952) / wing color: purple (0.749) | tail color: yellow (0.395) |
| `marker_localization` | "…marked with the star?" | nape | crown (0.922) | tail (0.000) |
| `visibility_side` | "…which sides visible?" | Both | Left (0.500) | None (0.000) |
| `visibility_partset` | "…which parts visible?" | {beak, tail} | {beak} (soft-Jaccard partial) | disjoint (0.000) |
| `category_mcq` | "…what type of bird?" | sparrow | near coarse type | unrelated type |
| `open_common_name` | "…common name?" | song sparrow | fox sparrow (0.700) | american crow (0.000) |
| `open_scientific_name` | "…scientific name?" | Melospiza melodia | Melospiza georgiana (0.575) | Corvus brachyrhynchos (0.054) |
| `open_genus` | "…genus?" | Melospiza | Melospiza (1.000) | Passerella (0.000) |
| ordinal (size) | "…size?" | small (5–9 in) | medium 9–16 in (0.750) | very large 32–72 in (0.250) |
| pattern | "…wing pattern?" | spotted | striped (0.500) | solid (0.000) |

---

<p align="center">
  <sub>Part of the <b>BioVQA</b> benchmark · Axis (ii) — Fine-Grained Visual Attribute Grounding · Imageomics Institute</sub><br>
  <sub>All leaderboard/matrix values reproduced from <code>Results/</code>; all kernel example values computed by <code>evaluate_soft.py</code> under the default configuration.</sub>
</p>
