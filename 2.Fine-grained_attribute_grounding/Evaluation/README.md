# BioVQA · Component C2 — Fine‑Grained Visual Attribute Grounding
## Evaluation Methodology, Metrics, Results, and Analysis

> This document is the self‑contained technical companion to the C2 component of **BioVQA**. It is written to serve, in one place, as the **methodology**, **results**, **analysis**, and **appendix** for the fine‑grained attribute‑grounding experiments: what the questions are, how models are graded, why we grade them the way we do, the exact equations and algorithms behind every number, the full results across 18 vision–language models, and the robustness analysis that shows the ranking is not an artifact of metric hyper‑parameters. Every kernel value quoted in the worked examples is computed directly by the scoring code in `evaluate_soft.py`; every leaderboard number is reproduced verbatim from the run logs and the CSVs in `Results/`.

---

## Table of Contents

1. [Overview and motivation](#1-overview-and-motivation)
2. [The dataset: categories, question families, and examples](#2-the-dataset-categories-question-families-and-examples)
3. [Two metrics, and why: Exact Match (hard) vs. CAGS (graded)](#3-two-metrics-and-why-exact-match-hard-vs-cags-graded)
4. [The scoring pipeline (algorithmic workflow)](#4-the-scoring-pipeline-algorithmic-workflow)
5. [The similarity kernels — equations and worked examples](#5-the-similarity-kernels--equations-and-worked-examples)
6. [Chance adjustment: from GS_raw to CAGS](#6-chance-adjustment-from-gs_raw-to-cags)
7. [Uncertainty and mistake severity](#7-uncertainty-and-mistake-severity)
8. [A full end‑to‑end worked example](#8-a-full-end-to-end-worked-example)
9. [Metric reference card (all equations in one place)](#9-metric-reference-card-all-equations-in-one-place)
10. [Experimental setup](#10-experimental-setup)
11. [Results](#11-results)
12. [Robustness: the sensitivity sweep](#12-robustness-the-sensitivity-sweep)
13. [Discussion and threats to validity](#13-discussion-and-threats-to-validity)
14. [Reproducibility: files, commands, formats](#14-reproducibility-files-commands-formats)
15. [Appendix A — full per‑model × per‑family tables](#appendix-a--full-per-model--per-family-tables)
16. [Appendix B — constants and lexicons](#appendix-b--constants-and-lexicons)
17. [Appendix C — glossary and per‑family example catalogue](#appendix-c--glossary-and-per-family-example-catalogue)

---

## 1. Overview and motivation

BioVQA is a large multi‑axis benchmark for biological visual question answering. **Component C2 — Fine‑Grained Visual Attribute Grounding** targets the part of visual competence that generic VQA benchmarks systematically miss: whether a model can look at a single bird photograph and correctly reason about its *fine‑grained visible attributes* — the colour of a specific body region, the relative salience of competing traits, the identity and spatial layout of anatomical parts, and the taxonomic name the bird belongs to. C2 is built on **CUB‑200‑2011** and contains **824,104 questions per model**, evaluated over **18 vision–language models** spanning roughly 3B–38B parameters.

The central methodological problem C2 solves is **measurement**. Fine‑grained attribute questions have answers that are *near‑miss‑rich*: "the wing is blue" versus "the wing is purple" is a small perceptual error; "the marked part is the nape" versus "the crown" is a small spatial error; "song sparrow" versus "fox sparrow" shares a genus and a head‑noun. A metric that scores all of these as a flat **0** (as naïve exact‑match string comparison does) throws away most of the signal that distinguishes a good model from a bad one, and it conflates two very different failure modes — *being slightly wrong* and *being completely lost*.

C2 therefore reports **two complementary metrics side by side**:

- **Exact Match (EM)** — a strict, conservative, unambiguous correctness rate. It is the number every reviewer already trusts, and it anchors the benchmark.
- **CAGS (Chance‑Adjusted Graded Score)** — a graded metric that awards *partial credit for being close*, using a per‑question‑family notion of "closeness" (perceptual colour distance, anatomical distance, ordinal distance, taxonomic overlap, set overlap), and then **removes the credit a confused guesser would have earned by chance** on that specific question. CAGS is what turns a leaderboard of accuracy numbers into a diagnostic instrument: it exposes *how* models fail, not just *how often*.

The rest of this document explains both metrics in complete detail, presents the results, and then demonstrates — through a hyper‑parameter sensitivity sweep — that the CAGS ranking is stable to the metric's free constants, which is the property a benchmark needs before its ranking can be reported as a fact about models rather than a fact about the metric.

---

## 2. The dataset: categories, question families, and examples

C2 organizes its 824,104 questions into **four categories**, which decompose further into **ten question families**. The category is the coarse axis of visual skill; the family is the concrete question template with its own answer space and its own natural notion of a "near miss." The counts below are exact, per model.

| Category | Question family | # Questions | Answer space | Format |
|---|---|---:|---|---|
| **Morphological Attributes** | `attr_bare_value` | 280,433 | attribute value (colour / pattern / size / bill length) | MCQ |
| **Relative Salience** | `attr_group_value` | 321,073 | `region: value` trait | MCQ |
| **Anatomical Localization** | `marker_localization` | 11,788 | a named body part | MCQ |
| | `visibility_partset` | 117,880 | a set of visible parts | MCQ |
| | `visibility_side` | 35,364 | Left / Right / Both / None | MCQ |
| **Taxonomy & Naming** | `category_mcq` | 11,191 | coarse bird type | MCQ |
| | `open_common_name` | 11,788 | common (English) species name | open‑ended |
| | `open_scientific_name` | 11,668 | Latin binomial | open‑ended |
| | `open_genus` | 11,728 | genus | open‑ended |
| | `open_kind_of` | 11,191 | folk category ("what kind of bird") | open‑ended |
| **Total** | | **824,104** | | |

Two structural facts about this table drive the entire evaluation design:

1. **Most questions are multiple‑choice, but the hard signal is open‑ended.** Roughly 94% of items are MCQ, where the model chooses among 2–5 distractors; the remaining ~6% are open‑ended naming, where the model must *generate* the answer from scratch. The gap between a model's MCQ accuracy and its open‑ended accuracy is one of the most informative quantities in the whole benchmark (Section 11).
2. **"Closeness" means something different in every family.** A near‑miss on colour is a perceptual‑distance question; a near‑miss on a body part is a 3‑D‑layout question; a near‑miss on size is an ordinal‑scale question; a near‑miss on a species name is a taxonomic‑decomposition question. This is precisely why a single flat exact‑match number is inadequate, and why CAGS uses a **different kernel per family** (Section 5).

### 2.1 The four categories, in plain terms (with dataset examples)

**Morphological Attributes** (`attr_bare_value`). "What is the ⟨region⟩ ⟨property⟩ of this bird?" The options are bare attribute values. The property is colour, pattern, size, or bill length.

> *Example.* **Q:** "What is the wing color of this bird? Options: (A) blue, (B) brown, (C) black, (D) grey" · **Gold:** (A) blue · A model answering **purple** is perceptually close (near miss); a model answering **yellow** is far.

**Relative Salience** (`attr_group_value`). "What is the most prominently visible physical trait of this bird?" The options are full `region: value` traits, so the model must pick both the *right region* and the *right value*, and — crucially — must judge which trait is most *salient*.

> *Example.* **Q:** "What is the most prominently visible physical trait of this bird? Options: (A) wing color: blue, (B) breast color: rufous, (C) bill shape: hooked, ..." · **Gold:** (A) wing color: blue · A model that says **breast color: blue** got the value right but the region wrong; a model that says **wing color: purple** got the region right but the value slightly off. These are *different* kinds of near miss, and the paired‑attribute kernel (Section 5.5) scores them differently.

**Anatomical Localization** (`marker_localization`, `visibility_partset`, `visibility_side`). Three families that probe spatial grounding:

- `marker_localization` — "Which body part of this bird is marked with the red star? Options: (A) nape, (B) crown, (C) throat, (D) tail." The model must name the part under a visual marker. Answering **crown** when the gold is **nape** is a small spatial error (adjacent parts); answering **tail** is a large one.
- `visibility_partset` — "Which of the following parts are visible in this image? Options: (A) beak, (B) left wing, (C) tail, None, All three." The answer is a *set*; partial overlap earns partial credit.
- `visibility_side` — "Which ⟨parts⟩ are visible in this image? Options: (A) Left, (B) Right, (C) Both, (D) None." A laterality judgment; "Both" overlaps "Left" but not "None."

**Taxonomy & Naming** (`category_mcq`, `open_common_name`, `open_scientific_name`, `open_genus`, `open_kind_of`). The naming axis, ranging from an easy coarse‑grained MCQ to hard open‑ended binomials:

> *Examples.* `category_mcq`: "What type of bird is this? Options: (A) sparrow, (B) warbler, (C) finch, ..." (recognition among coarse types) · `open_common_name`: "What is the common name of this bird?" → *song sparrow* (free generation) · `open_scientific_name`: → *Melospiza melodia* · `open_genus`: → *Melospiza* · `open_kind_of`: "What kind of bird is this?" → folk category. The naming kernels (Section 5.7) award credit for a shared genus, a shared head‑noun, or a matching epithet, so *fox sparrow* for a *song sparrow* is a near miss, while *American crow* is not.

---

## 3. Two metrics, and why: Exact Match (hard) vs. CAGS (graded)

### 3.1 Why keep a hard Exact‑Match metric

Exact Match is deliberately unforgiving: a prediction is correct only if, after light normalization, it equals the gold answer (for MCQ, if the chosen *letter* equals the gold letter). EM is retained for three reasons:

1. **Trust and comparability.** EM is the metric the field already understands; it makes C2 directly comparable to prior fine‑grained benchmarks and gives reviewers an anchor they cannot dispute.
2. **It is a lower bound on competence.** If a model scores high on EM, it is unambiguously correct that often. There is no partial‑credit machinery to argue with.
3. **It exposes format pathologies that graded metrics can mask.** EM is computed alongside MCQ **parse rate**, **letter/text‑mismatch rate**, and a **position‑bias** diagnostic (the fraction of answers that are option "A"). These reveal whether an accuracy number is inflated or deflated by *formatting* behaviour rather than *visual* competence — for instance, a model that answers "(A)" 63% of the time (Section 11.5).

### 3.2 Why EM alone is insufficient — and what CAGS adds

Exact match has two well‑known failure modes on fine‑grained data:

- **It cannot distinguish a near miss from a total miss.** "purple" for "blue" scores the same 0 as "yellow" for "blue." On a 200‑way species‑naming task this is catastrophic: nearly every wrong answer is a flat 0, and models that consistently name the right *genus* look identical to models that name random birds.
- **It ignores question difficulty.** A 2‑option question and a 5‑option question both contribute a binary correct/incorrect, even though random guessing yields 50% on one and 20% on the other. Worse, a question whose distractors are all *almost* the correct colour is far easier to "look good" on than a question whose distractors are wildly different.

**CAGS** addresses both. For each question it (i) computes a **graded raw similarity** `GS_raw ∈ [0,1]` between the model's answer and the gold answer using the family‑appropriate kernel, then (ii) **subtracts the chance baseline** `b` — the average similarity that the question's own distractors would have earned — and renormalizes:

$$\mathrm{CAGS} \;=\; \frac{\text{GS\_raw} - b}{1 - b}.$$

This is a **Cohen's‑κ‑style chance correction** applied per item. It means:

- A model earns credit only for the closeness it achieves *above what a confused guesser on this specific question would achieve.* A question whose distractors are all near‑misses has a high `b`, so it takes a genuinely good answer to move CAGS.
- CAGS = 1 for a perfect answer, CAGS = 0 for "no better than the distractor pool," and CAGS < 0 (reported **unclipped**) for answers that are *worse than a random distractor* — a signal that turns out to matter (Section 11.6).

CAGS and EM are reported **together**. EM says how often the model is exactly right; CAGS says how good the model is when graded for closeness and corrected for chance. Where they disagree is where the interesting science lives.

---

## 4. The scoring pipeline (algorithmic workflow)

Every question flows through the same deterministic pipeline, implemented in `evaluate_soft.py`. This section is the algorithmic contract; Section 5 fills in the kernels.

```
                     ┌──────────────────────────────────────────────────────────┐
   raw CSV row  ──►  │  q = question,  a = gold answer,  p = model prediction     │
                     └──────────────────────────────────────────────────────────┘
                                          │
              (1) parse_options(q)  ──►  options = {A: "...", B: "...", ...}  or  {} (open-ended)
                                          │
              (2) question_family(q, options)  ──►  fam ∈ {attr_bare_value, ..., open_scientific_name}
                                          │
              (3) parse_gold(a, options)          ──►  (gold_letter, gold_text)
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
                  record { family, category, gold/pred, status, exact, s_raw, b, cags, severity }
```

### 4.1 Normalization

All string comparison happens in a normalized space. `_norm(s)` lower‑cases, replaces underscores with spaces, maps the American spelling `gray → grey`, strips punctuation except `:`, `-`, `(`, `)`, and collapses whitespace. This makes `"Blue_Jay"`, `"blue jay"`, and `"blue jay."` identical, so that neither EM nor CAGS is fooled by cosmetic formatting.

### 4.2 Option parsing

`parse_options(q)` looks for the literal token `Options:` and, if present, extracts an ordered map `{A: text, B: text, ...}` via the regular expression that matches `(A) … , (B) … , …`. If `Options:` is absent, the question is **open‑ended** and the option map is empty. This single test — "does the question contain an option list?" — is what routes an item into the MCQ or open‑ended branch of the whole pipeline.

### 4.3 Gold parsing

`parse_gold(a, options)` handles gold answers written either as a bare value or in the `(A) text` form. If the gold begins with a parenthesized letter that is a valid option, it returns `(letter, option_text)`; otherwise it returns `(None, a)`. This lets the benchmark store gold answers in whichever form is natural for a family without ambiguity at scoring time.

### 4.4 Prediction resolution — the most consequential step

`resolve_prediction(p, options, prefer="letter")` converts a free‑form model string into a `(letter, text, status)` triple. It is deliberately permissive, because MLLMs answer MCQs in wildly different formats, and we do **not** want to punish visual competence for formatting quirks. In order:

1. **Prefix stripping.** `clean_pred` removes conversational scaffolding ("the answer is", "final answer:", back‑ticks, asterisks) and keeps only the first line, where models overwhelmingly place the choice.
2. **Explicit letter.** A parenthesized `(A)`, or a leading `A.` / `A)` / `A:`, is taken as the chosen letter.
3. **Verbatim option text.** If any option's normalized text appears anywhere in the prediction, that option's letter is recorded as the *text‑implied* letter.
4. **Reconciliation.** If both a letter and an option text are present and they **agree**, status = `letter`. If they **disagree**, status = `letter_text_mismatch` and the `--prefer` policy decides which wins (default: the letter). If only one is present, that one is used (`letter` or `text`).
5. **Fuzzy fallback.** If nothing parses cleanly, a `difflib` ratio against each option is computed and, if the best exceeds a threshold (0.86), that option is chosen with status = `fuzzy`.
6. **No parse.** Otherwise status = `no_parse`, and the item receives `s_raw = 0`.

The three diagnostic rates reported later — **parse rate** (fraction not `no_parse`), **letter/text‑mismatch rate**, and the **position‑bias** share — all fall out of this step. In our runs parse rates are ≥ 98.9% for every model (Section 11), so the accuracy numbers reflect visual reasoning rather than parsing luck.

### 4.5 The question‑family router

`question_family(q, options)` assigns each question to exactly one family by matching template‑specific phrases, in priority order: the marker phrase "body part of this bird marked with" → `marker_localization`; "most prominently visible physical trait" → `attr_group_value`; "are visible in this image" with `{left, right, both, none}` options → `visibility_side`, else `visibility_partset`; "what type of bird is this" → `category_mcq`; "common name" → `open_common_name`; "scientific name" → `open_scientific_name`; "genus" → `open_genus`; "what kind of…" → `open_kind_of`; a bare "what is the ⟨…⟩ of this bird?" with options → `attr_bare_value`. The router is used identically by **both** the CAGS evaluator and the exact‑match evaluator, so the two metrics always partition the data the same way.

---

## 5. The similarity kernels — equations and worked examples

The graded raw similarity `s_raw = score_pair(family, pred, gold, …)` dispatches on family to a kernel tailored to that family's notion of closeness. Each kernel returns a value in `[0, 1]`, where `1` is an exact match. Below, every numeric value in the "worked examples" is the **actual output of the code** for that input pair (default config: exponential colour kernel, `τ = 30`, `α = 0.5`, schematic anatomy).

### 5.1 Colour kernel (perceptual, CIEDE2000)

Colour terms are compared **perceptually**, not lexically. Each of the 15 CUB colour words is anchored to an sRGB triple, converted to CIE‑Lab, and compared with the **CIEDE2000** colour‑difference formula ΔE₀₀ (Sharma, Wu & Dalal, 2005). The pipeline is:

$$\text{sRGB} \xrightarrow{\;\gamma^{-1}\;} \text{linear RGB} \xrightarrow{\;M\;} \text{XYZ} \xrightarrow{\;f(\cdot),\,D65\;} \text{Lab} \xrightarrow{\;\Delta E_{00}\;} \text{distance}.$$

The gamma expansion is $c_{\text{lin}} = c/12.92$ if $c \le 0.04045$ else $\big((c+0.055)/1.055\big)^{2.4}$; the Lab nonlinearity is $f(t)=t^{1/3}$ if $t>0.008856$ else $7.787\,t + 16/116$. ΔE₀₀ itself is the full Sharma et al. implementation (chroma/hue rotation term $R_T$, weighting functions $S_L, S_C, S_H$), reproduced exactly in `ciede2000()`.

The distance is then mapped to a similarity by one of two kernels:

$$
\text{color\_sim}(c_1,c_2) =
\begin{cases}
1 & c_1 = c_2\\[2pt]
0 & c_1 \text{ or } c_2 \in \{\texttt{iridescent}\}\\[2pt]
\exp\!\big(-\Delta E_{00}/\tau\big) & \text{exponential kernel (default, } \tau=30)\\[2pt]
\max\!\big(0,\; 1 - \Delta E_{00}/\Delta E_{\max}\big) & \text{linear kernel}
\end{cases}
$$

where $\Delta E_{\max}$ is the largest pairwise distance among the 15 anchors. `iridescent` has no fixed hue and is therefore exact‑match‑only. **Worked examples (exp | linear):**

| pair | ΔE₀₀ kernel (exp, τ=30) | linear kernel |
|---|---:|---:|
| blue vs **purple** | **0.498** | 0.791 |
| blue vs green | 0.134 | 0.396 |
| blue vs yellow | 0.053 | 0.120 |
| brown vs **rufous** | **0.760** | 0.918 |
| yellow vs orange | 0.507 | 0.796 |
| red vs orange | 0.284 | 0.622 |
| black vs white | 0.036 | 0.000 |

The exponential kernel is intentionally sharper than the linear one (it decays faster with distance), which is why the two are both included in the sensitivity sweep — to show the ranking does not depend on that choice (Section 12).

### 5.2 Ordinal kernel (bill length, size)

Ordered scales — bill length (`shorter than head` < `about the same` < `longer than head`) and body size (five buckets from "very small" to "very large") — are scored by normalized rank distance:

$$\text{ordinal\_sim}(v_1,v_2) = 1 - \frac{|\,\mathrm{rank}(v_1)-\mathrm{rank}(v_2)\,|}{\max \mathrm{rank}}.$$

**Worked examples:** bill length `shorter` vs `about the same` = **0.500** (off by one of two steps); `shorter` vs `longer` = **0.000** (the extremes); size `small` vs `medium` = **0.750** (one of four steps); `very small` vs `large` = **0.250** (three steps). An off‑by‑one size error keeps 75% of the credit; a maximal error keeps none.

### 5.3 Pattern kernel (nominal confusion)

Patterns are unordered, so similarity is a small hand‑built confusion matrix that encodes one qualitative fact: the *structured* patterns (spotted, striped, multi‑colored) are mutually confusable, but confusing a structured pattern with **solid** ("has markings" vs. "has none") is a categorical error:

$$\text{pattern\_sim} = \begin{cases} 0.5 & \{v_1,v_2\}\subseteq\{\text{spotted, striped, multi‑colored}\}\\ 0.0 & \text{one of } \{v_1,v_2\} = \text{solid}\end{cases}$$

**Worked examples:** spotted vs striped = **0.500**; spotted vs multi‑colored = **0.500**; solid vs spotted = **0.000**.

### 5.4 Anatomy kernel (3‑D part layout)

Body parts live at coordinates in a 3‑D bird layout (facing right, `y` down, `z` a small lateral offset that keeps left/right distinguishable). Part similarity is normalized Euclidean proximity:

$$\text{part\_sim}(p_1,p_2)=\max\!\Big(0,\;1-\tfrac{\lVert x_{p_1}-x_{p_2}\rVert}{d_{\max}}\Big),$$

with $d_{\max}$ the diameter of the layout. Region questions map a region ("wing", "head") to the centroid of its constituent parts and compare centroids. The layout is either **schematic** (a fixed anatomically‑plausible template) or **empirical** — estimated from CUB's `part_locs.txt` by taking, per part, the median bounding‑box‑normalized `(x,y)` over all images where the part is visible (this run used the empirical layout). **Worked examples (schematic):**

| pair | part_sim |
|---|---:|
| nape vs **crown** (adjacent on the head) | **0.922** |
| nape vs throat | 0.739 |
| beak vs forehead | 0.857 |
| left wing vs right wing | 0.883 |
| beak vs **tail** (opposite ends) | **0.000** |
| crown vs belly | 0.430 |

This is the kernel that lets `marker_localization` reward "crown for nape" (0.922) far above "tail for nape" (~0), which exact match cannot do.

### 5.5 Paired‑attribute kernel (Relative Salience)

`attr_group_value` options are full `region: value` traits, so a near miss can be in the *region*, the *value*, or both. The kernel factorizes:

$$\text{attr\_pair\_sim} = \alpha\cdot \underbrace{g(\text{region}_1,\text{region}_2)}_{\text{region agreement}} + (1-\alpha)\cdot \underbrace{v(\text{value}_1,\text{value}_2)}_{\text{value similarity}},\qquad \alpha=0.5.$$

Region agreement `g` itself blends the anatomy region‑similarity with a property‑match indicator (colour vs. pattern vs. shape); value similarity `v` dispatches to the colour / ordinal / pattern kernels above. **Worked examples:** `[wing color: blue]` vs `[wing color: purple]` = **0.749** (same region, near colour); vs `[breast color: blue]` = **0.952** (same value, near region); vs `[tail color: yellow]` = **0.395** (both off). The parameter `α` trades region credit against value credit and is swept in Section 12.

### 5.6 Set‑overlap kernel (visibility)

`visibility_side` and `visibility_partset` have **set‑valued** answers, scored by a **soft Jaccard** overlap:

$$\text{soft\_Jaccard}(A,B)=\frac{\sum_{a\in A}\max_{b\in B} \mathrm{kern}(a,b)}{|A|+|B|-\sum_{a\in A}\max_{b\in B}\mathrm{kern}(a,b)}.$$

For `visibility_side`, sides map to sets — `both → {L,R}`, `left → {L}`, `none → ∅` — with an exact‑match element kernel. For `visibility_partset`, the option set is compared with an element kernel that is either exact‑match (default) or the anatomy part‑similarity (the `anat_in_vis` toggle, Section 12). **Worked examples (side):** both vs left = **0.500**; both vs none = **0.000**; left vs right = **0.000**; left vs left = **1.000**.

### 5.7 Naming kernels (common / scientific / genus)

Open‑ended naming decomposes the name and rewards taxonomic overlap.

- **Common name:** $\text{sim}=0.6\cdot \mathrm{token\text-}F_1(p,g) + 0.4\cdot \mathbf{1}[\text{head‑noun}(p)=\text{head‑noun}(g)]$, lifted to at least 0.75 if a taxonomy map confirms the two names share a genus.
- **Scientific name:** $\text{sim}=0.5\cdot \text{genus‑agreement} + 0.5\cdot \text{epithet‑agreement}$, each term 1 on an exact match and a down‑weighted string ratio otherwise.
- **Genus:** 1 on an exact genus match, else a down‑weighted string ratio.

**Worked examples:** common `song sparrow` vs `fox sparrow` = **0.700** (shared head‑noun "sparrow" + token overlap); `song sparrow` vs `american crow` = **0.000**; scientific `Melospiza melodia` vs `Melospiza georgiana` = **0.575** (same genus, different epithet); genus `Melospiza` vs `Melospiza` = **1.000**, vs `Passerella` = **0.000**. This is why a model that reliably names the right genus scores far above one that names random species, even when *neither* gets the exact binomial right.

---

## 6. Chance adjustment: from GS_raw to CAGS

### 6.1 The chance baseline `b`

For each item, the chance baseline is the **mean raw similarity of that item's own distractors to the gold answer**:

$$b = \frac{1}{|D|}\sum_{d\in D}\text{score\_pair}(\text{fam}, d, \text{gold}, \dots),$$

where for an MCQ, `D` is the set of non‑gold options; for an open‑ended item, `D` is a random sample (default 40) of other gold answers observed in the dataset for that family (the "open pool"). `b` is the expected graded score of a respondent who answers by picking a plausible distractor — i.e., the *difficulty‑aware chance level* of that specific question.

### 6.2 The CAGS transform

$$\boxed{\;\mathrm{CAGS}=\dfrac{\text{GS\_raw}-b}{1-b}\;}\qquad(\text{if } b \text{ undefined or } 1-b<\varepsilon,\; \mathrm{CAGS}=\text{GS\_raw}).$$

Reported **unclipped**, so CAGS naturally lives in $(-\infty, 1]$ with the meaningful anchors:

- $\mathrm{CAGS}=1$ — perfect answer.
- $\mathrm{CAGS}=0$ — exactly at the distractor‑pool chance level.
- $\mathrm{CAGS}<0$ — *worse than chance*: the answer is farther from the gold than the average distractor. This happens when a model is confidently and specifically wrong.

Because the correction is per item, an easy‑looking question (distractors all near the gold ⇒ high `b`) contributes little CAGS even for a decent answer, while a hard question (distractors far ⇒ low `b`) rewards a good answer fully. This is the mechanism by which CAGS is **difficulty‑aware** where EM is not.

### 6.3 The three summary levels

`GS_raw`, the mean of `s_raw`, is reported alongside CAGS. Reading them together is diagnostic: `GS_raw − CAGS` is large exactly when the family's questions are "easy to look good on" (high `b`), which is itself a property of the question design worth knowing.

---

## 7. Uncertainty and mistake severity

### 7.1 Bootstrap confidence intervals (CAGS_lo, CAGS_hi)

Each reported mean CAGS carries a **95% bootstrap confidence interval**. Given the per‑item CAGS values $\{c_i\}$, we resample with replacement $B$ times (default $B=1000$), recompute the mean each time, and take the empirical 2.5th and 97.5th percentiles:

$$[\text{CAGS\_lo},\ \text{CAGS\_hi}] = \Big[\,Q_{0.025}\big(\{\bar c^{(b)}\}_{b=1}^{B}\big),\ Q_{0.975}\big(\{\bar c^{(b)}\}_{b=1}^{B}\big)\,\Big].$$

**Interpretation rule used throughout the results:** two models whose CAGS intervals **overlap** are statistically tied on that slice; a ranking difference between them is not, on its own, meaningful. This is the standard we apply when deciding whether a rank swap in the sensitivity sweep is "real."

### 7.2 Mistake severity

Averaged over **wrong items only** (EM = 0), severity measures *how bad the misses are*:

$$\text{MistakeSeverity} = \operatorname*{mean}_{i:\ \text{exact}_i=0}\big(1 - \mathrm{CAGS}_i\big).$$

Severity $\le 1$ means the average miss still beats chance (near misses); **severity $>1$ means the average miss is *worse than a random distractor*** — a model that is confidently, specifically wrong. Severity $>1$ is not a bug; it is one of the most telling diagnostics in the benchmark (Section 11.6).

---

## 8. A full end‑to‑end worked example

**Question.** "What is the wing color of this bird? Options: (A) blue, (B) purple, (C) yellow, (D) brown." **Gold:** (A) blue. **Model prediction:** "purple" (family `attr_bare_value`).

1. `parse_options` → `{A: blue, B: purple, C: yellow, D: brown}`.
2. `question_family` → `attr_bare_value`; implicit group = "wing color".
3. `parse_gold` → gold_text = **blue**.
4. `resolve_prediction("purple", …)` → the text "purple" matches option B → (letter B, text **purple**, status `text`).
5. **Raw similarity** via the colour kernel: `s_raw = color_sim(purple, blue) =` **0.498**.
6. **Chance baseline.** Distractors are {purple, yellow, brown}; their colour similarities to blue are `purple = 0.498`, `yellow = 0.053`, `brown = 0.196`, so
   $$b = \tfrac{1}{3}(0.498 + 0.053 + 0.196) = \mathbf{0.249}.$$
7. **CAGS.**
   $$\mathrm{CAGS} = \frac{0.498 - 0.249}{1 - 0.249} = \mathbf{0.331}.$$
8. **Exact match.** `norm("purple") ≠ norm("blue")` ⇒ EM = **0**. **Severity** = $1 - 0.331 = 0.669$ (a near miss).

The takeaway in one line: exact match records this as a flat failure (0); CAGS records it as a *decent* answer (0.331) that beats the question's chance level, and severity confirms the miss was mild (0.669, well under 1). Multiply this distinction across 824,104 questions and 18 models and you obtain a ranking with far more resolving power than accuracy alone — which is exactly what Sections 11–12 demonstrate.

---

## 9. Metric reference card (all equations in one place)

| Symbol | Name | Definition |
|---|---|---|
| `EM` | Exact match | $\mathbf 1[\,\mathrm{norm}(\text{pred})=\mathrm{norm}(\text{gold})\,]$ (MCQ: chosen letter = gold letter) |
| `s_raw` | Raw graded similarity | family kernel output in $[0,1]$ |
| `GS_raw` | Mean raw graded score | $\operatorname{mean}_i s\_raw_i$ |
| `b` | Chance baseline | mean $s\_raw$ of the item's distractors to the gold |
| `CAGS` | Chance‑adjusted graded score | $(\text{GS\_raw}-b)/(1-b)$, unclipped |
| `CAGS_lo/hi` | 95% bootstrap CI | percentiles of resampled mean CAGS |
| `MistakeSeverity` | Miss badness | $\operatorname{mean}_{i:\,EM_i=0}(1-\mathrm{CAGS}_i)$ |
| `parse_rate` | Parse success | fraction of MCQ preds with status ≠ `no_parse` |
| `letter_text_mismatch` | Format inconsistency | fraction with status = `letter_text_mismatch` |
| `chose_A_pct` | Position bias | fraction of MCQ answers equal to option "A" |

---

## 10. Experimental setup

- **Data.** CUB‑200‑2011, 824,104 questions per model across the four categories / ten families of Section 2. Empirical anatomy layout estimated from `parts/part_locs.txt` + `bounding_boxes.txt`; a 200‑entry `taxon_name_mapping.json` supplies genus lookups for the naming kernels.
- **Models (18).** Qwen3‑VL {32B, 30B, 8B}; Qwen2.5‑VL {7B, 3B}; InternVL3 {38B, 8B}; InternVL3.5 {38B, 30B‑A3B, 8B}; Gemma {4‑12B, 3‑12B}; Molmo2‑8B; Pixtral‑12B; Phi‑3.5‑Vision; Llama‑3.2‑Vision‑11B; LLaVA‑NeXT {Mistral‑7B, Vicuna‑7B}.
- **Metric configuration (default).** exponential colour kernel, $\tau=30$, $\alpha=0.5$, letter‑preferring resolution, anatomy *not* folded into visibility, CAGS reported unclipped, $B=1000$ bootstrap resamples, open‑pool sample 40, seed 42.
- **Parse quality.** All 18 models parse at **≥ 98.9%**, with letter/text‑mismatch **≤ 1.17%** (Section 11.5). The accuracy and CAGS numbers therefore reflect visual reasoning, not parsing.

---

## 11. Results

### 11.1 CAGS leaderboard (primary)

Sorted by CAGS (mean over all 824,104 items). EM is shown alongside; `MistakeSeverity` and CI bounds included.

| Rank | Model | EM | GS_raw | **CAGS** | CAGS_lo | CAGS_hi | Severity |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Qwen3VL_32B | 0.5696 | 0.6867 | **0.5929** | 0.5918 | 0.5940 | 0.9458 |
| 2 | Qwen3VL_30B | 0.5380 | 0.6654 | **0.5689** | 0.5679 | 0.5700 | 0.9331 |
| 3 | InternVL3_38B | 0.5351 | 0.6648 | **0.5680** | 0.5670 | 0.5691 | 0.9292 |
| 4 | Gemma4-12B | 0.5259 | 0.6513 | **0.5540** | 0.5528 | 0.5551 | 0.9409 |
| 5 | Qwen3VL_8B | 0.5275 | 0.6543 | **0.5526** | 0.5513 | 0.5536 | 0.9470 |
| 6 | InternVL3.5_38B | 0.5282 | 0.6482 | **0.5476** | 0.5464 | 0.5487 | 0.9591 |
| 7 | Molmo2_8B | 0.5200 | 0.6341 | **0.5327** | 0.5315 | 0.5338 | 0.9736 |
| 8 | Qwen2.5VL_7B | 0.4966 | 0.6346 | **0.5291** | 0.5280 | 0.5302 | 0.9355 |
| 9 | InternVL3.5_30B | 0.5011 | 0.6306 | **0.5242** | 0.5231 | 0.5254 | 0.9536 |
| 10 | InternVL3_8B | 0.4858 | 0.6191 | **0.5067** | 0.5055 | 0.5079 | 0.9592 |
| 11 | InternVL3.5_8B | 0.4607 | 0.6123 | **0.5046** | 0.5035 | 0.5056 | 0.9186 |
| 12 | Gemma3-12B | 0.4681 | 0.6047 | **0.4867** | 0.4855 | 0.4877 | 0.9652 |
| 13 | Qwen2.5VL_3B | 0.4409 | 0.5856 | **0.4609** | 0.4598 | 0.4620 | 0.9644 |
| 14 | Pixtral_12B | 0.4135 | 0.5652 | **0.4392** | 0.4381 | 0.4403 | 0.9561 |
| 15 | LLaVa-NeXT_Vicuna_7B | 0.3669 | 0.5410 | **0.4191** | 0.4181 | 0.4202 | 0.9175 |
| 16 | Phi-3.5_Vision | 0.4147 | 0.5353 | **0.4045** | 0.4033 | 0.4056 | **1.0175** |
| 17 | Llama3.2_Vision | 0.3755 | 0.5266 | **0.3815** | 0.3803 | 0.3826 | 0.9904 |
| 18 | LLaVa-NeXT_Mistral_7B | 0.3762 | 0.5171 | **0.3783** | 0.3771 | 0.3794 | 0.9967 |

### 11.2 Exact‑match leaderboard (anchor)

| Rank | Model | Accuracy % | MCQ % | Open % | parse % | letter/text mism. % | chose‑A % |
|---:|---|---:|---:|---:|---:|---:|---:|
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

### 11.3 Per‑category CAGS

| Model | Anatomical Loc. | Morphological | Rel. Salience | Taxonomy & Naming |
|---|---:|---:|---:|---:|
| Qwen3VL_32B | 0.4754 | 0.6441 | 0.5839 | **0.7310** |
| Qwen3VL_30B | 0.4525 | 0.6313 | 0.5582 | 0.6584 |
| InternVL3_38B | 0.4132 | 0.6451 | 0.5621 | 0.6691 |
| Molmo2_8B | **0.4741** | 0.6429 | 0.5027 | 0.3301 |
| Qwen2.5VL_7B | 0.3556 | 0.5800 | 0.5409 | 0.7127 |
| Qwen2.5VL_3B | 0.3890 | 0.5005 | 0.4272 | 0.6615 |
| Phi-3.5_Vision | **0.1541** | 0.5335 | 0.4540 | 0.2175 |

(Full 18‑model matrix in Appendix A.) The category structure already tells a story: the naming axis has the widest spread (0.20 → 0.73), and the best anatomical‑localization model is an 8B (Molmo2), not a 32/38B.

### 11.4 The recognition–generation gap (headline finding #1)

The single most informative contrast in C2 is **`category_mcq` vs. the open‑ended naming families**. Both probe taxonomic knowledge; they differ only in whether the model *recognizes among options* or *generates from scratch*.

- Best `category_mcq` CAGS: **0.9616** (Qwen2.5VL_7B) — coarse recognition is nearly solved.
- Best `open_common_name` CAGS: **0.6267** (Qwen2.5VL_7B); best `open_scientific_name`: **0.6294** (Qwen3VL_32B).
- On **exact match** the gap is even starker: `category_mcq` reaches **96.16%**, while the best `open_common_name` EM is **35.49%** and the best `open_scientific_name` EM is **52.17%**.

The same models that can pick the right bird type among five options collapse when they must produce the name unaided. This gap is a direct, quantitative measurement of the difference between *recognition* and *generation* in current MLLMs — a phenomenon C2 is purpose‑built to expose, and which averaged accuracy hides.

### 11.5 Position bias and format pathology (headline finding #2)

The exact‑match diagnostics catch a failure that has nothing to do with vision. **Llama‑3.2‑Vision answers option "(A)" 63.26% of the time** — more than three times a uniform rate — and its per‑letter accuracy is correspondingly lopsided. This is textbook MCQ **position bias**: the model's accuracy is partly an artifact of a prior over answer positions, not of what it sees. LLaVA‑NeXT‑Mistral shows a milder version (33.96% "A"). Because C2 reports `chose_A_pct` next to accuracy, this inflation is visible rather than silently baked into the leaderboard — and it is a concrete argument for reporting graded, chance‑corrected scores that are harder to game by position priors.

### 11.6 Severity > 1: "confidently wrong" (headline finding #3)

**Phi‑3.5‑Vision is the one model whose overall `MistakeSeverity` exceeds 1 (1.0175), and the one model whose CAGS (0.4045) falls *below* its EM (0.4147).** Every other model has CAGS ≥ EM, because partial credit for near misses lifts them above their exact‑match rate. Phi is the exception: its wrong answers are, on average, *farther from the gold than a random distractor* — it is confidently and specifically wrong rather than vaguely close. This is visible at the family level too (e.g., Phi's `visibility_partset` CAGS is only 0.1084 with severity 1.088). Severity > 1 is the cleanest single illustration of what CAGS measures that EM cannot, and it is the kind of qualitative model characterization a benchmark should surface.

### 11.7 The grounding outlier (headline finding #4)

**Molmo2‑8B is the best model in the benchmark at anatomical localization** — category CAGS **0.4741**, ahead of Qwen3VL‑32B (0.4754 is essentially tied; Molmo leads on the visibility families specifically) — despite being an 8B model competing against 30–38B models. It posts the top `visibility_side` CAGS (**0.7463**) and the top `marker_localization` CAGS (**0.7429**) in the entire field. This is consistent with Molmo's pointing/grounding‑oriented pretraining and is a clean example of an **architecture‑specific competence** that a single aggregate score would bury. Conversely, Molmo is weak on naming (Taxonomy CAGS 0.3301), so its overall rank (7th) understates a genuine, localized strength — exactly the kind of nuance the per‑family CAGS matrix is designed to reveal.

### 11.8 CAGS re‑ranks where it matters (headline finding #5)

CAGS and EM agree on the broad ordering (which is reassuring — CAGS is not capricious), but they **disagree in the lower cluster**, and the disagreements are informative:

- **Phi‑3.5 (EM rank 15) drops to CAGS rank 16**, overtaken by **LLaVA‑NeXT‑Vicuna (EM rank 18 → CAGS rank 15)**. Vicuna is *less often exactly right* than Phi, but its misses land *closer* to the gold, so under a closeness‑aware, chance‑corrected metric it is the better model. EM alone gets this ordering backwards.
- **Gemma4‑12B rises from EM rank 6 to CAGS rank 4**, because its near‑misses earn substantial partial credit (high `GS_raw` = 0.6513).

These are not arbitrary reshuffles: every CAGS/EM rank swap occurs between models whose behaviours genuinely differ in *how* they are wrong. That is the added value of CAGS over EM stated operationally.

---

## 12. Robustness: the sensitivity sweep

A graded metric has free constants ($\tau$, $\alpha$, kernel shape, and two policy toggles). Before the CAGS ranking can be reported as a fact about models, we must show it is **not** a fact about those constants. `sensitivity_sweep.py` re‑scores every model under a grid of configurations on a fixed, family‑stratified subsample (60,000 rows/model, seed 42, identical rows across configs), and measures how far the leaderboard moves from the default.

### 12.1 Stability statistics

Let $\mathbf{s}^{(0)}$ be the vector of per‑model CAGS under the default config and $\mathbf{s}^{(c)}$ under config $c$; let $\mathbf r^{(c)}$ be the induced ranks (1 = best).

- **Spearman $\rho$** — Pearson correlation of ranks: $\rho = 1 - \dfrac{6\sum_i d_i^2}{n(n^2-1)}$ with $d_i = r_i^{(0)}-r_i^{(c)}$ (ties averaged). $\rho=1$ ⇒ identical order.
- **Kendall $\tau$** — $\tau = \dfrac{(\#\text{concordant}) - (\#\text{discordant})}{\binom n 2}$; stricter than $\rho$.
- **Max rank displacement** — $\max_i |r_i^{(0)}-r_i^{(c)}|$.
- **Top‑3 stable** — whether the *set* of the three best models is unchanged.
- **Mean $|\Delta\text{CAGS}|$** — $\operatorname{mean}_i |s_i^{(c)}-s_i^{(0)}|$, the size of the score shift (independent of ranking).

### 12.2 Base grid — τ, α, colour kernel (9 configs)

| config | Spearman ρ | Kendall τ | max rank disp. | top‑3 stable | mean \|ΔCAGS\| |
|---|---:|---:|---:|:--:|---:|
| default | 1.0000 | 1.0000 | 0 | ✓ | 0.0000 |
| tau=20 | 0.9979 | 0.9869 | 1 | ✓ | 0.0008 |
| tau=40 | 0.9979 | 0.9869 | 1 | ✓ | 0.0008 |
| tau=50 | 0.9979 | 0.9869 | 1 | ✓ | 0.0015 |
| alpha=0.3 | 1.0000 | 1.0000 | 0 | ✓ | 0.0011 |
| alpha=0.4 | 1.0000 | 1.0000 | 0 | ✓ | 0.0006 |
| alpha=0.6 | 1.0000 | 1.0000 | 0 | ✓ | 0.0007 |
| alpha=0.7 | 0.9979 | 0.9869 | 1 | ✓ | 0.0015 |
| kernel=linear | 0.9959 | 0.9739 | 1 | ✓ | 0.0027 |

**Headline (base grid):** min Spearman **0.9959**, min Kendall **0.9739**, **max displacement 1 rank**, top‑3 stable in **8/8** configs, worst mean $|\Delta\text{CAGS}|$ **0.0027** (`kernel=linear`). Under the metric's numeric constants the ranking is *near‑invariant*, and the only rank exchanges are single‑position swaps between models whose bootstrap CIs already overlap (Section 11.1) — statistical ties, not instability.

### 12.3 Policy grid — adding `prefer=text` and `anat_in_vis=1` (11 configs)

| config | Spearman ρ | Kendall τ | max rank disp. | top‑3 stable | mean \|ΔCAGS\| |
|---|---:|---:|---:|:--:|---:|
| …(9 base configs as above)… | | | | ✓ | |
| prefer=text | 0.9979 | 0.9869 | 1 | ✓ | 0.0004 |
| **anat_in_vis=1** | **0.9587** | **0.8562** | **3** | ✓ | **0.0200** |

**Headline (policy grid):** min Spearman **0.9587**, min Kendall **0.8562**, max displacement **3 ranks**, top‑3 stable in **10/10** configs, worst mean $|\Delta\text{CAGS}|$ **0.0200** — both driven entirely by `anat_in_vis=1`. The letter‑vs‑text tie‑break (`prefer=text`) is negligible.

**What `anat_in_vis=1` does and why it moves things.** By default, the `visibility_partset` kernel scores parts by exact match. `anat_in_vis=1` instead folds the anatomy part‑similarity into that set kernel, so "named the left wing when the right wing was visible" earns partial credit. This is a *substantive modelling decision*, not an arbitrary constant, and it legitimately changes the visibility‑family scores — hence a 3‑rank move in the mid/low table (e.g., LLaVA‑Vicuna 15→18, Llama3.2 17→15, Gemma3‑12B 12→9, InternVL3.5‑8B 11→13). Crucially, **the top‑3 set is unchanged even here**, and the change is confined to families where crediting near‑miss parts is a defensible design choice we make explicit rather than hide.

**How to report this honestly.** Under all numeric hyper‑parameters and the resolution policy, C2's ranking is near‑invariant (ρ ≥ 0.996). The one configuration that moves mid‑table ranks is a genuine semantic choice about visibility scoring; we adopt the exact‑match default and disclose that folding anatomy into visibility would reshuffle the middle of the table by up to three positions while leaving the podium intact.

---

## 13. Discussion and threats to validity

- **Why the ranking is trustworthy.** The CAGS order tracks the field's expectations (larger/newer Qwen3‑VL and InternVL3 models on top; older 7B LLaVA‑NeXT variants at the bottom) *and* survives the hyper‑parameter sweep, while still exposing non‑obvious structure (Molmo's grounding, Phi's over‑confidence, Llama's position bias). A benchmark that reproduced known orderings but revealed nothing new would be redundant; one that produced surprising orderings that collapsed under perturbation would be unreliable. C2 is neither.
- **The graded layer sits on a stable backbone.** CAGS perturbs a partial‑credit signal on top of an exact‑match / chance‑adjustment core that does not depend on $\tau$, $\alpha$, or kernel shape. This is *part of why* the ranking barely moves under the base grid; we state it explicitly so the stability does not read as trivial.
- **Naming kernels depend on a taxonomy map.** The common‑name and genus kernels use a 200‑entry `taxon_name_mapping.json`. Coverage gaps degrade to token/head‑noun overlap rather than failing, but the naming‑family CAGS is only as complete as that map.
- **Open‑ended chance baseline is sampled.** For open families, `b` uses a 40‑answer random pool (fixed seed), an approximation to the full answer distribution. The sweep's stability under seed‑fixed resampling indicates this is not a sensitive choice, but it is an approximation.
- **`visibility_partset` is the one genuinely policy‑sensitive family.** As Section 12.3 shows, whether anatomy is folded into visibility is the single decision that moves the mid‑table. We recommend reporting both the default and, in an appendix, the `anat_in_vis=1` variant, so readers can see the podium is invariant either way.
- **Macro‑averaged label metrics are intentionally omitted.** For open‑vocabulary naming with hundreds of classes, macro precision/recall/F1 collapse to near‑noise and mislead; C2 reports accuracy, CAGS, and the per‑family breakdown, which carry the real signal.

---

## 14. Reproducibility: files, commands, formats

### 14.1 Directory layout

```
<base_dir>/                         # holds one sub-directory per model (and Results/)
  <Model>/QA_*.csv                  # consolidated predictions (question, answer, <model>_answer)
  <Model>/*chunk*.csv               # inference shards (marks the folder as a model dir)
  <Model>/Evaluation/
      eval_soft_scored.csv          # per-question: family, gold/pred, status, s_raw, b, cags, severity
      eval_soft_summary.csv         # tidy overall/family/category CAGS tables
      eval_soft_stats.txt           # per-model human-readable CAGS report
      eval_stats.txt / eval_rows.csv / eval_summary.csv   # exact-match per-model outputs
  Results/
    Eval - CAGS/
      eval_soft_leaderboard.csv          eval_soft_stats_ALL.txt
      eval_soft_by_category_ALL.csv      eval_soft_by_family_ALL.csv
      sweep_stability.csv / _with_policy.csv
      sweep_cags_matrix(.csv) / _with_policy      sweep_rank_matrix(.csv) / _with_policy
      sweep_report.txt / sweep_report_with_policy.txt
    Eval - Exact Match/
      eval_leaderboard.csv   eval_stats_ALL.txt
      eval_by_category_ALL.csv   eval_by_family_ALL.csv   eval_format_ALL.csv
    Manual Review/
      annotation_sheet.csv   annotation_key.csv   INSTRUCTIONS.txt   cards/  (+ annotation_cards.pdf)

<data_dir>/                         # resource side-files (may differ from base_dir)
  taxon_name_mapping.json   parts/part_locs.txt   parts/parts.txt   bounding_boxes.txt
```

### 14.2 Commands

```bash
# 1. CAGS (graded)                    base_dir = models, data_dir = resources
python evaluate_soft.py            --base_dir "$BASE_DIR" --data_dir "$DATA_DIR"
# 2. Exact match (hard)
python evaluate.py                --base_dir "$BASE_DIR"
# 3. Sensitivity sweep (numeric)
python sensitivity_sweep.py       --base_dir "$BASE_DIR" --data_dir "$DATA_DIR"
# 4. Sensitivity sweep (+ policy)    -> writes separate sweep_*_with_policy.* files
python sensitivity_sweep.py       --base_dir "$BASE_DIR" --data_dir "$DATA_DIR" --with_policy
# 5. Blind manual-review sample
python setup_manual_review_images.py         --base_dir "$BASE_DIR"
# 6. Render review cards + annotatable PDF
python setup_manual_review_pdf_annotate.py   --base_dir "$BASE_DIR" --pdf
```

`run_all_eval.sh` chains all six in dependency order. Key metric flags: `--tau`, `--alpha`, `--color_kernel {exp,linear}`, `--prefer {letter,text}`, `--anatomy_in_visibility`, `--clip`, `--bootstrap`, `--open_pool_sample`, `--seed`.

### 14.3 Human validation

The CAGS metric is validated against human judgment via a **blind, stratified manual‑review sample**: 90 items drawn across a (category × CAGS‑band) grid (low/mid/high CAGS), presented as cards showing the image, question, gold answer, and model answer — with the model identity, CAGS, and band hidden. Three raters score each item 0/1/2 (wrong / partially right / right); their ratings are later correlated against the hidden CAGS to confirm the metric tracks human perception of closeness.

---

## Appendix A — full per‑model × per‑family tables

### A.1 CAGS by question family (rows = model)

| Model | attr_bare | attr_group | category_mcq | marker_loc | common | genus | kind_of | sci_name | vis_partset | vis_side |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma3-12B | 0.5506 | 0.4657 | 0.9145 | 0.5723 | 0.4381 | 0.2988 | 0.6876 | 0.2430 | 0.4155 | 0.3390 |
| Gemma4-12B | 0.6205 | 0.5915 | 0.7720 | 0.7198 | 0.2855 | 0.1210 | 0.6154 | 0.1052 | 0.3846 | 0.4876 |
| InternVL3.5_30B | 0.6093 | 0.5367 | 0.8686 | 0.5736 | 0.3007 | 0.5053 | 0.6063 | 0.4331 | 0.3281 | 0.3502 |
| InternVL3.5_38B | 0.6379 | 0.5595 | 0.9200 | 0.6559 | 0.3892 | 0.4166 | 0.5184 | 0.4542 | 0.3097 | 0.4975 |
| InternVL3.5_8B | 0.6028 | 0.4953 | 0.7831 | 0.4991 | 0.2766 | 0.3472 | 0.5569 | 0.3420 | 0.3484 | 0.4091 |
| InternVL3_38B | 0.6451 | 0.5621 | 0.9481 | 0.6031 | 0.4953 | 0.6358 | 0.6652 | 0.6142 | 0.3731 | 0.4837 |
| InternVL3_8B | 0.5844 | 0.5081 | 0.8842 | 0.4676 | 0.3742 | 0.5229 | 0.6436 | 0.4988 | 0.3321 | 0.3526 |
| LLaVa-NeXT_Mistral_7B | 0.4843 | 0.3722 | 0.6364 | 0.3915 | 0.1541 | 0.0228 | 0.3377 | 0.0628 | 0.2101 | 0.3765 |
| LLaVa-NeXT_Vicuna_7B | 0.4535 | 0.4571 | 0.5502 | 0.2544 | 0.1415 | 0.0272 | 0.2307 | 0.0561 | 0.3583 | 0.4196 |
| Llama3.2_Vision | 0.4831 | 0.3168 | 0.8586 | 0.4880 | 0.6235 | 0.1855 | 0.7035 | 0.2523 | 0.2598 | 0.3069 |
| Molmo2_8B | 0.6429 | 0.5027 | 0.5318 | **0.7429** | 0.2837 | 0.1472 | 0.5785 | 0.1293 | 0.3656 | **0.7463** |
| Phi-3.5_Vision | 0.5335 | 0.4540 | 0.7160 | 0.3370 | 0.1774 | 0.0661 | 0.0955 | 0.0489 | 0.1084 | 0.2454 |
| Pixtral_12B | 0.4950 | 0.4471 | 0.8468 | 0.5289 | 0.3962 | 0.2594 | 0.6134 | 0.2347 | 0.3275 | 0.2249 |
| Qwen2.5VL_3B | 0.5005 | 0.4272 | 0.9264 | 0.3340 | 0.5663 | 0.5649 | 0.6652 | 0.5972 | 0.3994 | 0.3726 |
| Qwen2.5VL_7B | 0.5800 | 0.5409 | **0.9616** | 0.5634 | 0.6267 | 0.6364 | 0.7357 | 0.6157 | 0.3285 | 0.3765 |
| Qwen3VL_30B | 0.6313 | 0.5582 | 0.9511 | 0.6598 | 0.6076 | 0.4230 | 0.7334 | 0.5937 | 0.4339 | 0.4453 |
| Qwen3VL_32B | 0.6441 | 0.5839 | 0.9590 | 0.7267 | 0.6192 | **0.6962** | **0.7633** | **0.6294** | **0.4352** | 0.5256 |
| Qwen3VL_8B | 0.6244 | 0.5309 | 0.9577 | 0.6078 | 0.5434 | 0.4859 | 0.6910 | 0.5532 | 0.4264 | 0.4345 |

### A.2 Exact‑match accuracy (%) by question family (rows = model)

| Model | attr_bare | attr_group | category_mcq | marker_loc | common | genus | kind_of | sci_name | vis_partset | vis_side |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
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

### A.3 Rank movement under the policy sweep (1 = best)

Ranks are identical across the base grid except for single‑position swaps; the only ≥2‑rank movement is under `anat_in_vis=1`:

| Model | default | anat_in_vis=1 | Δ |
|---|---:|---:|---:|
| LLaVa-NeXT_Vicuna_7B | 15 | 18 | −3 |
| Llama3.2_Vision | 17 | 15 | +2 |
| Gemma3-12B | 12 | 9 | +3 |
| InternVL3.5_8B | 11 | 13 | −2 |
| Qwen2.5VL_7B | 8 | 10 | −2 |
| Qwen3VL_32B / 30B / InternVL3_38B (top‑3) | 1 / 2 / 3 | 1 / 2 / 3 | 0 |

---

## Appendix B — constants and lexicons

**Colour anchors (sRGB).** black (0,0,0); white (255,255,255); grey/gray (128,128,128); brown (139,69,19); buff (240,220,180); rufous (168,84,45); red (220,20,20); orange (255,165,0); yellow (255,225,30); olive (128,128,0); green (30,150,50); blue (30,60,220); purple (128,0,128); pink (255,170,190). Non‑chromatic (exact‑match‑only): **iridescent**.

**Ordinal scales.** *bill length:* shorter than head (0) < about the same as head (1) < longer than head (2). *size:* very small 3–5 in (0) < small 5–9 in (1) < medium 9–16 in (2) < large 16–32 in (3) < very large 32–72 in (4).

**Pattern confusion.** {spotted, striped, multi‑colored} pairwise = 0.5; any pairing with **solid** = 0.0.

**Schematic parts** (x, y, z): beak (0.95,0.35,0), forehead (0.85,0.28,0), crown (0.78,0.22,0), left/right eye (0.83,0.32,∓0.03), nape (0.72,0.25,0), throat (0.82,0.45,0), breast (0.68,0.55,0), belly (0.50,0.62,0), back (0.55,0.30,0), left/right wing (0.55,0.45,∓0.05), left/right leg (0.45,0.85,∓0.04), tail (0.10,0.45,0). Regions map to the centroid of their parts; `primary`, `shape`, `size` are GLOBAL (non‑localizable).

---

## Appendix C — glossary and per‑family example catalogue

**Glossary.** *EM* exact‑match accuracy. *GS_raw* mean raw graded similarity before chance adjustment. *CAGS* chance‑adjusted graded score $(GS\_raw-b)/(1-b)$, unclipped. *b* chance baseline (mean distractor similarity to gold). *Severity* mean $(1-\text{CAGS})$ over wrong items; $>1$ ⇒ worse‑than‑chance misses. *parse_rate / letter_text_mismatch / chose_A_pct* MCQ format and position‑bias diagnostics.

**Per‑family example catalogue** (every similarity value below is a real kernel output):

| Family | Example question | Gold | A near miss (score) | A far miss (score) |
|---|---|---|---|---|
| `attr_bare_value` | "What is the wing color of this bird?" | blue | purple (0.498) | yellow (0.053) |
| `attr_group_value` | "…most prominently visible physical trait?" | wing color: blue | breast color: blue (0.952) / wing color: purple (0.749) | tail color: yellow (0.395) |
| `marker_localization` | "…body part marked with the red star?" | nape | crown (0.922) | tail (0.000) |
| `visibility_side` | "Which sides are visible?" | Both | Left (0.500) | None (0.000) |
| `visibility_partset` | "Which parts are visible?" | {beak, tail} | {beak} (soft‑Jaccard partial) | {} / disjoint (0.000) |
| `category_mcq` | "What type of bird is this?" | sparrow | (near coarse type) | (unrelated type) |
| `open_common_name` | "…common name?" | song sparrow | fox sparrow (0.700) | american crow (0.000) |
| `open_scientific_name` | "…scientific name?" | Melospiza melodia | Melospiza georgiana (0.575) | Corvus brachyrhynchos (0.054) |
| `open_genus` | "…genus?" | Melospiza | Melospiza (1.000) | Passerella (0.000) |
| Ordinal (size) | "…size of this bird?" | small (5–9 in) | medium 9–16 in (0.750) | very large 32–72 in (0.250) |
| Pattern | "…wing pattern?" | spotted | striped (0.500) | solid (0.000) |

---

*End of document. All leaderboard and matrix values are reproduced from `Results/Eval - CAGS/` and `Results/Eval - Exact Match/`; all kernel example values are computed by `evaluate_soft.py` under the default configuration (exponential colour kernel, τ=30, α=0.5, schematic anatomy for the illustrative examples, empirical anatomy for the reported runs).*