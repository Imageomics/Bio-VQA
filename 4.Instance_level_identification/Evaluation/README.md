<h1 align="center"> Bio-VQA &rarr; Individual Re-Identification &rarr; Evaluation </h1>

<p align="center">
  <b>Signal-detection scoring of eighteen vision&ndash;language models under adversarial pair selection</b>
</p>
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="Models" src="https://img.shields.io/badge/models-18-orange">
  <img alt="Cells" src="https://img.shields.io/badge/cells-270-purple">
  <img alt="Answers" src="https://img.shields.io/badge/answers-270%2C000-lightgrey">
  <img alt="Metric" src="https://img.shields.io/badge/metric-d%E2%80%B2%20%2B%20criterion%20c-red">
  <img alt="Unparsed" src="https://img.shields.io/badge/unparsed-0.000%25-brightgreen">
</p>
<p align="center">
  We score discriminability and response bias separately, because on a balanced binary task<br>
  accuracy cannot tell a model that perceives identity from one that always answers &ldquo;Yes&rdquo;.<br>
  Six of our eighteen models make that distinction necessary.
</p>

---

## Table of Contents

1. [Overview](#1-overview)
2. [Why We Do Not Report Accuracy Alone](#2-why-we-do-not-report-accuracy-alone)
3. [Signal Detection Theory: d&prime; and Criterion c](#3-signal-detection-theory-d-and-criterion-c)
4. [Tiered Answer Parsing](#4-tiered-answer-parsing)
5. [The Scoring Pipeline](#5-the-scoring-pipeline)
6. [Metric Reference](#6-metric-reference)
7. [Running the Evaluation](#7-running-the-evaluation)
8. [Output Artifacts](#8-output-artifacts)
9. [Result: Parsing Integrity](#9-result-parsing-integrity)
10. [Result: Discriminability](#10-result-discriminability)
11. [Result: Response Bias](#11-result-response-bias)
12. [Result: The Difficulty Manipulation](#12-result-the-difficulty-manipulation)
13. [Result: By Species](#13-result-by-species)
14. [The Specialist Baseline](#14-the-specialist-baseline)
15. [Why Below Chance Is Not Deliberate Error](#15-why-below-chance-is-not-deliberate-error)
16. [What We Conclude](#16-what-we-conclude)
17. [Limitations](#17-limitations)
18. [Reproducibility](#18-reproducibility)

---

## 1. Overview

This stage turns model answers into measurements. The corpus and the inference runners are documented
separately.

The task is a balanced binary verification: 500 same-individual and 500 different-individual pairs
per species × condition. That balance makes raw accuracy uninformative on its own — a model answering
"Yes" to everything scores exactly 0.500, which accuracy cannot distinguish from a coin flip or from
genuine partial competence.

We therefore score in the framework of signal detection theory, reporting discriminability (d&prime;)
separately from response bias (criterion *c*), alongside balanced accuracy, MCC, Cohen's &kappa;, and
the full confusion matrix. We compute bootstrapped confidence intervals per cell and test the
predicted `easy ≥ random ≥ hard` ordering.

Scope: **18 models × 5 species × 3 conditions = 270 cells, 270,000 answers.**

```
<species>/instance_identification_1000_<condition>/QA_<model>.csv
            │
            ▼
   evaluate_c4.py   ──►  QA_all_results.csv        (270 cells × 41 columns)
                    ──►  QA_<model>_results.csv     (per model)
                    ──►  QA_summary_by_model.csv    (pooled + effect sizes)
                    ──►  QA_report.txt              (narrative, overall)
                    ──►  QA_<model>_report.txt      (narrative, per model)
                    ──►  QA_unparsed_samples.csv    (audit trail)

   baseline_reid.py ──►  baseline_results.csv       (specialist + construction check)
                    ──►  baseline_report.txt
```

---

## 2. Why We Do Not Report Accuracy Alone

Three hypothetical models on a balanced set:

| Model | Behaviour | Accuracy | d&prime; | criterion *c* |
| --- | --- | ---: | ---: | ---: |
| A | answers "Yes" always | 0.500 | 0.00 | extreme |
| B | flips a fair coin | 0.500 | 0.00 | 0.00 |
| C | discriminates weakly, leans "Yes" | 0.520 | 0.35 | −0.90 |

Accuracy calls A and B identical and ranks C barely above both. d&prime; separates them: A and B have
no discriminability and differ only in bias, while C has real if weak sensitivity that accuracy hides
behind its bias.

This is our situation. `llava_next_vicuna_7B` answers "Yes" to essentially every item — accuracy
0.500, d&prime; = 0.000, criterion *c* = −3.09. Its accuracy reflects the class balance, not
perception.

---

## 3. Signal Detection Theory: d&prime; and Criterion c

We treat "same individual" as signal present and compute two rates per cell:

- **hit rate** `H = P(says Yes | same individual)`
- **false-alarm rate** `F = P(says Yes | different individual)`

Then

```
d′ = z(H) − z(F)                        discriminability
c  = −½ · ( z(H) + z(F) )               response criterion
```

where `z` is the inverse standard normal CDF.

**d&prime;** measures the separation between the model's evidence distributions in standard
deviations. It is invariant to where the model sets its threshold, so a biased but perceptive model
scores as perceptive.

**Criterion *c*** is the bias d&prime; discards. `c < 0` means a standing preference for "Yes",
`c > 0` for "No".

| d&prime; | balanced accuracy | reading |
| ---: | ---: | --- |
| 0.0 | 50% | no discrimination |
| 0.5 | 60% | weak |
| 1.0 | 69% | moderate |
| 2.0 | 84% | strong |
| 3.0 | 93% | near-ceiling |

**Negative d&prime;** means `F > H`: the model says "Yes" more often to different-individual pairs
than to same-individual pairs. It separates the classes and assigns them the opposite answers. This
is the central diagnostic of the component; §15 covers it.

We apply the log-linear correction `(x + 0.5) / (n + 1)` to both rates unconditionally rather than
only at ceiling, since applying it selectively biases comparisons between cells.

When we report a model's figure across species, we sum the confusion counts first and compute
d&prime; once from the pooled counts, rather than averaging per-species d&prime; values.

---

## 4. Tiered Answer Parsing

The graded-similarity metric used elsewhere in Bio-VQA (CAGS) does not transfer here: there is no
synonym set for "Yes" and no near-miss between two mutually exclusive labels. Our analogue is tiered
parsing — we resolve each answer by the strictest rule that applies and record which rule was used.

| Tier | Rule | Example |
| --- | --- | --- |
| `exact` | whole answer is yes/no after stripping markdown, quotes, trailing punctuation | `Yes` · `no.` · `**No**` |
| `prefix` | answer begins with yes/no as a word | `Yes, they are the same individual` |
| `contains_unique` | exactly one of the two words appears, as a word | `The answer is no` |
| `negated` | explicit same/different phrasing, no bare yes/no token | `not the same individual` |
| `unparsed` | anything else, including answers containing both words | `Yes or No` · `I cannot determine` |

Two guards matter. `"Yes or No"` is the model echoing the prompt, and without an explicit rule it
resolves to `Yes` by prefix match. `"I am not sure"` contains *not* and resolves to `No` under
`contains_unique`, turning an abstention into a confident wrong answer — so we match refusal
phrasings first and route them to `unparsed`.

Tier counts appear as columns in every output row, so parser leniency is a reported number rather
than an assumption.

---

## 5. The Scoring Pipeline

1. **Discovery.** Walk `--base_dir` for `instance_identification_<N>_<cond>` directories. Every
   `QA_<model>.csv` inside is a model result; we take the model name from the filename, so new models
   are picked up without a registry.
2. **Column resolution.** Prefer `<model>_answer`, else the unique non-reserved column.
3. **Ground truth.** Parse `qa_pair` JSON per row; count and exclude unreadable rows.
4. **Answer resolution.** Tiered parsing, tier tallied.
5. **Confusion matrix.** TP / FN / FP / TN over resolved items.
6. **Metrics.** Accuracy, balanced accuracy, precision, recall, specificity, NPV, F1, MCC,
   Cohen's &kappa;, hit rate, false-alarm rate, yes-rate, d&prime;, criterion *c*.
7. **Confidence intervals.** Seeded percentile bootstrap, 2,000 resamples, for accuracy and balanced
   accuracy.
8. **Aggregation.** Pooled per model × condition from summed counts; `easy − hard` with a
   two-proportion difference CI; per-species monotonicity count.
9. **Reporting.** CSVs plus narrative reports generated from the numbers.

---

## 6. Metric Reference

| Metric | Range | Chance | Bias-robust | Notes |
| --- | --- | ---: | :---: | --- |
| `accuracy` | 0–1 | 0.500 | ✗ | resolved items only |
| `accuracy_all_items` | 0–1 | — | ✗ | unresolved counted as errors |
| `balanced_accuracy` | 0–1 | 0.500 | ✓ | mean of sensitivity and specificity |
| `d_prime` | −∞…+∞ | 0.000 | ✓ | primary metric |
| `criterion_c` | −∞…+∞ | 0.000 | — | negative = Yes-leaning |
| `mcc` | −1…+1 | 0.000 | ✓ | correlation form of the confusion matrix |
| `cohens_kappa` | −1…+1 | 0.000 | ✓ | chance-corrected agreement |
| `f1` | 0–1 | — | ✗ | positive class only |
| `hit_rate`, `false_alarm_rate` | 0–1 | — | — | the rates d&prime; is built from |
| `yes_rate` | 0–1 | 0.500 | — | degeneracy indicator |
| `coverage` | 0–1 | — | — | fraction of items resolved |

---

## 7. Running the Evaluation

```bash
python evaluate_c4.py \
  --base_dir /path/to/4-Instance-Level-Identification \
  --exclude internvl3_5_38B internvl3_5_30B_A3B
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--base_dir` | required | root containing the per-species directories |
| `--key_string` | `QA_` | result-file prefix |
| `--dir_pattern` | `instance_identification_1000_` | task-directory prefix |
| `--out_dir` | `<base_dir>/evaluation` | destination |
| `--exclude` | — | model names to skip |
| `--n_boot` | 2000 | bootstrap resamples per cell |
| `--seed` | 0 | bootstrap seed |
| `--expect_models` | — | names the sweep should contain; absentees listed in the report |
| `--no_reports` | off | CSVs only |

Runtime is seconds, CPU only. We use `--exclude` for the two superseded InternVL runs rather than
deleting their files, so the raw outputs stay on disk.

The specialist baseline runs in two stages:

```bash
python baseline_reid.py --base_dir <root> --mode extract    # GPU
python baseline_reid.py --base_dir <root> --mode evaluate   # CPU, seconds
```

---

## 8. Output Artifacts

| File | Contents |
| --- | --- |
| `QA_all_results.csv` | 270 rows × 41 columns; every metric for every cell |
| `QA_<model>_results.csv` | the 15 rows for one model |
| `QA_summary_by_model.csv` | pooled per condition, `easy − hard` effects, monotonicity counts |
| `QA_report.txt` | overall narrative: scope, parsing, discriminability, bias, manipulation, species |
| `QA_<model>_report.txt` | per-model narrative with verdict and per-species ordering tests |
| `QA_unparsed_samples.csv` | raw strings that failed to parse, with ground truth and tier |
| `baseline_results.csv` | specialist and construction-check metrics per species × condition |

---

## 9. Result: Parsing Integrity

```
0 of 270,000 answers unresolved (0.000%)
exact = 269,903    contains_unique = 97    prefix = 0    negated = 0    unparsed = 0
```

Every answer resolved, and 99.96% at the strictest tier. Answer format is not a source of measurement
error here, and no result below depends on parser leniency.

---

## 10. Result: Discriminability

Pooled across species. We key verdicts on the **random** condition, since easy-condition pairs are
separable by appearance alone and grading on them would be circular.

| Model | easy | random | hard | Verdict |
| --- | ---: | ---: | ---: | --- |
| qwenVL3_32B | 1.455 | **0.696** | −0.604 | weak discrimination |
| internvl3_8B-Instruct | 2.062 | 0.684 | −0.738 | weak discrimination |
| qwenVL2_5_3B | 2.290 | 0.655 | −0.764 | weak discrimination |
| internvl3_5_30B_A3B-Instruct | 1.563 | 0.568 | −0.529 | weak discrimination |
| phi35_vision | 1.919 | 0.562 | −0.932 | weak discrimination |
| qwenVL3_30B | 1.281 | 0.518 | −0.248 | weak discrimination |
| qwen3vl_8B | 1.635 | 0.492 | −0.589 | weak discrimination |
| molmo2_8B | 1.820 | 0.491 | −0.411 | weak discrimination |
| llava_next_mistral_7B | 1.738 | 0.476 | −0.550 | weak discrimination |
| internvl3_38B-Instruct | 1.597 | 0.463 | −0.725 | weak discrimination |
| qwenVL2_5_7B | 1.335 | 0.442 | −0.395 | weak discrimination |
| gemma4_12B | 1.304 | 0.351 | −0.414 | weak discrimination |
| internvl3_5_38B-Instruct | 1.278 | 0.258 | −0.460 | weak discrimination |
| internvl3_5_8B | 0.905 | 0.236 | −0.286 | weak discrimination |
| gemma3_12B | 0.946 | 0.120 | −0.091 | degenerate |
| pixtral_12B | 0.651 | 0.119 | −0.240 | at chance |
| llama3_2_11B | 0.864 | 0.000 | −0.208 | degenerate |
| llava_next_vicuna_7B | 0.000 | 0.000 | 0.000 | degenerate |

Two findings.

**A ceiling.** The highest neutral-condition d&prime; across all eighteen models is **0.696**, about
64% balanced accuracy. No model reaches d&prime; = 1.0. Individual re-identification is close to
unsolved for this generation of vision–language models, independently of any manipulation.

**Below chance under adversarial pairing.** **17 of 18** models have negative d&prime; in the `hard`
condition, from −0.932 to −0.091. The one exception emits a constant answer and cannot be moved by
any manipulation.

Model scale does not predict performance: the 3B Qwen2.5 ranks third, above the 38B InternVL.

---

## 11. Result: Response Bias

Criterion *c*, pooled; negative means a standing preference for "Yes".

| Model | *c* easy | *c* random | *c* hard |
| --- | ---: | ---: | ---: |
| llava_next_vicuna_7B | −3.09 | −3.09 | −3.09 |
| llava_next_mistral_7B | −2.41 | −2.46 | −2.54 |
| gemma3_12B | −1.60 | −1.99 | −1.99 |
| llama3_2_11B | −1.13 | −1.59 | −1.59 |
| qwenVL2_5_3B | −0.88 | −1.32 | −1.53 |
| qwenVL2_5_7B | −1.18 | −1.34 | −1.51 |
| … | | | |
| qwenVL3_30B | +0.13 | +0.15 | −0.06 |

Seventeen of eighteen models lean toward "Yes", and the lean deepens from `easy` to `hard` in almost
every case. As pairs become more confusable, models become more affirmative rather than more
cautious. This holds across families and scales, so it is a general tendency rather than a property
of one architecture.

Six models exceed |*c*| = 1.0 in the neutral condition. For those we report d&prime; and balanced
accuracy rather than raw accuracy.

---

## 12. Result: The Difficulty Manipulation

Accuracy difference `easy − hard`, pooled, with 95% two-proportion difference intervals.

| Model | easy − hard (pts) | 95% CI | Verdict |
| --- | ---: | --- | --- |
| internvl3_8B-Instruct | +38.8 | [+37.0, +40.6] | confirmed |
| internvl3_5_30B_A3B-Instruct | +38.4 | [+36.6, +40.1] | confirmed |
| internvl3_38B-Instruct | +38.4 | [+36.6, +40.2] | confirmed |
| phi35_vision | +37.0 | [+35.2, +38.8] | confirmed |
| qwenVL3_32B | +36.9 | [+35.1, +38.7] | confirmed |
| molmo2_8B | +36.0 | [+34.2, +37.7] | confirmed |
| qwenVL2_5_3B | +35.3 | [+33.5, +37.1] | confirmed |
| … | | | |
| llava_next_mistral_7B | +5.2 | [+3.2, +7.1] | confirmed |
| llava_next_vicuna_7B | +0.0 | [−2.0, +2.0] | flat |

**17 confirmed · 1 flat · 0 reversed.**

Per-species: **77 of 90** model × species combinations show the full `easy ≥ random ≥ hard` ordering
on balanced accuracy. Under a null where condition is irrelevant, a random ordering of three
conditions is monotone with probability 1/6; the exact binomial probability of 77 or more from 90 is
**1.9 × 10⁻⁴⁶**.

The one flat case is the model that emits a constant answer, which by construction cannot respond to
any manipulation. Its flatness confirms the effect runs through the model's perception rather than
through anything in the item format.

---

## 13. Result: By Species

Mean d&prime; across models.

| Species | easy | random | hard | easy − hard |
| --- | ---: | ---: | ---: | ---: |
| Whale shark | 2.812 | 0.276 | −0.906 | 3.718 |
| Spotted hyena | 1.743 | 0.672 | −0.126 | 1.869 |
| Grévy's zebra | 1.346 | 0.391 | −0.596 | 1.942 |
| Leopard | 1.280 | 0.517 | −0.188 | 1.468 |
| Beluga whale | 0.508 | 0.383 | −0.758 | 1.267 |

Species differ in how strongly the manipulation bites, and beluga is diagnostically useful. Its
easy-condition separation is by far the weakest — 0.508 against 1.28–2.81 elsewhere. We predicted
this from the corpus before running any model: beluga is the only species whose `random` condition
has a negative similarity margin (−0.0613), meaning that under uniform sampling different-individual
pairs are already more similar than same-individual pairs. DINOv2's embedding geometry for belugas is
not organised by individual identity, so a manipulation defined in that space has less to work with.

---

## 14. The Specialist Baseline

`baseline_reid.py` scores the same pairs with a purpose-built wildlife re-identification embedding
(**MegaDescriptor-L-384**, via `timm`), thresholded on cosine similarity. We run it to establish
whether the `hard` condition defeats appearance-based matching in general or vision–language models
specifically.

AUC is threshold-free and converts to the same units as the VLM table via `d′ = √2 · z(AUC)`. Where
we report accuracy, we fit the threshold on the `random` condition and apply it unchanged to `easy`
and `hard`.

**MegaDescriptor**, Grévy's zebra:

| Condition | AUC | d&prime; | accuracy |
| --- | ---: | ---: | ---: |
| easy | 0.927 | 2.055 | 0.902 |
| random | 0.698 | **0.733** | 0.683 |
| hard | 0.247 | **−0.969** | 0.317 |

The specialist is above chance on `easy` and `random` and below chance on `hard`, at a magnitude
exceeding every vision–language model we tested. That system is a cosine similarity compared against
a threshold — no language model, no reasoning component — so its sign reversal comes from the
anti-correlation the condition induces and nothing else. This is our strongest evidence that the
`hard` condition targets appearance-based matching rather than penalising language models.

We make no claim of a VLM-versus-specialist capability gap. On neutral pairs the specialist reaches
d&prime; = 0.733 against the best VLM's 0.696 — a tie, and neither figure is strong in absolute
terms. These pairs are difficult for appearance-based methods generally.

**Construction check.** The same script scores the DINOv2 similarities stored in
`pairs_metadata.csv`. These are not a baseline — we built the pairs by thresholding exactly these
values — and we report them to quantify the circularity:

| Species | easy AUC | random AUC | hard AUC |
| --- | ---: | ---: | ---: |
| Leopard | 1.000 | 0.835 | 0.114 |
| Whale shark | 1.000 | 0.718 | 0.053 |
| Spotted hyena | 1.000 | 0.784 | 0.095 |
| Grévy's zebra | 1.000 | 0.658 | 0.068 |
| Beluga whale | 0.990 | 0.386 | 0.010 |

An easy-condition AUC of 1.000 is the construction doing exactly what it claims.

---

## 15. Why Below Chance Is Not Deliberate Error

The natural objection is that 0.500 should be a floor on a binary task, so below-chance performance
implies the model knows the answer and reports the opposite.

0.500 is the expected accuracy of a responder whose output is independent of the stimulus. It is a
floor only for coin-flippers. Any system whose output depends on the image can be driven below 0.500
by selecting images adversarially, and that requires no knowledge of the label.

**The mechanism.** Models apply a decision rule approximating *"answer Yes if the two animals look
alike."* That rule correlates with truth in general, which is why the same models score above chance
in `random`. The `hard` condition selects pairs so that within the sample, visual similarity is
anti-correlated with identity. The rule is applied unchanged, the rule is sound, and the sample is
arranged so the rule points the wrong way.

**What negative d&prime; does and does not establish.** It establishes that the model's decision
variable is anti-correlated with the label in this sample. It does not establish that the variable is
identity. We claim the variable is appearance similarity — a quantity that normally correlates with
identity and which this condition anti-correlates with it. On this account the model never forms an
identity judgment, so there is no identity judgment being inverted.

**Two lines of evidence.**

A system incapable of deliberation shows the same effect: MegaDescriptor reaches d&prime; = −0.969 on
`hard`. A thresholded dot product cannot know an answer and choose otherwise.

Sensitivity magnitude is preserved while only the sign flips:

| Model | d&prime; random | d&prime; hard | \|d&prime;\| |
| --- | ---: | ---: | --- |
| internvl3_8B-Instruct | 0.684 | −0.738 | 0.68 → 0.74 |
| qwenVL3_32B | 0.696 | −0.604 | 0.70 → 0.60 |
| qwenVL2_5_3B | 0.655 | −0.764 | 0.66 → 0.76 |
| molmo2_8B | 0.491 | −0.411 | 0.49 → 0.41 |

This rules out "the model has no signal and is guessing", which drives d&prime; toward 0 rather than
to −0.7. It is the signature of a constant mechanism meeting inverted labels.

---

## 16. What We Conclude

**Individual re-identification is largely unsolved.** No model exceeds d&prime; = 0.70 on neutral
pairs, and four of eighteen sit at or below the floor. The finest granularity of visual
discrimination in Bio-VQA is where current vision–language models perform worst.

**Models judge identity by resemblance.** The manipulation moves seventeen of eighteen models in the
predicted direction, with 77 of 90 model × species orderings monotone. The size of `easy − hard`
gives a per-model measure of how far each relies on that shortcut, ranging from +38.8 points down to
+5.2.

**The failure is systematic, not noisy.** Seventeen models fall below chance under adversarial
pairing while preserving their sensitivity magnitude. They are not confused by hard pairs; they are
reliably misled by them.

**The manipulation targets appearance matching, not language models.** A specialist re-identification
embedding is driven below chance on the same pairs, further than any VLM.

**Response bias is a general tendency.** Seventeen of eighteen models lean toward "Yes", and the lean
deepens as pairs become harder.

---

## 17. Limitations

**Circularity.** We define the conditions by DINOv2 cosine similarity, and several models we evaluate
share vision-encoder lineage with DINOv2 or related self-supervised objectives, so the manipulation
partly indexes shared embedding geometry rather than a general property of visual identity
representation. We do not replicate the construction in a second embedding space. §14 quantifies the
upper bound of this concern.

**No achievable ceiling.** The specialist ties rather than exceeds the best VLM on neutral pairs, so
we do not establish the absolute difficulty of these pairs. Our claims are comparative across models
and conditions. We also evaluate the specialist on full frames rather than bounding-box crops, which
understates it, since it is trained on tight crops.

**Single prompt.** We use one instruction phrasing throughout. The near-universal negative criterion
may be an architectural tendency or an artifact of wording, and this design cannot separate them.

**Composite format.** Pairs are one side-by-side image rather than two, adding a spatial-parsing
burden to the identity judgment. Constant across models and conditions, so it does not confound the
difficulty effect, but it lowers absolute levels.

**Multiple comparisons.** 18 models × 5 species × 3 pairwise contrasts yields several hundred
comparisons, so some nominally significant per-cell results are expected by chance. We treat per-cell
results as exploratory and the pooled tests as confirmatory.

**Item non-independence.** Some source images appear in more than one pair, so bootstrap intervals
computed under independence are mildly optimistic.

**d&prime; assumptions.** We assume equal-variance Gaussian evidence distributions and use d&prime;
as a bias-corrected effect size, not as a claim about internal representations.

---

## 18. Reproducibility

| Aspect | Guarantee |
| --- | --- |
| Determinism | Bootstrap seeded; repeated runs give identical intervals. |
| Discovery | Models found by filename; no registry to drift. |
| Parsing | Rules fixed in code; tier counts reported per cell. |
| Provenance | Every row records `source_file` and `answer_column`. |
| Both readings | `accuracy` and `accuracy_all_items` both emitted. |
| Thresholds | Verdict constants declared at module top. |
| Baseline | Extraction idempotent and shardable; threshold fitted on `random`, applied out-of-fit elsewhere. |

Full regeneration from stored model answers:

```bash
python evaluate_c4.py --base_dir <root> --exclude internvl3_5_38B internvl3_5_30B_A3B
python baseline_reid.py --base_dir <root> --mode evaluate
```

Both stages read only committed CSVs. No model inference is needed to reproduce any number here.