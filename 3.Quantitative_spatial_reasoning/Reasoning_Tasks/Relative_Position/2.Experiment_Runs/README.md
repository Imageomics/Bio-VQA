<h1 align="center"> Relative-Position &rarr; Stage 3 &rarr; Inference Across 18 Models </h1>

<p align="center">
  <b>Byte-identical harness, closed-vocabulary prompting, longest-first direction parsing</b>
</p>

<p align="center">
  <img alt="Stage" src="https://img.shields.io/badge/stage-3%20of%204-blue">
  <img alt="Models" src="https://img.shields.io/badge/models-18-purple">
  <img alt="GPU" src="https://img.shields.io/badge/GPU--hours-~3.2-brightgreen">
  <img alt="Tokens" src="https://img.shields.io/badge/max__new__tokens-32-orange">
  <img alt="Parse" src="https://img.shields.io/badge/parse%20rate-1.000-success">
</p>

---

## Overview

> **Design contract:** all eighteen runners are **byte-identical outside a single `MODEL BLOCK`**, and
> the question text is **byte-identical across every item**. The runner appends only a format
> instruction, so a condition comparison could never be confounded by a prompt difference.

```bash
for f in *.py; do sed -n '/^# HARNESS/,$p' "$f" | md5sum; done | sort -u | wc -l
# → 1
```

---

## Table of Contents

- [Prompt Construction](#prompt-construction)
- [Three Prompt Decisions](#three-prompt-decisions)
- [Answer Extraction](#answer-extraction)
- [Output Columns](#output-columns)
- [Vision Budget](#vision-budget)
- [Execution](#execution)
- [Abort Conditions](#abort-conditions)
- [FAQ](#faq)

---

## Prompt Construction

The question is generated in Stage 1 and is identical for every row. The runner appends:

```python
INSTRUCTION = (
    "\n\n"
    "Answer with one of the four options exactly as written above, in lower "
    "case, and nothing else.\n"
    "Do not describe the boxes, do not explain your reasoning, do not restate "
    "the question.\n"
    "For example, if the answer is the option bottom-left, reply: bottom-left"
)
```

Rendered in full:

```
Where is the specimen inside the red box located with respect to the specimen
inside the blue box?

Use the specimen in the blue box as the reference point and name the direction
of the specimen in the red box from it. If the red specimen is beside the blue
one and roughly level with it, answer left or right. If it is above or below and
roughly in line, answer top or bottom. If it is clearly offset in both directions
at once, use the hyphenated name - top-left means above and to the left.

Answer with exactly one of these eight words: left, right, top, bottom,
top-left, top-right, bottom-left, bottom-right
```

---

## Three Prompt Decisions

### 1 &mdash; No box inventory

Earlier versions opened with:

> *"This image is marked with six boxes: two red, two blue, one green and one cyan. Exactly one red box
> and exactly one blue box contain a specimen; the other red box and the other blue box are empty. The
> green and cyan boxes are irrelevant to the question."*

**Removed.** *"The specimen inside the red box"* is a definite description that already denotes
uniquely &mdash; it denotes only if exactly one red box contains a specimen, which is true. Spelling
out the structure was redundant and handed the model a search strategy it should derive.

### 2 &mdash; No per-row option list

Earlier versions listed four options, which were **exactly the gold answer plus the three
mis-pairings**. The option set itself revealed that the decoy boxes mattered.

| Consequence of removal | Effect |
| :--- | :--- |
| Task becomes **eight-way open** | Chance drops 0.250 &rarr; **0.125** |
| `errOther` becomes informative | With four options the attribution was exhaustive and `errOther` was zero by construction |
| Option-position controls become undefined | `optPosBias` and `errOffOption` are dropped by Stage 4 rather than printed as NaN |

### 3 &mdash; The vocabulary is closed instead

Without options, an open-ended question against an **unstated** label set would score `"upper left"` or
`"northwest"` as wrong for wording rather than perception &mdash; the same failure as asking for a
viewpoint without naming the labels.

The eight words are stated once, identically on every row. **This gives the answer *format*, never the
answer.**

`MAX_NEW_TOKENS = 32` &mdash; `bottom-right` does not fit in 8, and an 8-token cap cost one model 36%
of its replies in an earlier C3 task.

---

## Answer Extraction

```python
DIRECTIONS = ["bottom-right", "bottom-left", "top-right", "top-left",
              "bottom", "right", "left", "top"]          # LONGEST FIRST
DIRS_BY_LEN = sorted(DIRECTIONS, key=len, reverse=True)
_DIR_ALT = "|".join(d.replace("-", r"[\s\-_]*") for d in DIRS_BY_LEN)

ANSWER_MARK_RE = re.compile(
    r"(?:answer|option|choice|reply|located|is)\b[^A-Za-z]{0,12}\**\s*("
    + _DIR_ALT + r")\b", re.I)
LEAD_RE = re.compile(r"^\W*(" + _DIR_ALT + r")\b", re.I)
ANY_RE  = re.compile(r"(" + _DIR_ALT + r")\b", re.I)
```

| Priority | Rule |
| :---: | :--- |
| 1 | explicit answer marker, **last** match |
| 2 | direction at the very start of the reply |
| 3 | **last** direction mentioned anywhere |

### Two traps

| Trap | Why it matters |
| :--- | :--- |
| **`left` is a substring of `bottom-left` and `top-left`** | Longest-first matching is mandatory; otherwise `bottom-left` resolves to `left`. |
| **Last, not first** | The direction guide names all eight directions, so a model echoing any part of the question would be scored on the first word it repeated rather than on its conclusion. |

Separator-tolerant: `Bottom Left`, `bottom_left`, `bottom-left` all canonicalise to `bottom-left`.

Verified against 12 realistic reply forms &mdash; **12/12** &mdash; including `**right**`,
`The answer is bottom-right.`, and
`Options: left, bottom-right, top-right, right. The specimen is left.`

**Observed parse rate on the released run: `1.000` for all 18 models.**

---

## Output Columns

| Column | Contents |
| :--- | :--- |
| `<KEY>_answer_unfiltered` | raw model string, verbatim |
| `<KEY>_answer` | resolved direction, or empty |

`qa_<KEY>.summary.json` per chunk records the model id, the prompt instruction, vision budget,
`off_option` count, and indicative accuracy.

> `off_option` &mdash; answering a direction not among the offered options &mdash; is recorded but is
> **not defined** now that options are not shown. It appears as `NaN`, and Stage 4 drops the column
> rather than printing a column of NaNs.

---

## Vision Budget

| Family | Other C3 tasks | Here |
| :--- | :--- | :--- |
| Qwen | `max_pixels = 1280·28·28` | **`2560·28·28`** (~2.0 MP) |
| InternVL | 6 tiles | **12 tiles** |

The subjects are two marked beetles on a plate and the answer depends on their relative centre
positions, so the boxes must remain distinguishable after downsampling. Recorded per run in
`summary.json`.

---

## Execution

```bash
bash run_all.sh --dry --lanes 3
bash run_all.sh --lanes 3
```

Each array task builds a disposable venv in node-local `$TMPDIR`, removed by a `trap` on exit &mdash;
including on failure, timeout and `scancel`. Package pins cannot collide between models.

| Package | Failure it fixes |
| :--- | :--- |
| `torch>=2.6` | Gemma-3 `or_mask_function` requirement |
| `sentencepiece` | LLaVA-NeXT Vicuna slow-tokenizer instantiation |
| `protobuf` | the follow-on `ImportError` |
| `einops` | InternVL remote code imports `rearrange` at load |
| `--timeout 60` | a truncated PyPI read killed one task in an earlier sweep |

**Runtime: ~3.2 GPU-hours serial** &mdash; the cheapest sweep in Component 3, since the corpus is only
1,000 items.

---

## Abort Conditions

| Condition | Behaviour |
| :--- | :--- |
| no image resolves on disk | abort before GPU work |
| none of the first 20 replies name a direction | abort &mdash; misconfigured model |
| more than half the batches raise | abort without writing a partial CSV |
| output already exists | skip unless `OVERWRITE=1` |