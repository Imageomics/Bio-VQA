<h1 align="center"> Counting-Relative &rarr; Stage 3 &rarr; Inference Across 18 Models </h1>

<p align="center">
  <b>Byte-identical harness, prompt construction, answer extraction, SLURM orchestration</b>
</p>

<p align="center">
  <img alt="Stage" src="https://img.shields.io/badge/stage-3%20of%204-blue">
  <img alt="Models" src="https://img.shields.io/badge/models-18-purple">
  <img alt="Params" src="https://img.shields.io/badge/params-3.75B%E2%80%9338.4B-informational">
  <img alt="GPU" src="https://img.shields.io/badge/GPU--hours-~189-red">
  <img alt="Harness" src="https://img.shields.io/badge/harness-md5%20verified-brightgreen">
</p>

---

## Overview

| Component | Count | Role |
| :--- | :---: | :--- |
| `<model>.py` | 18 | Inference runner, one per model |
| `Preprocess/split_qa.py` | 1 | Manifest assembly &amp; sharding |
| `Preprocess/stats.py` | 1 | Pre-run audit |
| SLURM array tasks | 18 &times; 6 | One per (model, chunk) |

> **Design contract:** all eighteen runners are **byte-identical outside a single `MODEL BLOCK`**.
> Everything that could influence a score &mdash; prompt, parser, batching, abort conditions, summary
> record &mdash; is shared. Only `load_model()` and `run_batch()` differ.

---

## Table of Contents

- [The Controlled-Experiment Principle](#the-controlled-experiment-principle)
- [Model Registry](#model-registry)
- [Prompt Construction](#prompt-construction)
- [Answer Extraction](#answer-extraction)
- [Output Columns](#output-columns)
- [Vision Budget](#vision-budget)
- [Execution](#execution)
- [Abort Conditions](#abort-conditions)
- [Verification](#verification)
- [FAQ](#faq)

---

## The Controlled-Experiment Principle

```bash
for f in *.py; do sed -n '/^# HARNESS/,$p' "$f" | md5sum; done | sort -u | wc -l
# → 1
```

**A single hash means no model received different treatment.** If this ever returns more than 1, the
comparison is invalid and must not be reported.

```
┌──────────────────────────────────────────────────────────────┐
│  HEADER          docstring, imports, PIL config              │  shared
├──────────────────────────────────────────────────────────────┤
│  EXPERIMENT      FRAME, INSTRUCTION, build_prompt()          │  shared
│  CONSTANTS                                                   │
├──────────────────────────────────────────────────────────────┤
│  MODEL BLOCK     MODEL_ID, MODEL_KEY, DEFAULT_BATCH_SIZE     │  ← ONLY
│                  load_model(), run_batch()                   │    difference
├──────────────────────────────────────────────────────────────┤
│  HARNESS         parser, run_all(), summary.json, main()     │  shared
└──────────────────────────────────────────────────────────────┘
```

---

## Model Registry

| Model file | Params | Family | `MODEL_KEY` |
| :--- | ---: | :--- | :--- |
| `qwenVL2_5_3B` | 3.75B | Qwen2.5-VL | `qwenVL2_5_3B` |
| `phi_3_5_vision` | 4.2B | Phi-3.5 | `phi35_vision` |
| `llava_next_vicuna_7B` | 7.1B | LLaVA-NeXT | `llava_next_vicuna_7B` |
| `llava_next_mistral_7B` | 7.6B | LLaVA-NeXT | `llava_next_mistral_7B` |
| `molmo_8B` | 8.0B | Molmo2 | `molmo2_8B` |
| `intern3_8B` | 8.1B | InternVL3 | `internvl3_8B-Instruct` |
| `intern3_5_8B` | 8.1B | InternVL3.5 | `internvl3_5_8B-Instruct` |
| `qwenVL2_5_7B` | 8.3B | Qwen2.5-VL | `qwenVL2_5_7B` |
| `qwen3_8B` | 8.8B | Qwen3-VL | `qwen3vl_8B` |
| `llama_3_2_vision_11B` | 10.7B | Mllama | `llama3_2_11B` |
| `gemma3_12B` | 12.2B | Gemma-3 | `gemma3_12B` |
| `gemma4_12B` | 12.2B | Gemma-4 | `gemma4_12B` |
| `pixtral_12B` | 12.4B | Pixtral | `pixtral_12B` |
| `intern3_5_30B` | 30.5B | InternVL3.5 MoE | `internvl3_5_30B_A3B-Instruct` |
| `qwenVL3_30B` | 30.5B | Qwen3-VL MoE | `qwenVL3_30B` |
| `qwenVL3_32B` | 32.8B | Qwen3-VL | `qwenVL3_32B` |
| `intern3_38B` | 38.4B | InternVL3 | `internvl3_38B-Instruct` |
| `intern_3_5_38B` | 38.4B | InternVL3.5 | `internvl3_5_38B-Instruct` |

> `MODEL_KEY` is the column prefix in every output CSV and the join key for cross-task correlation.
> **It must not change after a run.**

---

## Prompt Construction

```python
FRAME = (
    "This image is a 2x2 grid of four separate photographs:\n"
    "  Image A - top-left      Image B - top-right\n"
    "  Image C - bottom-left   Image D - bottom-right\n\n"
)

INSTRUCTION = (
    "\n\n"
    "Count the animals in each of the four photographs and compare the "
    "totals. The photographs are shown at different scales, so compare how "
    "many animals there are, not how large they appear.\n"
    "Exactly one photograph is the answer. Choose it even if the difference "
    "is small.\n"
    "Answer immediately with the letter. Do not describe the photographs and "
    "do not explain your reasoning.\n"
    "Respond with one letter and nothing else: A, B, C or D."
)

def build_prompt(question):
    return FRAME + str(question).strip() + INSTRUCTION
```

### Why each clause earns its place

| Clause | Failure it prevents |
| :--- | :--- |
| **Grid legend** | The question says "which of these 4 images", but the canvas is itself an image. Without the frame, "Image A" is ambiguous between a quadrant and the whole picture. |
| **Scale sentence** | Panels are pasted at different magnifications, so apparent animal size does not track count. Load-bearing: six models still score below chance on `different`, consistent with size comparison despite the warning. |
| **"even if the difference is small"** | Suppresses hedging on close quadruples. |
| **"Answer immediately… do not explain"** | With `MAX_NEW_TOKENS = 32`. An 8-token cap cost one model 36% of its replies in an earlier C3 task when it began `**Step 1:`. |

> The frame is applied to **both conditions unchanged**. A prompt difference between conditions would
> confound the `same` vs `different` contrast, which is the task's primary result.

---

## Answer Extraction

Two traps, both encountered in practice.

### Trap 1 &mdash; "Image" contains a lower-case `a`

```python
re.search(r"([ABCD])", "Image B (top-right)", re.I).group(1)   # → 'A'   ✗
```

Every option label resolves to `A`, silently. All patterns are anchored or case-sensitive.

### Trap 2 &mdash; narration enumerates before concluding

```
"Image A has 3, Image B has 12, Image D has 5. The answer is B."
```

A first-match rule returns `A`. **Every pattern takes the last match.**

### Resolution ladder

```python
POSITIONS = {"topleft": "A", "topright": "B",
             "bottomleft": "C", "bottomright": "D"}

ANSWER_MARK_RE = re.compile(
    r"(?:answer|choice|selection|pick|most|least|fewest|greatest|largest)\b"
    r"[^A-Za-z0-9]{0,12}(?:is\s+)?(?:image\s*)?\**\s*([ABCD])(?![A-Za-z])",
    re.I)                        # re.I is REQUIRED — "**Answer:** B" is capitalised
LEAD_RE   = re.compile(r"^\W*(?:image\s+)?([ABCD])(?![A-Za-z])", re.I)
IMAGE_RE  = re.compile(r"[Ii]mage\s*\**\s*([ABCD])(?![A-Za-z])")
POS_RE    = re.compile(r"\b(top|bottom)[\s\-_]*(left|right)\b", re.I)
LETTER_RE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")
```

| Priority | Rule | Example matched |
| :---: | :--- | :--- |
| 1 | explicit answer marker, last match | `**Answer:** B` |
| 2 | letter at the very start of the reply | `D` |
| 3 | last `Image X` mention | `The answer is Image C.` |
| 4 | last quadrant name &rarr; mapped | `the bottom-right photograph` &rarr; `D` |
| 5 | last standalone letter | `… C` |

> **Quadrant names are mapped, not discarded.** `bottom-left` &rarr; `C` is a correct answer in the
> wrong format; scoring it zero would measure format compliance rather than comparison.

Verified against 17 realistic reply forms. The `re.I` on `ANSWER_MARK_RE` is not cosmetic: without it,
`**Answer:** B` fails the marker test and falls through to rule 3, returning `D` from the enumeration.

---

## Output Columns

| Column | Contents |
| :--- | :--- |
| `<KEY>_answer_unfiltered` | the raw model string, verbatim |
| `<KEY>_answer` | the resolved letter, or empty |

Nothing is lost, and Stage 4 never re-parses free text. The raw column also supports post-hoc audits
not anticipated at run time &mdash; the prompt-echo check in the cross-task analysis reads it directly.

Per chunk, `qa_<KEY>.summary.json`:

```json
{
  "model_key": "qwenVL3_32B",
  "model_id": "Qwen/Qwen3-VL-32B-Instruct",
  "task": "counting_relative",
  "rows": 7069, "scored": 7069, "unparsed": 0,
  "batch_size": 2, "max_new_tokens": 32,
  "vision_budget": "qwen max_pixels=2560*28*28",
  "prompt_frame": "…", "prompt_instruction": "…",
  "indicative_accuracy": 0.555, "indicative_pred_top": "C",
  "seconds": 4113.2, "finished_utc": "…"
}
```

This is the permanent run record; SLURM `.out` files are convenience only.

---

## Vision Budget

Four photographs share one canvas, so each panel receives roughly a quarter of the token budget, and
the animals are often small aerial subjects to begin with.

| Family | Other C3 tasks | Here |
| :--- | :--- | :--- |
| Qwen | `max_pixels = 1280·28·28` | **`2560·28·28`** (~2.0 MP) |
| InternVL | 6 tiles | **12 tiles** |
| Phi-3.5 | `num_crops = 16` | 16 (unchanged) |

Applied identically within each family and recorded per run. **Results must never be compared across
different budgets.**

---

## Execution

```bash
bash run_all.sh --dry --lanes 3     # print the schedule, submit nothing
bash run_all.sh --lanes 3           # 3 models in flight, each over 6 chunks
bash run_all.sh --only gemma3_12B   # rerun one model
```

Each array task builds a **disposable virtual environment** in node-local `$TMPDIR`, removed by a
`trap` on exit &mdash; including on failure, timeout and `scancel`. Nothing is shared between tasks, so
`transformers` pins cannot collide: Gemma-4 requires git HEAD, the Qwen3 family requires current
`transformers`, everything else is pinned to `4.57.1`.

| Package in the common install line | Failure it fixes |
| :--- | :--- |
| `torch>=2.6` | Gemma-3: `Using or_mask_function … require torch>=2.6` |
| `sentencepiece` | LLaVA-NeXT Vicuna: `Cannot instantiate this tokenizer from a slow version` |
| `protobuf` | the follow-on `ImportError` from the same path |
| `einops` | InternVL remote code imports `rearrange` at module load |
| `--timeout 60` | a truncated PyPI read killed one array task in an earlier sweep |

| Resource | Value |
| :--- | ---: |
| Serial GPU-hours | **~189** |
| Wall clock, 6 chunks &times; 3 lanes | ~10.5 h |

---

## Abort Conditions

The harness stops rather than writing a plausible-looking but meaningless CSV.

| Condition | Behaviour |
| :--- | :--- |
| no image resolves on disk | abort before any GPU work |
| none of the first 20 replies contain an option letter | abort &mdash; misconfigured model, wrong class, or broken chat template |
| more than half the batches raise | abort without writing a partial CSV |
| output CSV already exists | skip, unless `OVERWRITE=1` |

> The second is the important one. A model loaded with randomly initialised mismatched layers produces
> fluent garbage; failing at row 20 is far better than discovering it after an hour of inference.

---

## Verification

```bash
# all harnesses identical
for f in *.py; do sed -n '/^# HARNESS/,$p' "$f" | md5sum; done | sort -u | wc -l   # 1

# no double-backslash in the pip line (a real bug from an earlier task)
grep -c '\\\\' *.sh                                                                # 0

# every model wrote every chunk
python ../3.Evaluate/merge_results.py --out_dir <RESULTS_DIR>                      # all complete
```

