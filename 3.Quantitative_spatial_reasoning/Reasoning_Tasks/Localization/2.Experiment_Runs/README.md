<h1 align="center"> Localization &rarr; Stage 3 &rarr; Inference Across 18 Models </h1>

<p align="center">
  <b>Dimension-aware prompting, multi-convention box resolution, SLURM orchestration</b>
</p>

<p align="center">
  <img alt="Stage" src="https://img.shields.io/badge/stage-3%20of%204-blue">
  <img alt="Models" src="https://img.shields.io/badge/models-18-purple">
  <img alt="Params" src="https://img.shields.io/badge/params-3.75B%E2%80%9338.4B-informational">
  <img alt="GPU" src="https://img.shields.io/badge/GPU--hours-~11.5-brightgreen">
  <img alt="Tokens" src="https://img.shields.io/badge/max__new__tokens-48-orange">
</p>

---

## Overview

> **Design contract:** all eighteen runners are **byte-identical outside a single `MODEL BLOCK`**. This
> matters more here than in the categorical tasks &mdash; bounding-box output is sensitive to the format
> instruction, the token budget and the vision budget, and any of those differing between models would
> invalidate the comparison.

```bash
for f in *.py; do sed -n '/^# HARNESS/,$p' "$f" | md5sum; done | sort -u | wc -l
# → 1
```

---

## Table of Contents

- [Prompt Construction](#prompt-construction)
- [The Worked-Example Hazard](#the-worked-example-hazard)
- [Box Resolution](#box-resolution)
- [Output Columns](#output-columns)
- [Vision Budget](#vision-budget)
- [Model Registry](#model-registry)
- [Execution](#execution)
- [Abort Conditions](#abort-conditions)
- [FAQ](#faq)

---

## Prompt Construction

```python
INSTRUCTION_HEAD = (
    "\n\n"
    "Respond with four numbers in this exact format and nothing else:\n"
    "[x, y, width, height]\n"
    "where x and y are the pixel coordinates of the top-left corner of the "
    "box, measured from the top-left corner of the WHOLE image described "
    "above, and width and height are the size of the box in pixels.\n"
    "Do not use fractions, percentages or normalised coordinates. Do not "
    "explain.\n"
    "Example: [1240, 88, 540, 410]"
)

def build_prompt(question, n_panels, width, height):
    frame  = PANEL_FRAME if int(n_panels) > 1 else ""
    frame += ("The whole image is %d pixels wide and %d pixels tall.\n\n"
              % (int(width), int(height)))
    return frame + str(question).strip() + INSTRUCTION_HEAD
```

Rendered:

```
The whole image is 2400 pixels wide and 1920 pixels tall.

Provide bounding box coordinates for the subject animal (zebra) in this image.

Respond with four numbers in this exact format and nothing else:
[x, y, width, height]
where x and y are the pixel coordinates of the top-left corner of the box,
measured from the top-left corner of the WHOLE image described above, and width
and height are the size of the box in pixels.
Do not use fractions, percentages or normalised coordinates. Do not explain.
Example: [1240, 88, 540, 410]
```

| Element | Why |
| :--- | :--- |
| **Canvas dimensions, per row** | Without them "in pixels" has no referent, and the resolver has nothing to normalise against. Read from each file header once at split time. |
| **Explicit corner semantics** | `[x, y, w, h]` is ambiguous between top-left+size and centre+size. |
| **"Do not use fractions…"** | An attempt to enforce one convention. It does not succeed &mdash; see [Box Resolution](#box-resolution). |
| **`MAX_NEW_TOKENS = 48`** | Four numbers with decimals do not fit in 8. |

### The `PANEL_FRAME` that never fires

The prompt builder retains a three-panel clause even though this tier is single-image, so the
scaffolding stays **byte-identical** if multi-panel tiers are added later. A prompt difference between
tiers would contaminate the tier comparison.

---

## The Worked-Example Hazard

`Example: [1240, 88, 540, 410]` was added to fix output formatting. It caused a different problem
&mdash; **five models echo it verbatim**:

| Model | Replies exactly equal to the example |
| :--- | ---: |
| phi35_vision | **75.8%** |
| molmo2_8B | **61.9%** |
| llava_next_mistral_7B | 37.2% |
| llama3_2_11B | 29.7% |
| internvl3_5_8B-Instruct | 13.0% |

The gold base rate of that exact box is **0.0%**, so every occurrence is an echo rather than a
coincidence.

> ⚠️ Those cells are **not ability estimates**. `molmo2_8B` is explicitly pointing-trained and should be
> strong at grounding; its last-place finish is an *instruction-following* failure, not a grounding
> failure. Retained here as-run for fidelity; a future revision should test whether the example can be
> dropped without losing format compliance.

The raw-output column makes this audit possible after the fact, which is the argument for retaining it.

---

## Box Resolution

Models do not agree on a coordinate convention. Scoring a correct localisation as zero because of its
units would measure format compliance rather than grounding.

### Candidate space &mdash; four scales &times; two formats

| Scale | Multiplier | Format | Interpretation |
| :--- | :--- | :--- | :--- |
| `abs` | &times;1 | `xywh` | pixels, top-left + size |
| `norm1000` | &times;W/1000, &times;H/1000 | `xyxy` | two corners |
| `norm01` | &times;W, &times;H | | |
| `pct` | &times;W/100, &times;H/100 | | |

### Ordered first-valid-wins, by magnitude

```python
mx = max(abs(t) for t in v)
if   mx <= 1.02:                            order = [N01, PCT, N1K, ABS]
elif mx <= 1000.0 and mx < 0.25 * max(W,H): order = [N1K, PCT, ABS, N01]
elif mx <= 100.0:                           order = [PCT, N1K, ABS, N01]
else:                                       order = [ABS, N1K, PCT, N01]

for name, sx, sy in order:
    for fmt in ("xywh", "xyxy"):
        if _valid(...):
            return box, name + "_" + fmt
```

A candidate is valid if it has positive area, lies within the canvas (2% tolerance), and covers between
$10^{-5}$ and $1.0$ of it.

> **First valid wins, deliberately.** An earlier version scored candidates and kept the best, with a
> mild preference for smaller boxes &mdash; which handed every Qwen-style 0–1000 answer to the absolute
> reading, because a box in the corner "scores better" than the correct one. **Ordering must decide,
> not scoring.**

Verified 14/14 across conventions and canvas sizes:

| Input | Resolved as | IoU vs gold |
| :--- | :--- | ---: |
| `[9803, 163, 2792, 2446]` | `abs_xywh` | 1.000 |
| `[9803, 163, 12595, 2609]` | `abs_xyxy` | 1.000 |
| `[778, 47, 222, 699]` | `norm1000_xywh` | 0.997 |
| `[0.778, 0.047, 0.222, 0.699]` | `norm01_xywh` | 0.997 |
| `Box: [9803, 163, 2792, 2446].` | `abs_xywh` | 1.000 |

**Nothing is repaired.** No clipping to bounds, no aspect correction, no snapping to a nearby plausible
box. A wrong box stays wrong, or the error rate stops meaning anything.

### The per-row decision is not final

On a 2400-pixel canvas, `600` could be either convention, and no single reply can separate them. The
row-level mode is **recorded**, and Stage 4 re-resolves each model under a convention chosen from its
whole set of replies.

---

## Output Columns

| Column | Contents |
| :--- | :--- |
| `<KEY>_answer_unfiltered` | the raw model string, verbatim |
| `<KEY>_answer` | the resolved box as JSON `[x, y, w, h]` in pixels |
| `<KEY>_parse_mode` | `norm1000_xyxy`, `abs_xywh`, `unparsed`, `invalid`, … |

> `<KEY>_parse_mode` is **unique to this task** and is what makes the model-level convention correction
> auditable. The raw column supports the prompt-echo audit above.

`qa_<KEY>.summary.json` per chunk records the model id, both prompt fragments, the vision budget, the
parse-mode histogram, and indicative IoU.

---

## Vision Budget

| Family | Other C3 tasks | Here |
| :--- | :--- | :--- |
| Qwen | `max_pixels = 1280·28·28` | **`2560·28·28`** (~2.0 MP) |
| InternVL | 6 tiles | **12 tiles** |
| Phi-3.5 | `num_crops = 16` | 16 |

Raised because the target can be a small animal and the coordinate answer depends on resolving its
edges. Recorded per run; **results must never be compared across budgets**.

---

## Model Registry

3.75B &ndash; 38.4B across eleven families: Qwen2.5-VL, Qwen3-VL (dense and MoE), InternVL3,
InternVL3.5 (dense and MoE), LLaVA-NeXT (Vicuna and Mistral), Gemma-3, Gemma-4, Pixtral, Phi-3.5,
Molmo2, Llama-3.2-Vision.

`MODEL_KEY` is the column prefix in every output CSV and the join key for cross-task correlation.
**It must not change after a run.**

---

## Execution

```bash
bash run_all.sh --dry --lanes 3
bash run_all.sh --lanes 3
bash run_all.sh --only qwenVL3_32B
```

Each array task builds a disposable venv in node-local `$TMPDIR`, removed by a `trap` on exit
&mdash; including on failure, timeout and `scancel`. Package pins cannot collide: Gemma-4 needs git
HEAD, the Qwen3 family needs current `transformers`, everything else is pinned to `4.57.1`.

| Package | Failure it fixes |
| :--- | :--- |
| `torch>=2.6` | Gemma-3 `or_mask_function` requirement |
| `sentencepiece` | LLaVA-NeXT Vicuna slow-tokenizer instantiation |
| `protobuf` | the follow-on `ImportError` |
| `einops` | InternVL remote code imports `rearrange` at load |
| `--timeout 60` | a truncated PyPI read killed one task in an earlier sweep |

| Resource | Value |
| :--- | ---: |
| Serial GPU-hours | **~11.5** |
| Wall clock | ~36 min |

---

## Abort Conditions

| Condition | Behaviour |
| :--- | :--- |
| no image resolves on disk | abort before GPU work |
| none of the first 20 replies contain four numbers | abort &mdash; misconfigured model |
| more than half the batches raise | abort without writing a partial CSV |
| output already exists | skip unless `OVERWRITE=1` |