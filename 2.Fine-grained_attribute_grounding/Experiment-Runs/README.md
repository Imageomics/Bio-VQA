<h1 align="center">Bio-VQA &rarr; Fine-Grained Visual Attribute Grounding</h1>

<p align="center">
  <b>CUB-200-2011 &mdash; Vision&ndash;Language Model Inference Runners</b>
</p>
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="Source dataset" src="https://img.shields.io/badge/source-CUB--200--2011-green">
  <img alt="Models" src="https://img.shields.io/badge/models-8%20families-orange">
  <img alt="Decoding" src="https://img.shields.io/badge/decoding-greedy-lightgrey">
  <img alt="Release" src="https://img.shields.io/badge/release-v1.0-brightgreen">
</p>
<p align="center">
  A uniform batch-inference harness for evaluating vision&ndash;language models on the<br>
  Bio-VQA C2 question corpus &mdash; one script per model family, one CSV in, one CSV out.
</p>

---

## Table of Contents

1. [Overview](#1-overview)
2. [Design Principles](#2-design-principles)
3. [Runner Contents](#3-runner-contents)
4. [Input and Output Schemas](#4-input-and-output-schemas)
5. [Environment and Dependencies](#5-environment-and-dependencies)
6. [Command-Line Interface](#6-command-line-interface)
7. [Prompt Construction](#7-prompt-construction)
8. [Robustness](#8-robustness)
9. [Running on a SLURM Cluster](#9-running-on-a-slurm-cluster)
10. [Post-Processing: Llama-3.2](#10-post-processing-llama-32)
11. [End-to-End Pipeline](#11-end-to-end-pipeline)
12. [Reproducibility](#12-reproducibility)

---

## 1. Overview

This component runs each candidate **vision–language model (VLM)** over the Bio-VQA C2 question
corpus (`QA.csv`, produced by the CUB-200-2011 generation pipeline) and records its answers. Every
runner does the same three things: load one model family, stream the question CSV through it in
deterministic batches, and append a single new column of raw answers.

The corpus spans four competencies — **Relative Salience**, **Morphological Attributes**,
**Anatomical Localization**, and **Taxonomy & Naming** — mixing 5-way MCQ items with open-ended
naming. A single runner handles all four without per-category configuration: the prompt suffix is
selected from the question text at runtime (Section 7).

```
QA.csv
   │
   ▼
 run_<model>.py   ──►  QA_<model_tag>.csv        (adds one <tag>_answer column)
   │
   ├─ (sharded)   ──►  QA_chunk{k}_<tag>.csv      one file per SLURM array task
   │
   └─ Llama only  ──►  run_llama3_2_fix_answers.py  ──►  QA_llama3.2_11B_clean.csv
```

---

## 2. Design Principles

**One script per model family, no shared state.** Each runner is self-contained: it owns its loader,
its generation call, and its output column. Swapping models is a one-line change in the job script,
and a failure in one runner cannot corrupt another's output. There is no orchestration layer and no
hidden config.

**Raw output is preserved verbatim.** The runner writes the model's exact string to
`<tag>_answer` and modifies nothing else in the CSV. Downstream graded evaluation needs to
distinguish a near-miss (`Phoebastria immutabilis` for `Phoebastria nigripes`) from a coarse answer
(`albatross`) from a confidently wrong one (`PACHYPTILA`) — so nothing is silently normalized at
inference time. All cleaning is additive, auditable, and lives in separate columns.

**Sharding is a deterministic contiguous split.** `--num_chunks N --chunk_id k` slices the dataframe
into `N` equal contiguous blocks and processes block `k`. The concatenation of all shards is
row-identical to a single-process run, which is what makes the SLURM array (Section 9) a pure
throughput optimization rather than a semantic change.

**The harness survives dirty data.** CUB and iNaturalist-adjacent sources include oversized JPEGs,
truncated files, and `.jpg` files that are secretly video containers. Every runner tolerates these
(Section 8) and skips-with-logging rather than aborting a shard on a single bad row.

**Decoding is deterministic across every model.** `do_sample=False`, `max_new_tokens=30`,
left-padded tokenizers throughout. No seed is required for answer generation, and re-running a shard
reproduces it exactly.

---

## 3. Runner Contents

| Script | Model(s) | Loader |
| --- | --- | --- |
| `run_qwen.py` | Qwen2.5-VL / Qwen3-VL (3B → 32B) | `Qwen2_5_VLForConditionalGeneration` + `qwen_vl_utils` |
| `run_InternVL.py` | InternVL3.5 (8B / 30B / 38B) | `AutoModel` (`trust_remote_code`), dynamic tiling |
| `run_gemma.py` | Gemma 3 / Gemma 4 (12B) | `Gemma3ForConditionalGeneration` |
| `run_llava_next.py` | LLaVA-NeXT (Mistral-7B / Vicuna-7B) | `LlavaNextForConditionalGeneration` |
| `run_pixtral12B.py` | Pixtral-12B | `LlavaForConditionalGeneration` |
| `run_Molmo2_8B.py` | Molmo2-8B | `AutoModelForImageTextToText` (`trust_remote_code`) |
| `run_phi_3_5_vision.py` | Phi-3.5-vision | `AutoModelForCausalLM` (`trust_remote_code`) |
| `run_llama3_2_11B.py` | Llama-3.2-11B-Vision-Instruct | `MllamaForConditionalGeneration` |
| `run_llama3_2_fix_answers.py` | *(post-processor)* | Canonicalizes free-form Llama output — Section 10 |

Runners that serve multiple checkpoints (`run_qwen.py`, `run_InternVL.py`, `run_gemma.py`,
`run_llava_next.py`) take `--model_id` on the command line; single-model runners hard-code it.

---

## 4. Input and Output Schemas

### 4.1 Input — `QA.csv`

| Column | Description |
| --- | --- |
| `image_id` | CUB integer index, or a 5-hex UUID for marker-annotated images |
| `filename` | Image path, **relative to `--dataset_dir`** (absolute paths are also honored) |
| `question` | Question stem; MCQ items carry an inline `Options: (A) …, (B) …` suffix |
| `answer` | Ground-truth answer string |
| `category` | One of `Taxonomy & Naming`, `Morphological Attributes`, `Relative Salience`, `Anatomical Localization` |

**Representative rows:**

```csv
image_id,filename,question,answer,category
1,images/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg,What is the scientific name of the bird?,Phoebastria nigripes,Taxonomy & Naming
1,images/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg,"What is the eye color of this bird? Options: (A) purple, (B) orange, (C) blue, (D) white, (E) brown",(E) brown,Morphological Attributes
1,images/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg,"Which of the following is the most prominently visible physical trait of the bird in this image? Options: (A) throat color: buff, (B) eye color: red, (C) leg color: olive, (D) forehead color: orange, (E) underparts color: brown",(A) throat color: buff,Relative Salience
63ac2,Images_Annotated/Black_Footed_Albatross_0046_18_63ac2.jpg,"What is the body part of this bird marked with the yellow star? Options: (A) crown, (B) left eye, (C) right leg, (D) left leg, (E) belly",(A) crown,Anatomical Localization
```

The last row is a **visually-prompted localization** item: the image is a star-annotated derivative
(`Images_Annotated/`) carrying its own UUID `image_id`.

### 4.2 Output — `QA_<tag>.csv`

The runner appends **one column**, `<model_tag>_answer`, holding the raw model string, and writes to
`<csv_basename>[_chunk<k>]_<tag>.csv`. Nothing else changes. The `<tag>` is derived filesystem-safely
from the checkpoint name (e.g. `Qwen2.5-VL-7B-Instruct` → `Qwen2_5_VL_7B_Instruct`).

**Representative output** (`qwen3_30B_answer`):

```csv
image_id,…,question,answer,category,qwen3_30B_answer
1,…,What is the common name of the bird?,Black footed Albatross,Taxonomy & Naming,albatross
1,…,"What type of bird is this? Options: (A) Tanager, (B) Gull, (C) Albatross, (D) Cormorant, (E) Waterthrush",(C) Albatross,Taxonomy & Naming,(C) Albatross
1,…,What is the scientific name of the bird?,Phoebastria nigripes,Taxonomy & Naming,Phoebastria immutabilis
1,…,What is the genus of the bird?,Phoebastria,Taxonomy & Naming,PACHYPTILA
1,…,"What is the eye color of this bird? Options: … (E) brown",(E) brown,Morphological Attributes,(E) brown
```

The failure modes here — near-miss species, coarse-grained naming, confidently wrong genus — are
exactly what raw preservation keeps legible for the evaluator.

---

## 5. Environment and Dependencies

Baseline for most runners:

```bash
pip install torch torchvision transformers accelerate pandas pillow opencv-python-headless einops
```

Per-model additions and pins:

| Model | Requirement |
| --- | --- |
| Qwen | `qwen_vl_utils` |
| Molmo2 | `transformers==4.57.1` (pinned upstream by Ai2) |
| InternVL | `trust_remote_code=True`; optional `flash-attn` |
| Phi-3.5-vision | `trust_remote_code=True`; the runner ships shims for newer `transformers` (Section 8) |
| Gemma 3/4 | `torch>=2.6`, `transformers>=4.49` |
| Pixtral | use the `transformers`-native mirror (`mistral-community/pixtral-12b`) — the original repo ships consolidated weights not loadable via `from_pretrained` |

> **Environment isolation.** `opencv-python-headless` is preferred over `opencv-python` on compute
> nodes to avoid `libGL.so` errors. Molmo2's `transformers==4.57.1` pin is incompatible with the
> Gemma `>=2.6 / >=4.49` requirement, so those two need **separate virtual environments**;
> everything else co-exists in one.

---

## 6. Command-Line Interface

Every runner accepts the same core arguments:

| Flag | Default | Description |
| --- | --- | --- |
| `--dataset_dir` | *required* | Root directory that `filename` is resolved against |
| `--csv_path` | *required* | Path to `QA.csv` |
| `--model_id` | varies | HF checkpoint (required for the multi-checkpoint runners; hard-coded otherwise) |
| `--batch_size` | 4–8 | Questions per generation call |
| `--chunk_id` | 0 | Index of this shard |
| `--num_chunks` | 1 | Total shards (rows split contiguously and evenly) |

Model-specific extras:

| Flag | Script | Description |
| --- | --- | --- |
| `--max_num` | `run_InternVL.py` | Max image tiles (default 12; drop to 6 for 30B/38B to save memory) |
| `--use_flash_attn` | `run_InternVL.py` | Enable FlashAttention (must be installed) |
| `--num_crops` | `run_phi_3_5_vision.py` | 16 is best for single images; lower to save memory |
| `--attn` | `run_phi_3_5_vision.py` | `eager` (safe default) \| `sdpa` \| `flash_attention_2` |

### 6.1 Single-process run

```bash
python run_qwen.py \
  --dataset_dir /path/to/CUB-200 \
  --csv_path    /path/to/CUB-200/QA.csv \
  --model_id    Qwen/Qwen2.5-VL-7B-Instruct \
  --batch_size  8
```

### 6.2 Concatenating shards

Each shard writes its own `_chunk{k}_` file. After the array finishes, stitch them back:

```python
import glob, pandas as pd
pd.concat([pd.read_csv(f) for f in sorted(glob.glob("QA_chunk*_Qwen2_5_VL_7B_Instruct.csv"))]) \
  .to_csv("QA_Qwen2_5_VL_7B_Instruct.csv", index=False)
```

Because the split is deterministic and contiguous, the stitched CSV is row-identical to a
single-process run.

---

## 7. Prompt Construction

Each runner appends a suffix instruction selected from the question text, so one CSV covers all four
axes without per-category bookkeeping:

| Question contains | Instruction appended |
| --- | --- |
| `options:` | A six-point directive: locate the `Options:` block, pick one choice, and echo **the letter in parentheses plus its exact trailing text** (`(C) Albatross`) — never the bare letter, never invented words, no markdown |
| `scientific name` | Answer with only the scientific name |
| `genus` | Answer with only the capitalized genus |
| `common name` / `what kind` | Answer with only the everyday English common name |
| *(fallback)* | Answer concisely with only the exact terms requested |

Generation is deterministic across all runners: `do_sample=False`, `max_new_tokens=30`, and
tokenizers are left-padded for batched decoding.

> InternVL additionally prepends `Please answer in English.` to every instruction — without it the
> model intermittently answers in Chinese.

---

## 8. Robustness

Shared across every runner:

**Tolerant image loading.** `Image.MAX_IMAGE_PIXELS = None` and
`ImageFile.LOAD_TRUNCATED_IMAGES = True` accept oversized and truncated JPEGs that would otherwise
raise.

**Hidden-video recovery.** Some `.jpg` files are actually video containers. Each runner calls
`Image.verify()`; on failure it falls back to OpenCV, extracts frame 0 to a temp file, and proceeds.
Temp frames are namespaced by `chunk_id` and cleaned up per batch, so parallel shards never collide.

**Skip-with-logging.** Missing images are logged and skipped rather than aborting the shard.
`run_llama3_2_11B.py` additionally wraps `from_pretrained` in a 3-attempt retry with a 30 s backoff,
absorbing transient contention when many array tasks touch the weight cache at once.

**Live progress logs.** Every 10th question prints file, query, answer, and ground truth, plus a
running count of unique images seen — enough to catch a broken prompt within seconds of launch.

**Phi-3.5 cache shims.** `run_phi_3_5_vision.py` re-attaches legacy `DynamicCache` methods
(`seen_tokens`, `get_max_length`, `get_usable_length`) that newer `transformers` removed but
Phi-3.5's remote code still calls — letting it run in the same environment as the other models
instead of a pinned one.

---

## 9. Running on a SLURM Cluster

The sharding flags map directly onto a SLURM **job array**: one array task per chunk, with
`SLURM_ARRAY_TASK_ID` supplying `--chunk_id`. The template below is deliberately cluster-agnostic —
adapt partition, account, module names, and paths to your own site.

```bash
#!/bin/bash
#SBATCH --job-name=cub_qwen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-24                 # 25 shards -> NUM_CHUNKS=25
#SBATCH --output=logs/cub_qwen_%A_%a.out
#SBATCH --error=logs/cub_qwen_%A_%a.err

set -e
module purge
module load cuda/12.4.1
source /path/to/your/venv/bin/activate

# Read from one warm cache; go offline once weights are local.
export HF_HOME=/path/to/shared/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATASET_DIR=/path/to/CUB-200
CSV_PATH="$DATASET_DIR/QA.csv"
NUM_CHUNKS=25

# Stagger task starts so N processes don't hit the cache simultaneously.
sleep $((SLURM_ARRAY_TASK_ID * 5))

python run_qwen.py \
  --dataset_dir "$DATASET_DIR" \
  --csv_path    "$CSV_PATH" \
  --model_id    Qwen/Qwen2.5-VL-7B-Instruct \
  --chunk_id    ${SLURM_ARRAY_TASK_ID} \
  --num_chunks  ${NUM_CHUNKS} \
  --batch_size  8
```

Submit with `sbatch job.sh`; monitor with `squeue -u $USER`; cancel a single shard with
`scancel <jobid>_<taskid>`.

### 9.1 Practical notes

| Concern | Guidance |
| --- | --- |
| **Array width** | `NUM_CHUNKS` must equal the array size (`0-(N-1)`). Larger `N` → shorter per-task runtime and better queue throughput, until model-load overhead dominates. 25–50 suits this workload. |
| **Weight caching** | Point `HF_HOME` at shared storage and pre-download each checkpoint **once** before launching. Setting `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` afterward stops dozens of tasks stampeding the Hub. |
| **Staggering** | `sleep $((SLURM_ARRAY_TASK_ID * 5))` avoids a thundering herd against the cache and filesystem at t=0. |
| **Env races** | For pinned-dependency runners (e.g. Molmo2 → `transformers==4.57.1`), build that venv **before** submitting. If you must build in-job, guard it with an atomic `mkdir` lock plus a `.ready` sentinel so one task builds and the rest wait, or build into a per-task `$TMPDIR`. |
| **Batch size** | Scale down as parameters grow: ~8 for ≤8B, 4 for 12B, 1–2 for 30B+. InternVL's tiling multiplies tokens per image — lower `--max_num` before lowering the batch size. |

---

## 10. Post-Processing: Llama-3.2

Every model receives identical prompts, and most comply.
**Llama-3.2-11B-Vision-Instruct does not.** It routinely ignores the format instructions — emitting
markdown, injected headers, prose preambles, trailing periods, bundled extra names, and occasionally
non-English text. Because C2 scores against a **closed CUB label set**, these answers are
*substantively* correct but *mechanically* unscorable: exact match reports them wrong, and even
graded metrics see a garbled string.

`run_llama3_2_fix_answers.py` exists solely to strip this wrapper.
**It is not a general normalizer and must not be applied to other models** — their raw output
already parses.

### 10.1 Observed failure modes

**(a) Markdown bold, injected headers, prose scaffolding.**

```
Query:   What kind of Wren is the bird in this image?
Answer:  **Bird Identification** * The bird in the image is a **Bewick's Wren**.
G-Truth: Bewick Wren
```
```
Query:   What is the scientific name of the bird?
Answer:  **Scientific Name:** **Troglodytes aedon**
G-Truth: Troglodytes aedon
```
```
Query:   What is the genus of the bird?
Answer:  The genus of the bird is **Trochilus**.
G-Truth: Salpinctes
```

**(b) Trailing punctuation on MCQ answers.** A single period turns a correct selection into a mismatch.

```
Query:   Which of the following is the most prominently visible physical trait…
         Options: (A) leg color: brown, (B) upperparts color: rufous, …
Answer:  (A) leg color: brown.
G-Truth: (A) leg color: brown
```

**(c) Wrong register / bundled names.** Told not to write sentences, not to use markdown, and not to
add words outside the answer — it does all three.

```
Query:   What is the scientific name of the bird?
Answer:  **Marsh Wren** (Cistothorus palustris)
G-Truth: Cistothorus palustris
```
```
Query:   What is the scientific name of the bird?
Answer:  **Scientific Name:** Rock Wren (Salpinctes obsoletus)
G-Truth: Salpinctes obsoletus
```

Cleaning does **not** rescue genuinely wrong answers, by design — it only makes them scorable:

```
Query:   What is the scientific name of the bird?
Answer:  **Cantorchilus modestus**   →  cleans to "Cantorchilus modestus"
G-Truth: Salpinctes obsoletus            (still wrong, now legible to the evaluator)
```

### 10.2 How the fix works

Two paths, chosen by whether the question contains an `Options:` block:

- **MCQ** — extract the first `(X)` letter, look it up in the options parsed from the *question*, and
  rebuild the canonical `(X) full option text`. If no valid letter is found, the de-markdowned text
  is kept and flagged `no_letter`.
- **Free-form (Taxonomy & Naming)** — strip markdown (`*` `**` `` ` `` `#`), peel preambles
  (`The genus of the bird is`, `Scientific Name:`, `This is a`, …) up to three times, drop trailing
  punctuation, then match against the **vocabulary of ground-truth labels ever used for that exact
  question text**. Matching is normalization-insensitive across case, possessives, and hyphens
  (`Bewick's Wren` → `bewick wren`). No match → keep the cleaned text, flagged `unmatched`.

The vocabulary is the union of GT answers **per question text**, never the row's own ground truth —
so a wrong answer stays wrong; it just becomes readable.

### 10.3 Usage

```bash
python run_llama3_2_fix_answers.py --csv QA_llama3.2_11B.csv            # → QA_llama3.2_11B_clean.csv
python run_llama3_2_fix_answers.py --csv QA_llama3.2_11B.csv --inplace  # overwrite in place
```

| Flag | Default | Description |
| --- | --- | --- |
| `--csv` | *required* | CSV containing a `*_answer` prediction column |
| `--pred_col` | auto-detected | Explicit prediction column (first ending in `_answer`) |
| `--inplace` | off | Overwrite the input instead of writing `*_clean.csv` |

Two columns are added: `<tag>_clean` (canonical answer) and `<tag>_method`
(`letter` / `no_letter` / `vocab` / `unmatched` / `empty`). The script prints raw vs. clean exact
match, a method histogram, and the first ten rows needing manual review — inspect the `no_letter`
and `unmatched` counts before trusting the output.

---

## 11. End-to-End Pipeline

```
                              QA.csv
                                │
        ┌───────────────────────┼───────────────────────────┐
        │                       │                           │
   run_qwen.py            run_InternVL.py      …     run_llama3_2_11B.py
   (× N shards)            (× N shards)                (× N shards)
        │                       │                           │
   concat shards           concat shards               concat shards
        │                       │                           │
   QA_qwen…csv            QA_InternVL…csv          QA_llama3.2_11B.csv
        │                       │                           │
        │                       │              run_llama3_2_fix_answers.py
        │                       │                           │
        └───────────┬───────────┘                  QA_llama3.2_11B_clean.csv
                    │                                       │
                    └──────────────► graded evaluation (CAGS) ◄──────────────┘
```

All models except Llama-3.2 feed the evaluator directly from their raw `<tag>_answer` column;
Llama-3.2 passes through the cleaner first.

---

## 12. Reproducibility

- **Greedy decoding** (`do_sample=False`, `max_new_tokens=30`) throughout — no seed needed for
  answer generation.
- **Deterministic sharding.** The contiguous split makes the concatenation of all shards
  row-identical to a single-process run.
- **Verbatim raw output.** The `<tag>_answer` column preserves the exact model string; every
  cleaning transformation is additive, lives in separate columns, and is auditable against the
  original.

---

<p align="center">
  <sub>Part of the <b>Bio-VQA</b> benchmark · Imageomics Institute</sub>
</p>