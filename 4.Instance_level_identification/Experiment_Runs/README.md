<h1 align="center"> Bio-VQA &rarr; Individual Re-Identification &rarr; VLM Inference </h1>

<p align="center">
  <b>Eighteen vision&ndash;language models &middot; Five species &middot; Three conditions &middot; One protocol</b>
</p>
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/torch-bfloat16-orange">
  <img alt="Transformers" src="https://img.shields.io/badge/HF-transformers-yellow">
  <img alt="Scheduler" src="https://img.shields.io/badge/scheduler-SLURM-red">
  <img alt="Models" src="https://img.shields.io/badge/models-18-brightgreen">
  <img alt="Answers" src="https://img.shields.io/badge/answers-270%2C000-purple">
</p>
<p align="center">
  We run eighteen models over fifteen task manifests under one prompting protocol.<br>
  The only thing that differs between runners is the model &mdash; everything else is<br>
  generated from a single template and is byte-identical by checksum.
</p>

---

## Table of Contents

1. [Overview](#1-overview)
2. [How We Keep the Protocol Constant](#2-how-we-keep-the-protocol-constant)
3. [Model Roster](#3-model-roster)
4. [Anatomy of a Runner](#4-anatomy-of-a-runner)
5. [The Prompt](#5-the-prompt)
6. [Decoding Configuration](#6-decoding-configuration)
7. [Input and Output](#7-input-and-output)
8. [Command-Line Interface](#8-command-line-interface)
9. [Task Enumeration](#9-task-enumeration)
10. [Answer Alignment](#10-answer-alignment)
11. [Sanity Guard](#11-sanity-guard)
12. [Throughput](#12-throughput)
13. [Environments](#13-environments)
14. [Running on SLURM](#14-running-on-slurm)
15. [Job Arrays](#15-job-arrays)
16. [Vision-Token Budgets](#16-vision-token-budgets)
17. [Engineering Notes](#17-engineering-notes)
18. [Reproducibility](#18-reproducibility)

---

## 1. Overview

We evaluate eighteen vision–language models against the fifteen QA manifests produced by the
[Data-Preparation pipeline](../Data-Preparation) — five species × three difficulty conditions — and
append one column of raw model answers to each.

Every runner implements the same protocol: identical prompt text, identical decoding configuration,
identical batching semantics, identical answer alignment, identical output schema. Runners differ
only in the model-loading and message-formatting code each architecture requires.

Each model loads once per job and streams over every task in scope rather than reloading per task.
Across eighteen models this saves several hours of redundant initialisation.

Total volume: **18 models × 5 species × 3 conditions × 1,000 pairs = 270,000 answers.**

---

## 2. How We Keep the Protocol Constant

The protocol is the controlled variable, so we enforce it structurally rather than by convention. We
generate every runner from one template. The region below the model block is byte-identical across
all eighteen scripts, and we verify this by checksum:

```bash
python - <<'PY'
import hashlib, glob
for f in sorted(glob.glob("*.py")):
    t = open(f).read()
    if "# HARNESS - identical" not in t: continue
    print(hashlib.md5(t[t.index("# HARNESS - identical"):].encode()).hexdigest(), f)
PY
```

All runners print the same digest. A mismatch means a script has drifted and its results are not
protocol-comparable.

Three further choices follow from the same principle:

**One prompt, no per-model tuning.** The instruction text is a module-level constant. We do not adapt
it to any model's quirks. Per-model prompt engineering would make a difference in results
attributable to prompts rather than models.

**We never disclose the label prior.** The prompt does not say the set is balanced or suggest an
expected proportion of "Yes" answers. Disclosing it turns perception into bookkeeping.

**We store raw answers and defer parsing.** Runners write the model's unmodified output string — no
normalisation, no yes/no coercion, no truncation. All interpretation happens in the Evaluation stage
where the rules are stated once and applied uniformly.

---

## 3. Model Roster

Eighteen open-weight models across nine families, 3B–38B parameters. We use open weights only:
closed API models cannot be pinned to a version, re-run reproducibly, or inspected for the
load-integrity failures described in §17.

| Runner | Model | Params | Class | `transformers` |
| --- | --- | ---: | --- | --- |
| `qwenVL2_5_3B.py` | Qwen2.5-VL-3B-Instruct | 3B | `AutoModelForImageTextToText` | 4.57.1 |
| `qwenVL2_5_7B.py` | Qwen2.5-VL-7B-Instruct | 7B | `AutoModelForImageTextToText` | 4.57.1 |
| `qwen3_8B.py` | Qwen3-VL-8B-Instruct | 8B | `AutoModelForImageTextToText` | current |
| `qwenVL3_30B.py` | Qwen3-VL-30B-A3B-Instruct | 30B MoE | `AutoModelForImageTextToText` | current |
| `qwenVL3_32B.py` | Qwen3-VL-32B-Instruct | 32B | `AutoModelForImageTextToText` | current |
| `intern3_8B.py` | InternVL3.5-8B-Instruct | 8B | `AutoModel` (remote code) | 4.57.1 |
| `intern3_5_30B_a3b.py` | InternVL3.5-30B-A3B-Instruct | 30B MoE | `AutoModel` (remote code) | 4.57.1 |
| `intern3_38B.py` | InternVL3.5-38B-Instruct | 38B | `AutoModel` (remote code) | 4.57.1 |
| `gemma3_12B.py` | Gemma-3-12B-it | 12B | `AutoModelForImageTextToText` | 4.57.1 |
| `gemma4_12B.py` | Gemma-4-12B-it | 12B | `AutoModelForImageTextToText` | git HEAD |
| `molmo_8B.py` | Molmo2-8B | 8B | `AutoModelForImageTextToText` (remote code) | 4.57.1 |
| `llava_next_mistral_7B.py` | LLaVA-NeXT Mistral-7B | 7B | `LlavaNextForConditionalGeneration` | current |
| `llava_next_vicuna_7B.py` | LLaVA-NeXT Vicuna-7B | 7B | `LlavaNextForConditionalGeneration` | current |
| `run_phi_3_5_vision.py` | Phi-3.5-vision-instruct | 4.2B | `AutoModelForCausalLM` (remote code) | 4.57.1 |
| `run_pixtral12B.py` | Pixtral-12B | 12B | `LlavaForConditionalGeneration` | 4.57.1 |
| `run_llama3_2V_11B.py` | Llama-3.2-11B-Vision-Instruct | 11B | `MllamaForConditionalGeneration` | 4.57.1 |

We pin `transformers` wherever a model's remote code targets the 4.x API and breaks under the 5.x
rewrite. Gemma-4 needs a source install because `gemma4_unified` is not in any released version.

---

## 4. Anatomy of a Runner

Each script has three regions separated by banner comments:

```
┌─ EXPERIMENT CONSTANTS ──────────────────────────────┐
│  INSTRUCTION, DATASETS, DIFFICULTIES, build_prompt  │  identical, checksum-verified
├─ MODEL BLOCK ───────────────────────────────────────┤
│  MODEL_ID, MODEL_KEY, DEFAULT_BATCH_SIZE,           │  the only per-model region
│  MAX_NEW_TOKENS, load_model(), run_batch()          │
├─ HARNESS ───────────────────────────────────────────┤
│  task enumeration, batching, alignment, guard,      │  identical, checksum-verified
│  summary, CLI                                       │
└─────────────────────────────────────────────────────┘
```

The model block exposes two functions:

```python
def load_model():
    """Return an opaque handle. Called once per job."""

def run_batch(handle, batch):
    """batch: list of (image_path, prompt).
       Returns raw answer strings, same length and order."""
```

Adding a model means writing those two functions and changing four constants.

We keep heavy imports (`torch`, `transformers`) inside the functions rather than at module scope, so
a runner imports and its harness runs without a GPU or model weights present. That is how we test the
harness.

---

## 5. The Prompt

We read the question verbatim from `qa_pair["Q"]` and append a fixed instruction suffix:

```
<question from QA.csv>

* The image shows two photographs placed side by side.
* Each photograph may contain more than one animal - consider only the main subject in the foreground.
* Judge only from what is visible in the two photographs.
* Answer with one word: Yes or No. Do not explain.
```

Each clause does specific work:

| Clause | Purpose |
| --- | --- |
| "two photographs placed side by side" | The item is a composite; without this a model may read it as one scene. |
| "consider only the main subject in the foreground" | Source frames often contain several animals. |
| "Judge only from what is visible" | Discourages appeals to metadata or speculation. |
| "one word: Yes or No" | Constrains format so parsing does not become the measurement. |

We print the suffix to the job log at startup, so every run records the exact protocol it used.

---

## 6. Decoding Configuration

Greedy throughout: `do_sample=False`, `max_new_tokens=15`. Deterministic decoding removes sampling
variance as a source of cross-model difference, and fifteen tokens is ample for a one-word answer
while leaving room for a short preamble the parser still resolves.

We run `bfloat16` with `device_map="auto"`, SDPA attention where supported, and `eager` for
Phi-3.5-vision whose remote code requires it. Padding side is `left` for every batched runner.

---

## 7. Input and Output

**Input** — `<task_dir>/QA.csv`:

| Column | Description |
| --- | --- |
| `image_path` | composite PNG, relative to the task directory |
| `qa_pair` | JSON `{"Q": …, "A": "Yes"｜"No"}` |

**Output** — `<task_dir>/QA_<MODEL_KEY>.csv`: the input file with one column appended.

| Column | Description |
| --- | --- |
| `<MODEL_KEY>_answer` | the model's raw output string |

Writing beside the source manifest keeps every model's answers for a task in one directory and lets
the Evaluation stage discover models by filename without a registry.

Runs are resumable: we skip a task whose output CSV exists unless `--overwrite` is given.

---

## 8. Command-Line Interface

Every runner takes the same arguments.

```bash
python <runner>.py --data_root /path/to/4-Instance-Level-Identification
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--data_root` | configured | root containing the per-species directories |
| `--difficulties` | `easy hard random` | restrict the sweep |
| `--num` | 500 | pairs per class at generation; determines directory names |
| `--batch_size` | per model | inference batch size |
| `--io_workers` | 4 | background threads prefetching images; 0 disables |
| `--task_index` | −1 | run only the *N*th grid entry; −1 runs all |
| `--limit` | 0 | first *N* rows of each task |
| `--overwrite` | off | re-run completed tasks |

We smoke-test each model before its full sweep:

```bash
LIMIT=8 sbatch <runner>.sh
```

---

## 9. Task Enumeration

Runners build their own task list from `--data_root`; no external file supplies it.

```python
DATASETS = ["BelugaID-2022/beluga.coco", "HyenaID-2022/hyena.coco",
            "LeopardID-2022/leopard.coco", "WhalesharkID/whaleshark.coco",
            "GrevysZebra"]
DIFFICULTIES = ["easy", "hard", "random"]
```

The grid is the full 5 × 3 product in fixed order, each entry tagged with whether its `QA.csv`
exists. `--task_index` indexes this grid, not the subset that happens to be present.

That distinction matters for array jobs. If the index addressed only existing tasks, an ungenerated
condition would shift every later index and an array element would write one species' answers into
another species' file. Indexing the full grid makes `--task_index N` mean the same thing on every
submission. An element addressing an ungenerated entry logs it and exits with status 0.

Grid order is species-major, condition-minor: index 0 is beluga/easy, 1 beluga/hard, 2 beluga/random,
3 hyena/easy, through to 14 = zebra/random. Each element logs its resolved path at startup.

---

## 10. Answer Alignment

We write answers back by row position, not by append order:

```python
work.append((pos, image_path, prompt))          # position travels with the item
...
for (pos, _, _), answer in zip(chunk, answers):
    df.iat[pos, col_idx] = answer               # written to the row it came from
```

This closes a silent corruption mode. Accumulating answers in a list and assigning
`df["answer"] = answers` with truncation means any skipped row — a missing image, a malformed
`qa_pair` — shifts every subsequent answer by one, and the truncation hides it.

We measured this on a controlled test with alternating labels and one missing image among twenty
rows: append-and-truncate scores 0.105, position-indexed write-back scores 1.000 on identical model
outputs. Rows whose item was skipped stay `NA` and are counted separately downstream.

---

## 11. Sanity Guard

After the first ~20 items of a task, and again at task end, we check whether any answer resolves to
Yes/No. If none does, the job aborts and prints the raw sample:

```
ABORTING: none of the first 20 answers parsed as Yes/No.
  raw sample: ['\x00\x01 !!!', ...]
  The model is almost certainly misconfigured. Check the transformers LOAD REPORT
  above for MISSING weights, and confirm the model class matches the checkpoint.
```

We added this after a dense checkpoint loaded with a Mixture-of-Experts class silently discarded
every MLP weight and randomly initialised replacements. That job ran to completion and wrote 8,000
rows of noise. The guard now stops it in seconds.

We also log the first raw outputs of every task unconditionally, so each model's answer format is on
record.

---

## 12. Throughput

Generation preserves source resolution, so composites are large — several exceed 4,000 px. For the
smaller models, image decode and filesystem read dominate wall-clock rather than GPU compute.

A background `ThreadPoolExecutor` reads upcoming batches into the OS page cache while the GPU works
on the current one:

```python
target = min(len(work), start + batch_size * (1 + IO_PREFETCH_BATCHES))
while warm_ptr < target:
    pool.submit(warm_file, work[warm_ptr][1])
    warm_ptr += 1
```

PIL's decode releases the GIL, so this overlaps genuinely. Controlled by `--io_workers`
(`IO_WORKERS=8 sbatch …`), disabled with `0`.

We set batch size per model to the largest value that fits VRAM given the vision-token budget: 8 for
the 3B, 4 for 7–8B, 2 for 12B and larger, 1 for Phi-3.5-vision which does not batch cleanly and
generates sequentially.

---

## 13. Environments

Each batch script builds a private virtual environment on node-local storage and deletes it via an
`EXIT` trap, so cleanup happens on success, failure, timeout and cancellation alike:

```bash
VENV="${TMPDIR:-/tmp}/c4_<model>_${SLURM_JOB_ID}"
cleanup() { rm -rf "${VENV}"; }
trap cleanup EXIT
```

The eighteen models require mutually incompatible dependency sets. Molmo2 pins
`transformers==4.57.1` while Gemma-4 needs a git-HEAD install; InternVL's remote code targets the 4.x
API while Qwen3-VL needs a recent release. No single environment satisfies them, and a shared
environment mutated per model would make results depend on submission order.

`HF_HOME` and `PIP_CACHE_DIR` point at scratch, so model weights and wheels persist while only the
environment is disposable. Without the wheel cache each job re-downloads about 2.5 GB of `torch`.

Every script asserts GPU visibility before doing work:

```python
if not torch.cuda.is_available():
    sys.exit("ERROR: no GPU visible - check --gpus-per-node in the SBATCH header")
```

---

## 14. Running on SLURM

One batch script per model, matching the runner name:

```bash
mkdir -p Runs/logs
sbatch qwen3_8B.sh
```

Resources we use:

| Model class | GPUs | `--mem` | Walltime |
| --- | ---: | ---: | --- |
| 3B–8B dense | 1 | 64 G | 3–6 h |
| 11B–12B | 1 | 64–96 G | 8–12 h |
| 30B MoE, 32B, 38B dense | 2 | 64 G | 6–16 h |

`--mem` is host RAM, not VRAM. With `low_cpu_mem_usage=True` and `device_map="auto"`, safetensors
shards stream directly to GPU and host peak stays near one shard, so 64 G suffices even for the 38B.
The binding constraint for large models is VRAM — 38B in bf16 is about 76 GB — which is why they take
two GPUs.

Environment overrides every batch script accepts:

```bash
LIMIT=8 sbatch <model>.sh                      # smoke test
DIFFICULTIES="easy hard" sbatch <model>.sh     # restrict the sweep
BATCH_SIZE=1 sbatch <model>.sh                 # override the per-model default
OVERWRITE=1 sbatch <model>.sh                  # re-run completed tasks
MODEL_ID=<repo> sbatch <model>.sh              # override the checkpoint id
```

---

## 15. Job Arrays

For the larger models a single serial job over fifteen tasks is impractical, so we submit one grid
entry per array element:

```bash
#SBATCH --array=0-14%3
...
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  ARGS+=(--task_index "${SLURM_ARRAY_TASK_ID}")
fi
```

`0-14` is fifteen elements — SLURM array ranges are inclusive at both ends. `%3` caps concurrency at
three.

Each element writes its own complete `QA_<model>.csv` for one species × condition, so **no merge step
is required** — unlike the row-level chunking used in other Bio-VQA components. Task-level
granularity is the natural unit here because the fifteen tasks are independent by construction.

We pre-warm the weight cache before submitting an array:

```bash
HF_HOME=/path/to/hf_cache huggingface-cli download <repo>
```

An array element that read shards while a sibling was still writing them loaded partially-written
tensors and produced degenerate output with no error raised.

---

## 16. Vision-Token Budgets

Dynamic-resolution models expand a large composite into an enormous number of vision tokens. A
4,000 × 2,000 image at a 28-px patch grid is about 10,000 tokens for one item — slow, and an
out-of-memory risk at any useful batch size.

We apply two bounds, both identical across all conditions so neither confounds the difficulty
comparison:

```python
# Qwen family
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(MODEL_ID,
                                          min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

# InternVL family — tiles per image; 12 is the default
MAX_NUM_TILES = 6
```

We hold the InternVL tile budget at 6 across all three model sizes so tiling is not a hidden variable
between them.

These bounds lower the effective resolution every model sees, which is the regime where
individual-specific markings are hardest to resolve. Because the reduction is constant across
conditions it leaves the difficulty effect intact, but it does lower absolute performance.

---

## 17. Engineering Notes

We record the failures that shaped the current implementation.

**Dense checkpoint loaded with a MoE class.** `Qwen3-VL-8B-Instruct` is dense; instantiating it with
`Qwen3VLMoeForConditionalGeneration` made `transformers` discard all 36 MLP weight tensors
(`UNEXPECTED`) and randomly initialise MoE experts (`MISSING`), with no exception raised. The job
completed and wrote 8,000 rows of noise. We now use `AutoModelForImageTextToText` so the architecture
resolves from the checkpoint config, print the resolved class on load, and run the sanity guard.

**Missing `--gpus-per-node`.** A submission without the directive ran 8B inference on CPU for six
hours. Every batch script now asserts `torch.cuda.is_available()` immediately after building the
environment.

**`transformers` 5.x kwarg rename.** Unpinned installs resolved to 5.x, where `torch_dtype=` became
`dtype=`. We use `dtype=` throughout and pin where a model needs 4.x.

**Cold HF cache race.** Concurrent array elements downloading the same weights onto parallel Lustre
read partially-written shards. We pre-warm the cache, or serialise the download behind a lock file.

**Answer misalignment.** See §10.

**Phi-3.5-vision cache API.** Its remote code targets an old `DynamicCache` API; newer `transformers`
removed `seen_tokens`, `get_max_length`, and `get_usable_length`. We re-attach the legacy names as
shims in `load_model`.

**Mllama image nesting.** `MllamaForConditionalGeneration` expects a list of images per sample, not a
bare image.

**Pixtral pad token.** The checkpoint ships no pad token and batched padding raises; we set
`pad_token = eos_token` before setting padding side.

---

## 18. Reproducibility

| Aspect | Guarantee |
| --- | --- |
| Decoding | Greedy; no sampling variance. |
| Prompt | Module-level constant, checksum-identical across runners, echoed to every job log. |
| Harness | Generated from one template; byte-identical below the model block. |
| Environment | Rebuilt per job; versions echoed at startup; pinned where the model requires it. |
| Answers | Raw strings stored verbatim; interpretation deferred to Evaluation. |
| Resumption | Task-level; completed outputs skipped unless `--overwrite`. |
| Task addressing | `--task_index` indexes the full 5 × 3 grid, stable regardless of which tasks exist. |