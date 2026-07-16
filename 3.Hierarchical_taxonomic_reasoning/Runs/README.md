<h1 align="center">VLM Inference &mdash; Hierarchical Taxonomic Reasoning</h1>
 
<p align="center">
  <b>Eighteen vision–language models · Ten organismal groups · One prompting protocol</b>
</p>
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/torch-bfloat16-orange">
  <img alt="Transformers" src="https://img.shields.io/badge/HF-transformers-yellow">
  <img alt="Scheduler" src="https://img.shields.io/badge/scheduler-SLURM-red">
  <img alt="Models" src="https://img.shields.io/badge/models-18-brightgreen">
</p>
---

## Table of Contents
 
- [Model Roster](#model-roster)
- [The Shared Protocol](#the-shared-protocol)
- [Common Interface](#common-interface)
- [Input and Output Contract](#input-and-output-contract)
- [Environments](#environments)
- [Running](#running)
- [Chunking and SLURM Arrays](#chunking-and-slurm-arrays)
---

 
## Overview
 
This directory contains the inference stage of the **hierarchical taxonomic reasoning** component of
Bio-VQA. Each script evaluates one vision–language model (VLM) against the QA manifests produced by the
[Tree-of-Life-200M pipeline](../Tree-of-Life-200M), emitting a per-model answer column joined back onto
the source rows.
 
Every runner implements the **same experimental protocol** — identical prompts, identical decoding
configuration, identical post-processing — and differs only in the model loading and message-formatting
code that each architecture requires. This is deliberate: the protocol is the controlled variable, and
any behavioral difference in the results must be attributable to the model rather than to the harness.
 
Shell submission scripts are **not committed**. Templates appear in [Running](#running) below; adapt the
paths to your environment.
 
---

## Model Roster
 
Eighteen checkpoints spanning 3.75B to 38.4B parameters, seven model families, and both dense and
mixture-of-experts architectures.
 
| Script | Checkpoint | Params | LLM backbone | Vision encoder |
| :--- | :--- | :--- | :--- | :--- |
| `run_qwenVL2_5_3B.py` | `Qwen/Qwen2.5-VL-3B-Instruct` | 3.75B dense | Qwen2.5-3B | Native-resolution ViT (~0.67B), window attention |
| `run_qwenVL2_5_7B.py` | `Qwen/Qwen2.5-VL-7B-Instruct` | 8.29B dense | Qwen2.5-7B | Native-resolution ViT (~0.67B), window attention |
| `run_qwenVL3_8B.py` | `Qwen/Qwen3-VL-8B-Instruct` | ~8.8B dense | Qwen3-8B | Qwen3-VL ViT, DeepStack fusion |
| `run_qwenVL3_30B.py` | `Qwen/Qwen3-VL-30B-A3B-Instruct` | 30.5B total / ~3B active (MoE) | Qwen3-30B-A3B | Qwen3-VL ViT, DeepStack fusion |
| `run_qwenVL3_32B.py` | `Qwen/Qwen3-VL-32B-Instruct` | ~32B dense | Qwen3-32B | Qwen3-VL ViT, DeepStack fusion |
| `run_InternVL_3_8B.py` | `OpenGVLab/InternVL3-8B-Instruct` | ~8B dense | Qwen2.5-7B | InternViT-300M-448px-V2.5 |
| `run_InternVL_3_38B.py` | `OpenGVLab/InternVL3-38B-Instruct` | ~38B dense | Qwen2.5-32B | InternViT-6B-448px-V2.5 |
| `run_InternVL_3_5_8B.py` | `OpenGVLab/InternVL3_5-8B-Instruct` | ~8B dense | Qwen3-8B | InternViT-300M-448px-V2.5 |
| `run_InternVL_3_5_30B.py` | `OpenGVLab/InternVL3_5-30B-A3B-Instruct` | ~31B total / ~3B active (MoE) | Qwen3-30B-A3B | InternViT-300M-448px-V2.5 |
| `run_InternVL_3_5_38B.py` | `OpenGVLab/InternVL3_5-38B-Instruct` | 38.4B dense | Qwen3-32B | InternViT-6B-448px-V2.5 |
| `run_gemma3_12B.py` | `google/gemma-3-12b-it` | ~12B dense | Gemma 3 12B | SigLIP-So400m/14-896, pan-and-scan |
| `run_gemma4_12B.py` | `google/gemma-4-12B-it` | ~12B dense | Gemma 4 12B (same decoder structure as Gemma 4 31B Dense) | **None — encoder-free.** 35M vision embedder: raw 48×48×3 RGB patches projected into the LLM embedding space by a single matmul, with 2D coordinate positional embeddings and a LayerNorm |
| `run_llama3_2V_11B.py` | `meta-llama/Llama-3.2-11B-Vision-Instruct` | 10.6B dense | Llama 3.1 8B (frozen) | ViT-H/14 (~0.63B) + cross-attention adapter |
| `run_llava_next_mistral_7B.py` | `llava-hf/llava-v1.6-mistral-7b-hf` | 7.57B dense | Mistral-7B-Instruct-v0.2 | CLIP ViT-L/14-336, AnyRes tiling |
| `run_llava_next_vicuna_7B.py` | `llava-hf/llava-v1.6-vicuna-7b-hf` | 7.06B dense | Vicuna-7B-v1.5 (LLaMA-2) | CLIP ViT-L/14-336, AnyRes tiling |
| `run_pixtral12B.py` | `mistral-community/pixtral-12b` | 12.4B dense | Mistral NeMo 12B | Pixtral-ViT (~0.4B), trained from scratch, variable resolution |
| `run_Molmo2_8B.py` | `allenai/Molmo2-8B` | 8.66B dense | Qwen3-8B | SigLIP 2 |
| `run_phi_3_5_vision.py` | `microsoft/Phi-3.5-vision-instruct` | 4.2B dense | Phi-3.5-mini (3.8B) | CLIP ViT-L/14-336, multi-crop |
 
Figures are as reported by the respective model cards and technical reports; counts marked `~` are
rounded.

### What the roster is controlling for
 
The selection is not a convenience sample. Three axes are crossed deliberately:
 
- **Backbone lineage.** Three checkpoints share the **Qwen3-8B** backbone under completely different
  vision stacks — Qwen3-VL-8B (native-resolution ViT), InternVL3_5-8B (InternViT-300M), and Molmo2-8B
  (SigLIP 2) — and Qwen2.5-32B / Qwen3-32B anchor the two InternVL 38B generations. Others hold the
  vision stack fixed and vary the backbone: the two LLaVA-NeXT runners differ **only** in the language
  model. This permits partial attribution of performance to the language versus vision half.
- **Vision encoder scale and design.** The roster spans a ~0.3B InternViT, a 6B InternViT, frozen CLIP
  ViT-L/14 towers, SigLIP and SigLIP 2, a from-scratch variable-resolution encoder (Pixtral),
  native-resolution dynamic-tiling designs (Qwen2.5-VL onward), and — at the far end — **no encoder at
  all** (Gemma 4 12B). Encoder capacity varies by more than two orders of magnitude at comparable total
  parameter counts.
- **Dense versus sparse.** Two MoE checkpoints (Qwen3-VL-30B-A3B, InternVL3_5-30B-A3B) sit at ~30B total
  but ~3B active parameters, adjacent to dense models of both sizes. Reporting them alongside dense 8B
  and 32B models separates capacity from compute.
**Architectural lineage is more informative than parameter count.** Qwen3-VL-8B, InternVL3_5-8B, and
Molmo2-8B are all "8B models" running the *same* language backbone; the gaps between them are vision-stack
and training-data results, not scale results. Any table reporting only parameter counts obscures this.
 
**Checkpoint selection notes.**
 
- **Gemma 4 12B is encoder-free**, and is the only such model in the roster. Where every other checkpoint
  routes images through a dedicated vision tower (150M–6B parameters), Gemma 4 12B projects raw 48×48×3
  RGB patches straight into the LLM embedding space through a single ~35M-parameter matmul, with spatial
  position supplied by factorized 2D coordinate embeddings rather than learned by an attention-based
  encoder. Google's own report frames this as replacing the 550M encoder used by the other medium-sized
  Gemma 4 models. For a benchmark that probes fine-grained visual attributes and anatomical localization,
  this is a substantive design contrast worth reporting rather than a footnote: it shifts all visual
  processing burden onto the decoder.
- **Molmo2-8B** pairs Qwen3-8B with SigLIP 2 and is trained by Ai2 on a video-centric mixture with
  dedicated pointing, counting, and tracking collections. Its grounding-oriented supervision is the
  natural hypothesis for any localization-task outlier behavior — and because it shares a backbone with
  two other 8B entries, that hypothesis is partially testable within the roster.
- **Pixtral** loads `mistral-community/pixtral-12b`, a transformers-native mirror. The upstream
  `mistralai/Pixtral-12B-2409` repository ships consolidated weights intended for vLLM or
  mistral-inference and is not loadable through `from_pretrained`.
- **Qwen3-VL-30B-A3B** requires `Qwen3VLMoeForConditionalGeneration`; loading it through the dense class
  fails to map the experts. The 32B checkpoint is dense and uses the standard class.
- **InternVL3_5-30B-A3B** is likewise an MoE variant, but InternVL's remote code dispatches internally,
  so `AutoModel` resolves the correct class without intervention.
- **Llama-3.2-11B-Vision** is architecturally distinct from every other model here: the vision
  information enters through gated cross-attention layers inserted into a frozen text backbone, rather
  than by projecting visual tokens into the LLM's input sequence. Its text-only behavior is preserved by
  construction, which is a plausible confound when interpreting its taxonomic recall.
- **Phi-3.5-vision** does not batch cleanly through its remote code and is executed sequentially;
  `--batch_size` is accepted for interface parity but only affects log grouping.
### Batch-size defaults
 
Tuned per architecture and set in each script; override with `--batch_size`.
 
| Default | Scripts |
| :---: | :--- |
| 32 | `run_qwenVL3_8B.py` |
| 9 | `run_qwenVL2_5_3B.py` |
| 8 | `run_qwenVL3_30B.py`, all five InternVL runners |
| 4 | everything else |
| *n/a* | `run_phi_3_5_vision.py` (sequential) |

 
---



## The Shared Protocol
 
All eighteen runners implement the following pipeline identically. Any deviation would confound the
cross-model comparison, so changes here must be applied to all scripts simultaneously.
 
### 1. Question-type routing
 
The question string is lowercased and scanned against an ordered keyword list:
 
```
phylum → class → order → family → genus → species epithet
       → scientific name → among the four options → common name
```
 
The **first** keyword found determines the response-format instruction appended to the question. Order
matters: the list is scanned sequentially and breaks on first match.
 
### 2. Format instructions
 
A rank-specific instruction is concatenated to the question, constraining the output to the bare
taxonomic term. Representative examples:
 
| Rank | Instruction (abridged) |
| :--- | :--- |
| phylum | Answer with ONLY the Phylum name (e.g., Arthropoda). Do not write full sentences. |
| genus | Answer with ONLY the Genus name (capitalized, e.g., Argiope). Do not include the species epithet. |
| species epithet | Answer with ONLY the species epithet (lowercase second word, e.g., aurantia). Do NOT include the Genus. |
| scientific name | Answer with ONLY the full binomial (e.g., Argiope aurantia). |
| among the four options | Answer with ONLY the exact correct option from the four choices provided. |
| common name | Answer with ONLY the everyday English common name. Do NOT provide the Latin name. |
 
An unrouted question receives a generic fallback instruction. The instruction is appended to the
**prompt only** — the `question` recorded in the manifest and used for scoring remains the unmodified
original.
 
> **InternVL variant.** The InternVL 3 8B and 38B runners prefix every instruction with *"Please answer
> in English."* This was added in response to observed Chinese-language output on those checkpoints. It
> is a deliberate, documented deviation from the shared instruction text and should be reported as such;
> the remaining sixteen runners use the unprefixed instructions.


---

### 5. Input robustness
 
Every runner applies the same defenses against a corpus assembled from heterogeneous web sources:
 
| Guard | Purpose |
| :--- | :--- |
| `Image.MAX_IMAGE_PIXELS = None` | Disables the decompression-bomb ceiling; some herbarium scans legitimately exceed it. |
| `ImageFile.LOAD_TRUNCATED_IMAGES = True` | Permits decoding of incompletely transferred files. |
| `img.verify()` + OpenCV fallback | Files that fail PIL verification are re-tried as video containers; the first frame is extracted to a chunk-scoped temporary file. Some records served under an image URL are in fact video. |
| Chunk-scoped temp filenames | `temp_frame_chunk{chunk_id}_row{idx}.jpg` prevents collisions between concurrent array tasks sharing a working directory. |
| Per-row `try/except` | An unreadable row logs and continues rather than aborting the job. **See [Known Issues](#known-issues)** — the recovery path has an alignment defect. |
 
---
 
## Common Interface
 
Every runner exposes the same four core arguments:
 
| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset_dir` | path | *required* | Taxon directory containing `Images/`. Relative `local_path` values are resolved against it; absolute values are used as-is. |
| `--csv_path` | path | *required* | The `*-10K-QA.csv` manifest. Also determines the output location. |
| `--batch_size` | int | *varies* | Generation batch size. Defaults are tuned per architecture — see the [roster](#model-roster). |
| `--chunk_id` | int | `0` | Zero-indexed shard identifier. |
| `--num_chunks` | int | `1` | Total shards. `> 1` activates row-range slicing. |
 
Model-specific additions:
 
| Flag | Scripts | Default | Description |
| :--- | :--- | :---: | :--- |
| `--model_id` | InternVL (all), LLaVA-NeXT (both) | *per script* | Override the checkpoint. The output directory and column name are derived from it. |
| `--max_num` | InternVL (all) | `12` | Maximum dynamic image tiles. Lower to `6` to reduce memory on the 30B/38B checkpoints. |
| `--use_flash_attn` | InternVL (all) | off | Enable flash-attention. Requires `flash-attn` to be installed. |
| `--num_crops` | Phi-3.5-vision | `16` | Crop count. 16 is the vendor recommendation for single images. |
| `--attn` | Phi-3.5-vision | `eager` | `eager` (safe) \| `sdpa` \| `flash_attention_2`. |
 
---
 
## Input and Output Contract
 
### Input
 
Consumes the QA manifest produced by `generateQA.py`:
 
| Column | Use |
| :--- | :--- |
| `local_path` | Resolved against `--dataset_dir` to locate the image. |
| `qa_pairs` | JSON string; `Q` is the prompt, `A` the ground truth. |
 
All other columns are carried through untouched.
 
### Output
 
Written **next to the input manifest**, in a model-named subdirectory:
 
```
<BASE_DIR>/Amphibians-10K/
├── Images/
├── Amphibians-10K.csv
├── Amphibians-10K-QA.csv            ← --csv_path
├── Qwen3-VL-32B/
│   ├── Amphibians-10K-QA_chunk0_qwen3_32b.csv
│   ├── Amphibians-10K-QA_chunk1_qwen3_32b.csv
│   └── ...
├── Gemma-3-12B/
├── Molmo2-8B/
└── ...
```
 
The output CSV is the input rows plus one column carrying that model's answers. With `--num_chunks > 1`
the filename carries `_chunk{chunk_id}`; with a single chunk the infix is omitted.
 
> **No merge step is provided.** Concatenating shards is left to the consumer. Preserve chunk order —
> shard *k* holds rows `[k·⌈N/K⌉, (k+1)·⌈N/K⌉)` of the input.
 
---
 
## Environments
 
The runners impose mutually incompatible dependency constraints and therefore require **two virtual
environments**. Do not attempt to unify them.
 
### Primary environment
 
Covers Qwen, Gemma, Llama, LLaVA-NeXT, Pixtral, InternVL, and Phi-3.5-vision.
 
```bash
python -m venv <VENV_PRIMARY>
source <VENV_PRIMARY>/bin/activate
pip install torch torchvision transformers accelerate
pip install pillow opencv-python pandas
pip install qwen-vl-utils          # Qwen runners only
pip install flash-attn --no-build-isolation   # optional; enables --use_flash_attn
```
 
### Molmo2 environment
 
Molmo2 ships custom modeling code pinned by the vendor to a specific transformers release. Installing it
into the primary environment would break the other runners.
 
```bash
python -m venv <VENV_MOLMO2>
source <VENV_MOLMO2>/bin/activate
pip install transformers==4.57.1 torch pillow einops torchvision accelerate
```
 
### Compatibility notes
 
- **Phi-3.5-vision** targets a transformers `Cache` API that later releases removed. The runner installs
  three monkey-patches at import time (`seen_tokens`, `get_max_length`, `get_usable_length`) that
  re-attach the legacy names to the current API. This lets the script run on old or new transformers
  **without modifying the environment**. If the shim fails, a warning is printed and generation will
  likely raise `'DynamicCache' object has no attribute 'seen_tokens'`.
- **InternVL, Molmo2, Phi-3.5-vision** all require `trust_remote_code=True`. Model code is executed from
  the Hub repository. Pin checkpoint revisions if reproducibility across time matters to you.
- **Gemma 4** ships a built-in thinking mode. The runner passes `enable_thinking=False` so that the
  15-token budget is spent on the answer rather than on reasoning traces, falling back gracefully via
  `TypeError` if the installed build does not accept the kwarg. **Verify the fallback did not silently
  engage** — if it did, that model's answers are being truncated mid-thought and are not comparable.
---
 
## Running
 
The submission scripts are not committed. The template below is the canonical form; **the only value
that changes across the ten organismal groups is `GROUP`.**
 
### Standard runner
 
```bash
#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=qwen3_32B_Amphibians
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --gpus-per-node=<N_GPU>
#SBATCH --cpus-per-task=8
#SBATCH --mem=<MEM>
#SBATCH --time=24:00:00
#SBATCH --output=logs/qwen3_32B_Amphibians_%j.out
#SBATCH --error=logs/qwen3_32B_Amphibians_%j.err
 
GROUP="Amphibians"                                   # <-- the only per-taxon change
 
BASE_DIR="<BASE_DIR>"
PYTHON_BIN="<VENV_PRIMARY>/bin/python"
SCRIPT_PATH="<REPO>/<INFERENCE_DIR>/run_qwenVL3_32B.py"
 
DATASET_DIR="${BASE_DIR}/${GROUP}-10K"
CSV_PATH="${DATASET_DIR}/${GROUP}-10K-QA.csv"
 
$PYTHON_BIN $SCRIPT_PATH \
  --dataset_dir "$DATASET_DIR" \
  --csv_path    "$CSV_PATH" \
  --batch_size  4
 
echo "[DONE] ${GROUP} inference complete."
```
 
Substitute `GROUP` with any of:
 
```
Amphibians · Angiosperms · Arachnids · Birds · Bryophytes
Fish · Gymnosperms · Insects · Mammals · Reptiles
```
 
### InternVL
 
Passes the checkpoint explicitly and exposes the tiling budget:
 
```bash
$PYTHON_BIN "<REPO>/<INFERENCE_DIR>/run_InternVL_3_5_38B.py" \
  --dataset_dir "$DATASET_DIR" \
  --csv_path    "$CSV_PATH" \
  --model_id    "OpenGVLab/InternVL3_5-38B-Instruct" \
  --batch_size  4 \
  --max_num     6 \
  --use_flash_attn
```
 
`--max_num 6` halves the tile budget; use it when the 30B/38B checkpoints exhaust memory. Omit
`--use_flash_attn` unless `flash-attn` is installed in the environment.
 
### Molmo2
 
Identical arguments, different interpreter:
 
```bash
PYTHON_BIN="<VENV_MOLMO2>/bin/python"
 
$PYTHON_BIN "<REPO>/<INFERENCE_DIR>/run_Molmo2_8B.py" \
  --dataset_dir "$DATASET_DIR" \
  --csv_path    "$CSV_PATH" \
  --batch_size  4
```
 
### Phi-3.5-vision
 
Sequential; expect substantially longer wall time and budget accordingly.
 
```bash
$PYTHON_BIN "<REPO>/<INFERENCE_DIR>/run_phi_3_5_vision.py" \
  --dataset_dir "$DATASET_DIR" \
  --csv_path    "$CSV_PATH" \
  --num_crops   16 \
  --attn        eager
```
 
### LLaVA-NeXT
 
The two runners are identical apart from the default checkpoint; either accepts `--model_id` for any
`llava-v1.6` variant.
 
```bash
$PYTHON_BIN "<REPO>/<INFERENCE_DIR>/run_llava_next_vicuna_7B.py" \
  --dataset_dir "$DATASET_DIR" \
  --csv_path    "$CSV_PATH" \
  --model_id    "llava-hf/llava-v1.6-vicuna-7b-hf" \
  --batch_size  4
```
 
---
 
## Chunking and SLURM Arrays
 
A single manifest holds roughly 90K questions. `--num_chunks` / `--chunk_id` shard it across array
tasks; each task writes its own output file.
 
```bash
#!/bin/bash
#SBATCH --job-name=qwen3_8B_Birds
#SBATCH --array=0-7
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen3_8B_Birds_%A_%a.out
#SBATCH --error=logs/qwen3_8B_Birds_%A_%a.err
 
GROUP="Birds"
NUM_CHUNKS=8
 
BASE_DIR="<BASE_DIR>"
DATASET_DIR="${BASE_DIR}/${GROUP}-10K"
CSV_PATH="${DATASET_DIR}/${GROUP}-10K-QA.csv"
 
<VENV_PRIMARY>/bin/python "<REPO>/<INFERENCE_DIR>/run_qwenVL3_8B.py" \
  --dataset_dir "$DATASET_DIR" \
  --csv_path    "$CSV_PATH" \
  --batch_size  32 \
  --num_chunks  $NUM_CHUNKS \
  --chunk_id    $SLURM_ARRAY_TASK_ID
```
 
Sharding is by contiguous row range: `chunk_size = ceil(N / num_chunks)`, and task *k* processes
`[k·chunk_size, min((k+1)·chunk_size, N))`. The array size and `NUM_CHUNKS` must agree — a mismatch
silently drops the tail of the manifest.
 
Because temporary frame files are chunk-scoped, array tasks may share a working directory safely.
 
---


 
<p align="center">
  <sub>Part of the <b>Bio-VQA</b> benchmark · Imageomics Institute</sub>
</p>
 

