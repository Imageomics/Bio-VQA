<h1 align="center"> Counting-Relative &rarr; Stage 2 &rarr; Manifest Assembly &amp; Pre-Run Audit </h1>

<p align="center">
  <b>Condition merging, gold extraction, deterministic sharding, four standing checks</b>
</p>

<p align="center">
  <img alt="Stage" src="https://img.shields.io/badge/stage-2%20of%204-blue">
  <img alt="Compute" src="https://img.shields.io/badge/compute-login%20node-lightgrey">
  <img alt="Chunks" src="https://img.shields.io/badge/chunks-6-green">
  <img alt="Rows" src="https://img.shields.io/badge/rows-42%2C412-informational">
</p>

---

## Overview

| Script | Role |
| :--- | :--- |
| `split_qa.py` | Combine `Same/` and `Different/`, parse fields, shard into 6 chunks |
| `stats.py` | Read-only audit: key balance, polarity pairing, clustering, runtime estimate |

> **Design contract:** every issue this stage can catch &mdash; an unresolvable path, a collapsed
> answer key, an unpaired grid &mdash; would otherwise surface only *after* inference and require
> rerunning all 18 models. The audit costs seconds; the sweep costs ~189 GPU-hours.

---

## Table of Contents

- [Why One Combined Manifest](#why-one-combined-manifest)
- [Columns Added](#columns-added)
- [Gold-Letter Extraction](#gold-letter-extraction)
- [Deterministic Sharding](#deterministic-sharding)
- [The Four Audit Checks](#the-four-audit-checks)
- [Usage](#usage)
- [FAQ](#faq)

---

## Why One Combined Manifest

```
   Same/qa.csv  ─┐
                 ├─→  split_qa.py  ─→  chunks/chunk00 … chunk05/qa.csv
Different/qa.csv ─┘                     each ~7,069 rows, both conditions mixed
```

Three reasons:

1. **One model load per array task** instead of two.
2. **Identical conditions guaranteed.** Both are evaluated under the same loading, batching and prompt
   configuration. Separate jobs would leave open the possibility of a difference, confounding the
   task's primary contrast.
3. **Failure isolation.** Rows are interleaved after a seeded shuffle, so every chunk holds the same
   condition and polarity mix. One failed array task cannot wipe out a whole condition.

---

## Columns Added

| Column | Purpose |
| :--- | :--- |
| `row_id` | Assigned **before** shuffling; restores original order at merge |
| `condition` | `same` \| `different` |
| `polarity` | `most` \| `least` |
| `answer_letter` | `A`/`B`/`C`/`D`, extracted from `"Image C (bottom-left)"` |
| `source_dataset` | `same` only, from the `quad_<Dataset>_` filename |
| `animal_term` | The noun the question asks about |
| `image_abspath` | Resolved once, absolutely |
| `n_options`, `answer_in_options`, `answer_pos` | Option-order audit |

### Why `image_abspath` is not optional

Chunk CSVs live in `chunks/chunkNN/` while images stay in `<condition>/Images/`. A relative path is
unresolvable from every chunk directory. Resolving once at split time also means a missing file is
caught **here, on a login node**, rather than after a job has queued.

---

## Gold-Letter Extraction

```python
GOLD_RE = re.compile(r"[Ii]mage\s*([ABCD])\b")     # anchored on the word
POS_RE  = re.compile(r"\b(top|bottom)[\s\-_]*(left|right)\b", re.I)
```

A naive search returns the wrong answer for every row:

```python
re.search(r"([ABCD])", "Image B (top-right)", re.I).group(1)   # → 'A'   ✗
#                        ↑ the lower-case 'a' in "Image"
```

Fallback chain: anchored `Image X` &rarr; quadrant name &rarr; standalone letter.

---

## Deterministic Sharding

```python
df.insert(0, "row_id", range(len(df)))                 # id FIRST
df = df.sample(frac=1.0, random_state=args.seed)       # THEN shuffle
for i in range(n_chunks):
    df.iloc[i::n_chunks].to_csv(f"chunks/chunk{i:02d}/qa.csv")
```

Rows arrive grouped by condition and alternating in polarity. A plain `i::6` stride on that ordering
can produce chunks that are entirely one polarity. `row_id` is assigned first, so the original order
remains recoverable at merge.

---

## The Four Audit Checks

### 1 &mdash; Is the answer key balanced?

| Letter | `different`/least | `different`/most | `same`/least | `same`/most |
| :---: | ---: | ---: | ---: | ---: |
| A | 0.249 | 0.248 | 0.245 | 0.245 |
| B | 0.251 | 0.250 | 0.253 | 0.255 |
| C | 0.249 | 0.255 | 0.254 | 0.250 |
| D | 0.251 | 0.247 | 0.249 | 0.250 |

All within 0.245&ndash;0.255. Chance is 0.250 and the majority baseline coincides with it, so plain
accuracy is meaningful for the `same` condition.

> ⚠️ **A single letter at 1.000 would indicate the `Image`&rarr;`a` extraction bug.** This table is the
> standing regression check.

### 2 &mdash; Is every grid paired?

```
images with both most and least : 20000
images with only one polarity   : 0
```

Required for the `incoherence` metric &mdash; an unpaired grid cannot contribute to a consistency check.

### 3 &mdash; How badly are rows clustered?

```
rows per image: mean 2.12  max 6
```

Two questions per grid, except GrevysZebra grids which carry three animal terms
(`animals`, `grevys zebras`, `zebras`) &times; two polarities = **six rows**.

> **Effective sample size is the 20,000 images**, not the 42,412 rows. `evaluate.py` bootstraps over
> images accordingly. This is also why `incoherence` pairs on `(image, animal_term)`: pairing on the
> image alone would compare *"most animals"* against *"least grevys zebras"* &mdash; which may
> legitimately share an answer &mdash; and silently discard the other two pairs.

### 4 &mdash; What is the `different`-condition floor?

`stats.py` prints the Monte-Carlo prior baselines and the exact flags to pass:

```
--prior_different_most 0.530 --prior_different_least 0.548
```

It does **not** recompute them, because that requires per-panel source datasets which the manifest
deliberately does not carry for the `different` condition.

---

## Usage

```bash
python split_qa.py --task_root /path/to/Counting-Relative
python split_qa.py --task_root ... --n_chunks 8 --seed 1234 --force

python stats.py
python stats.py --run_dir /path/to/6.Counting_Relative
```

`--force` is required to overwrite an existing `chunks/`, because re-splitting would orphan any model
outputs already written there.

### Reading order

```
split_qa.py  →  stats.py  →  [inspect the four checks]  →  run_all.sh
```

---