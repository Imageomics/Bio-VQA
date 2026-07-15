<h1 align="center">Tree-of-Life-200M &rarr; Bio-VQA</h1>

<p align="center">
  <b>Hierarchical Taxonomic Reasoning &mdash; Data Acquisition &amp; QA Generation Pipeline</b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="DuckDB" src="https://img.shields.io/badge/engine-DuckDB-yellow">
  <img alt="Dataset" src="https://img.shields.io/badge/source-TreeOfLife--200M-green">
  <img alt="Scheduler" src="https://img.shields.io/badge/scheduler-SLURM-red">
  <img alt="Status" src="https://img.shields.io/badge/status-active-brightgreen">
</p>

---

## Overview

This module builds the **hierarchical taxonomic reasoning** split of **Bio-VQA**: ten taxon-balanced,
metadata-complete image corpora of ~10K images each, drawn from the
[TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M) catalog, and the
visual-question-answering pairs derived from their taxonomic labels.

The pipeline is deliberately **three small, single-responsibility scripts** rather than one monolith:

| Stage | Script | Responsibility |
| :---: | :--- | :--- |
| 1 | `download.py` | Query the catalog, select species with **complete** taxonomy, download images concurrently, emit a manifest CSV. |
| 2 | `verify.py` | Audit every manifest for missing / `NaN` / empty taxonomic ranks before anything downstream runs. |
| 3 | `generateQA.py` | Turn each verified row into open-ended and multiple-choice QA pairs across eight taxonomic levels. |

> **Design contract:** a row exists in the manifest **if and only if** its image is on disk **and** every
> taxonomic rank used by the QA generator is non-null. Stage 2 exists to prove that contract holds.

---

## Table of Contents

- [Repository Layout](#repository-layout)
- [Output Layout](#output-layout)
- [Prerequisites](#prerequisites)
- [Data Source](#data-source)
- [Stage 1 &mdash; `download.py`](#stage-1--downloadpy)
- [Stage 2 &mdash; `verify.py`](#stage-2--verifypy)
- [Stage 3 &mdash; `generateQA.py`](#stage-3--generateqapy)
- [Taxon Configurations](#taxon-configurations)
- [Running on a SLURM Cluster](#running-on-a-slurm-cluster)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Citation](#citation)

---

## Repository Layout

```
3.Hierarchical_taxonomic_reasoning/
└── Tree-of-Life-200M/
    ├── README.md          # you are here
    ├── download.py        # Stage 1 — catalog query + concurrent image download
    ├── verify.py          # Stage 2 — metadata completeness audit
    └── generateQA.py      # Stage 3 — QA pair synthesis
```

No shell scripts are committed. All launch commands live in this README — copy the ones you need into
your own scheduler script, adapting paths to your environment.

---

## Output Layout

Each taxon produces one self-contained directory. Everything is relative; nothing in the manifest
hard-codes a machine-specific path.

```
<BASE_DIR>/
├── Amphibians-10K/
│   ├── Images/
│   │   ├── <uuid>.jpg
│   │   ├── <uuid>.png
│   │   └── ...                       # exactly one file per manifest row
│   ├── Amphibians-10K.csv            # manifest  (catalog columns + local_path)
│   └── Amphibians-10K-QA.csv         # QA pairs   (Stage 3 output)
├── Angiosperms-10K/
├── Arachnids-10K/
├── Birds-10K/
├── Bryophytes-10K/
├── Fish-10K/
├── Gymnosperms-10K/
├── Insects-10K/
├── Mammals-10K/
└── Reptiles-10K/
```

**Manifest schema** = every column of the TreeOfLife-200M catalog, plus:

| Column | Description |
| :--- | :--- |
| `local_path` | Relative path of the form `Images/<uuid>.<ext>` — resolve against the taxon directory. |

**QA schema** (`*-QA.csv`):

| Column | Description |
| :--- | :--- |
| `uuid` | Image identifier, joins back to the manifest. |
| `source_url` | Original image URL. |
| `local_path` | `Images/<uuid>.<ext>`. |
| `qa_pairs` | JSON string: `{"Q": "...", "A": "..."}`. One row per question. |

---

## Prerequisites

```bash
python -m venv myenv
source myenv/bin/activate
pip install duckdb pandas requests tqdm pyarrow
```

| Package | Why |
| :--- | :--- |
| `duckdb` | Out-of-core SQL over the Parquet catalog — no need to load 200M rows into RAM. |
| `pyarrow` | Parquet backend. |
| `pandas` | Grouping, sampling, CSV IO. |
| `requests` | Pooled HTTP session for downloads. |
| `tqdm` | Progress bar. |

**Hardware guidance.** Stage 1 is network- and I/O-bound, not compute-bound: request CPU nodes, not
GPUs. ~12–24 CPUs and ≥128 GB total memory is comfortable; the memory headroom is for the DuckDB
materialization of candidate rows, not the downloads themselves.

---

## Data Source

The pipeline reads a **single Parquet file**, the TreeOfLife-200M catalog:

```
https://huggingface.co/datasets/imageomics/TreeOfLife-200M/blob/main/dataset/catalog.parquet
```

Fetch it once, anywhere you like, and pass the path via `--catalog`:

```bash
huggingface-cli download imageomics/TreeOfLife-200M \
  dataset/catalog.parquet \
  --repo-type dataset \
  --local-dir ./tol200m
```

The catalog contains metadata and `source_url` only — **images are not redistributed**, which is
precisely why Stage 1 downloads them from the original providers.

Taxonomic columns consumed by this pipeline:

```
kingdom · phylum · class · order · family · genus · species · scientific_name · common
```

> `order` is a reserved SQL keyword; `download.py` quotes it as `"order"` inside DuckDB.
> `common` is renamed to `common_name` by `generateQA.py` for readability.

---

## Stage 1 &mdash; `download.py`

### What it does

1. **Species discovery.** Runs a DuckDB aggregate over the catalog, keeping only rows where the taxon
   filter matches, `source_url` is present, **and every taxonomic rank is non-null**. Species are
   returned ordered by frequency (most-photographed first).
2. **Candidate materialization.** Pulls the full rows for the top species into memory and groups by
   `scientific_name`.
3. **Concurrent download.** A `ThreadPoolExecutor` dispatches one task per species. Each task shuffles
   its candidate rows and downloads until it has `--images_per_species` successes.
4. **Quality gate.** A download counts as a success only if the file lands on disk **and** exceeds
   `--min_size_kb`. Undersized or failed files are deleted immediately — this filters thumbnails,
   placeholder images, and truncated transfers.
5. **Global stop.** A lock-guarded counter halts every worker the instant `--stop_limit` successes are
   recorded, so you get *exactly* N rows, not N-ish.
6. **Stray cleanup.** Any file in `Images/` not referenced by the manifest is removed — this catches
   images that landed after the stop signal fired.
7. **Sanity check.** Prints `CSV rows` vs `files on disk` and asserts they match.

### Arguments

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--catalog` | path | *required* | Path to `catalog.parquet`. |
| `--output_dir` | path | *required* | Taxon directory; `Images/` is created inside it. |
| `--output_csv` | path | *required* | Manifest destination. |
| `--taxon_rank` | str | *required* | Column to filter on (`class`, `phylum`, `order`, …). |
| `--taxon_value` | str | *required* | Single value (`Aves`) **or** a SQL tuple (`('Magnoliopsida','Liliopsida')`) — a value containing `(` switches the filter to `IN`. |
| `--images_per_species` | int | `10` | Per-species cap. Raises diversity; lower it for species-rich taxa. |
| `--stop_limit` | int | `10000` | Global image budget. |
| `--min_size_kb` | float | `1024` | Minimum accepted file size, in KB. |
| `--threads` | int | `24` | Worker threads. |
| `--skip_order` | flag | off | Drops `order` from the NOT-NULL completeness check. |

### Usage

```bash
python download.py \
  --catalog             "<PATH>/catalog.parquet" \
  --output_dir          "<BASE_DIR>/Birds-10K" \
  --output_csv          "<BASE_DIR>/Birds-10K/Birds-10K.csv" \
  --taxon_rank          "class" \
  --taxon_value         "Aves" \
  --images_per_species  20 \
  --stop_limit          10000 \
  --min_size_kb         1024 \
  --threads             24
```

### Tuning the two knobs that matter

`--images_per_species` and `--min_size_kb` jointly determine whether you reach `--stop_limit`.

- **Species-rich, well-photographed taxa** (Birds, Insects, Mammals): a low per-species cap and a high
  size floor both hold — there are plenty of species to draw from.
- **Species-poor taxa** (Gymnosperms: ~1K accepted species worldwide): 10K images cannot be reached at
  20/species. Raise the cap to 100 and drop the floor to 256 KB.

Rule of thumb: `stop_limit ≲ n_valid_species × images_per_species × success_rate`, where `success_rate`
is typically 0.5–0.8 depending on how many source URLs are dead or undersized. If Stage 1 finishes
short, that inequality was violated.

---

## Stage 2 &mdash; `verify.py`

### What it does

Walks `--base_dir` recursively, opens every `*.csv` that is **not** a `*-QA.csv`, and reports, per file
and per taxonomic rank, how many rows are missing a value. "Missing" is checked three ways, because CSV
round-tripping produces all three:

- a true `NaN`,
- the literal string `"nan"` (case-insensitive),
- an empty / whitespace-only string.

A column absent from the file entirely is reported as `COLUMN MISSING` rather than crashing.

### Usage

```bash
python verify.py --base_dir "<BASE_DIR>"
```

### Expected output

```
============================================================
FILE: Birds-10K.csv
Total Rows: 10000
  - phylum         : 0 missing
  - class          : 0 missing
  - order          : 0 missing
  - family         : 0 missing
  - genus          : 0 missing
  - species        : 0 missing
  - scientific_name: 0 missing
  - common         : 0 missing
============================================================
[DONE] Verification process complete.
```

> **Read this as a gate, not a report.** Every count should be `0`, because Stage 1's SQL already
> enforced non-nullity. A non-zero count means the manifest and the query disagree — investigate before
> running Stage 3. The one sanctioned exception is `order` for taxa downloaded with `--skip_order`.

---

## Stage 3 &mdash; `generateQA.py`

### What it does

For every manifest row, walks eight taxonomic levels:

```
phylum → class → order → family → genus → species → scientific_name → common_name
```

and emits one question per non-null level. Levels that are null are silently skipped, so no question is
ever generated without a ground-truth answer.

**Label conditioning.** The question addresses the subject as an *animal*, *plant*, or *organism*,
inferred from `kingdom` (`Animalia` → animal, `Plantae` → plant, else organism). This keeps phrasing
natural across a corpus spanning bryophytes and mammals.

**Template pools.** Five paraphrases for ordinary ranks, three for name-like ranks
(`scientific_name`, `common_name`) where "which … does it belong to" reads wrong. One is sampled per
row, so the benchmark is not a single memorizable string.

**Species epithet.** The `species` level is verbalized as *"species epithet"* — TreeOfLife-200M stores
the epithet alone, not the binomial, and asking for "the species" invites the binomial as an answer.

**Multiple-choice item.** At `common_name`, an additional 4-way MCQ is emitted. Distractors are drawn
from a **taxonomically graded pool** — other common names sharing the row's genus, then family, then
order, then class. These are *hard negatives*: a model must distinguish congeners, not merely tell a
frog from a fern. If fewer than three siblings exist, the pool is topped up from the dataset-wide name
list. Options are shuffled, so correct-answer position carries no signal.

### Arguments

| Flag | Type | Description |
| :--- | :--- | :--- |
| `--base_dir` | path | Root to walk. Every file ending in `-10K.csv` is processed. |

### Usage

```bash
python generateQA.py --base_dir "<BASE_DIR>"
```

Discovery is by suffix: `*-10K.csv` in, `*-10K-QA.csv` out, alongside the input. Already-generated
`-QA.csv` files are not re-processed (they don't match the `-10K.csv` suffix).

### Sample output

```json
{"Q": "What is the family of the animal shown in this image?", "A": "Ranidae"}
{"Q": "What is the species epithet of the animal in this image?", "A": "catesbeianus"}
{"Q": "Identify the scientific name of the animal shown in this image.", "A": "Lithobates catesbeianus"}
{"Q": "Among the four options, what is the common name of the animal shown in this image? Green Frog, American Bullfrog, Pickerel Frog, Wood Frog", "A": "American Bullfrog"}
```

**Yield.** Expect ~9 QA pairs per image (8 ranks + 1 MCQ), i.e. roughly **90K pairs per taxon** and
**~900K across all ten** — modulo rows whose `order` or `common` is absent.

---

## Taxon Configurations

The ten splits, with the parameters each requires. Deviations from the defaults are called out.

| Split | `--taxon_rank` | `--taxon_value` | `--images_per_species` | `--min_size_kb` | Notes |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `Amphibians-10K` | `class` | `Amphibia` | 20 | 512 | |
| `Angiosperms-10K` | `class` | `('Magnoliopsida','Liliopsida')` | 20 | 512 | Dicots + monocots. |
| `Arachnids-10K` | `class` | `Arachnida` | 20 | 512 | |
| `Birds-10K` | `class` | `Aves` | 20 | 1024 | Highest floor — bird imagery is abundant and high-res. |
| `Bryophytes-10K` | `phylum` | `Bryophyta` | 20 | 512 | Filters on **phylum**, not class. |
| `Fish-10K` | `class` | `('Actinopterygii','Teleostei')` | 20 | 512 | |
| `Gymnosperms-10K` | `class` | `Pinopsida` | **100** | **256** | Species-poor — both knobs relaxed to reach 10K. |
| `Insects-10K` | `class` | `Insecta` | 20 | 512 | |
| `Mammals-10K` | `class` | `Mammalia` | 20 | 512 | |
| `Reptiles-10K` | `order` | `('Squamata','Testudines','Crocodylia','Sphenodontia')` | 20 | 512 | See note below. Use `--skip_order`. |

> ### ⚠️ The Reptilia problem
>
> Reptilia is **paraphyletic** — no single modern class value captures it, so the split is defined as a
> union of the four extant reptile *orders*. Two consequences:
>
> 1. **`--taxon_rank` must be `order`**, not `class`. Filtering `class IN ('Squamata',…)` matches
>    nothing, and Stage 1 will happily report *"Found 0 species"* and produce an empty manifest.
> 2. Pass **`--skip_order`** if you also want to relax the `order IS NOT NULL` completeness check for
>    this split. Note that when filtering *on* order, the flag is redundant — a row matching the filter
>    necessarily has a non-null order. It is retained as an escape hatch for any taxon where `order` is
>    sparsely populated upstream.
>
> If a reptile run returns zero species, check the rank before anything else.

---

## Running on a SLURM Cluster

Stages 1 and 3 are long-running; submit them rather than running them on a login node.

### Stage 1 — one job per taxon

```bash
#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=Amphibians
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/Amphibians_%j.out
#SBATCH --error=logs/Amphibians_%j.err

CATALOG="<PATH>/catalog.parquet"
BASE_DIR="<BASE_DIR>"
PYTHON_BIN="<PATH>/myenv/bin/python"
SCRIPT_PATH="<REPO>/Tree-of-Life-200M/download.py"

TAXON_RANK="class"
TAXON_VALUE="Amphibia"
IMG_PER_SPECIES=20
MIN_SIZE=512

SUBDIR="Amphibians-10K"
OUTPUT_DIR="${BASE_DIR}/${SUBDIR}"
OUTPUT_CSV="${OUTPUT_DIR}/${SUBDIR}.csv"
mkdir -p "$OUTPUT_DIR"

$PYTHON_BIN $SCRIPT_PATH \
  --catalog            "$CATALOG" \
  --output_dir         "$OUTPUT_DIR" \
  --output_csv         "$OUTPUT_CSV" \
  --taxon_rank         "$TAXON_RANK" \
  --taxon_value        "$TAXON_VALUE" \
  --images_per_species $IMG_PER_SPECIES \
  --min_size_kb        $MIN_SIZE \
  --threads            $SLURM_CPUS_PER_TASK

echo "[DONE] Download process complete."
```

To produce the other nine splits, change only the block between `TAXON_RANK` and `SUBDIR`, per the
[configuration table](#taxon-configurations).

### Scheduler notes

- **`--nodes=1`, always.** `download.py` is a single Python process using a thread pool; it has no
  multi-node awareness. Requesting two nodes wastes an allocation — the second sits idle.
- **`--threads $SLURM_CPUS_PER_TASK`** keeps thread count tied to the allocation. Because the work is
  network-bound, you can safely oversubscribe (e.g. `2 × CPUs`) if the source hosts tolerate it.
- **CPU partition.** No GPU is touched at any stage.
- **Wall time.** 8 h is generous for 10K images at 20+ threads; species-poor taxa that scan many
  candidates for few successes take longest.
- **Be a good citizen.** Thread count is your rate limit against third-party image hosts. Ten
  concurrent 24-thread jobs is 240 simultaneous requests to a partly overlapping set of providers.
  Stagger submissions.

### Stages 2 and 3 — once, over everything

```bash
$PYTHON_BIN <REPO>/Tree-of-Life-200M/verify.py     --base_dir "$BASE_DIR"
$PYTHON_BIN <REPO>/Tree-of-Life-200M/generateQA.py --base_dir "$BASE_DIR"
```

Run `verify.py` interactively — it is fast and you want to read it. Only proceed to `generateQA.py`
once the audit is clean.

---

## Reproducibility

| Component | Deterministic? | Mechanism |
| :--- | :---: | :--- |
| Species selection (Stage 1) | ✅ | Deterministic SQL, ordered by frequency. |
| Image selection (Stage 1) | ❌ | `df.sample(frac=1)` is unseeded, and thread interleaving decides which downloads win the race to `stop_limit`. **Two runs will not give identical manifests.** |
| Verification (Stage 2) | ✅ | Pure read. |
| QA generation (Stage 3) | ✅ | `RANDOM_SEED = 42`, **re-seeded per file** before processing, so a given manifest always yields the same templates, distractors, and option order — regardless of how many files precede it in the walk. |

**Implication:** the manifest CSVs are the artifact of record. Given a manifest, Stage 3 is exactly
reproducible; the download stage is not. Archive the manifests (and, where licensing permits, the
images) rather than expecting to regenerate them byte-for-byte.

If you need reproducible downloads, seed the sampler — `group_df.sample(frac=1, random_state=42)` — and
run single-threaded. Expect it to be dramatically slower.

---

## Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `[INFO] Found 0 species with valid metadata.` | Rank/value mismatch — most often Reptiles filtered on `class`. | Verify the value exists at that rank: `SELECT DISTINCT class FROM read_parquet('…') LIMIT 50`. |
| Final CSV rows &lt; `stop_limit`; `❌ ERROR: Mismatch` | Candidate pool exhausted: too few species × cap, or the size floor rejected too much. | Raise `--images_per_species`, lower `--min_size_kb`, or accept a smaller split. |
| Job OOM-kills during `[STEP 1]` | The materialization step pulls all candidate rows into a pandas DataFrame; broad taxa (Insecta) are large. | Increase `--mem-per-cpu`, or narrow the taxon value. |
| Downloads crawl; many failures | Source hosts throttling, or dead URLs (the catalog indexes historical records). | Lower `--threads`; a nontrivial failure rate is expected and normal. |
| `verify.py` reports missing `order` | Split was downloaded with `--skip_order`. | Expected. Any *other* missing rank is a real bug — Stage 1's SQL should have excluded those rows. |
| `generateQA.py` skips a file | Discovery matches the literal `-10K.csv` suffix only. | Rename the manifest to `<Taxon>-10K.csv`. |
| `KeyError: 'common_name'` | Manifest lacks the `common` column entirely. | Confirm you're reading the TreeOfLife-200M catalog and not a re-exported subset that dropped columns. |
| Truncated / corrupt images | A file passed the size floor but the transfer was incomplete. | The size gate is a heuristic, not a decoder. Add a Pillow `verify()` pass if it bites. |

---

## FAQ

**Why cap images per species?**
Without a cap, frequency-ordered selection is brutally long-tailed — a handful of charismatic species
would dominate the split and the benchmark would measure recall of *Turdus migratorius*, not taxonomic
reasoning. The cap forces breadth.

**Why the file-size floor?**
`source_url` fields point at whatever the provider served, including 3 KB thumbnails and 404 pages
returned with a 200. Size is a cheap, effective first-pass filter. Tune it per taxon: bird photography
is high-resolution; herbarium bryophyte scans often are not.

**Why does the sanity check compare CSV rows to file count?**
Because the failure mode is silent. Threads racing toward `stop_limit` can write files after the stop
signal, and a crashed writer can leave rows without files. Stage 1's cleanup plus the count assertion
turn that into a loud, immediate failure.

**Can I re-run a single taxon?**
Yes — each taxon directory is independent. `download.py` opens the manifest in `'w'` mode, so a re-run
truncates and rebuilds it, and stray-image cleanup removes the previous run's orphans.

**Why is `species` asked as "species epithet"?**
Because the catalog's `species` column holds the epithet (`catesbeianus`), not the binomial. Asking
"what is the species?" would score a correct binomial as wrong.

**Are the MCQ distractors really hard?**
As hard as the split allows. The pool widens genus → family → order → class and stops as soon as it has
three candidates, so a species with congeners in the corpus gets congeneric distractors. Rare species
fall back further, and in the worst case to a global sample — those items are easy. If you need a
uniform difficulty guarantee, filter the MCQ items by distractor rank.

---

## Citation

If you use this pipeline or the resulting splits, please cite the underlying dataset:

```bibtex
@misc{treeoflife200m,
  title  = {TreeOfLife-200M},
  author = {Imageomics Institute},
  year   = {2025},
  url    = {https://huggingface.co/datasets/imageomics/TreeOfLife-200M}
}
```

Images are downloaded from their original providers and retain their original licenses. This repository
distributes **code and metadata only** — verify licensing before redistributing any downloaded imagery.

---

<p align="center">
  <sub>Part of the <b>Bio-VQA</b> benchmark · Imageomics Institute</sub>
</p>
