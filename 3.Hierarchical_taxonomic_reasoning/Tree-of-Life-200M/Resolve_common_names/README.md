<h2 align="center"> Hierarchical Taxonomic Reasoning &rarr; TOL-200M &rarr; Common-Name Resolution </h2>

<p align="center">
  <b>Detecting, Quantifying &amp; Repairing Multilingual Vernacular-Name Contamination in the Common-Name Question Set</b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="Source" src="https://img.shields.io/badge/source-TreeOfLife--200M-green">
  <img alt="Backbone" src="https://img.shields.io/badge/backbone-GBIF-805ad5">
  <img alt="Authorities" src="https://img.shields.io/badge/authorities-GBIF%20%C2%B7%20iNaturalist%20%C2%B7%20Wikipedia-teal">
  <img alt="Status" src="https://img.shields.io/badge/answer%20key-frozen-dd6b20">
</p>

---

## Overview

The vernacular ("common") names shipped in the TreeOfLife-200M catalog are inherited from the
**GBIF Backbone Taxonomy**, whose `vernacularName` field is **multilingual by design**. As a result,
a non-trivial fraction of this component's *common-name* questions were generated with a
**non-English gold answer** &mdash; e.g. `grönfläckig padda` (Swedish) instead of *European green
toad* &mdash; or with a **non-English distractor** inside a multiple-choice option set.

This module detects that contamination, resolves each affected taxon to one canonical English common
name through a three-stage pipeline of decreasing automation, and freezes the result into a single
answer key, `common_names_resolved.csv`. Every design decision below is stated **together with the
failure mode it prevents**, so the procedure can be audited rather than trusted.

| Stage | Script | Responsibility |
| :---: | :--- | :--- |
| 1&ndash;2 | `Resolve_common_names.py` | Offline language triage (GBIF codes + English wordlist), then online API resolution (GBIF / iNaturalist / Wikipedia). |
| 3 | *(manual)* | Hand-curate the residual taxa the network could not resolve, with a cited source. |
| Merge | `Resolve_common_names_Finalize.py` | Deterministically merge the three tiers into the frozen key; log exclusions and conflicts. |


> **Provenance note &mdash; no party is at fault.** GBIF's `vernacularName` extension is multilingual
> by specification; TreeOfLife-200M inherits that field unchanged; and the images are distributed with
> all taxonomic fields already populated, with **no per-record language indicator surfaced at download
> time**. The non-English strings are therefore not an error at any layer &mdash; each field is
> correct in its own context. They are *latent*: the multilingual character of the field only becomes
> visible, and only becomes a problem, once the field is consumed as an English gold label. Detecting
> and correcting that latent property is the entire purpose of this module.

---

## Table of Contents

- [Why Multilingual Gold Labels Break VQA](#why-multilingual-gold-labels-break-vqa)
- [The Contamination, With Examples](#the-contamination-with-examples)
- [Pipeline at a Glance](#pipeline-at-a-glance)
- [GBIF Backbone &mdash; Download &amp; Version Pinning](#gbif-backbone--download--version-pinning)
- [Stage 1 &mdash; Offline Language Triage](#stage-1--offline-language-triage)
- [Stage 2 &mdash; API Resolution](#stage-2--api-resolution)
- [Stage 3 &mdash; Manual Curation](#stage-3--manual-curation)
- [Finalize &mdash; Merge &amp; Precedence](#finalize--merge--precedence)
- [A Worked Trace: One Taxon Through Every Stage](#a-worked-trace-one-taxon-through-every-stage)
- [The `is_english` Decision Procedure](#the-is_english-decision-procedure)
- [Taxa With No Formal English Name: Genus Fallback](#taxa-with-no-formal-english-name-genus-fallback)
- [Report Files: What Each Stores &amp; How to Regenerate](#report-files-what-each-stores--how-to-regenerate)
- [Scope &amp; Measured Impact](#scope--measured-impact)
- [Applying the Fix to the Question Set](#applying-the-fix-to-the-question-set)
- [Reproducibility](#reproducibility)
- [Limitations &amp; Threats to Validity](#limitations--threats-to-validity)
- [FAQ](#faq)

---

## Why Multilingual Gold Labels Break VQA

A benchmark answer key is a measurement instrument. A non-English gold label degrades it in three
distinct ways, each requiring a different remedy.

**1. Unanswerable items inject noise, not signal.** When the gold answer is `en` or `åkergroda`, a
model that correctly recognizes the organism and emits the correct English name (*Common juniper*,
*Moor frog*) is scored **wrong**. The item stops measuring biological recognition and starts measuring
whether the model reproduces a specific foreign string. Aggregated over thousands of items this
depresses every model's common-name accuracy &mdash; and not uniformly: models with more
European-language pre-training are spuriously advantaged.

**2. Language-cued multiple choice leaks the answer.** In the four-option common-name MCQ, distractors
are sampled from other taxa's `common` fields. When one option is a conspicuous non-English string in
an otherwise-English list (or vice versa), the *orthographic register* of the options correlates with
correctness. A model can pick "the odd one out" without recognizing the organism at all &mdash;
converting a visual item into a partly textual one.

**3. It confounds construct validity.** This benchmark's central claim rests on whether visual
competence transfers across abstraction levels. If a slice of the common-name axis is silently
measuring *language identification*, the cross-component correlation the benchmark reports is
confounded. Removing this confound is a prerequisite for the correlation to mean what we claim.

> **Principle.** A vision&ndash;language item must be answerable *from the image*, and must not be
> answerable *without* it. Multilingual gold labels violate the first; language-cued option sets
> violate the second. Both are repaired here.

---

## The Contamination, With Examples

The contaminating strings are overwhelmingly North-Germanic (Swedish, Danish, Norwegian) with a long
tail of other European languages, reflecting the checklist composition of the underlying occurrence
records. Representative gold-answer repairs *(shipped &rarr; resolved)*:

| `scientific_name` | `common` as shipped | language | resolved English name |
| :--- | :--- | :---: | :--- |
| *Bufotes viridis* | `grönfläckig padda` | Swedish | European green toad |
| *Rana arvalis* | `åkergroda` | Swedish | Moor frog |
| *Rana arvalis* | `Spidssnudet frø` | Danish | Moor frog |
| *Rana temporaria* | `Butsnudet frø` | Danish | Common frog |
| *Rana dalmatina* | `långbensgroda` | Swedish | Agile frog |
| *Lissotriton vulgaris* | `Lille vandsalamander` | Danish | Smooth newt |
| *Lissotriton vulgaris* | `mindre vattensalamander` | Swedish | Smooth newt |
| *Triturus cristatus* | `Stor vandsalamander` | Danish | Crested newt |
| *Bombina bombina* | `klockgroda` | Swedish | European fire-bellied toad |
| *Ichthyosaura alpestris* | `bergsalamander` | Swedish | Alpine newt |
| *Juniperus communis* | `en` | Swedish / Danish | Common juniper |
| *Picea abies* | `gran` | Swedish | Norway spruce |
| *Pinus sylvestris* | `tall` | Swedish | Scots pine |
| *Anguis fragilis* | `kopparödla` | Swedish | Slow worm |

Two structural facts drive the whole design:

- **A single taxon can carry several foreign spellings across the corpus.** *Lissotriton vulgaris*
  appears with both a Swedish and a Danish name above. The resolver therefore keys on the
  **scientific name**, not on any one `common` string, so every occurrence variant maps to one
  canonical English answer.
- **A foreign string can appear as a *distractor* on an image whose own gold answer is already
  correct.** `en`, `gran`, and `tall` (two-to-four-letter Scandinavian names) are especially insidious
  as options. Repairing only the affected images would leave those strings sitting in otherwise-clean
  option lists, so the fix rewrites contaminated **options** wherever they occur, not only gold
  answers.

The sharpest single case is `en` (*Juniperus communis*): as an English gold label it is not merely
wrong, it is **unanswerable** &mdash; no model or human would emit the two-letter token `en` as the
common name of a conifer. Its danger as a distractor is measured in [Scope &amp; Measured
Impact](#scope--measured-impact).

---

## Pipeline at a Glance

```
                                                   ┌──────────────────────────────────────────────────────────┐
                                                   │           STAGE 1  offline triage        (no network)    │
                                    manifests ──▶  │ GBIF backbone language codes + English wordlist          │
                      (scientific_name + common)   │  ├─ resolve_common_name_english_ascii.csv  (resolved)    │
                                                   │  ├─ taxa_to_resolve.txt                    (handoff)     │
                                                   │  └─ taxa_to_resolve_details.txt            (human read)  │
                                                   └───────────────────────┬──────────────────────────────────┘
                                                                           │
                                                                           │ unresolved taxa only
                                                                           │
                                                                           ▼
                                                   ┌──────────────────────────────────────────────────────────┐
                                                   │                STAGE 2  API resolution                   │
                                    network ────▶  │   iNaturalist → GBIF vernacular → Wikipedia lead         │
                                                   │    ├─ resolve_common_name_inat_wiki.csv                  │
                                                   │    └─ .api_cache.jsonl  (fetch cache)                    │
                                                   └───────────────────────┬──────────────────────────────────┘
                                                                           │
                                                                           │ still-unresolved taxa (≈ {N_manual})
                                                                           │
                                                                           ▼
                                                   ┌──────────────────────────────────────────────────────────┐
                                                   │               STAGE 3  manual curation                   │
                                    human ──────▶  │   manually-entered, with proper references               │
                                                   │    └─ resolve_common_name_manual.csv                     │
                                                   └───────────────────────┬──────────────────────────────────┘
                                                                           │
                                                                           ▼
                                                   ┌──────────────────────────────────────────────────────────┐
                                                   │ FINALIZE  merge of three disjoint tiers                  │
                                                   │  ├─ common_names_resolved.csv    (Finalized for Eval)    │
                                                   │  ├─ excluded_no_english_name.csv (dropped taxa): NONE    │
                                                   │  └─ name_conflicts.csv           (tier disagreements): 0 │
                                                   └──────────────────────────────────────────────────────────┘

```

The staged design exists so that the **cheapest sufficient evidence resolves each taxon**: an
authoritative GBIF language tag settles a name for free, an API call is spent only when offline
evidence is inconclusive, and a human is spent only when the network cannot answer. Because the stages
are **sequential and each taxon exits the pipeline the moment it is resolved**, the three output tiers
are **disjoint by construction** &mdash; a taxon resolved in Stage 1 is never re-queried in Stage 2,
and only what Stages 1&ndash;2 leave behind reaches the manual sheet. This disjointness is why the
merge produces **no conflicts and no exclusions** in practice (see [Finalize](#finalize--merge--precedence)).

---

## GBIF Backbone &mdash; Download &amp; Version Pinning

Stage 1 needs the GBIF Backbone's `VernacularName.tsv`, which maps each vernacular string to its
language code(s). **This file is not committed to the repository** &mdash; the backbone archive is
multiple gigabytes, far past any sane Git limit &mdash; so the pipeline downloads and caches it on
first use.

The backbone is served, by dated version, from:

```
GBIF_BASE = "https://hosted-datasets.gbif.org/datasets/backbone"
```

and `--gbif` accepts a version string matching:

```python
VERSION_RE = re.compile(r"^(?:current|\d{4}-\d{2}-\d{2})$")
```

i.e. a dated snapshot such as `2023-08-28`, or the literal `current`.

**Pass a dated version, never `current`.** `current` is a moving pointer: the file behind it changes
whenever GBIF republishes, so an answer key regenerated six months apart from `current` would silently
differ. A **dated snapshot is immutable and DOI-citable**, which is exactly what a frozen benchmark
answer key requires. Concretely:

```bash
# downloads backbone.zip once, extracts VernacularName.tsv, caches under <out-dir>/gbif/<version>/
python Resolve_common_names.py <BASE_DIR> \
    --gbif 2023-08-28 \
    --out-dir Reports/ \
    --contact you@example.org
```

`--gbif` also accepts a URL or a path to an already-extracted directory / `.zip`, for air-gapped
clusters where the file is staged out-of-band. Whatever the source, `ensure_gbif()` writes the
extracted TSV to `<gbif-cache>/<version>/VernacularName.tsv` and reuses it on subsequent runs, so the
multi-GB download happens exactly once.

> **Which version to pin.** Use the **most recent dated backbone release available at the time the
> benchmark is frozen** &mdash; the latest stable snapshot, not `current`. This maximizes taxonomic and
> vernacular coverage while keeping the key reproducible. Record the exact version string alongside the
> frozen key; it is the single fact a third party needs to regenerate the language verdicts
> byte-for-byte.

---

## Stage 1 &mdash; Offline Language Triage

Stage 1 walks every manifest CSV, aggregates the distinct `common` strings observed per
`scientific_name` (weighted by row frequency), and assigns each a verdict of **English /
not-English / unknown** from two evidence sources, in order:

1. **GBIF backbone language codes.** `VernacularName.tsv` is loaded once into a map from normalized
   name &rarr; set of language subtags. Tagged `en`/`eng` &rarr; accepted as English on authoritative
   evidence. Tagged foreign with no English tag &rarr; evidence *against* English.
2. **English wordlist heuristic.** For names GBIF does not tag, the fraction of tokens present in an
   English dictionary (`web2`, or a system wordlist) is computed: &ge; 0.5 &rArr; English; 0.0 &rArr;
   not English; strictly between &rArr; unknown.

Three subtleties, each corresponding to a real failure observed on this corpus:

- **A blank GBIF language tag means *unknown*, not *foreign*.** A large share of backbone rows leave
  `language` empty. Treating blank as non-English would bury obvious English names (e.g.
  *Three-striped arrow-poison frog*) that simply were not tagged. Blank tags therefore never veto the
  wordlist.
- **Non-ASCII letters outrank the wordlist.** A name containing non-ASCII letters (`grönfläckig`,
  `frø`) is rejected *before* the wordlist runs. This enforces **precision over recall for the answer
  key**: a false *positive* (a foreign name wrongly accepted as English) poisons the gold standard,
  whereas a false *negative* merely costs one Stage-2 lookup. The cheaper error is chosen deliberately.
- **Eponyms are language-neutral and never altered.** *Peters' dwarf frog* and *Nordmann's
  vargspindel* both carry a possessive proper noun. The eponym token is skipped **only when counting
  dictionary words**, so the verdict rests on the remainder: *Peters' dwarf frog* passes on "dwarf
  frog", while *Nordmann's vargspindel* correctly fails on "vargspindel" instead of being auto-passed
  by the apostrophe. **The stored name is always the full, unmodified string.**

**Worked examples through Stage 1:**

| Input `common` | scientific | verdict | evidence | outcome |
| :--- | :--- | :---: | :--- | :--- |
| `American toad` | *Anaxyrus americanus* | yes | `wordlist` (2/2 English) | resolved here |
| `Silver fir` | *Abies alba* | yes | `gbif` (tagged `en`) | resolved here |
| `Salticidae` | *Salticidae* | yes | `genus` (name = the genus) | resolved here &rarr; *"jumping spider" style group answer* |
| `grönfläckig padda` | *Bufotes viridis* | **no** | `non_ascii` (ö, ä) | &rarr; `taxa_to_resolve.txt` |
| `en` | *Juniperus communis* | **no** | `wordlist` (0/1 English) | &rarr; `taxa_to_resolve.txt` |
| `Nordmann's vargspindel` | *Nordmann's...* | **no** | `wordlist` (0/1 after eponym drop) | &rarr; `taxa_to_resolve.txt` |
| `Peters' dwarf frog` | *Physalaemus petersi* | yes | `wordlist` (2/2 after eponym drop) | resolved here |

Everything with a `yes`/`unknown` verdict is written to `resolve_common_name_english_ascii.csv`
(carrying the `evidence` code that decided it). Everything that drops out is handed to Stage 2 via
`taxa_to_resolve.txt`, with a human-readable companion `taxa_to_resolve_details.txt` grouping the
unresolved taxa by source file with every candidate form and its verdict.

> **On the `unknown` verdict.** `unknown` is never a final state. Any taxon whose best offline verdict
> is `unknown` is passed forward to Stage 2, and if the APIs cannot settle it, to the Stage-3 manual
> check &mdash; where it is resolved to a species name, or, in the worst case, to a group/genus name
> (see [Genus Fallback](#taxa-with-no-formal-english-name-genus-fallback)). No `unknown` survives into
> the frozen key unadjudicated.

---

## Stage 2 &mdash; API Resolution

Taxa unresolved by Stage 1 (`grönfläckig padda`, `en`, `kopparödla`, `gran`, `tall`, &hellip;) are
queried against three authorities, taking the first clean English candidate in priority order:
**iNaturalist preferred common name &rarr; GBIF English vernacular &rarr; Wikipedia lead sentence**.

- **iNaturalist** `/v1/taxa` is queried with an English locale; the `preferred_common_name` of an
  exact, active, species-rank match is used.
- **GBIF** `/species/match` resolves the name to a usage key (**following synonyms to the accepted
  name** when the queried name is itself a synonym), then `/vernacularNames` is filtered to English
  entries.
- **Wikipedia** is the last automated resort: the article is fetched and the **lead sentence** is
  parsed with a small set of anchored patterns (*"X, commonly known as the **Y**, is &hellip;"*;
  *"The **Y** (X) &hellip;"*). Only the lead is used and the scientific name is masked before matching,
  to avoid harvesting arbitrary body text.

Every candidate passes a `clean_candidate` gate that rejects strings that are too long, contain rank
words (*species*, *genus*, *family*), end in taxonomic qualifiers (*group*, *complex*, *sensu lato*),
echo the scientific name itself, or contain characters outside a conservative Latin set. Free-text
sources readily return non-names (taxonomic prose, URLs, higher-rank labels); the gate errs toward
**rejecting a borderline candidate and deferring the taxon to manual review**.

**Worked examples through Stage 2:**

| Unresolved taxon | iNat | GBIF vern. | Wikipedia | resolved | source recorded |
| :--- | :--- | :--- | :--- | :--- | :--- |
| *Bufotes viridis* | European green toad | European green toad | &mdash; | **European green toad** | iNat + GBIF URLs |
| *Juniperus communis* | Common juniper | Common juniper | &mdash; | **Common juniper** | iNat URL |
| *Anguis fragilis* | Slow worm | Slow worm | Slow worm | **Slow worm** | iNat URL |
| *Rana arvalis* | &mdash; | Moor frog | Moor frog | **Moor frog** | GBIF URL |
| *(obscure mite, no page)* | &mdash; | &mdash; | &mdash; | *(none)* | &rarr; manual |

The stage is **cached** to `.api_cache.jsonl`, keyed by taxon and carrying a schema version. Re-runs
consult the cache and fetch only genuinely new taxa; a schema bump auto-invalidates stale records.
Traffic is rate-limited (`--sleep`) and carries a contact User-Agent (`--contact`), per each provider's
usage policy. What Stage 2 still cannot resolve falls through to Stage 3.

---

## Stage 3 &mdash; Manual Curation

The residual taxa &mdash; on the order of **&asymp; {N_manual}**, typically obscure invertebrates and
newly-described species with no English name in any queried authority &mdash; are resolved by hand into
`resolve_common_name_manual.csv` (`taxa, resolved_common_name, source`), **each with a cited source**.
Two conventions keep a partially-completed sheet safe:

- A row whose `resolved_common_name` is literally **`N/A`** is an **explicit exclusion**: the taxon is
  dropped and lower tiers are *not* consulted. This is the correct entry when a curator has checked and
  confirmed **no English common name exists**.
- A row left **blank** falls through to the lower automated tiers, and the count of such blanks is
  reported at finalize time &mdash; so a half-finished sheet is *visible* rather than silently treated
  as complete.

**Worked examples through Stage 3:**

| Taxon | manual entry | source | effect |
| :--- | :--- | :--- | :--- |
| *Opiliones sp.* | `harvestman` | group-level English name, cited | genus/group fallback (see below) |
| *Salticidae sp.* | `jumping spider` | family-level English name, cited | genus/group fallback |

Stage 3 is where the **genus-fallback** names of the next-but-one section are entered when the
appropriate English answer is a group-level name rather than a species-level binomial vernacular. In
this corpus every residual taxon received a species-level or group-level English name at this step, so
the exclusion path (`N/A`) was **available but never exercised**.

---

## Finalize &mdash; Merge &amp; Precedence

`Resolve_common_names_Finalize.py` merges the three tier CSVs into the frozen key. A precedence order
(**manual > inat_wiki > ascii**) is defined as a formal tie-break, but **it never fires in practice**:
because the stages are sequential and a taxon exits the pipeline the moment it is resolved, the three
tiers are **disjoint** &mdash; no taxon is present in more than one. Stage 2 only ever sees what Stage 1
could not resolve; the manual sheet only ever sees what Stages 1&ndash;2 could not. There is therefore
nothing for two tiers to disagree *about*.

The merge nonetheless emits two audit artifacts, by design, so that the *absence* of problems is
proven rather than assumed:

- **`excluded_no_english_name.csv`** &mdash; would list every taxon dropped for having no English name
  at any level (explicit `N/A`, or no name from any tier). **In this run it is empty: no taxon was
  excluded**, because every residual reached a species-level or group/genus-level English name at
  Stage 3.
- **`name_conflicts.csv`** &mdash; would list any taxon where two tiers supplied materially different
  names (comparison case-, spacing-, and punctuation-insensitive, so *"Four-spot Orbweaver"* vs
  *"four-spot orb-weaver"* is not counted). **In this run it is empty: there are no conflicts**, which
  is the expected consequence of the disjoint, sequential design above &mdash; the two files exist to
  make that guarantee auditable, not because either was populated.


Both audit files are written only when non-empty; since neither was populated this run, neither is 
present on disk, and their absence is itself the evidence that the merge was clean. The logging is 
retained deliberately: should a future backbone version, an updated GBIF/iNaturalist API response, 
or an expanded taxon set ever introduce a genuine tier disagreement or an un-nameable taxon, 
these files will surface it automatically rather than letting it pass silently into the frozen key.


---

## A Worked Trace: One Taxon Through Every Stage

Following ***Juniperus communis*** end-to-end makes the hand-offs concrete:

```
shipped manifest       scientific_name = Juniperus communis
                       common          = "en"                       (Swedish/Danish)

STAGE 1  triage        norm("en") → GBIF map: not tagged en/eng
                       genus? "en" ≠ "juniperus"            → no
                       non-ASCII? no
                       wordlist fraction of {"en"} = 0/1 = 0.0
                       has_foreign? (blank)                 → verdict: NO / wordlist
                       ⇒ dropped to taxa_to_resolve.txt

STAGE 2  APIs          iNaturalist  q="Juniperus communis" → "Common juniper"  ✓ (species rank)
                       (first clean English hit wins; GBIF/Wiki not needed)
                       ⇒ resolve_common_name_inat_wiki.csv
                         resolved_common_name = "Common juniper"
                         source = https://www.inaturalist.org/taxa/<id>
                       cached in .api_cache.jsonl

STAGE 3  manual        not needed — Stage 2 resolved it

FINALIZE  merge        present in exactly one tier (inat_wiki) → "Common juniper"
                       ⇒ common_names_resolved.csv
                         Juniperus communis, Common juniper, https://www.inaturalist.org/taxa/<id>
```

Contrast with a taxon that only the manual stage can settle:

```
shipped manifest       scientific_name = <arachnid with no species-level English name>
                       common          = "<non-English or genus string>"

STAGE 1                non-ASCII / wordlist 0.0 → NO → taxa_to_resolve.txt
STAGE 2                iNat ∅ · GBIF ∅ · Wikipedia ∅ → (none) → manual
STAGE 3                curator assigns the group name, cited → "harvestman"
FINALIZE               present in exactly one tier (manual) → "harvestman"
                       ⇒ common_names_resolved.csv   (group/genus fallback, no foreign string retained)
```

---

## The `is_english` Decision Procedure

The exact verdict cascade in `english_verdict()`, in order:

```
1. GBIF says English (en/eng tag present)            → yes      [gbif]
2. name equals the genus of the scientific name      → yes      [genus]
3. name contains non-ASCII letters                   → no       [non_ascii]
4. wordlist fraction ≥ 0.5                            → yes      [wordlist]
   wordlist fraction is 0.0 AND GBIF says foreign     → no       [gbif]
   wordlist fraction is 0.0                           → no       [wordlist]
   wordlist fraction strictly in (0, 0.5)             → unknown  [wordlist]
5. GBIF says foreign (and wordlist gave no verdict)   → no       [gbif]
6. otherwise                                          → unknown  [none]
```

The ordering is not arbitrary. **English evidence is checked before foreign evidence**, and
**non-ASCII rejection is checked before the permissive wordlist**, so the two error types trade off in
the direction that protects the gold standard. A verdict of `unknown` is *conservative*: such taxa are
handed to Stage 2 rather than guessed, and any `unknown` that Stage 2 cannot settle is adjudicated by
the Stage-3 manual check &mdash; so no `unknown` reaches the frozen key.

---

## Taxa With No Formal English Name: Genus Fallback

**Not every taxon has a formal, species-level English common name.** This is a fact about the English
language and about biology, not a defect in the pipeline, and it is represented honestly rather than
papered over.

Many arthropods &mdash; and **Arachnids most acutely** &mdash; are known in English only at a *group*
level. A given species of harvestman (Opiliones) or jumping spider (Salticidae) frequently has no
distinct English vernacular; the only natural English answer to *"what is the common name of this
organism?"* is the group name **harvestman** or **jumping spider**. This is why Arachnids is the group
where the *absence-of-English-name* problem concentrates &mdash; a **different** phenomenon from the
*foreign-name* problem that concentrates in the European-checklist amphibian, gymnosperm, and insect
records.

The policy, encoded across the pipeline, is **genus / group fallback as the floor**:

- When a taxon has a species-level English name, that name is used.
- When it has no species-level English name but a well-established **group-level English name applies**
  (*harvestman*, *jumping spider*, *orb-weaver*, &hellip;), that group name is accepted as the
  canonical answer. Stage 1 admits the bare-genus case automatically (verdict `genus`); Stage 3 admits
  the broader group-name case by manual entry with a cited source. The question is thereby answerable
  at the granularity at which English actually operates.

The `N/A` exclusion path exists as a formal floor beneath even the group name &mdash; for a taxon with
no English name at *any* level &mdash; but in this corpus it was never needed: every residual taxon
resolved to a species-level or group-level English name.

> **Why this is correct.** A group-level answer is not a degradation of the task; it is the task stated
> at the level English supports. Forcing a species-level English label that does not exist would either
> invent data or re-introduce a foreign string &mdash; both worse than answering *"jumping spider"*.

---

## Report Files: What Each Stores &amp; How to Regenerate

All artifacts live in:

```
3.Hierarchical_taxonomic_reasoning/Tree-of-Life-200M/Resolve_common_names/Reports/
```

| File | Produced by | Contents | Schema |
| :--- | :--- | :--- | :--- |
| `resolve_common_name_english_ascii.csv` | Stage 1 | Taxa resolved offline, with the observed name forms and the evidence code that decided each. | `taxa, common_name, resolved_common_name, evidence` |
| `taxa_to_resolve.txt` | Stage 1 | Machine hand-off: one scientific name per line, the taxa Stage 1 could not resolve. | one name / line |
| `taxa_to_resolve_details.txt` | Stage 1 | Human-readable audit of the above, grouped by source file, listing every candidate form with its verdict and row count. | free text |
| `resolve_common_name_inat_wiki.csv` | Stage 2 | API resolutions, keeping the per-authority candidates alongside the chosen name and its source URL(s). | `taxa, common_name, inat_common_name, wiki_common_name, resolved_common_name, source` |
| `.api_cache.jsonl` | Stage 2 | Per-taxon cache of raw API results (schema-versioned), so re-runs never re-query resolved taxa. | JSONL, one record / taxon |
| `resolve_common_name_manual.csv` | Stage 3 *(human)* | Hand-curated names for taxa the network could not resolve; `N/A` marks explicit exclusions. | `taxa, resolved_common_name, source` |
| **`common_names_resolved.csv`** | Finalize | **The frozen answer key.** One canonical English name per resolvable taxon, with provenance. | `taxa, resolved_common_name, source` |
| `excluded_no_english_name.csv` | Finalize | Every taxon dropped from the task, with the reason. *(written only when non-empty; empty this run)* | `taxa, reason` |
| `name_conflicts.csv` | Finalize | Every taxon where the tiers disagreed, showing all candidates and the winning tier. *(written only when non-empty; empty this run)* | `taxa, conflict_kind, winning_tier, tier, name, source` |

> **BOM note.** All CSVs are written with a UTF-8 BOM (`utf-8-sig`). Downstream consumers **must** open
> them with a BOM-aware reader, or the first column header parses as `\ufefftaxa` instead of `taxa`.

### Regenerating from scratch

```bash
cd 3.Hierarchical_taxonomic_reasoning/Tree-of-Life-200M/Resolve_common_names/

# Stages 1 + 2  (downloads & caches the pinned GBIF backbone on first run)
python Resolve_common_names.py <BASE_DIR> \
    --gbif 2023-08-28 \
    --out-dir Reports/ \
    --contact you@example.org

# Stage 3  — edit Reports/resolve_common_name_manual.csv by hand for the residual taxa
#            (fill resolved_common_name + source, or N/A to exclude)

# Finalize  — merge the three tiers into the frozen key
python Resolve_common_names_Finalize.py --dir Reports/
```

Useful switches: `--skip-api` runs Stage 1 only (no network); `--refresh` ignores `.api_cache.jsonl`
and re-fetches; `--no-wordlist` / `--no-compound` ablate the dictionary heuristics;
`--show-conflicts N` controls how many conflicts print to the console (the CSV always holds them all).

---

## Scope &amp; Measured Impact

> Counts are emitted by the pipeline's own summary logs and inserted here verbatim from the frozen run;
> they are placeholders in this template.

| Quantity | Value |
| :--- | :--- |
| Distinct taxa with a resolved English name (`common_names_resolved.csv`) | `{N_resolved_taxa}` |
| &nbsp;&nbsp;&mdash; resolved in Stage 1 (offline) | `{N_stage1}` |
| &nbsp;&nbsp;&mdash; resolved in Stage 2 (API) | `{N_stage2}` |
| &nbsp;&nbsp;&mdash; resolved in Stage 3 (manual) | `{N_manual}` |
| Taxa excluded (no English name at any level) | `0` |
| Tier conflicts at merge | `0` |
| Taxa whose shipped `common` field required correction | `{N_taxa_fixed}` |
| Images whose gold common-name answer changed | `{N_images_fixed}` (`{PCT_images_fixed}%` of the pool) |
| Common-name MCQ option sets containing &ge;1 non-English string | `{N_mcq_contaminated}` (`{PCT_mcq_contaminated}%`) |

**The language-cue leakage is measurable, not hypothetical.** For *Juniperus communis*, whose correct
option was the two-letter token `en` in an otherwise-English list, **{N_juniper_pick} of {N_models}**
evaluated models selected it &mdash; a rate far above chance for a two-letter answer, explicable only
as models exploiting the odd-token-out cue rather than recognizing juniper foliage. This is precisely
the failure mode [described above](#why-multilingual-gold-labels-break-vqa), caught in the data.

**Contamination is uneven across the ten groups**, tracking the linguistic provenance of each group's
occurrence records: **Amphibians, Gymnosperms, and Insects** carry the largest share of Nordic
gold-label contamination, while **Arachnids** is dominated instead by the *absence-of-English-name*
case handled by genus fallback.

---

## Applying the Fix to the Question Set

The frozen key drives a separate, **non-destructive** rewrite of the question set. For each
common-name question of an affected taxon, the **gold answer** is replaced with the resolved English
name; for multiple-choice items, **every contaminated option** (including foreign distractors borrowed
from other taxa) is replaced and the option list rebuilt. Model prediction columns are updated with the
same old&rarr;new substitution, so a prediction that was correct under the old label stays correct and a
wrong one stays wrong. **Short foreign tokens** (`en`, `gran`, `tall`) are substituted only when they
constitute the *entire* answer, never as substrings, so legitimate English text (`green`, `basking`) is
never corrupted. The corrected question set is written to new CSV files; the original question CSVs are
preserved untouched, so the correction is fully reversible and auditable.

The applier's mechanics, safety guards, and reviewed skip-list are documented alongside that script and
are out of scope for this card.

---

## Reproducibility

| Component | Deterministic? | Mechanism |
| :--- | :---: | :--- |
| Stage 1 language verdicts | ✅ | Pinned dated GBIF backbone + specifiable wordlist. |
| Stage 2 candidates | ✅ *(after first fetch)* | `.api_cache.jsonl` replays resolved taxa without re-querying; live network only for genuinely new taxa. |
| Stage 3 manual sheet | ✅ | Checked-in artifact with per-row cited sources. |
| Finalize merge | ✅ | Given the three input CSVs, the key, exclusions, and conflicts are fully determined. |

The only non-deterministic element is **live API content at first fetch**. Once `.api_cache.jsonl` is
populated and committed, the pipeline is fully reproducible offline. Pin the GBIF version and archive
the cache, and any third party regenerates `common_names_resolved.csv` exactly.

---

## Limitations &amp; Threats to Validity

- **Genus/group-level answers reduce granularity for a minority of items**, predominantly in Arachnids.
  These test recognition to the group English speakers actually name, not to species &mdash; a
  deliberate, disclosed trade-off.
- **The wordlist heuristic can misjudge rare or highly compound English names.** It is tertiary
  evidence only, used after GBIF codes and before any name reaches the key; every `unknown`-verdict
  taxon is carried forward to Stage 2 and, if needed, the manual check rather than being trusted on the
  wordlist alone.
- **Stage-2 free-text extraction (Wikipedia) is inherently noisier than structured sources.** It is
  the lowest-priority automated authority, consulted only after iNaturalist and GBIF, and every
  candidate must pass the conservative `clean_candidate` gate before it is accepted &mdash; doubtful
  cases are deferred to manual review. This trades recall for precision by design, which is the correct
  trade for an answer key, but it does mean Wikipedia contributes proportionally few final names.
- **The GBIF backbone is a fixed snapshot.** Revisions after the pinned date are not reflected &mdash;
  the intended cost of reproducibility for a frozen benchmark.

None of these residuals reintroduce a non-English gold label or a language-cued option set into the
frozen key.

---

## FAQ

**Why not just take the English row from GBIF directly and skip the wordlist entirely?**
Because a large share of backbone `vernacularName` rows carry a **blank** language tag. Trusting only
explicitly `en`-tagged rows would drop thousands of obviously-English names that were never tagged,
inflating the Stage-2 workload enormously. The wordlist recovers those cheaply and offline; GBIF tags
still win when present.

**Why is `en` (juniper) so much more dangerous than a long Swedish name?**
A long non-ASCII string like `grönfläckig padda` is trivially caught by the non-ASCII rule and is an
obvious odd-one-out to a human. A short ASCII token like `en`, `gran`, or `tall` passes the eye, is
pure ASCII, and &mdash; as a distractor &mdash; can look like a plausible truncated English word. It is
both the hardest to detect automatically (hence the wordlist-fraction rule) and the most effective
illegitimate cue in an MCQ (hence the measured juniper pick-rate).

**Were there any tier conflicts or exclusions?**
No. The three tiers are disjoint by construction &mdash; each taxon is resolved by exactly one stage and
leaves the pipeline &mdash; so there is nothing for tiers to disagree about, and every residual taxon
received an English name (species-level, or group/genus-level in the worst case). The conflict and
exclusion logs were generated as a check and both came back empty; their absence on disk is the proof.

**Does resolving to a group name ("jumping spider") make the item easier?**
Yes, marginally, and that is the point: the item is now answerable at the level English actually names
the organism. The alternative &mdash; a fabricated species-level English name, or an untranslated
foreign string &mdash; is strictly worse for validity. The trade-off is disclosed in Limitations.

**What if GBIF publishes a newer backbone after we freeze?**
Nothing changes for this release: the version is pinned and recorded. A future benchmark revision may
pin a newer dated snapshot and regenerate; the cache and version string make the delta auditable.

---

## Data Sources &amp; Acknowledgements

Vernacular names and language codes derive from the **GBIF Backbone Taxonomy** (pinned, versioned
release), served from `https://hosted-datasets.gbif.org/datasets/backbone`. Species-level common names
additionally draw on **iNaturalist** and **English Wikipedia**, with the specific source URL recorded
per row in the answer key's `source` column. Please attribute these sources when redistributing the
resolved names.

---

<p align="center">
  <sub>Part of the <b>Bio-VQA</b> benchmark &middot; Imageomics Institute</sub>
</p>
