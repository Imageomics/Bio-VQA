<div align="center">

# Common-Name Resolution for BioVQA

### Detecting, quantifying, and repairing multilingual vernacular-name contamination in the *Hierarchical Taxonomic Reasoning* question set

<a href="#"><img src="https://img.shields.io/badge/component-Hierarchical%20Taxonomic%20Reasoning-2b6cb0"></a>
<a href="#"><img src="https://img.shields.io/badge/source-TreeOfLife--200M%20%2F%20GBIF-38a169"></a>
<a href="#"><img src="https://img.shields.io/badge/backbone-GBIF-805ad5"></a>
<a href="#"><img src="https://img.shields.io/badge/status-answer%20key%20frozen-dd6b20"></a>

</div>

---

> **TL;DR.** The vernacular ("common") names shipped with the source imagery are drawn from the GBIF Backbone Taxonomy, whose `vernacularName` field is *multilingual by design*. A non-trivial fraction of C1's common-name questions therefore carried a **non-English gold answer** (e.g. `grönfläckig padda` for *Bufotes viridis*) or a **non-English distractor** in the multiple-choice option set. Left uncorrected, these items are either unanswerable as written or answerable through a spurious *language* cue rather than *visual* recognition. This document specifies (i) why the contamination exists, (ii) why it is a validity threat specific to vision–language evaluation, (iii) the three-stage resolution pipeline that produces the frozen answer key `common_names_resolved.csv`, and (iv) the residual cases we accept and disclose. Every design decision is stated together with the failure mode it is meant to prevent, so that the procedure can be audited rather than trusted.

---

## Table of contents

1. [Motivation: where the names come from](#1-motivation-where-the-names-come-from)
2. [Why multilingual gold labels are invalid for VQA](#2-why-multilingual-gold-labels-are-invalid-for-vqa)
3. [Illustrative contamination](#3-illustrative-contamination)
4. [Scope and measured impact](#4-scope-and-measured-impact)
5. [Resolution pipeline overview](#5-resolution-pipeline-overview)
6. [Stage 1 — offline language triage](#6-stage-1--offline-language-triage-gbif-codes--english-wordlist)
7. [Stage 2 — API resolution](#7-stage-2--api-resolution-gbif--inaturalist--wikipedia)
8. [Stage 3 — manual curation](#8-stage-3--manual-curation-last-resort)
9. [Finalization and tier precedence](#9-finalization-and-tier-precedence)
10. [The `is_english` decision procedure](#10-the-is_english-decision-procedure)
11. [Taxa with no formal English name: the genus-fallback policy](#11-taxa-with-no-formal-english-name-the-genus-fallback-policy)
12. [Reproducibility](#12-reproducibility)
13. [Artifacts and schemas](#13-artifacts-and-schemas)
14. [Limitations and threats to validity](#14-limitations-and-threats-to-validity)
15. [Applying the fix to the C1 question set](#15-applying-the-fix-to-the-c1-question-set)
16. [Citation](#16-citation)

---

## 1. Motivation: where the names come from

*Hierarchical Taxonomic Reasoning* (C1) is built on imagery and metadata from **TreeOfLife-200M**, whose occurrence records and taxonomy are anchored to the **GBIF Backbone Taxonomy**. Each image record carries a `scientific_name` and a `common` field. The scientific name is a controlled binomial and is not at issue here. The `common` field, however, is populated from GBIF's `VernacularName` extension.

**The `VernacularName` extension is a union of vernacular labels contributed by GBIF's constituent checklists, in every language those checklists record.** Each row pairs a `vernacularName` string with an ISO `language` tag, but (a) many rows leave the language tag blank, and (b) for a given taxon the *first* or *most frequent* vernacular string is frequently not the English one — it may be Swedish, Danish, Norwegian, German, or another language, simply because a Nordic or Central-European checklist happened to supply the most records for that occurrence.

Consequently, when C1's question generator asked *"What is the common name of the organism in this image?"* and lifted the answer from the `common` field, it sometimes wrote down a gold answer such as:

| `scientific_name` | `common` (as shipped) | language | correct English name |
|---|---|---|---|
| *Bufotes viridis* | `grönfläckig padda` | Swedish | European green toad |
| *Rana arvalis* | `åkergroda` | Swedish | Moor frog |
| *Rana arvalis* | `Spidssnudet frø` | Danish | Moor frog |
| *Lissotriton vulgaris* | `mindre vattensalamander` | Swedish | Smooth newt |
| *Triturus cristatus* | `Stor vandsalamander` | Danish | Crested newt |
| *Rana dalmatina* | `långbensgroda` | Swedish | Agile frog |
| *Juniperus communis* | `en` | Swedish/Danish | Common juniper |

The last row is the sharpest illustration: `en` is the Scandinavian common name for juniper, but as an English gold label it is not merely wrong, it is *unanswerable* — no vision–language model can be expected to emit the two-letter token `en` as the "common name" of a conifer, and no human annotator would either.

> **Design note.** This is not a data-entry error in TreeOfLife-200M or GBIF; both are behaving as specified. The defect is entirely on the consumer side: our C1 generator treated a multilingual field as if it were an English field. The fix therefore belongs in C1's answer key, not upstream.

---

## 2. Why multilingual gold labels are invalid for VQA

A benchmark answer key is a measurement instrument. For the common-name task, a non-English gold label degrades the instrument in three distinct ways, each of which we call out because they require different remedies.

**(a) Unanswerable items inflate the error rate with noise, not signal.** When the gold answer is `en` or `åkergroda`, a model that correctly recognizes the organism and emits the correct *English* name ("Common juniper", "Moor frog") is scored **wrong**. The item now measures whether the model reproduces a specific foreign string, which is unrelated to biological visual understanding. Aggregated over thousands of items, this depresses the common-name accuracy of *every* model roughly uniformly, but not exactly uniformly — models with more European-language training data are spuriously advantaged. The metric stops being a clean measure of recognition.

**(b) Language-cued multiple choice leaks the answer.** In the four-option common-name MCQ, the distractors are sampled from other taxa's `common` fields. When the gold option is an English string and one or more distractors are English while exactly one option is a conspicuous non-English string (or vice versa), the *orthographic register* of the options correlates with correctness. A model can then select the correct option by detecting "the option that looks like the others / the odd one out" without recognizing the organism at all. This converts a visual-recognition item into a partially textual one. Empirically we observed exactly this leakage: for *Juniperus communis*, whose correct option was the token `en` embedded in an otherwise-English list, **{N_juniper_pick} of {N_models} evaluated models selected it** — a selection rate far above chance for a two-letter answer, which is only explicable as models exploiting the odd-token-out cue rather than identifying juniper foliage.

**(c) It contaminates the dataset's construct validity.** C1's scientific claim rests on *generality* — that a model's competence transfers across abstraction levels. If a slice of the common-name axis is silently measuring language identification, the cross-component correlation that the benchmark reports as its central result is confounded. Removing the confound is a prerequisite for the correlation to mean what we claim it means.

> **Principle.** A vision–language item must be answerable *from the image*, and must not be answerable *without* it. Multilingual gold labels violate the first property; language-cued option sets violate the second. Both must be repaired before the common-name axis can be scored.

---

## 3. Illustrative contamination

The contaminating strings are overwhelmingly North-Germanic (Swedish, Danish, Norwegian) with a long tail of other European languages, reflecting the checklist composition of the underlying occurrence records. Representative gold-answer repairs — *(prev → resolved)* — include:

```
Bufotes viridis        grönfläckig padda      → European green toad
Rana arvalis           åkergroda              → Moor frog
Rana arvalis           Spidssnudet frø        → Moor frog
Rana temporaria        Butsnudet frø          → Common frog
Lissotriton vulgaris   Lille vandsalamander   → Smooth newt
Lissotriton vulgaris   mindre vattensalamander→ Smooth newt
Triturus cristatus     Stor vandsalamander    → Crested newt
Rana dalmatina         långbensgroda          → Agile frog
Juniperus communis     en                     → Common juniper
```

Two structural observations that matter for the fix:

- **A single taxon can carry several different foreign spellings** across the corpus (*Lissotriton vulgaris* appears with both a Swedish and a Danish name above). The resolver therefore keys on the *scientific name*, not on any one `common` string, so that every occurrence variant maps to one canonical English answer.
- **A foreign string can appear as a distractor on an image whose own gold answer is already correct.** Repairing only the affected images would leave those foreign strings sitting in otherwise-clean option lists. The fix therefore also rewrites contaminated *options* wherever they occur, not only the gold answers.

---

## 4. Scope and measured impact

> Counts below are emitted by the pipeline's own summary logs and are inserted here verbatim from the frozen run; they are placeholders in this template.

| Quantity | Value |
|---|---|
| Distinct taxa in C1 with a resolved English name (`common_names_resolved.csv`) | `{N_resolved_taxa}` |
| Taxa whose shipped `common` field required correction | `{N_taxa_fixed}` |
| C1 images whose gold common-name answer changed | `{N_images_fixed}` (`{PCT_images_fixed}%` of the C1 image pool) |
| Common-name MCQ option sets containing ≥1 non-English string | `{N_mcq_contaminated}` (`{PCT_mcq_contaminated}%` of common-name MCQ items) |
| Groups affected | all ten, but concentration is uneven — see below |

**The contamination is not uniform across the ten organismal groups.** It tracks the geographic and linguistic provenance of each group's occurrence records. Amphibians and other European-checklist-heavy groups carry the largest share of Nordic gold-label contamination. Separately — and for a *different* reason discussed in §11 — **Arachnids** is the group most affected by the *absence* of a formal species-level English name, which the genus-fallback policy addresses. These two phenomena (foreign name present vs. no English name available) are distinct and are handled by distinct parts of the pipeline.

---

## 5. Resolution pipeline overview

The objective is a single frozen answer key, `common_names_resolved.csv`, mapping every in-scope taxon to one canonical English common name, with provenance. Resolution proceeds in three stages of decreasing automation and increasing cost, followed by a deterministic merge.

```
                 ┌──────────────────────────────────────────────────────────┐
   manifests ───▶│  STAGE 1  offline triage                                  │
 (scientific_name│  GBIF backbone language codes  +  English wordlist        │
  + common)      │  → resolve_common_name_english_ascii.csv   (resolved)     │
                 │  → taxa_to_resolve.txt                      (handoff)      │
                 └───────────────┬──────────────────────────────────────────┘
                                 │ unresolved taxa only
                                 ▼
                 ┌──────────────────────────────────────────────────────────┐
   network   ───▶│  STAGE 2  API resolution                                  │
                 │  GBIF vernacular  +  iNaturalist  +  Wikipedia lead        │
                 │  → resolve_common_name_inat_wiki.csv                       │
                 └───────────────┬──────────────────────────────────────────┘
                                 │ still-unresolved taxa  (~{N_manual})
                                 ▼
                 ┌──────────────────────────────────────────────────────────┐
   human     ───▶│  STAGE 3  manual curation                                 │
                 │  hand-entered with a cited source                         │
                 │  → resolve_common_name_manual_check.csv                    │
                 └───────────────┬──────────────────────────────────────────┘
                                 ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  FINALIZE  deterministic merge, precedence manual >        │
                 │  inat_wiki > ascii; conflicts + exclusions logged         │
                 │  → common_names_resolved.csv          (FROZEN ANSWER KEY)  │
                 │  → excluded_no_english_name.csv                            │
                 │  → name_conflicts.csv                                      │
                 └──────────────────────────────────────────────────────────┘
```

Stages 1–2 are implemented in `resolve_common_names_pipeline.py`; the merge is `finalize_resolve_common_names.py`. The staged design exists so that the **cheapest sufficient evidence resolves each taxon**: an authoritative GBIF language tag settles a name for free, an API call is spent only when offline evidence is inconclusive, and a human is spent only when the network cannot answer.

---

## 6. Stage 1 — offline language triage (GBIF codes + English wordlist)

Stage 1 walks every manifest CSV, aggregates the distinct `common` strings observed per `scientific_name` (weighted by row frequency), and assigns each a verdict of **English / not-English / unknown** using two evidence sources, in order:

1. **GBIF backbone language codes.** The GBIF `VernacularName.tsv` is loaded once into a map from normalized name → set of language subtags. If the name is tagged `en`/`eng`, it is accepted as English on authoritative evidence. If it is tagged with a foreign language *and no English tag*, it is evidence *against* English.
2. **English wordlist heuristic.** For names GBIF does not tag, the fraction of tokens present in an English dictionary (`web2`, or a system wordlist) is computed. ≥ 0.5 ⇒ English; 0.0 ⇒ not English; strictly between ⇒ unknown.

Three subtleties are deliberately encoded because each corresponds to a real failure we observed:

- **A blank GBIF language tag means *unknown*, not *foreign*.** A large share of backbone rows leave `language` empty. Treating blank as non-English would bury obvious English names (e.g. "Three-striped arrow-poison frog") that simply were not tagged. Blank tags therefore never veto the wordlist.
- **Non-ASCII letters outrank the wordlist.** If a name contains non-ASCII letters (`grönfläckig`, `frø`), it is rejected as non-English *before* the wordlist is consulted. This enforces **precision over recall for the answer key**: a false *positive* (a foreign name wrongly accepted as English) would poison the gold standard, whereas a false *negative* merely costs one Stage-2 lookup. We accept the cheaper error.
- **Eponyms are language-neutral and are never altered.** "Peters' dwarf frog" and "Nordmann's vargspindel" both contain a possessive proper noun. The eponym token is skipped *only when counting dictionary words*, so the verdict rests on the remainder: "Peters' dwarf frog" passes on "dwarf frog", while "Nordmann's vargspindel" correctly fails on "vargspindel" rather than being auto-passed by the apostrophe. **The stored name is always the full, unmodified string.**

Stage 1 emits the taxa it resolves to `resolve_common_name_english_ascii.csv` (with the `evidence` code that decided each), and the taxa it cannot resolve to `taxa_to_resolve.txt` for Stage 2, plus a human-readable `taxa_to_resolve_details.txt` grouping the unresolved taxa by source file with every candidate name and its verdict.

---

## 7. Stage 2 — API resolution (GBIF · iNaturalist · Wikipedia)

Taxa unresolved by Stage 1 are queried against three authorities, and the first clean English candidate is taken, in this priority: **iNaturalist preferred common name → GBIF English vernacular → Wikipedia lead sentence**.

- **iNaturalist** `/v1/taxa` is queried with an English locale; the `preferred_common_name` of an exact, active, species-rank match is used.
- **GBIF** `/species/match` resolves the name to a usage key (following synonyms to the accepted name when necessary), then `/vernacularNames` is filtered to English entries.
- **Wikipedia** is a last automated resort: the article is fetched and the *lead sentence* is parsed with a small set of anchored patterns ("*X*, commonly known as the **Y**, is …"; "The **Y** (*X*) …") to extract the common name. Only the lead is used, and the scientific name is masked before matching, to avoid harvesting arbitrary body text.

Every candidate passes a `clean_candidate` gate that rejects strings that are too long, contain rank words ("species", "genus", "family"), end in taxonomic qualifiers ("group", "complex", "sensu lato"), contain the scientific name itself, or contain characters outside a conservative Latin set. This gate exists because free-text sources readily return non-names (taxonomic prose, URLs, higher-rank labels); it errs toward rejecting a borderline candidate and deferring the taxon to manual review.

Stage 2 is **cached** to a JSONL file keyed by taxon, with a schema version. Re-runs consult the cache and only fetch genuinely new taxa; a schema bump invalidates stale records automatically. API traffic is rate-limited and carries a contact User-Agent, per each provider's usage policy.

---

## 8. Stage 3 — manual curation (last resort)

The taxa still unresolved after Stage 2 — on the order of **~{N_manual}** — are resolved by hand into `resolve_common_name_manual_check.csv` (`taxa, resolved_common_name, source`), each with a **cited source**. Two conventions make a partially-completed sheet safe:

- A row whose `resolved_common_name` is literally `N/A` is an **explicit exclusion**: the taxon is dropped and lower tiers are *not* consulted. This is the correct entry when a curator has checked and confirmed that **no English common name exists**.
- A row left **blank** falls through to the lower automated tiers, and the count of such blanks is reported at finalize time — so a half-finished sheet is visible rather than silently treated as complete.

The manual stage is where the genus-fallback names of §11 are entered when a curator judges that the appropriate English answer is a group-level name (e.g. "harvestman", "jumping spider") rather than a species-level binomial vernacular.

---

## 9. Finalization and tier precedence

`finalize_resolve_common_names.py` merges the three CSVs into the frozen key under a strict precedence, highest first:

1. **manual** — hand-curated, cited.
2. **inat_wiki** — Stage-2 API resolution.
3. **ascii** — Stage-1 offline triage.

For each taxon the highest-precedence non-blank tier supplies the answer; an explicit `manual N/A` drops the taxon outright. The merge additionally emits two audit artifacts:

- **`excluded_no_english_name.csv`** — every dropped taxon and the reason (explicit `N/A`, or no name from any tier). These taxa are removed from the common-name task entirely.
- **`name_conflicts.csv`** — every taxon where two tiers supply *materially different* names, or where one tier says `N/A` while another supplies a name. Comparison is case-, spacing-, and punctuation-insensitive, so "Four-spot Orbweaver" vs. "four-spot orb-weaver" is **not** flagged; only genuine disagreements are. Higher precedence silently wins at merge time, so the conflict log exists precisely to make those silent overrides *visible and auditable*.

Both audit files are written **only when non-empty**, and a stale copy from a previous run is deleted when it no longer applies — an empty report left on disk reads as a current finding and is worse than no report.

---

## 10. The `is_english` decision procedure

For completeness and reviewer scrutiny, the exact verdict cascade in `english_verdict()` is, in order:

```
1. GBIF says English (en/eng tag present)           → yes   [gbif]
2. name equals the genus of the scientific name     → yes   [genus]
3. name contains non-ASCII letters                  → no    [non_ascii]
4. wordlist fraction ≥ 0.5                           → yes   [wordlist]
   wordlist fraction is 0.0 and GBIF says foreign    → no    [gbif]
   wordlist fraction is 0.0                          → no    [wordlist]
   wordlist fraction strictly in (0, 0.5)            → unknown [wordlist]
5. GBIF says foreign (and wordlist gave no verdict)  → no    [gbif]
6. otherwise                                         → unknown [none]
```

The ordering is not arbitrary. **English evidence is checked before foreign evidence**, and **non-ASCII rejection is checked before the permissive wordlist**, so that the two error types are traded off in the direction that protects the gold standard (see §6). A verdict of `unknown` is *conservative*: such taxa are handed to Stage 2 rather than guessed, and any `unknown` that nonetheless reaches the key on wordlist evidence is flagged in the run summary for spot-checking.

---

## 11. Taxa with no formal English name: the genus-fallback policy

**Not every taxon has a formal, species-level English common name.** This is a fact about the English language and about biology, not a defect in the pipeline, and it must be represented honestly in the answer key rather than papered over.

Many arthropods — and **Arachnids most acutely** — are known in English only at a *group* level. A given species of harvestman (Opiliones) or jumping spider (Salticidae) frequently has no distinct English vernacular; the only natural English answer to "what is the common name of this organism?" is the group name **harvestman** or **jumping spider**. This is why Arachnids is the group where the *absence-of-English-name* problem concentrates, in contrast to the *foreign-name* problem that concentrates in the European-checklist amphibian and insect records.

Our policy, encoded across the pipeline, is **genus / group fallback with explicit exclusion as the floor**:

- When a taxon has no species-level English name but a well-established **group-level English name applies** (harvestman, jumping spider, orb-weaver, and so on), that group name is accepted as the canonical answer. Stage 1 admits the bare-genus case automatically (verdict `genus`); Stage 3 admits the broader group-name case by manual entry with a cited source. The common-name question for such a taxon is thereby answerable at the granularity at which English actually operates.
- When **no English name exists at any level** that a curator is willing to cite, the taxon is marked `N/A` and **excluded** from the common-name task (recorded in `excluded_no_english_name.csv`). We do not fabricate a name, and we do not leave a foreign string in place.

> **Why this is the correct call.** A group-level answer is not a degradation of the task; it is the task stated at the level of description that English supports. Forcing a species-level English label that does not exist would either invent data or re-introduce a foreign string — both worse than answering "jumping spider". Excluding the genuinely un-nameable taxa keeps the remaining items clean and is disclosed in full.

---

## 12. Reproducibility

The answer key is reproducible from fixed inputs:

- **The GBIF backbone is pinned to a dated version** (e.g. `--gbif 2022-11-23`), not `current`. A dated backbone is downloaded once, cached, and is DOI-citable; `current` moves over time and would make the key irreproducible. This single choice is what lets a third party regenerate the language verdicts exactly.
- **The English wordlist is specifiable** (`--wordlist`), defaulting to the `web2` set, so the dictionary heuristic is not tied to an undocumented system file.
- **Stage-2 API responses are cached** with a schema version, so a re-run reproduces the same candidates without re-querying, and a schema change invalidates the cache rather than silently mixing formats.
- **The manual sheet is a checked-in artifact** with per-row cited sources.
- **The merge is deterministic**: given the three input CSVs, `common_names_resolved.csv`, the exclusion log, and the conflict log are fully determined.

The only non-deterministic element is live API content at Stage-2 fetch time; once the cache is populated and committed, the pipeline is fully reproducible offline.

---

## 13. Artifacts and schemas

| File | Producer | Schema |
|---|---|---|
| `resolve_common_name_english_ascii.csv` | Stage 1 | `taxa, common_name, resolved_common_name, evidence` |
| `taxa_to_resolve.txt` | Stage 1 | one scientific name per line (handoff) |
| `taxa_to_resolve_details.txt` | Stage 1 | human-readable, grouped by source file |
| `resolve_common_name_inat_wiki.csv` | Stage 2 | `taxa, common_name, inat_common_name, wiki_common_name, resolved_common_name, source` |
| `resolve_common_name_manual_check.csv` | Stage 3 (human) | `taxa, resolved_common_name, source` |
| **`common_names_resolved.csv`** | Finalize | **`taxa, resolved_common_name, source`** ← frozen answer key |
| `excluded_no_english_name.csv` | Finalize | `taxa, reason` |
| `name_conflicts.csv` | Finalize | `taxa, conflict_kind, winning_tier, tier, name, source` |

The frozen key `common_names_resolved.csv` is written with a UTF-8 BOM (`utf-8-sig`); downstream consumers must open it with a BOM-aware reader so the first column header is `taxa` and not `\ufefftaxa`.

---

## 14. Limitations and threats to validity

We state the residual weaknesses plainly.

- **Genus/group-level answers reduce the granularity of a minority of common-name items**, predominantly in Arachnids (§11). These items test recognition to the group English speakers actually name, not to species. This is a deliberate, disclosed trade-off, not an oversight.
- **A small number of taxa are excluded entirely** for having no English name at any level (`excluded_no_english_name.csv`). Exclusion removes items but does not bias the surviving ones.
- **The wordlist heuristic can misjudge rare or highly compound English names.** It is used only as tertiary evidence after GBIF codes, and every `unknown`-verdict resolution that reaches the key is flagged for spot-check; nonetheless a handful of borderline names rest on it.
- **Stage-2 free-text extraction (Wikipedia) is inherently noisier than structured sources.** The `clean_candidate` gate is conservative and defers doubtful cases to manual review, trading recall for precision.
- **The GBIF backbone version is a fixed snapshot.** Taxonomic or vernacular revisions after the pinned date are not reflected; this is the cost of reproducibility and is the correct trade for a frozen benchmark.
- **A very small set of near-synonymous or spelling/case-only "corrections" were reviewed and deliberately *not* applied**, to avoid perturbing a correct English answer into a different equally-correct one. These are recorded so the decision is auditable rather than hidden.

None of these residuals reintroduce a non-English gold label or a language-cued option set — the two properties §2 requires — into the frozen key.

---

## 15. Applying the fix to the C1 question set

The frozen key drives a separate, non-destructive rewrite of the C1 question files. For each common-name question of an affected taxon, the **gold answer** is replaced with the resolved English name; for multiple-choice items, **every contaminated option** (including foreign distractors borrowed from other taxa) is replaced, and the option list is rebuilt. Model prediction columns are updated with the same old→new substitution so that a prediction which was correct under the old label remains correct under the new one and a wrong one remains wrong. Short foreign tokens (e.g. `en`) are substituted only when they constitute the entire answer, never as substrings, so that legitimate English text (`green`, `basking`) is never corrupted. The corrected files are written under a new `QA_finalized` name; the original `QA_difficulty` files are preserved untouched.

The precise mechanics, safety guards, and the reviewed skip-list are documented alongside the applier and are out of scope for this data card.

---

## 16. Citation

If you use the C1 common-name answer key or this resolution procedure, please cite the BioVQA benchmark and the pinned GBIF Backbone Taxonomy release used to generate the key:

```bibtex
@misc{biovqa_c1_commonnames,
  title  = {Common-Name Resolution for BioVQA C1: Multilingual Vernacular
            Contamination Detection and Repair},
  author = {{BioVQA Team}},
  year   = {{YEAR}},
  note   = {Answer key frozen against GBIF Backbone Taxonomy version
            {GBIF_VERSION}}
}
```

<div align="center">

---

Produced under the **Imageomics Institute**.
Backbone taxonomy: **GBIF Secretariat, GBIF Backbone Taxonomy** (pinned, versioned release).

</div>
