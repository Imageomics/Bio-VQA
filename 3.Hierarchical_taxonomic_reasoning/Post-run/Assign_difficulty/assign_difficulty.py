#!/usr/bin/env python3
"""
assign_difficulty.py
====================
Assign a taxonomy-aware difficulty tier to every QA item in the Hierarchical
Taxonomic Reasoning component.

Every CSV under --base_dir whose filename contains --keystring is discovered,
scored, and written back with the difficulty columns attached.

  --mode same       add/update the columns in the source CSV, in place
  --mode separate   write <name>_tiered.csv next to the source CSV

Re-running is safe: existing difficulty columns are UPDATED in place (no
duplicate columns, no reordering).

Constructs
----------
OPEN items (phylum..genus, species epithet, scientific name, common name)
    difficulty = prevalence tier.
    exposure c(t) = # catalog images in the subtree of the target taxon t.
    log_exposure  = log10(c + 1).
    tier = item-weighted, taxon-deterministic tertile of log_exposure within
           the (group, rank_class) cell.
    rank_class collapses {species epithet, scientific name, common name} into
    "species_level".

MCQ items (4 options)
    difficulty = confusability tier from Nearest-Distractor Distance.
    d_j = depth(GT) - depth(LCA(GT, distractor_j)) = ranks climbed to meet j.
    NDD = min_j d_j.   NDD<=1 Hard, ==2 Medium, >=3 Easy.
    mcq_confusability = mean_j exp(-ln2 * d_j).

Resolution policy
-----------------
Every item resolves to a node; there is no "Unresolved" outcome. Each image's
answer key carries its own lineage, so ranks are resolved from that lineage
rather than by looking the answer string up in isolation:

  1. exact       the answer names a node filed at the asked rank
  2. vernacular  a common name, resolved through an index built from the corpus
                 answer keys; a name labelling several species resolves to their
                 LCA, so 'Orb weavers' lands on Araneidae and 'Viviparous
                 Lizard' on the single species it denotes
  3. lineage     the asked rank read off this image's own resolved lineage
                 (covers rank-name disagreements such as the CoL 2022 promotion
                 of Squamata / Testudines / Crocodylia to class)
  4. anchor      the lowest resolvable node in this image's lineage

How each item resolved is recorded in the `resolution` column so the share of
non-exact resolutions can be reported and excluded in a robustness check.

Incomplete lineages
-------------------
Only genus + species are required of a catalog row. Intermediate ranks may be
null: post-2022 CoL reptiles carry class=Squamata and order=NULL, and requiring
every rank would delete the clade outright. A family whose order is null
attaches directly to its class; depth is always read from the rank name, so LCA
distances stay correct across the gap.

There is deliberately no global epithet fallback. Matching a bare epithet
against the most photographed species that happens to share it produces
confident wrong taxa ('carolinensis' -> Sciurus carolinensis for an Anolis
image).

Degenerate cells
----------------
A cell with fewer than MIN_TAXA distinct taxa cannot support tertiles (every
phylum answer in one group is 'Chordata'). Such a cell is re-cut against that
rank_class pooled across all groups; only if that still has too few taxa is the
cell labelled 'Degenerate'.

Usage
-----
python assign_difficulty.py \
    --base_dir   /path/to/3-Hierarchical-Taxonomic-Reasoning \
    --parquet    /path/to/catalog.parquet \
    --tree_cache tree.pkl \
    --keystring  QA_finalized \
    --mode       same
"""

import os
import re
import json
import math
import pickle
import argparse
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Knobs
# --------------------------------------------------------------------------- #
RANKS = ["phylum", "class", "order", "family", "genus", "species"]
RANK_DEPTH = {r: i + 1 for i, r in enumerate(RANKS)}
REQUIRED_RANKS = ["genus", "species"]        # every other rank may be null
CLIMB = ["species", "genus", "family", "order", "class", "phylum"]  # low -> high
NULLISH = ("", "nan", "none", "null", "<na>")
ROOT = 0
MIN_TAXA = 6                 # cells with fewer distinct taxa -> pool, then Degenerate
DECAY = math.log(2)          # one rank of distance halves MCQ credit
TIER_SUFFIX = "_tiered.csv"  # --mode separate output suffix

DIFF_COLS = ["target_taxon", "target_rank", "rank_class", "resolution",
             "exposure_count", "log_exposure", "prevalence_pct",
             "prevalence_tier", "mcq_ndd", "mcq_nearest_lca_rank",
             "mcq_confusability", "mcq_tier", "difficulty"]
NUM_COLS = {"exposure_count", "log_exposure", "prevalence_pct",
            "mcq_ndd", "mcq_confusability"}

# word-boundary patterns, most specific first
QUESTION_PATTERNS = [
    ("scientific_name", r"scientific\s+name"),
    ("common_name",     r"common\s+name"),
    ("species",         r"species\s+epithet|\bspecies\b"),
    ("phylum",          r"\bphylum\b|\bdivision\b"),
    ("class",           r"\bclass\b"),
    ("order",           r"\border\b"),
    ("family",          r"\bfamily\b"),
    ("genus",           r"\bgenus\b"),
]
QUESTION_PATTERNS = [(r, re.compile(p)) for r, p in QUESTION_PATTERNS]
MCQ_RE = re.compile(r"among the four options|four options|which of the following")

RANK_CLASS = {
    "phylum": "phylum", "class": "class", "order": "order",
    "family": "family", "genus": "genus",
    "species": "species_level", "scientific_name": "species_level",
    "common_name": "species_level",
    "unknown": "species_level",
}

# --------------------------------------------------------------------------- #
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")
_PARENS = re.compile(r"\([^)]*\)")


def norm(s) -> str:
    if s is None or (isinstance(s, float) and s != s):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = _PARENS.sub(" ", s).replace("-", " ").replace("_", " ").replace("/", " ")
    return _WS.sub(" ", _PUNCT.sub(" ", s)).strip()


def blank(v) -> bool:
    """True if a catalog cell carries no usable name."""
    return str(v).strip().lower() in NULLISH


def qkey(q) -> str:
    """Whitespace-normalised question, used as the join key."""
    return _WS.sub(" ", str(q)).strip()


def detect_rank(question):
    q = str(question).lower()
    is_mcq = bool(MCQ_RE.search(q))
    for rank, pat in QUESTION_PATTERNS:
        if pat.search(q):
            return rank, is_mcq
    return "unknown", is_mcq


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
class Taxonomy:
    """Rooted Linnaean tree with global subtree image counts (exposure).

    Intermediate ranks may be absent; a node attaches to the nearest populated
    ancestor. Depth always reflects the node's own rank.
    """

    def __init__(self):
        self.name = [""]; self.rank = ["root"]; self.parent = [-1]; self.count = [0]
        self.by_rank = defaultdict(dict)      # rank -> normname -> id
        self.key2id = {}
        self.common2sp = {}                   # normname -> species id
        self.epithet2sp = defaultdict(list)   # normepithet -> [species ids]
        self.N = 0

    def _add(self, rank, disp, key, parent):
        k = (rank, key)
        if k in self.key2id:
            return self.key2id[k]
        i = len(self.name)
        self.key2id[k] = i
        self.name.append(disp); self.rank.append(rank)
        self.parent.append(parent); self.count.append(0)
        self.by_rank[rank][key] = i
        return i

    @classmethod
    def from_parquet(cls, path):
        import duckdb
        cols = ('"phylum","class","order","family","genus","species",'
                '"scientific_name","common"')
        print("[tree] aggregating catalog via duckdb ...", flush=True)
        grp = duckdb.connect().execute(
            f"SELECT {cols}, COUNT(*) AS n FROM read_parquet(?) GROUP BY ALL",
            [path]).fetch_df()

        before = len(grp)
        grp = grp.dropna(subset=REQUIRED_RANKS)
        print(f"[tree] {len(grp):,} tuples kept "
              f"({before - len(grp):,} dropped: no genus/species)")

        gaps = {r: int(grp[r].isna().sum())
                for r in RANKS if r not in REQUIRED_RANKS}
        gaps = {r: c for r, c in gaps.items() if c}
        if gaps:
            print("[tree] tuples with a missing intermediate rank: "
                  + ", ".join(f"{r}={c:,}" for r, c in gaps.items()))

        t = cls()
        keys = RANKS + ["scientific_name", "common", "n"]
        streams = [grp[c].values for c in keys]
        for rec in zip(*streams):
            row = dict(zip(keys, rec))
            n = int(row["n"]); t.N += n

            parent = ROOT
            for rank in RANKS[:-1]:
                disp = str(row[rank]).strip()
                if blank(disp):
                    continue                  # gap: keep descending
                parent = t._add(rank, disp, norm(disp), parent)
                t.count[parent] += n

            epi = str(row["species"]).strip()
            sci = str(row.get("scientific_name") or "").strip()
            if blank(sci):
                sci = f"{str(row['genus']).strip()} {epi}"
            sp = t._add("species", sci, norm(sci), parent)
            t.count[sp] += n
            t.epithet2sp[norm(epi)].append(sp)

            com = str(row.get("common") or "").strip()
            if not blank(com):
                cn = norm(com)
                if cn not in t.common2sp or t.count[t.common2sp[cn]] < t.count[sp]:
                    t.common2sp[cn] = sp

        t.count[ROOT] = t.N
        t.epithet2sp = {k: sorted(set(v)) for k, v in t.epithet2sp.items()}
        t.count = np.asarray(t.count, dtype=np.float64)
        print(f"[tree] nodes={len(t.name):,}  images={t.N:,}  "
              f"catalog vernaculars={len(t.common2sp):,}")
        return t

    # structure --------------------------------------------------------------
    def depth(self, i):
        return 0 if i == ROOT else RANK_DEPTH[self.rank[i]]

    def lineage(self, i):
        out = []
        while i not in (ROOT, -1):
            out.append(i); i = self.parent[i]
        return out[::-1]

    def ancestor_at(self, i, rank):
        """Nearest ancestor filed at `rank`, or ROOT if that rank is absent."""
        target = RANK_DEPTH.get(rank, 0)
        while i != ROOT and self.depth(i) > target:
            i = self.parent[i]
        return i if i != ROOT and self.depth(i) == target else ROOT

    def lca(self, a, b):
        if a == b:
            return a
        anc = set(self.lineage(a)) | {ROOT}
        while b != ROOT and b not in anc:
            b = self.parent[b]
        return b if b in anc else ROOT

    def get(self, rank, name):
        return self.by_rank.get(rank, {}).get(norm(name))

    def species_from_binomial(self, s):
        n = norm(s)
        if not n:
            return None
        i = self.get("species", n)
        if i is not None:
            return i
        toks = n.split()
        return self.get("species", " ".join(toks[:2])) if len(toks) >= 2 else None


def load_tree(parquet, cache):
    """Load the cached tree, rebuilding if the parquet has changed."""
    stamp = None
    if parquet and os.path.exists(parquet):
        st = os.stat(parquet)
        stamp = (os.path.abspath(parquet), int(st.st_mtime), st.st_size)

    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            blob = pickle.load(f)
        if isinstance(blob, dict) and "tax" in blob:
            if stamp is None or blob.get("stamp") == stamp:
                print(f"[tree] loaded cache {cache}")
                return blob["tax"]
            print(f"[tree] cache {cache} is stale for {parquet}; rebuilding")
        else:
            print(f"[tree] cache {cache} predates stamping; rebuilding")

    if not parquet:
        raise SystemExit("--parquet is required when no valid --tree_cache exists")
    tax = Taxonomy.from_parquet(parquet)
    if cache:
        with open(cache, "wb") as f:
            pickle.dump({"stamp": stamp, "tax": tax}, f)
        print(f"[tree] cached -> {cache}")
    return tax


# --------------------------------------------------------------------------- #
# Per-image lineage + corpus vernacular index
# --------------------------------------------------------------------------- #
def _lca_many(tax, nodes):
    it = iter(nodes)
    a = next(it)
    for b in it:
        a = tax.lca(a, b)
        if a == ROOT:
            return ROOT
    return a


def build_image_index(items, tax):
    """Resolve every image once, from its own answer key.

    Returns
      lin    : (group, uuid) -> {rank: node_id}
      anchor : (group, uuid) -> lowest resolvable node id (or None)
      vern   : normalised common name -> node id
    """
    lin, anchor, members = {}, {}, defaultdict(set)

    for gu, g in items.groupby(["group", "uuid"], sort=False):
        ans = {}
        for r, gt, m in zip(g["_drank"], g["gt"], g["_mcq"]):
            if not m:
                ans.setdefault(r, str(gt))

        nodes = {}
        for rank in RANKS[:-1]:                       # phylum..genus, exact
            if ans.get(rank):
                i = tax.get(rank, ans[rank])
                if i is not None:
                    nodes[rank] = i

        sp = None
        if ans.get("scientific_name"):
            sp = tax.species_from_binomial(ans["scientific_name"])
        if sp is None and ans.get("genus") and ans.get("species"):
            sp = tax.get("species", f"{norm(ans['genus'])} {norm(ans['species'])}")
        if sp is None and "genus" in nodes and ans.get("species"):
            gname = norm(tax.name[nodes["genus"]])
            sp = tax.get("species", f"{gname} {norm(ans['species'])}")
        if sp is not None:
            nodes["species"] = sp
            for rank in RANKS[:-1]:                   # fill gaps from lineage
                if rank not in nodes:
                    a = tax.ancestor_at(sp, rank)
                    if a != ROOT:
                        nodes[rank] = a

        lin[gu] = nodes
        anchor[gu] = next((nodes[r] for r in CLIMB if r in nodes), None)

        cn = ans.get("common_name")
        if cn and anchor[gu] is not None:
            members[norm(cn)].add(nodes.get("species", anchor[gu]))

    vern = {k: _lca_many(tax, v) for k, v in members.items()}
    vern = {k: v for k, v in vern.items() if v != ROOT}

    n_sp = sum("species" in v for v in lin.values())
    n_no = sum(v is None for v in anchor.values())
    print(f"[lineage] {len(lin):,} images | species node {n_sp:,} "
          f"| no node at any rank {n_no:,}")
    print(f"[lineage] {len(vern):,} vernaculars indexed from answer keys")
    if n_no:
        print(f"[warn] {n_no:,} images resolve to nothing; their items stay unresolved")
    return lin, anchor, vern


def make_vernacular_lookup(tax, vern):
    """Corpus index first, catalog second, singular/plural tolerant."""
    def look(s):
        n = norm(s)
        if not n:
            return None
        cands = [n, n[:-1] if n.endswith("s") else n + "s"]
        for k in cands:
            v = vern.get(k)
            if v:
                return v
        for k in cands:
            v = tax.common2sp.get(k)
            if v:
                return v
        return None
    return look


# --------------------------------------------------------------------------- #
# Difficulty
# --------------------------------------------------------------------------- #
def parse_mcq_options(question):
    tail = str(question).rsplit("?", 1)[-1]
    return [o.strip() for o in tail.split(",") if o.strip()]


def compute_mcq(tax, question, gt_answer, look, gt_hint=None):
    gt_node = look(gt_answer) or gt_hint
    opts = parse_mcq_options(question)
    if gt_node is None:
        return dict(mcq_ndd=np.nan, mcq_nearest_lca_rank="unresolved",
                    mcq_confusability=np.nan, mcq_tier="Unresolved",
                    resolution="unresolved"), None

    gd = tax.depth(gt_node)
    ngt = norm(gt_answer)
    cand = []
    for o in opts:
        if norm(o) == ngt:
            continue
        node = look(o)
        if node is None:
            continue
        cand.append((gd - tax.depth(tax.lca(gt_node, node)), node))

    if cand:
        ndd, near = min(cand, key=lambda t: t[0])
        lca_rank = tax.rank[tax.lca(gt_node, near)]
        conf = float(np.mean([math.exp(-DECAY * d) for d, _ in cand]))
        how = "mcq"
    else:
        # No option could be placed in the tree. The most that can be said is
        # that nothing shares a clade with the answer: LCA at root.
        ndd, lca_rank = gd, "root"
        conf = math.exp(-DECAY * gd)
        how = "mcq_no_distractor"

    tier = "Hard" if ndd <= 1 else ("Medium" if ndd == 2 else "Easy")
    return dict(mcq_ndd=float(ndd), mcq_nearest_lca_rank=lca_rank,
                mcq_confusability=conf, mcq_tier=tier, resolution=how), gt_node


def resolve_open(tax, drank, gt, nodes, anch, look):
    """(node, how) for a non-MCQ item. Never returns (None, _) if anch exists."""
    if drank in ("phylum", "class", "order", "family", "genus"):
        i = tax.get(drank, gt)
        if i is not None:
            return i, "exact"
        if drank in nodes:
            return nodes[drank], "lineage"

    elif drank == "scientific_name":
        i = tax.species_from_binomial(gt)
        if i is not None:
            return i, "exact"
        if "species" in nodes:
            return nodes["species"], "lineage"

    elif drank == "common_name":
        i = look(gt)
        if i is not None:
            return i, "vernacular"
        if "species" in nodes:
            return nodes["species"], "lineage"

    elif drank == "species":
        if "species" in nodes:
            return nodes["species"], "exact"

    return (anch, "anchor") if anch is not None else (None, "unresolved")


def score_items(items, tax):
    """items: DataFrame[group, uuid, qkey, question, gt] -> difficulty columns."""
    dranks, mcqs = zip(*(detect_rank(q) for q in items["question"]))
    items = items.copy()
    items["_drank"] = dranks
    items["_mcq"] = mcqs

    n_unknown = int((items["_drank"] == "unknown").sum())
    if n_unknown:
        print(f"[warn] {n_unknown:,} items matched no rank pattern")

    lin, anchor, vern = build_image_index(items, tax)
    look = make_vernacular_lookup(tax, vern)

    recs = []
    for r in items.to_dict("records"):
        drank, is_mcq = r["_drank"], r["_mcq"]
        gu = (r["group"], r["uuid"])
        nodes, anch = lin.get(gu, {}), anchor.get(gu)

        rec = dict(target_taxon="", target_rank=drank,
                   rank_class=("mcq" if is_mcq else RANK_CLASS.get(drank, "species_level")),
                   resolution="", exposure_count=np.nan, log_exposure=np.nan,
                   prevalence_pct=np.nan, prevalence_tier="",
                   mcq_ndd=np.nan, mcq_nearest_lca_rank="",
                   mcq_confusability=np.nan, mcq_tier="")

        if is_mcq:
            mcq, node = compute_mcq(tax, r["question"], r["gt"], look,
                                    nodes.get("species", anch))
            rec.update(mcq)
            if node is not None:
                rec["target_taxon"] = tax.name[node]
        else:
            node, how = resolve_open(tax, drank, r["gt"], nodes, anch, look)
            rec["resolution"] = how
            if node is not None:
                rec["target_taxon"] = tax.name[node]
                rec["exposure_count"] = float(tax.count[node])
                rec["log_exposure"] = math.log10(rec["exposure_count"] + 1.0)
        recs.append(rec)

    out = pd.DataFrame(recs, index=items.index)
    out["group"] = items["group"].values
    _prevalence_tiers(out)
    out["difficulty"] = np.where(out["rank_class"] == "mcq",
                                 out["mcq_tier"], out["prevalence_tier"])
    return out.drop(columns=["group"])


def _tier_block(out, idx):
    """Cut item-weighted, taxon-deterministic tertiles over one index block."""
    sub = out.loc[idx]
    res = sub[sub["log_exposure"].notna()]
    if res.empty:
        return False
    taxa = (res.groupby("target_taxon")
               .agg(x=("log_exposure", "first"), n=("log_exposure", "size"))
               .reset_index()
               .sort_values(["x", "target_taxon"]))
    if len(taxa) < MIN_TAXA:
        return False
    total, cum, pct = taxa["n"].sum(), 0.0, {}
    for tx, n in zip(taxa["target_taxon"], taxa["n"]):
        pct[tx] = (cum + 0.5 * n) / total          # mid-rank -> deterministic
        cum += n
    p = res["target_taxon"].map(pct).astype(float)
    out.loc[res.index, "prevalence_pct"] = p.values
    out.loc[res.index, "prevalence_tier"] = np.where(
        p < 1 / 3, "Hard", np.where(p < 2 / 3, "Medium", "Easy"))
    out.loc[sub.index.difference(res.index), "prevalence_tier"] = "Unresolved"
    return True


def _prevalence_tiers(out):
    """Tier within (group, rank_class); escalate thin cells to a global re-cut."""
    open_idx = out.index[out["rank_class"] != "mcq"]
    if len(open_idx) == 0:
        return
    blocks = out.loc[open_idx].groupby(["group", "rank_class"], sort=False).groups

    thin = set()
    for (grp, cell), idx in blocks.items():
        if not _tier_block(out, idx):
            thin.add(cell)

    for cell in sorted(thin):
        idx = out.index[(out["rank_class"] == cell) & out.index.isin(open_idx)]
        if _tier_block(out, idx):
            print(f"[tiers] rank_class '{cell}' pooled across all groups "
                  f"({len(idx):,} items)")
        else:
            out.loc[idx, "prevalence_pct"] = np.nan
            out.loc[idx, "prevalence_tier"] = "Degenerate"
            n_taxa = out.loc[idx, "target_taxon"].nunique()
            print(f"[tiers] rank_class '{cell}' Degenerate even pooled "
                  f"({n_taxa} distinct taxa, {len(idx):,} items)")


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def find_csvs(base_dir, keystring):
    hits = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".csv") and keystring in f and not f.endswith(TIER_SUFFIX):
                hits.append(os.path.join(root, f))
    return sorted(hits)


def group_of(path, base_dir):
    """First path component under base_dir; files directly in base_dir pool
    together under '_root'."""
    rel = os.path.relpath(path, base_dir).split(os.sep)
    return rel[0] if len(rel) > 1 else "_root"


def out_path(path, mode):
    return path if mode == "same" else os.path.splitext(path)[0] + TIER_SUFFIX


def read_keys(path, base_dir):
    try:
        df = pd.read_csv(path, usecols=["uuid", "qa_pairs"])
    except ValueError as e:
        print(f"[skip] {path}: {e}")
        return None
    grp = group_of(path, base_dir)
    rows = []
    for uid, v in zip(df["uuid"], df["qa_pairs"]):
        d = json.loads(v) if isinstance(v, str) else v
        rows.append((grp, uid, qkey(d["Q"]), d["Q"], d["A"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--keystring", default="QA_difficulty")
    ap.add_argument("--mode", choices=["same", "separate"], default="separate")
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--tree_cache", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    csvs = find_csvs(args.base_dir, args.keystring)
    if not csvs:
        raise SystemExit(f"no CSVs matching '{args.keystring}' under {args.base_dir}")
    print(f"[scan] {len(csvs)} CSV(s) matching '{args.keystring}':")
    for p in csvs:
        print(f"        {os.path.relpath(p, args.base_dir)}  ->  "
              f"{os.path.basename(out_path(p, args.mode))}")
    if args.dry_run:
        return

    tax = load_tree(args.parquet, args.tree_cache)

    # pass 1 : unique item set across every matched CSV ----------------------
    rows, usable = [], []
    for p in csvs:
        r = read_keys(p, args.base_dir)
        if r is None:
            continue
        usable.append(p)
        rows.extend(r)
    items = (pd.DataFrame(rows, columns=["group", "uuid", "qkey", "question", "gt"])
               .drop_duplicates(subset=["group", "uuid", "qkey"])
               .reset_index(drop=True))
    print(f"[items] {len(items):,} unique (group, uuid, question) items")

    scored = score_items(items, tax)

    print("\n[tiers]\n" + scored["difficulty"].value_counts().to_string())
    print("\n[resolution]\n" + scored["resolution"].value_counts().to_string())
    n_bad = int((scored["difficulty"] == "Unresolved").sum())
    print(f"\n[check] unresolved items: {n_bad:,}")

    lut = {k: v for k, v in zip(
        zip(items["group"], items["uuid"], items["qkey"]),
        scored[DIFF_COLS].to_dict("records"))}

    # pass 2 : write columns onto every matched CSV --------------------------
    empty = {c: (np.nan if c in NUM_COLS else "NA") for c in DIFF_COLS}
    for p in usable:
        df = pd.read_csv(p)
        grp = group_of(p, args.base_dir)
        hits, miss = [], 0
        for uid, v in zip(df["uuid"], df["qa_pairs"]):
            d = json.loads(v) if isinstance(v, str) else v
            h = lut.get((grp, uid, qkey(d["Q"])))
            if h is None:
                miss += 1
                h = empty
            hits.append(h)
        add = pd.DataFrame(hits)
        for c in DIFF_COLS:                 # existing columns are overwritten
            df[c] = add[c].values
        dest = out_path(p, args.mode)
        df.to_csv(dest, index=False)
        print(f"[write] {os.path.relpath(dest, args.base_dir)}  "
              f"rows={len(df):,} miss={miss}")

    print("\n[done]")


if __name__ == "__main__":
    main()