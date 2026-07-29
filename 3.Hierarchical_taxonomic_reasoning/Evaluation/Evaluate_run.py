#!/usr/bin/env python3
"""
Evaluate.py
===========
Standalone. Imports nothing from assign_difficulty.py.

Scores every model's merged CSV in place and writes summary tables and reports.

Columns added to each model CSV
-------------------------------
    exact_match_correctness_hard   1/0  strict match after normalization
    exact_match_correctness_soft   1/0  hard, OR the answer is a GBIF synonym
                                        whose accepted form is the ground truth
    pred_taxon                     what the prediction resolved to
    resolution                     exact | epithet | legacy | wrong_rank |
                                   unresolved
    lca_rank                       rank at which GT and prediction meet
    lca_height                     ranks climbed from GT to the LCA; 0 = correct
    wu_palmer                      2*d(LCA) / (d(gt)+d(pred))      [Wu&Palmer 94]
    lin_ic                         2*IC(LCA) / (IC(gt)+IC(pred))   [Lin 1998]

IC(c) = -log(count(c)/N) from the global subtree counts.

Outputs (under --out_dir, default <base_dir>/Result)
----------------------------------------------------
    evaluate_stats.txt   leaderboard + pooled report + per-group reports
    result_ALL.txt       pooled report
    result_<Group>.txt   per-group reports
    csv/                 machine-readable summary tables

Requires a synonym table (synonyms.pkl) from build_synonyms.py for em_soft;
without it, em_soft == em_hard. Reuses tree.pkl if present and compatible;
otherwise rebuilds it from the parquet.

Run:
    python Evaluate.py --base_dir <dir> --parquet <catalog.parquet>
"""

import os

import re
import json
import math
import pickle
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
RANKS = ["phylum", "class", "order", "family", "genus", "species"]
RANK_DEPTH = {r: i + 1 for i, r in enumerate(RANKS)}
ROOT = 0

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def norm(s) -> str:
    if s is None:
        return ""
    if isinstance(s, float) and s != s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("-", " ").replace("_", " ")
    return _WS.sub(" ", _PUNCT.sub(" ", s)).strip()


def valid(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and v != v:
        return False
    s = str(v).strip()
    return s != "" and s.lower() not in ("nan", "none", "null")


QPATTERNS = [
    ("scientific_name", "scientific name"),
    ("common_name", "common name"),
    ("species", "species epithet"),
    ("species", "species"),
    ("phylum", "phylum"),
    ("class", "class"),
    ("order", "order"),
    ("family", "family"),
    ("genus", "genus"),
]
RANK_CLASS = {
    "phylum": "phylum", "class": "class", "order": "order",
    "family": "family", "genus": "genus",
    "species": "species_level", "scientific_name": "species_level",
    "common_name": "species_level",
}


def detect(question):
    """-> (asked_rank, is_mcq)"""
    q = question.lower()
    mcq = "options" in q
    for rank, pat in QPATTERNS:
        if pat in q:
            return rank, mcq
    return "common_name", mcq


# --------------------------------------------------------------------------- #
# Catalogue of Life promoted these former orders/subclasses to CLASS rank and
# dropped the paraphyletic parents. Models trained on textbook taxonomy answer
# with the parent. Accept it at CLASS rank only, and only for its own children.
# The old hand-written LEGACY_CLASS dict (reptiles/fish only) is gone. Soft-EM
# now credits ANY deprecated-but-correct name via the GBIF Backbone synonym
# table built by build_synonyms.py. SYNONYMS maps norm(old_name) -> norm(accepted
# _name); resolve() sends a name to its accepted form (identity if not a synonym).
# Empty until --synonyms is passed, in which case soft-EM == hard-EM (no crediting).
SYNONYMS = {}


def load_synonyms(path):
    global SYNONYMS
    with open(path, "rb") as fh:
        SYNONYMS = pickle.load(fh)["map"]
    return SYNONYMS


def resolve_syn(n):
    """norm name -> its GBIF-accepted norm name (identity if not a synonym)."""
    return SYNONYMS.get(n, n)


_LEAD = re.compile(
    r"^(the\s+)?(answer\s*[:\-]\s*|final answer\s*[:\-]\s*|"
    r"(common|scientific)\s+name\s+(is|of the \w+)?\s*[:\-]?\s*|"
    r"(phylum|class|order|family|genus|species)\s+(is|of)?\s*[:\-]?\s*)", re.I)


def clean(raw) -> str:
    """Strip the boilerplate VLMs emit despite terse-answer prompting."""
    if raw is None or (isinstance(raw, float) and raw != raw):
        return ""
    t = str(raw).strip().split("\n")[0].strip().strip("`*\"' ")
    t = re.sub(r"^\(?[A-Da-d]\)\s+", "", t)          # "(C) American toad"
    prev = None
    while prev != t:
        prev = t
        t = _LEAD.sub("", t).strip()
    return t.rstrip(".").strip()


# --------------------------------------------------------------------------- #
class Taxonomy:
    """
    Linnaean tree with global subtree image counts.  Layout is identical to the
    one assign_difficulty.py pickles, so tree.pkl is shared between the two
    scripts without either importing the other.

    Nodes are keyed by (parent, rank, normalized_name), so homonyms across
    lineages never collide.  NULL ranks are skipped -> skip-edges (reptiles
    have no order under the CoL classification).
    """

    def __init__(self):
        self.name = [""]
        self.rank = ["root"]
        self.parent = [-1]
        self.count = [0]
        self.child = {}                 # (parent, rank, key) -> id
        self.sp_by_genus_epi = {}       # (genus_id, norm_epithet) -> species_id
        self.common2sp = {}             # norm common name -> species_id
        self.N = 0

    def _add(self, rank, disp, key, parent):
        k = (parent, rank, key)
        i = self.child.get(k)
        if i is None:
            i = len(self.name)
            self.child[k] = i
            self.name.append(disp)
            self.rank.append(rank)
            self.parent.append(parent)
            self.count.append(0)
        return i

    @classmethod
    def build(cls, path):
        import duckdb
        q = ('SELECT "phylum","class","order","family","genus","species",'
             '"scientific_name","common", COUNT(*) AS n '
             f"FROM read_parquet('{path}') GROUP BY ALL")
        print("[tree] aggregating catalog ...")
        df = duckdb.connect().execute(q).fetch_df()
        print(f"[tree] {len(df):,} unique lineage tuples")

        t = cls()
        cols = ["phylum", "class", "order", "family", "genus",
                "species", "scientific_name", "common", "n"]
        for rec in zip(*[df[c].values for c in cols]):
            r = dict(zip(cols, rec))
            n = int(r["n"])
            t.N += n

            cur = ROOT
            for rank in ("phylum", "class", "order", "family", "genus"):
                v = r[rank]
                if not valid(v):
                    continue                       # skip-edge
                cur = t._add(rank, str(v).strip(), norm(v), cur)
                t.count[cur] += n
            gnode = cur if cur != ROOT and t.rank[cur] == "genus" else None

            epi = r["species"]
            if gnode is not None and valid(epi):
                sci = r["scientific_name"]
                if not valid(sci):
                    sci = f"{t.name[gnode]} {str(epi).strip()}"
                sci = str(sci).strip()
                sp = t._add("species", sci, norm(sci), gnode)
                t.count[sp] += n
                t.sp_by_genus_epi.setdefault((gnode, norm(epi)), sp)
                com = r["common"]
                if valid(com):
                    cn = norm(com)
                    prev = t.common2sp.get(cn)
                    if prev is None or t.count[prev] < t.count[sp]:
                        t.common2sp[cn] = sp

        t.count[ROOT] = t.N
        t.count = np.asarray(t.count, dtype=np.float64)
        print(f"[tree] nodes={len(t.name):,}  images={t.N:,}")
        return t

    # ---- structure --------------------------------------------------------
    def depth(self, i):
        return 0 if i == ROOT else RANK_DEPTH[self.rank[i]]

    def lineage(self, i):
        out = []
        while i not in (ROOT, -1):
            out.append(i)
            i = self.parent[i]
        return out

    def lca(self, a, b):
        if a == b:
            return a
        anc = set(self.lineage(a)) | {ROOT}
        while b not in (ROOT, -1) and b not in anc:
            b = self.parent[b]
        return b if b in anc else ROOT

    # ---- resolution -------------------------------------------------------
    def resolve_lineage(self, gt):
        """
        gt: {asked_rank: ground_truth_string} for ONE image.
        Walk down from the root using the image's own GT answers, so homonyms
        and bare epithets resolve exactly.  Returns {rank: node_id}, partial ok.
        """
        out = {}
        cur = ROOT
        for rank in ("phylum", "class", "order", "family", "genus"):
            if rank not in gt:
                continue                           # rank absent (reptile order)
            nxt = self.child.get((cur, rank, norm(gt[rank])))
            if nxt is None:
                return out
            out[rank] = nxt
            cur = nxt

        gnode = out.get("genus")
        if gnode is None:
            return out

        sp = None
        if "species" in gt:
            sp = self.sp_by_genus_epi.get((gnode, norm(gt["species"])))
        if sp is None and "scientific_name" in gt:
            sp = self.child.get((gnode, "species", norm(gt["scientific_name"])))
        if sp is None and "common_name" in gt:
            c = self.common2sp.get(norm(gt["common_name"]))
            if c is not None and self.parent[c] == gnode:
                sp = c
        if sp is not None:
            out["species"] = sp
        return out


def load_tree(parquet, tree_cache):
    """Reuse tree_cache when it is compatible; otherwise rebuild from parquet."""
    need = ("child", "sp_by_genus_epi", "common2sp", "count", "name",
            "rank", "parent", "N")
    if os.path.exists(tree_cache):
        try:
            with open(tree_cache, "rb") as fh:
                t = pickle.load(fh)
            missing = [a for a in need if not hasattr(t, a)]
            if missing:
                print(f"[tree] {tree_cache} incompatible (missing {missing}); "
                      f"rebuilding")
            else:
                print(f"[tree] loaded {tree_cache}  nodes={len(t.name):,}")
                return t
        except Exception as e:
            print(f"[tree] {tree_cache} unreadable ({e}); rebuilding")

    t = Taxonomy.build(parquet)
    with open(tree_cache, "wb") as fh:
        pickle.dump(t, fh)
    print(f"[tree] cached -> {tree_cache}")
    return t


# --------------------------------------------------------------------------- #
class Index:
    """Reverse indices + information content over a built Taxonomy."""

    def __init__(self, tax):
        self.tax = tax
        self.by_rank = defaultdict(dict)          # rank -> key -> [ids]
        for (_p, rank, key), i in tax.child.items():
            self.by_rank[rank].setdefault(key, []).append(i)
        self.epi = defaultdict(list)              # norm epithet -> [species ids]
        for (_g, e), sp in tax.sp_by_genus_epi.items():
            self.epi[e].append(sp)

        cnt = np.asarray(tax.count, dtype=np.float64)
        p = cnt / max(tax.N, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.ic = np.where(p > 0, -np.log(np.maximum(p, 1e-300)), 0.0)
        self.ic[ROOT] = 0.0

    def pick(self, cands, gt_nodes):
        """Disambiguate homonyms: prefer the GT's class, else the commonest."""
        if len(cands) == 1:
            return cands[0]
        gc = gt_nodes.get("class")
        if gc is not None:
            scoped = [c for c in cands if gc in set(self.tax.lineage(c))]
            if scoped:
                cands = scoped
        return max(cands, key=lambda c: self.tax.count[c])


def resolve_pred(idx, text, asked, gt_nodes, gt_str):
    """-> (node_id or None, resolution_flag)"""
    tax = idx.tax
    n = norm(clean(text))
    if not n:
        return None, "unresolved"

    gt_node = (gt_nodes.get("species") if RANK_CLASS[asked] == "species_level"
               else gt_nodes.get(asked))

    # Deprecated-but-correct name at ANY rank, via the GBIF synonym table.
    # Option 1 (strict, one-directional): the ANSWER must be a synonym whose
    # GBIF-accepted form is EXACTLY the ground truth (a true old->new, e.g.
    # Magnoliophyta->Tracheophyta, Reptilia->Squamata). This does NOT credit two
    # lateral synonyms that merely share a cluster, which is what over-credited
    # plants under the earlier symmetric rule.
    if asked in ("phylum", "class", "order", "family", "genus") \
            and gt_node is not None and n != norm(gt_str) \
            and resolve_syn(n) == norm(gt_str):
        return gt_node, "legacy"

    if asked in ("phylum", "class", "order", "family", "genus"):
        c = idx.by_rank[asked].get(n)
        if c:
            return idx.pick(c, gt_nodes), "exact"

    elif asked == "scientific_name":
        c = idx.by_rank["species"].get(n)
        if c:
            return idx.pick(c, gt_nodes), "exact"
        toks = n.split()
        if len(toks) >= 2:
            c = idx.by_rank["species"].get(" ".join(toks[:2]))
            if c:
                return idx.pick(c, gt_nodes), "exact"

    elif asked == "species":                       # bare epithet expected
        toks = n.split()
        if len(toks) >= 2:                         # model emitted a binomial
            c = idx.by_rank["species"].get(" ".join(toks[:2]))
            if c:
                return idx.pick(c, gt_nodes), "exact"
            n = toks[-1]
        g = gt_nodes.get("genus")
        if g is not None:
            sp = tax.sp_by_genus_epi.get((g, n))
            if sp is not None:
                return sp, "exact"
        c = idx.epi.get(n)
        if c:
            return idx.pick(c, gt_nodes), "epithet"

    elif asked == "common_name":
        sp = tax.common2sp.get(n)
        if sp is not None:
            return sp, "exact"

    # answered at some other rank entirely
    for r in ("species", "genus", "family", "order", "class", "phylum"):
        c = idx.by_rank[r].get(n)
        if c:
            return idx.pick(c, gt_nodes), "wrong_rank"
    sp = tax.common2sp.get(n)
    if sp is not None:
        return sp, "wrong_rank"
    return None, "unresolved"


def score(idx, gt, pred):
    """-> (lca_rank, lca_height, wu_palmer, lin_ic)"""
    tax = idx.tax
    if gt is None:
        return ("gt_unresolved", np.nan, np.nan, np.nan)
    if pred is None:
        return ("unresolved", np.nan, 0.0, 0.0)
    if gt == pred:
        # identical concepts are similarity 1 by definition; also guards the
        # 0/0 case when a taxon covers ~all of the catalog and its IC is 0
        return (tax.rank[gt], 0.0, 1.0, 1.0)
    l = tax.lca(gt, pred)
    dg, dp, dl = tax.depth(gt), tax.depth(pred), tax.depth(l)
    wup = (2.0 * dl / (dg + dp)) if (dg + dp) > 0 else 0.0
    icg, icp, icl = idx.ic[gt], idx.ic[pred], idx.ic[l]
    lin = (2.0 * icl / (icg + icp)) if (icg + icp) > 0 else 0.0
    return (tax.rank[l], float(dg - dl), float(wup), float(min(lin, 1.0)))


# --------------------------------------------------------------------------- #
def model_csvs(gdir, name_filter=""):
    """
    Find each model's merged prediction CSV under a group directory.

    A model dir is any directory (at any depth under gdir) that contains at
    least one '*chunk*.csv'; its deliverable is the sibling CSV WITHOUT 'chunk'
    in the name. Walking recursively supports nested layouts like
    Amphibians-10K/BioCLIP2/BioCLIP2_Kingdom/...csv.

    name_filter: if non-empty, keep only merged CSVs whose FILENAME contains
    this substring (case-insensitive). Empty (default) = every merged CSV is a
    candidate. E.g. name_filter="QA_finalized" restricts evaluation to
    ...-QA_finalized_*.csv and ignores QA_difficulty / QA_processed / etc.
    """
    nf = name_filter.lower()
    out = []
    for root, _dirs, files in os.walk(gdir):
        csvs = [f for f in files if f.endswith(".csv")]
        if not any("chunk" in f for f in csvs):
            continue                                   # not a model dir
        for f in sorted(csvs):
            if "chunk" in f:
                continue                               # a chunk, not the merge
            if nf and nf not in f.lower():
                continue                               # filtered out by name
            out.append(os.path.join(root, f))
    return out


def evaluate_csv(path, idx, group):
    tax = idx.tax
    df = pd.read_csv(path)
    acol = next((c for c in df.columns if c.endswith("_answer")), None)
    if acol is None:
        return None, None
    model = acol[:-len("_answer")]

    A, ranks, mcqs = [], [], []
    for v in df["qa_pairs"]:
        d = json.loads(v) if isinstance(v, str) else v
        A.append(d["A"])
        r, m = detect(d["Q"])
        ranks.append(r)
        mcqs.append(m)

    # GT lineage per image, from that image's own answers (as in difficulty)
    gt_by_uuid = defaultdict(dict)
    for uid, r, a, m in zip(df["uuid"], ranks, A, mcqs):
        if not m:
            gt_by_uuid[uid][r] = a
    nodes_by_uuid = {u: tax.resolve_lineage(g) for u, g in gt_by_uuid.items()}

    hard, soft, ptax, res, lrank, lh, wup, lin = [], [], [], [], [], [], [], []
    for uid, r, gt, pr in zip(df["uuid"], ranks, A, df[acol]):
        nodes = nodes_by_uuid.get(uid, {})
        gt_node = (nodes.get("species") if RANK_CLASS[r] == "species_level"
                   else nodes.get(r))

        pn, flag = resolve_pred(idx, pr, r, nodes, gt)
        gn, pnorm = norm(gt), norm(clean(pr))
        h = int(gn != "" and gn == pnorm)
        # soft credit (Option 1, strict): exact match, a resolve_pred 'legacy'
        # hit, OR the answer is a synonym whose GBIF-accepted form is EXACTLY the
        # GT. One-directional on purpose: resolve(answer)==gt, not
        # resolve(answer)==resolve(gt), so lateral synonym clusters don't count.
        s = int(h or flag == "legacy"
                or (pnorm != "" and pnorm != gn and resolve_syn(pnorm) == gn))

        lr, hh, w, li = score(idx, gt_node, pn)
        hard.append(h)
        soft.append(s)
        ptax.append(tax.name[pn] if pn is not None else "")
        res.append(flag)
        lrank.append(lr)
        lh.append(hh)
        wup.append(w)
        lin.append(li)

    df["exact_match_correctness_hard"] = hard
    df["exact_match_correctness_soft"] = soft
    df["pred_taxon"] = ptax
    df["resolution"] = res
    df["lca_rank"] = lrank
    df["lca_height"] = lh
    df["wu_palmer"] = wup
    df["lin_ic"] = lin
    df.to_csv(path, index=False)

    # For reporting, MCQ is its own rank so the recognition-vs-generation
    # contrast is visible. Scoring above still uses the plain rank.
    ranks_rep = [("common_name_MCQ" if m else r) for r, m in zip(ranks, mcqs)]

    stat = pd.DataFrame({
        "model": model, "group": group, "rank": ranks_rep,
        "difficulty": (df["difficulty"].values if "difficulty" in df.columns
                       else [""] * len(df)),
        "hard": hard, "soft": soft, "wu_palmer": wup, "lin_ic": lin,
        "_unres": [int(f == "unresolved") for f in res],
        "_sev": [hh if h == 0 else np.nan for hh, h in zip(lh, hard)],
    })
    return model, stat


# =========================================================================== #
#                            REPORTING / STATISTICS                           #
# =========================================================================== #
# Everything below derives from ONE small aggregate table ("the cube"), keyed
# by (group, model, rank, difficulty).  Every statistic is a groupby over it,
# so nothing is ever re-scored and memory stays flat.

MODEL_META = {   # params in billions; "active" params for MoE in parentheses
    "qwen2.5_3B": (3.0, "Qwen2.5-VL"),
    "qwen2.5_7B": (7.0, "Qwen2.5-VL"),
    "qwen3_8B": (8.0, "Qwen3-VL"),
    "qwen3_30B": (30.0, "Qwen3-VL"),
    "qwen3_32B": (32.0, "Qwen3-VL"),
    "InternVL3_8B_Instruct": (8.0, "InternVL3"),
    "InternVL3_38B_Instruct": (38.0, "InternVL3"),
    "InternVL3_5_8B_Instruct": (8.0, "InternVL3.5"),
    "InternVL3_5_30B_A3B_Instruct": (30.0, "InternVL3.5"),
    "InternVL3_5_38B_Instruct": (38.0, "InternVL3.5"),
    "gemma3_12B": (12.0, "Gemma3"),
    "gemma4_12B": (12.0, "Gemma4"),
    "llama3.2_11B": (11.0, "Llama3.2"),
    "molmo2_8B": (8.0, "Molmo2"),
    "phi35_vision": (4.2, "Phi3.5"),
    "pixtral_12B": (12.0, "Pixtral"),
    "llava_v1_6_mistral_7b_hf": (7.0, "LLaVA-1.6"),
    "llava_v1_6_vicuna_7b_hf": (7.0, "LLaVA-1.6"),
}

RANK_ORDER = ["phylum", "class", "order", "family", "genus", "species",
              "scientific_name", "common_name", "common_name_MCQ"]
DIFF_ORDER = ["Easy", "Medium", "Hard", "Degenerate", "Unresolved"]

SUMS = ["n", "hard_sum", "soft_sum", "wup_sum", "wup_n", "lin_sum", "lin_n",
        "sev_sum", "sev_n", "unres_sum"]


def spearman(x, y):
    """Rank correlation without a scipy dependency."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return np.nan
    rx = pd.Series(x[m]).rank().values
    ry = pd.Series(y[m]).rank().values
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def cube_of(stat):
    """Per-file aggregate keyed by (rank, difficulty)."""
    a = stat.groupby(["rank", "difficulty"], dropna=False, observed=True).agg(
        n=("hard", "size"), hard_sum=("hard", "sum"), soft_sum=("soft", "sum"),
        wup_sum=("wu_palmer", "sum"), wup_n=("wu_palmer", "count"),
        lin_sum=("lin_ic", "sum"), lin_n=("lin_ic", "count"),
        sev_sum=("_sev", "sum"), sev_n=("_sev", "count"),
        unres_sum=("_unres", "sum"),
    ).reset_index()
    a["group"] = stat["group"].iloc[0]
    a["model"] = stat["model"].iloc[0]
    return a


def roll(cube, keys):
    d = cube.groupby(keys, dropna=False, observed=True)[SUMS].sum().reset_index()
    o = pd.DataFrame({k: d[k] for k in keys})
    o["EM"] = d.hard_sum / d.n
    o["EM_soft"] = d.soft_sum / d.n
    o["EM_gap"] = o.EM_soft - o.EM
    o["severity"] = d.sev_sum / d.sev_n.replace(0, np.nan)
    o["unresolved"] = d.unres_sum / d.n
    o["wu_palmer"] = d.wup_sum / d.wup_n.replace(0, np.nan)
    o["lin_ic"] = d.lin_sum / d.lin_n.replace(0, np.nan)
    o["n"] = d.n
    return o


def pv(cube, index, col, val, order=None):
    d = roll(cube, [index, col])
    p = d.pivot(index=index, columns=col, values=val)
    if order:
        p = p[[c for c in order if c in p.columns]]
    return p


def fmt(df, f="{:.3f}"):
    x = df.copy()
    for c in x.columns:
        if pd.api.types.is_float_dtype(x[c]):
            x[c] = x[c].map(lambda v: "  -  " if pd.isna(v) else f.format(v))
    return x.to_string()


def sec(n, title, body, what="", implies="", paper=""):
    return "\n".join([f"\n{'=' * 78}", f"{n}. {title}", "=" * 78, "", body])


# --------------------------------------------------------------------------- #
def build_report(cube, label, is_all):
    L = ["#" * 78,
         f"# BioVQA  C1: Hierarchical Taxonomic Reasoning",
         f"# {label}",
         f"# items={int(cube.n.sum()):,}   models={cube.model.nunique()}   "
         f"groups={cube.group.nunique()}",
         "#" * 78]

    lb = roll(cube, ["model"]).sort_values("EM_soft", ascending=False)
    order = list(lb.model)
    N = len(order)

    # -- 1 ------------------------------------------------------------------
    L.append(sec(1, "LEADERBOARD",
        fmt(lb.set_index("model")[["EM", "EM_soft", "EM_gap", "severity",
                                   "unresolved", "wu_palmer", "lin_ic", "n"]]),
        what="",
        implies="",
        paper=""))

    # -- 2 ------------------------------------------------------------------
    pr = pv(cube, "model", "rank", "EM", RANK_ORDER).reindex(order)
    L.append(sec(2, "EM BY RANK  (per model)", fmt(pr),
        what="",
        paper=""))

    # -- 3 ------------------------------------------------------------------
    mr = roll(cube, ["rank"]).set_index("rank")["EM"]
    bars = []
    for r in RANK_ORDER:
        if r in mr.index:
            bars.append(f"  {r:<18} {mr[r]:.3f}  {'#' * int(round(mr[r] * 50))}")
    ctxt, cimp = "\n".join(bars), ""
    if "phylum" in mr.index and "species" in mr.index:
        a, b = mr["phylum"], mr["species"]
        cimp = ""
    L.append(sec(3, "THE COLLAPSE  (mean EM over all models, by rank)", ctxt,
        what="",
        implies="",
        paper=""))

    # -- 4 ------------------------------------------------------------------
    pdf = pv(cube, "model", "difficulty", "EM", DIFF_ORDER).reindex(order)
    L.append(sec(4, "EM BY DIFFICULTY  (per model)", fmt(pdf),
        what=""))

    # -- 5 ------------------------------------------------------------------
    if all(c in pdf.columns for c in ("Easy", "Medium", "Hard")):
        ok = (pdf["Easy"] > pdf["Medium"]) & (pdf["Medium"] > pdf["Hard"])
        k, frac = int(ok.sum()), ok.mean()
        verdict = ("PASS" if frac >= 0.8 else
                   "WEAK" if frac >= 0.5 else "FAIL")
        b = [f"  models with Easy > Medium > Hard : {k}/{len(ok)}   VERDICT: {verdict}"]
        if (~ok).any():
            b.append("\n  violations:")
            for m in pdf.index[~ok]:
                b.append(f"    {m:<32} E={pdf.loc[m,'Easy']:.3f} "
                         f"M={pdf.loc[m,'Medium']:.3f} H={pdf.loc[m,'Hard']:.3f}")
        d_all = roll(cube, ["difficulty"]).set_index("difficulty")["EM"]
        if all(x in d_all.index for x in ("Easy", "Medium", "Hard")):
            b.append(f"\n  mean over models:  Easy={d_all['Easy']:.3f}  "
                     f"Medium={d_all['Medium']:.3f}  Hard={d_all['Hard']:.3f}")
        imp = ""
        L.append(sec(5, "DIFFICULTY MONOTONICITY", "\n".join(b),
            what="",
            implies="",
            paper=""))

    # -- 6 ------------------------------------------------------------------
    rxd = roll(cube, ["rank", "difficulty"])
    rxd = rxd[~rxd.difficulty.isin(["Degenerate", "Unresolved"])]
    if len(rxd):
        p6 = rxd.pivot(index="rank", columns="difficulty", values="EM")
        p6 = p6[[c for c in ["Easy", "Medium", "Hard"] if c in p6.columns]]
        p6 = p6.reindex([r for r in RANK_ORDER if r in p6.index])
        L.append(sec(6, "RANK x DIFFICULTY  (mean EM; the interaction)", fmt(p6),
            what="",
            implies="",
            paper=""))

    # -- 7 ------------------------------------------------------------------
    if "common_name" in pr.columns and "common_name_MCQ" in pr.columns:
        g = pd.DataFrame({"open_ended": pr["common_name"],
                          "MCQ_4way": pr["common_name_MCQ"]})
        g["gap"] = g["MCQ_4way"] - g["open_ended"]
        g = g.sort_values("gap", ascending=False)
        L.append(sec(7, "RECOGNITION vs GENERATION", fmt(g),
            what="",
            implies="",
            paper=""))

    # -- 8 ------------------------------------------------------------------
    gap = lb.set_index("model")[["EM", "EM_soft", "EM_gap"]]
    gg = roll(cube, ["group"]).set_index("group")[["EM", "EM_soft", "EM_gap"]] \
        if is_all else None
    body8 = fmt(gap)
    if gg is not None:
        body8 += "\n\nBY GROUP (this is the control condition):\n" + fmt(gg)
    L.append(sec(8, "TAXONOMIC REVISION LAG  (EM_soft - EM)",
        body8,
        what="",
        implies="",
        paper=""))

    # -- 9 ------------------------------------------------------------------
    u = lb.set_index("model")[["unresolved", "severity", "EM"]] \
        .sort_values("unresolved", ascending=False)
    rho_u = spearman(lb.EM, lb.unresolved)
    L.append(sec(9, "UNRESOLVED ANSWERS", fmt(u),
        what="",
        implies="",
        paper=""))

    # -- 10 -----------------------------------------------------------------
    sv = pv(cube, "model", "rank", "severity", RANK_ORDER).reindex(order)
    L.append(sec(10, "SEVERITY BY RANK  (lower = closer misses)", fmt(sv, "{:.2f}"),
        what="",
        paper=""))

    # -- 11 -----------------------------------------------------------------
    d11 = lb.set_index("model")[["EM", "severity"]].copy()
    # nullable Int so an all-NaN severity column (e.g. a tiny filtered run with
    # no resolvable errors) does not crash .astype(int)
    def _rank_int(series, ascending=True):
        # ties yield fractional ranks (1.5); round before the integer cast.
        r = series.rank(ascending=ascending, method="min")
        arr = np.asarray(r, dtype="float64")
        return pd.array(np.round(arr), dtype="Int64")

    d11["EM_rank"] = _rank_int(d11.EM, ascending=False)
    d11["sev_rank"] = _rank_int(d11.severity)      # lower severity = better
    d11["rank_shift"] = d11.EM_rank - d11.sev_rank
    rho_s = spearman(lb.EM, lb.severity)
    if d11.severity.notna().any():
        best_sev = d11.severity.idxmin()
        implies11 = ""
    else:
        implies11 = ""
    L.append(sec(11, "EM vs SEVERITY DISSOCIATION",
        fmt(d11.sort_values("EM_rank"), "{:.3f}"),
        what="",
        implies="",
        paper=""))

    # -- 12 -----------------------------------------------------------------
    meta = lb.copy()
    meta["params"] = meta.model.map(lambda m: MODEL_META.get(m, (np.nan, ""))[0])
    meta["family"] = meta.model.map(lambda m: MODEL_META.get(m, (np.nan, ""))[1])
    mm = meta.dropna(subset=["params"])
    b12 = fmt(mm.sort_values("params")[["model", "family", "params", "EM",
                                        "severity"]].set_index("model"))
    rho_p = spearman(mm.params, mm.EM)
    fam = []
    for f, sub in mm.groupby("family"):
        if len(sub) >= 2:
            s = sub.sort_values("params")
            span = s.EM.iloc[-1] - s.EM.iloc[0]
            fam.append(f"  {f:<14} {s.params.iloc[0]:>5.1f}B EM={s.EM.iloc[0]:.3f}"
                       f"  ->{s.params.iloc[-1]:>5.1f}B EM={s.EM.iloc[-1]:.3f}"
                       f"   delta={span:+.3f}")
    L.append(sec(12, "SCALE  (does bigger help?)",
        b12 + ("\n\nWITHIN FAMILY (smallest -> largest):\n" + "\n".join(fam)
               if fam else ""),
        what="",
        implies="",
        paper=""))

    # -- 13 -----------------------------------------------------------------
    r_w, r_l = spearman(lb.EM, lb.wu_palmer), spearman(lb.EM, lb.lin_ic)
    rr = pd.DataFrame({"vs_EM_spearman": [r_w, r_l, rho_s, rho_u]},
                      index=["wu_palmer", "lin_ic", "severity", "unresolved"])
    verdict13 = ("REDUNDANT - demote to appendix" if abs(r_w) > 0.9
                 else "carries some independent signal - keep")
    L.append(sec(13, "METRIC REDUNDANCY  (do you need all these numbers?)",
        fmt(rr) + f"\n\n  wu_palmer verdict: {verdict13}",
        what="",
        implies="",
        paper=""))

    # -- 14 (ALL only) ------------------------------------------------------
    if is_all:
        gsum = roll(cube, ["group"]).sort_values("EM")
        L.append(sec(14, "WHICH DATASET IS HARDER?",
            fmt(gsum.set_index("group")[["EM", "EM_soft", "severity",
                                         "unresolved", "n"]]),
            what="",
            implies="",
            paper=""))

        pg = pv(cube, "model", "group", "EM").reindex(order)
        L.append(sec(15, "EM BY MODEL x GROUP", fmt(pg),
            what=""))

        # cross-group rank consistency
        rows = []
        gs = list(pg.columns)
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                rows.append(spearman(pg[gs[i]], pg[gs[j]]))
        rows = [r for r in rows if not np.isnan(r)]
        mrho = float(np.mean(rows)) if rows else np.nan
        cons = pd.DataFrame(index=gs, columns=gs, dtype=float)
        for a in gs:
            for b in gs:
                cons.loc[a, b] = 1.0 if a == b else spearman(pg[a], pg[b])
        L.append(sec(16, "CROSS-GROUP GENERALITY",
            fmt(cons, "{:.2f}") + f"\n\n  mean off-diagonal Spearman = {mrho:.3f}",
            what="",
            implies="",
            paper=""))

    # -- final --------------------------------------------------------------
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Score BioVQA C1 model CSVs and write reports.")
    ap.add_argument("--base_dir", required=True,
                    help="REQUIRED. Root holding the *-10K group directories. "
                         "Reports and the tree cache are written under it.")
    ap.add_argument("--parquet", required=True,
                    help="REQUIRED. TreeOfLife-200M catalog.parquet "
                         "(read only when the tree cache must be rebuilt).")
    ap.add_argument("--tree_cache", default=None,
                    help="pickled taxonomy; reused if compatible, else rebuilt "
                         "from --parquet. Default: <base_dir>/tree.pkl")
    ap.add_argument("--out_dir", default=None,
                    help="where reports and CSVs go. Default: <base_dir>/Result")
    ap.add_argument("--name_filter", default="",
                    help="only evaluate merged CSVs whose filename contains this "
                         "substring (case-insensitive). Default '' means "
                         "QA_finalized. Pass a different value (e.g. QA_difficulty, "
                         "QA_processed) to switch, or 'ALL' to evaluate every "
                         "nested merged CSV regardless of name.")
    ap.add_argument("--synonyms", default=None,
                    help="synonyms.pkl from build_synonyms.py. When present, "
                         "em_soft additionally credits deprecated-but-correct "
                         "names (Magnoliophyta==Tracheophyta, Reptilia's "
                         "children, ...). Default: <base_dir>/synonyms.pkl if it "
                         "exists, else none (em_soft == em_hard). em_hard is "
                         "never affected either way.")
    args = ap.parse_args()

    base_dir = args.base_dir
    if not os.path.isdir(base_dir):
        raise SystemExit(f"ERROR: --base_dir does not exist: {base_dir}")

    tree_cache = args.tree_cache or os.path.join(base_dir, "tree.pkl")
    out_dir = args.out_dir or os.path.join(base_dir, "ResultX")

    # synonyms default: <base_dir>/synonyms.pkl if it exists, else none.
    # An explicit --synonyms path is always honored (and must exist).
    if args.synonyms:
        synonyms_path = args.synonyms
    else:
        cand = os.path.join(base_dir, "synonyms.pkl")
        synonyms_path = cand if os.path.exists(cand) else None

    # Empty filter defaults to QA_finalized; the literal 'ALL' disables filtering.
    if args.name_filter == "":
        name_filter = "QA_finalized"
    elif args.name_filter.upper() == "ALL":
        name_filter = ""
    else:
        name_filter = args.name_filter

    print(f"[cfg] base_dir    = {base_dir}")
    print(f"[cfg] parquet     = {args.parquet}")
    print(f"[cfg] tree_cache  = {tree_cache}")
    print(f"[cfg] out_dir     = {out_dir}")
    print(f"[cfg] name_filter = {name_filter!r}"
          f"{'  (all merged CSVs)' if name_filter == '' else ''}")

    if synonyms_path:
        if not os.path.exists(synonyms_path):
            raise SystemExit(f"ERROR: --synonyms not found: {synonyms_path}")
        load_synonyms(synonyms_path)
        default_note = "" if args.synonyms else "  (default <base_dir>/synonyms.pkl)"
        print(f"[cfg] synonyms    = {synonyms_path}  "
              f"({len(SYNONYMS):,} pairs){default_note}")
    else:
        print(f"[cfg] synonyms    = (none; em_soft will equal em_hard)")
        print(f"                    no <base_dir>/synonyms.pkl and no --synonyms given")

    tax = load_tree(args.parquet, tree_cache)
    idx = Index(tax)
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    groups = sorted(d for d in os.listdir(base_dir)
                    if d.endswith("-10K")
                    and os.path.isdir(os.path.join(base_dir, d)))

    cubes = []
    for g in groups:
        gdir = os.path.join(base_dir, g)
        print(f"\n[{g}]")
        found = model_csvs(gdir, name_filter)
        if not found:
            print(f"   (no merged CSVs"
                  f"{f' matching {name_filter!r}' if name_filter else ''})")
        for mcsv in found:
            model, stat = evaluate_csv(mcsv, idx, g)
            if stat is None:
                print(f"   {os.path.basename(mcsv)}  no *_answer column, skipped")
                continue
            cubes.append(cube_of(stat))
            print(f"   {model:<34} EM={stat.hard.mean():.3f}  "
                  f"soft={stat.soft.mean():.3f}  "
                  f"sev={stat._sev.mean():.2f}  n={len(stat):,}")

    if not cubes:
        print("\n[warn] no model CSVs found; nothing written")
        return
    cube = pd.concat(cubes, ignore_index=True)

    print("\n[report] writing per-group reports ...")
    for g, sub in cube.groupby("group", observed=True):
        with open(os.path.join(out_dir, f"result_{g}.txt"), "w") as fh:
            fh.write(build_report(sub, g, is_all=False))
        roll(sub, ["model"]).to_csv(f"{csv_dir}/{g}_by_model.csv", index=False)
        roll(sub, ["model", "rank"]).to_csv(f"{csv_dir}/{g}_by_rank.csv", index=False)
        roll(sub, ["model", "difficulty"]).to_csv(f"{csv_dir}/{g}_by_difficulty.csv", index=False)
        roll(sub, ["model", "rank", "difficulty"]).to_csv(f"{csv_dir}/{g}_by_rank_x_difficulty.csv", index=False)
        print(f"   result_{g}.txt")

    all_report = build_report(cube, "ALL GROUPS POOLED", is_all=True)
    with open(os.path.join(out_dir, "result_ALL.txt"), "w") as fh:
        fh.write(all_report)
    roll(cube, ["model"]).to_csv(f"{csv_dir}/ALL_by_model.csv", index=False)
    roll(cube, ["model", "rank"]).to_csv(f"{csv_dir}/ALL_by_rank.csv", index=False)
    roll(cube, ["model", "difficulty"]).to_csv(f"{csv_dir}/ALL_by_difficulty.csv", index=False)
    roll(cube, ["model", "group"]).to_csv(f"{csv_dir}/ALL_by_group.csv", index=False)
    roll(cube, ["model", "rank", "difficulty"]).to_csv(f"{csv_dir}/ALL_by_rank_x_difficulty.csv", index=False)
    roll(cube, ["group"]).to_csv(f"{csv_dir}/ALL_group_summary.csv", index=False)
    roll(cube, ["rank"]).to_csv(f"{csv_dir}/ALL_rank_summary.csv", index=False)
    roll(cube, ["difficulty"]).to_csv(f"{csv_dir}/ALL_difficulty_summary.csv", index=False)
    cube.to_csv(f"{csv_dir}/cube_raw.csv", index=False)

    # full leaderboard, every reported metric, sorted by EM_soft high->low
    lb = roll(cube, ["model"]).sort_values("EM_soft", ascending=False)
    leaderboard = fmt(lb.set_index("model")[
        ["EM", "EM_soft", "EM_gap", "severity", "unresolved",
         "wu_palmer", "lin_ic", "n"]])

    # evaluate_stats.txt: the single file to read. Leaderboard first (sorted by
    # EM_soft), then the full pooled report, then each per-group report.
    stats_path = os.path.join(out_dir, "evaluate_stats.txt")
    with open(stats_path, "w") as fh:
        fh.write("#" * 78 + "\n")
        fh.write("# BioVQA C1 - evaluate_stats\n")
        fh.write(f"# items={int(cube.n.sum()):,}  models={cube.model.nunique()}"
                 f"  groups={cube.group.nunique()}\n")
        fh.write("# leaderboard sorted by EM_soft (high -> low)\n")
        fh.write("#" * 78 + "\n\n")
        fh.write("LEADERBOARD (pooled, sorted by EM_soft)\n")
        fh.write("  EM=strict  EM_soft=+GBIF synonyms  EM_gap=soft-hard\n")
        fh.write("  severity=mean ranks-to-LCA when wrong (lower better)\n")
        fh.write("  wu_palmer/lin_ic=hierarchical similarity  unresolved=unplaceable\n\n")
        fh.write(leaderboard + "\n\n")
        fh.write(all_report)
        for g, sub in cube.groupby("group", observed=True):
            fh.write("\n\n")
            fh.write(build_report(sub, g, is_all=False))

    print("\n" + leaderboard)
    print(f"\n[done]")
    print(f"   evaluate_stats.txt : {stats_path}   <- everything, sorted by EM_soft")
    print(f"   reports : {out_dir}/result_ALL.txt  + result_<Group>.txt x{len(groups)}")
    print(f"   csvs    : {csv_dir}/")
    print(f"\n   >>> READ SECTION 5 of result_ALL.txt FIRST - it is pass/fail for")
    print(f"       the whole difficulty axis.")


if __name__ == "__main__":
    main()