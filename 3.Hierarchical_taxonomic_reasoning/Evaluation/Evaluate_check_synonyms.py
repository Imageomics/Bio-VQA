#!/usr/bin/env python3
"""
audit_soft.py
=============
Audit every SOFT-only credit (soft==1 while hard==0) across all scored CSVs and
write the COMPLETE list to a CSV in base_dir. Each row is re-verified against the
GBIF synonym table, so the file itself says whether each credit is a legitimate
deprecated-name-for-the-same-taxon (Magnoliophyta->Tracheophyta) or something to
investigate.

Reads the already-scored CSVs; changes nothing.

Output:  <base_dir>/Evaluation_soft_audit.csv
  columns:
    no            1..N, ranked by count
    count         how many times this (predicted, ground_truth, rank) occurred
    rank          the taxonomic rank asked (phylum/class/.../scientific_name)
    ground_truth  the correct answer
    predicted     what the model answered
    resolved_predicted  GBIF-accepted form of `predicted`
    verdict       OK_synonym  = resolved(predicted) == ground_truth  (legit)
                  OK_exact_after_clean = matched once boilerplate was stripped
                  CHECK       = credited but does NOT resolve to GT (investigate)
                  NO_LOOKUP   = --synonyms not given, cannot verify
    note          short human-readable explanation
    group         organismal group

Console still prints the top-N per group for a quick look.

Run:
    python audit_soft.py --base_dir /fs/.../3-Hierarchical-Taxonomic-Reasoning \
                         --synonyms /fs/.../synonyms.pkl
    python audit_soft.py --base_dir ... --synonyms ... --name_filter QA_finalized --top 50
"""

import os
import re
import csv
import json
import pickle
import argparse
import unicodedata
from collections import Counter

csv.field_size_limit(2**31 - 1)

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")
# same leading-boilerplate stripper Evaluate.py uses
_LEAD = re.compile(
    r"^(the\s+)?(answer\s*[:\-]\s*|final answer\s*[:\-]\s*|"
    r"(common|scientific)\s+name\s+(is|of the \w+)?\s*[:\-]?\s*|"
    r"(phylum|class|order|family|genus|species)\s+(is|of)?\s*[:\-]?\s*)", re.I)


def norm(s):
    if s is None or (isinstance(s, float) and s != s):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("-", " ").replace("_", " ")
    return _WS.sub(" ", _PUNCT.sub(" ", s)).strip()


def clean(raw):
    if raw is None or (isinstance(raw, float) and raw != raw):
        return ""
    t = str(raw).strip().split("\n")[0].strip().strip("`*\"' ")
    t = re.sub(r"^\(?[A-Da-d]\)\s+", "", t)
    prev = None
    while prev != t:
        prev = t
        t = _LEAD.sub("", t).strip()
    return t.rstrip(".").strip()


def model_csvs(gdir, name_filter=""):
    nf = name_filter.lower()
    out = []
    for root, _d, files in os.walk(gdir):
        cs = [f for f in files if f.endswith(".csv")]
        if not any("chunk" in f for f in cs):
            continue
        for f in sorted(cs):
            if "chunk" not in f and (not nf or nf in f.lower()):
                out.append(os.path.join(root, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--synonyms", default=None,
                    help="synonyms.pkl from build_synonyms.py; enables the "
                         "verdict column. Strongly recommended.")
    ap.add_argument("--name_filter", default="QA_finalized",
                    help="'' or 'ALL' = every merged CSV; default QA_finalized")
    ap.add_argument("--top", type=int, default=25,
                    help="how many per-group pairs to print to console")
    ap.add_argument("--out", default=None,
                    help="output CSV (default <base_dir>/Evaluation_soft_audit.csv)")
    args = ap.parse_args()

    syn = {}
    if args.synonyms:
        with open(args.synonyms, "rb") as fh:
            raw = pickle.load(fh)["map"]
        for k, v in raw.items():
            syn[k] = set(v) if isinstance(v, (list, set, tuple)) else {v}
        multi = sum(1 for v in syn.values() if len(v) > 1)
        print(f"[lookup] {len(syn):,} synonym names ({multi:,} pro-parte)")
    else:
        print("[lookup] no --synonyms given; verdict column will be NO_LOOKUP")

    def targets(n):
        return syn.get(n, set())

    def matches(pn, gn):
        return gn in syn.get(pn, ())

    nf = "" if args.name_filter.upper() == "ALL" else args.name_filter
    out_path = args.out or os.path.join(args.base_dir, "Evaluation_soft_audit.csv")

    groups = sorted(d for d in os.listdir(args.base_dir)
                    if d.endswith("-10K")
                    and os.path.isdir(os.path.join(args.base_dir, d)))

    all_pairs = Counter()          # (predicted, gt, rank, group) -> count
    for g in groups:
        gdir = os.path.join(args.base_dir, g)
        pairs = Counter()
        n_soft_only = n_rows = 0
        for mcsv in model_csvs(gdir, nf):
            with open(mcsv, newline="") as fh:
                rdr = csv.DictReader(fh)
                if not rdr.fieldnames:
                    continue
                acol = next((c for c in rdr.fieldnames
                             if c.endswith("_answer")), None)
                if acol is None or \
                   "exact_match_correctness_hard" not in rdr.fieldnames:
                    continue
                for r in rdr:
                    n_rows += 1
                    try:
                        h = int(float(r["exact_match_correctness_hard"] or 0))
                        s = int(float(r["exact_match_correctness_soft"] or 0))
                    except ValueError:
                        continue
                    if s == 1 and h == 0:
                        n_soft_only += 1
                        gt = json.loads(r["qa_pairs"])["A"] if r["qa_pairs"] else ""
                        cat = r.get("category", "?")
                        pairs[(str(r[acol]).strip(), str(gt).strip(), cat)] += 1
        for k, c in pairs.items():
            all_pairs[k + (g,)] += c
        pct = 100 * n_soft_only / max(n_rows, 1)
        print(f"\n{'='*70}\n{g}: {n_soft_only:,} soft-only credits of "
              f"{n_rows:,} rows ({pct:.2f}%)\n{'='*70}")
        for (ans, gt, cat), c in pairs.most_common(args.top):
            print(f"  {c:6,}  [{cat:14s}] {ans!r:28s} -> {gt!r}")
        extra = len(pairs) - args.top
        if extra > 0:
            print(f"  ... {extra:,} more distinct pairs (full list in the CSV)")

    # ---- write the COMPLETE list -------------------------------------- #
    rows = []
    for (ans, gt, cat, grp), c in all_pairs.most_common():
        pn, gn = norm(clean(ans)), norm(gt)
        tgt = targets(pn)
        tgt_str = "|".join(sorted(tgt)) if tgt else pn
        if not syn:
            verdict, note = "NO_LOOKUP", "pass --synonyms to verify"
        elif pn == gn:
            verdict, note = "OK_exact_after_clean", "matched after boilerplate strip"
        elif matches(pn, gn):
            verdict = "OK_synonym"
            pp = "  (pro-parte)" if len(tgt) > 1 else ""
            note = f"GBIF: {pn} -> {gn}{pp}"
        else:
            verdict = "CHECK"
            note = f"does NOT resolve to GT (resolves to {tgt_str!r})"
        rows.append((c, cat, gt, ans, tgt_str, verdict, note, grp))

    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["no", "count", "rank", "ground_truth", "predicted",
                    "resolved_predicted", "verdict", "note", "group"])
        for i, (c, cat, gt, ans, rp, verdict, note, grp) in enumerate(rows, 1):
            w.writerow([i, c, cat, gt, ans, rp, verdict, note, grp])

    total = sum(r[0] for r in rows)
    ok = sum(r[0] for r in rows if r[5].startswith("OK"))
    check = sum(r[0] for r in rows if r[5] == "CHECK")
    print(f"\n{'#'*70}")
    print(f"wrote {len(rows):,} distinct credited pairs -> {out_path}")
    print(f"  total credited answers : {total:,}")
    if syn:
        print(f"  verified legit (OK_*)  : {ok:,} ({100*ok/max(total,1):.1f}%)")
        print(f"  need review (CHECK)    : {check:,} ({100*check/max(total,1):.1f}%)")
        if check:
            print(f"\n  >>> {check:,} credited answers do NOT resolve to their GT.")
            print(f"      Open the CSV, sort by verdict, inspect the CHECK rows.")
        else:
            print(f"\n  >>> every soft credit is a verified GBIF old->new name. Clean.")


if __name__ == "__main__":
    main()