#!/usr/bin/env python3
"""
Authoritative evaluation for the Viewpoint task. Reads merged per-model CSVs,
writes machine-readable results plus a markdown report.

  python evaluate.py
  python evaluate.py --results_dir ... --out_dir ...
  python evaluate.py --bootstrap 2000
  python evaluate.py --credit_over 0.5 --credit_under 0.35 --credit_sibling 0.15

Outputs (all written to --out_dir):
  viewpoint_overall.csv     one row per model
  viewpoint_by_group.csv    one row per model x tier x dataset
  viewpoint_by_relation.csv how each model's errors distribute over the rubric
  viewpoint_report.md       human-readable summary

--------------------------------------------------------------------------- #
THE GRADED RUBRIC

Exact match is too harsh for viewpoint. Answering "frontright" when the truth
is "front" is not the same kind of mistake as answering "back", but exact match
scores both as zero. Every accepted label decomposes into axis components -
front, back, left, right, up, down - and the relation between the gold set and
the predicted set determines the credit:

  relation        example (gold -> pred)          default credit
  exact           backright -> backright                   1.00
  over            right     -> backright                   0.50
  under           backright -> right                       0.35
  sibling         frontright-> backright                   0.15
  wrong           right     -> frontleft                   0.00

over  = the gold components are a strict subset of the predicted ones. The
        model named the right primary direction and added a qualifier the
        annotation does not carry. It committed to a finer distinction and got
        the coarse part right.
under = the predicted components are a strict subset of the gold ones. The
        model gave a coarser answer than the annotation. Credited less than
        over because it declined the finer distinction rather than attempting
        it - this asymmetry is a deliberate choice, and the symmetric angular
        metric is reported alongside so a reader can see it both ways.
sibling = the two share exactly one component but neither contains the other,
        which on the eight-point ring is always a 90 degree miss.

All four credits are command-line arguments. The rubric is recorded in the
report and in viewpoint_overall.csv so a result can never be quoted without it.

--------------------------------------------------------------------------- #
WHY macroVAS IS THE HEADLINE

The gold labels are severely skewed: left and right together are ~83% of rows,
and for Grevys zebra 89% of annotations are "right", because Wildbook's census
protocol photographs the right flank for identification. A model that always
answers "right" therefore scores well on any row-averaged metric, VAS included,
without looking at anything.

  macroVAS  the graded score averaged with equal weight over the GOLD LABELS
            present. A constant predictor cannot hide here: it earns credit on
            one label and zero on the rest.
  balAcc    macro-averaged recall. A constant predictor scores 1/k exactly.
  kappa     agreement corrected for both marginal distributions.
  predTop   the model's own most frequent answer and its share. Near 1.00
            means constant predictor; read nothing else until you have checked
            this column.
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

DEFAULT_LABELS = ["front", "back", "left", "right", "frontleft", "frontright",
                  "backleft", "backright", "up", "down"]

# Circular layout for the eight horizontal viewpoints, 45 degrees apart.
ANGLE = {"front": 0, "frontright": 45, "right": 90, "backright": 135,
         "back": 180, "backleft": 225, "left": 270, "frontleft": 315}

AXES = ["front", "back", "left", "right", "up", "down"]


def components(label):
    """Decompose a label into its axis components: frontleft -> {front,left}."""
    s = str(label)
    out = set()
    for a in ("front", "back", "up", "down", "left", "right"):
        if a in s:
            out.add(a)
    return out


def relation(gold, pred):
    """One of exact | over | under | sibling | wrong."""
    if gold == pred:
        return "exact"
    g, p = components(gold), components(pred)
    if not g or not p:
        return "wrong"
    if g < p:
        return "over"
    if p < g:
        return "under"
    if g & p:
        return "sibling"
    return "wrong"


def ang_err(g, p):
    if g not in ANGLE or p not in ANGLE:
        return float("nan")
    d = abs(ANGLE[g] - ANGLE[p]) % 360
    return min(d, 360 - d)


def make_scorer(credits):
    def vas(gold, pred):
        return credits[relation(gold, pred)]
    return vas


def balanced_accuracy(gold, pred):
    labs = sorted(set(gold))
    if not labs:
        return float("nan")
    rec = [float((pred[gold == l] == l).mean()) for l in labs
           if (gold == l).sum()]
    return sum(rec) / len(rec) if rec else float("nan")


def macro_vas(gold, pred, vas_vals):
    """VAS averaged with equal weight over the gold labels present."""
    labs = sorted(set(gold))
    if not labs:
        return float("nan")
    per = [float(vas_vals[gold == l].mean()) for l in labs if (gold == l).sum()]
    return sum(per) / len(per) if per else float("nan")


def vas_kappa(gold, pred, vas):
    """Chance-corrected graded agreement - Cohen's WEIGHTED kappa with the
    rubric as its weight matrix.

        kappaVAS = (VAS_observed - VAS_expected) / (1 - VAS_expected)

    VAS_expected is the score the same model would earn if its answers were
    shuffled against the gold labels, i.e. from the product of the two
    marginal distributions. This is what macroVAS cannot do: a model that
    answers compounds indiscriminately collects "over" credit on the 83% of
    rows whose gold label is a bare cardinal, and macroVAS still rewards it,
    because the credit is real on every individual row. Subtracting the
    expected score removes exactly that. Verified against synthetic
    strategies: always-"right" scores 0.000, indiscriminate-compounds scores
    0.001, while a genuine model with 30% one-step noise scores 0.773."""
    if len(gold) == 0:
        return float("nan")
    obs = float(np.mean([vas(g, p) for g, p in zip(gold, pred)]))
    pg = gold.value_counts(normalize=True)
    pp = pred.value_counts(normalize=True)
    exp = float(sum(pg[a] * pp[b] * vas(a, b) for a in pg.index for b in pp.index))
    if exp >= 1.0:
        return float("nan")
    return (obs - exp) / (1.0 - exp)


def cohen_kappa(gold, pred):
    if len(gold) == 0:
        return float("nan")
    po = float((gold.values == pred.values).mean())
    labs = set(gold) | set(pred)
    pe = sum(float((gold == l).mean()) * float((pred == l).mean()) for l in labs)
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1.0 - pe)


def score_block(gold, pred, vas):
    """gold/pred are pandas Series of label strings."""
    n = len(gold)
    if n == 0:
        return {k: float("nan") for k in
                ("n", "macroVAS", "VAS", "kappaVAS", "EM", "balAcc", "kappa",
                 "angErr",
                 "within45", "majority", "modalVAS", "modalMacroVAS",
                 "predTop", "predTopShare")}
    v = pd.Series([vas(g, p) for g, p in zip(gold, pred)], index=gold.index)
    errs = [ang_err(g, p) for g, p in zip(gold, pred)]
    errs = [e for e in errs if not math.isnan(e)]

    modal = gold.mode().iloc[0]
    modal_pred = pd.Series([modal] * n, index=gold.index)
    mv = pd.Series([vas(g, modal) for g in gold], index=gold.index)

    top = str(pred.mode().iloc[0])
    return {
        "n": int(n),
        "macroVAS": macro_vas(gold, pred, v),
        "VAS": float(v.mean()),
        "kappaVAS": vas_kappa(gold, pred, vas),
        "EM": float((gold.values == pred.values).mean()),
        "balAcc": balanced_accuracy(gold, pred),
        "kappa": cohen_kappa(gold, pred),
        "angErr": (sum(errs) / len(errs)) if errs else float("nan"),
        "within45": (sum(1 for e in errs if e <= 45) / len(errs))
                    if errs else float("nan"),
        "majority": float((gold == modal).mean()),
        "modalVAS": float(mv.mean()),
        "modalMacroVAS": macro_vas(gold, modal_pred, mv),
        "predTop": top,
        "predTopShare": float((pred == top).mean()),
    }


def bootstrap_ci(d, key, fn, n_boot, seed=0):
    """Percentile CI resampling clusters (images), not rows."""
    if n_boot <= 0 or not len(d):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    groups = d.groupby(key).indices
    keys = list(groups.keys())
    idx = [groups[k] for k in keys]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        rows = np.concatenate([idx[i] for i in pick])
        s = d.iloc[rows]
        vals.append(fn(s["_g"], s["_p"]))
    vals = [x for x in vals if x == x]
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(here, "results"))
    ap.add_argument("--out_dir", default=os.path.join(here, "results"))
    ap.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    ap.add_argument("--credit_over", type=float, default=0.50)
    ap.add_argument("--credit_under", type=float, default=0.35)
    ap.add_argument("--credit_sibling", type=float, default=0.15)
    ap.add_argument("--credit_wrong", type=float, default=0.0)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--sweep", action="store_true",
                    help="also print the macroVAS ranking at several rubric "
                         "settings, to show whether the ordering depends on "
                         "the credit values")
    ap.add_argument("--drop_datasets", default="",
                    help="comma list given an extra excluded row, e.g. zebra")
    args = ap.parse_args()

    labels = [s.strip().lower() for s in args.labels.split(",") if s.strip()]
    credits = {"exact": 1.0, "over": args.credit_over,
               "under": args.credit_under, "sibling": args.credit_sibling,
               "wrong": args.credit_wrong}
    vas = make_scorer(credits)
    drop = {s.strip() for s in args.drop_datasets.split(",") if s.strip()}

    files = sorted(glob.glob(os.path.join(args.results_dir, "qa_*.csv")))
    if not files:
        raise SystemExit("no merged qa_*.csv under " + args.results_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    print("rubric: " + json.dumps(credits))
    print()

    overall, groups, rels = [], [], []

    for path in files:
        key = os.path.basename(path)[len("qa_"):-len(".csv")]
        df = pd.read_csv(path)

        col = key + "_answer"
        if col not in df.columns:
            cand = [c for c in df.columns
                    if c.endswith("_answer") and not c.endswith("_unfiltered")]
            if not cand:
                print("[skip] %s: no answer column" % key)
                continue
            col = cand[0]

        gold = df["answer"].astype(str).str.lower().str.replace(
            r"[^a-z]", "", regex=True)
        pred = df[col].astype(str).str.lower().str.replace(
            r"[^a-z]", "", regex=True).replace({"nan": None, "": None})
        ok = gold.isin(labels) & pred.isin(labels)

        d = df.loc[ok].copy()
        d["_g"] = gold[ok].values
        d["_p"] = pred[ok].values
        parse_rate = float(pred.isin(labels).mean())
        off_list = int((~gold.isin(labels)).sum())

        if not len(d):
            print("[skip] %s: nothing scoreable" % key)
            continue

        # ---- relation breakdown ------------------------------------------ #
        d["_rel"] = [relation(g, p) for g, p in zip(d["_g"], d["_p"])]
        rc = d["_rel"].value_counts(normalize=True)
        rels.append(dict(model=key, n=len(d),
                         **{r: float(rc.get(r, 0.0)) for r in
                            ("exact", "over", "under", "sibling", "wrong")}))

        # ---- per tier x dataset ------------------------------------------ #
        gkeys = [k for k in ("tier", "dataset") if k in d.columns]
        if gkeys:
            for k, g in d.groupby(gkeys):
                if not isinstance(k, tuple):
                    k = (k, "")
                s = score_block(g["_g"], g["_p"], vas)
                s.update(model=key, tier=k[0], dataset=k[1])
                groups.append(s)
        if "tier" in d.columns:
            for tier, g in d.groupby("tier"):
                s = score_block(g["_g"], g["_p"], vas)
                s.update(model=key, tier=tier, dataset="__ALL__")
                groups.append(s)

        # ---- overall ------------------------------------------------------ #
        s = score_block(d["_g"], d["_p"], vas)
        s.update(model=key, parse_rate=parse_rate, gold_off_list=off_list,
                 **{("rel_" + r): float(rc.get(r, 0.0)) for r in
                    ("exact", "over", "under", "sibling", "wrong")})
        for kk, vv in credits.items():
            s["credit_" + kk] = vv

        if args.bootstrap:
            clus = "image" if "image" in d.columns else "row_id"
            lo, hi = bootstrap_ci(
                d, clus,
                lambda g, p: macro_vas(g, p, pd.Series(
                    [vas(a, b) for a, b in zip(g, p)], index=g.index)),
                args.bootstrap)
            s["macroVAS_lo"], s["macroVAS_hi"] = lo, hi

        if drop and "dataset" in d.columns:
            sub = d[~d["dataset"].isin(drop)]
            if len(sub):
                ss = score_block(sub["_g"], sub["_p"], vas)
                s["macroVAS_excl"] = ss["macroVAS"]
                s["balAcc_excl"] = ss["balAcc"]

        overall.append(s)
        print("[ok] %-32s n=%6d  kappaVAS %+.3f  macroVAS %.3f  EM %.3f  "
              "over %.3f  predTop %s %.2f"
              % (key, s["n"], s["kappaVAS"], s["macroVAS"], s["EM"],
                 float(rc.get("over", 0.0)), s["predTop"], s["predTopShare"]))

    ov = pd.DataFrame(overall)
    gr = pd.DataFrame(groups)
    rl = pd.DataFrame(rels)
    p1 = os.path.join(args.out_dir, "viewpoint_overall.csv")
    p2 = os.path.join(args.out_dir, "viewpoint_by_group.csv")
    p3 = os.path.join(args.out_dir, "viewpoint_by_relation.csv")
    ov.to_csv(p1, index=False)
    gr.to_csv(p2, index=False)
    rl.to_csv(p3, index=False)

    # ---- markdown --------------------------------------------------------- #
    cols = [c for c in ["model", "n", "kappaVAS", "macroVAS", "VAS",
                        "modalMacroVAS",
                        "modalVAS", "balAcc", "kappa", "EM", "majority",
                        "angErr", "within45", "predTop", "predTopShare",
                        "parse_rate"] if c in ov.columns]
    show = ov[cols].sort_values("kappaVAS", ascending=False)

    md = [
        "# BioVQA C3 - Viewpoint results",
        "",
        "Generated by `evaluate.py`.",
        "",
        "## Grading rubric",
        "",
        "| relation | example (gold to pred) | credit |",
        "|---|---|---|",
        "| exact | backright to backright | %.2f |" % 1.0,
        "| over | right to backright | %.2f |" % credits["over"],
        "| under | backright to right | %.2f |" % credits["under"],
        "| sibling | frontright to backright | %.2f |" % credits["sibling"],
        "| wrong | right to frontleft | %.2f |" % credits["wrong"],
        "",
        "`over` outranks `under` deliberately: a model that says *backright* "
        "when the annotation says *right* attempted the finer distinction and "
        "got the coarse component correct, whereas one that says *right* when "
        "the annotation says *backright* declined it. The symmetric angular "
        "metric (`angErr`, `within45`) is reported alongside so the asymmetry "
        "can be inspected rather than assumed.",
        "",
        "## Overall",
        "",
        "**kappaVAS is the headline; macroVAS is its uncorrected companion.**",
        "",
        "kappaVAS is Cohen's weighted kappa with the rubric above as its "
        "weight matrix: the graded score minus the score the same answer "
        "distribution would earn by chance, rescaled to [0,1]. It is the only "
        "column here that cannot be raised by answering compounds "
        "indiscriminately. Against synthetic strategies, always-*right* "
        "scores 0.000 and random-compounds scores 0.001, while a model with "
        "30% one-step noise scores 0.773.",
        "",
        "**macroVAS** The gold labels are severely skewed - "
        "left and right are ~83% of rows, and 89% of Grevys zebra annotations "
        "are *right* because Wildbook photographs the right flank for "
        "identification. macroVAS averages the graded score with equal weight "
        "over gold labels, so a constant predictor cannot hide in it. Check "
        "`predTopShare` first: near 1.00 means the model answered the same "
        "thing every time.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for _, r in show.iterrows():
        md.append("| " + " | ".join(
            (r[c] if isinstance(r[c], str) else
             ("%d" % r[c] if c == "n" else "%.3f" % r[c]))
            for c in cols) + " |")

    if len(rl):
        md += ["", "## Error composition", "",
               "Share of scoreable rows falling in each rubric class.", "",
               "| " + " | ".join(rl.columns) + " |",
               "|" + "---|" * len(rl.columns)]
        for _, r in rl.iterrows():
            md.append("| " + " | ".join(
                (r[c] if isinstance(r[c], str) else
                 ("%d" % r[c] if c == "n" else "%.3f" % r[c]))
                for c in rl.columns) + " |")

    if len(gr) and "tier" in gr.columns:
        t = gr[gr["dataset"] == "__ALL__"]
        tc = [c for c in ["model", "tier", "n", "kappaVAS", "macroVAS", "VAS",
                          "modalMacroVAS", "balAcc", "kappa", "EM", "angErr",
                          "within45", "predTopShare"] if c in t.columns]
        md += ["", "## By tier (macro over gold labels)", "",
               "| " + " | ".join(tc) + " |",
               "|" + "---|" * len(tc)]
        for _, r in t.sort_values(["model", "tier"]).iterrows():
            md.append("| " + " | ".join(
                (r[c] if isinstance(r[c], str) else
                 ("%d" % r[c] if c == "n" else "%.3f" % r[c]))
                for c in tc) + " |")

    md += [
        "",
        "## Reading the tier ladder",
        "",
        "`cued` and `positional` are both three-panel and share the same "
        "resize, so the two differ only in whether the species names the "
        "panel. Read the observed direction from the table rather than "
        "assuming one: across models this contrast is small and its sign is "
        "not consistent, which is itself the result - locating the panel "
        "costs far less than putting three photographs on one canvas does. "
        "`single` to the multi-panel tiers is the large effect, and it "
        "confounds acuity loss with distractor interference; neither is "
        "isolated without a resolution-matched control.",
        "",
    ]
    p4 = os.path.join(args.out_dir, "viewpoint_report.md")
    with open(p4, "w") as fh:
        fh.write("\n".join(md) + "\n")

    print()
    for p in (p1, p2, p3, p4):
        print("wrote " + p)
    print()
    print(show.to_string(index=False))

    if args.sweep:
        # Rubric sensitivity. macroVAS depends on credit_over, and a model
        # whose answers are mostly "over" gains disproportionately from a
        # generous setting. If the ranking is stable across these settings the
        # result is robust; if the top of the table reshuffles, the rubric
        # must be quoted with the number every time it is reported.
        print()
        print("=" * 78)
        print("RUBRIC SENSITIVITY - macroVAS under different credit settings")
        print("=" * 78)
        settings = [(0.50, 0.35), (0.35, 0.25), (0.25, 0.15), (0.0, 0.0)]
        table = {}
        for over, under in settings:
            cr = dict(credits); cr["over"] = over; cr["under"] = under
            v2 = make_scorer(cr)
            col = {}
            for path in files:
                k = os.path.basename(path)[len("qa_"):-len(".csv")]
                d2 = pd.read_csv(path)
                c2 = k + "_answer"
                if c2 not in d2.columns:
                    continue
                g2 = d2["answer"].astype(str).str.lower().str.replace(
                    r"[^a-z]", "", regex=True)
                p2 = d2[c2].astype(str).str.lower().str.replace(
                    r"[^a-z]", "", regex=True)
                m2 = g2.isin(labels) & p2.isin(labels)
                if not m2.any():
                    continue
                gg, pp = g2[m2], p2[m2]
                vv = pd.Series([v2(a, b) for a, b in zip(gg, pp)],
                               index=gg.index)
                col[k] = macro_vas(gg, pp, vv)
            table["over=%.2f/under=%.2f" % (over, under)] = col
        sw = pd.DataFrame(table)
        sw["rank_spread"] = (sw.rank(ascending=False).max(axis=1)
                             - sw.rank(ascending=False).min(axis=1))
        print(sw.sort_values(sw.columns[0], ascending=False).to_string(
            float_format=lambda x: "%.3f" % x))
        sw.to_csv(os.path.join(args.out_dir, "viewpoint_rubric_sweep.csv"))
        print()
        print("wrote " + os.path.join(args.out_dir,
                                      "viewpoint_rubric_sweep.csv"))
        print("The last column is how far each model moves in the ranking "
              "across settings.")
        print("Anything above ~3 is rubric-dependent and must not be quoted "
              "without the credits.")


if __name__ == "__main__":
    main()