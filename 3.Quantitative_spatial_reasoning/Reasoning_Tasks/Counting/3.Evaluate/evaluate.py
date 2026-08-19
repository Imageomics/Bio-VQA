#!/usr/bin/env python3
"""
Authoritative evaluation for the Counting task. Reads merged per-model CSVs,
writes machine-readable results plus a markdown report.

  python evaluate.py                       # results/ next to this script
  python evaluate.py --results_dir ... --out_dir ...
  python evaluate.py --bootstrap 2000      # clustered CIs (slower)

Outputs (all written to --out_dir):
  counting_overall.csv     one row per model
  counting_by_group.csv    one row per model x tier x source_dataset
  counting_report.md       human-readable summary
  counting_predictions.csv optional long-format dump with --dump_predictions

WHY NOT EXACT MATCH ALONE
Counting is numeric, so exact match throws away almost everything: answering 3
when the truth is 47 scores the same as answering 46. It is also brutally
sensitive to scene density - nobody counts 180 bees exactly - so EM collapses
to near zero on the dense datasets and stops discriminating between models.
The graded scheme below keeps EM as a reference point and adds metrics that
still separate models where EM cannot.

  EM        exact match. Reported, never the headline.
  CAS       Count Accuracy Score: mean of max(0, 1 - |pred-gold| / max(gold,1)).
            Bounded in [0,1]. Full credit for exact, linear decay, zero once
            the error reaches the true count. Scale-relative, so being 5 out on
            10 costs far more than being 5 out on 100 - which is how a
            biologist would read it.
  acc10     within 10% of the truth. The tolerance an ecologist would accept
  acc25     within 25% of the truth.       for an abundance estimate.
  MAE       mean absolute error, in animals. Interpretable, but dominated by
            the dense datasets, so read it per dataset and never pooled.
  RMSLE     root mean squared log error. Puts a 10 vs 20 miss on the same
            footing as 100 vs 200, which MAE does not.
  bias      mean log((pred+1)/(gold+1)). Negative = systematic undercounting.
            This is a finding in its own right, not a nuisance term.
  rho       Spearman correlation between predicted and true counts. The
            shortcut-immune metric: a model that always answers the modal
            count scores rho = 0 by construction while still posting a
            respectable EM. If rho is near zero, nothing was counted.

BASELINES
  modalEM / modalCAS  always answer that group's most common count. Computed
                      on the identical row subset as the model. A result that
                      does not clear its own baseline is not a measurement.

MACRO vs MICRO
Row counts differ ten-fold across source datasets, so micro-averaging lets the
largest dataset set the headline. macro* columns average the per-dataset score
with equal weight. Report macro.

CONFIDENCE INTERVALS
Bootstrap resamples IMAGES, not rows. In the two-panel tiers one canvas carries
several questions, and in the Single tier GrevysZebra contributes three
questions per photograph; those are not independent observations, and resampling
rows would produce intervals that are far too narrow.
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def cas(gold, pred):
    """Count Accuracy Score, elementwise."""
    denom = np.maximum(np.abs(gold), 1.0)
    return np.maximum(0.0, 1.0 - np.abs(pred - gold) / denom)


def spearman(a, b):
    """Rank correlation without a scipy dependency."""
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def score_block(gold, pred):
    """All metrics for one group. gold/pred are float arrays of equal length."""
    n = len(gold)
    if n == 0:
        return {k: float("nan") for k in
                ("n", "EM", "CAS", "acc10", "acc25", "MAE", "RMSLE", "bias",
                 "rho", "modalEM", "modalCAS")}
    err = np.abs(pred - gold)
    denom = np.maximum(np.abs(gold), 1.0)
    rel = err / denom
    logr = np.log((pred + 1.0) / (gold + 1.0))

    modal = float(pd.Series(gold).mode().iloc[0])
    modal_pred = np.full_like(gold, modal)

    return {
        "n": int(n),
        "EM": float((err == 0).mean()),
        "CAS": float(cas(gold, pred).mean()),
        "acc10": float((rel <= 0.10).mean()),
        "acc25": float((rel <= 0.25).mean()),
        "MAE": float(err.mean()),
        "RMSLE": float(np.sqrt((logr ** 2).mean())),
        "bias": float(logr.mean()),
        "rho": spearman(gold, pred),
        "modalEM": float((modal_pred == gold).mean()),
        "modalCAS": float(cas(gold, modal_pred).mean()),
    }


def bootstrap_ci(df, key, metric_fn, n_boot, seed=0):
    """Percentile CI resampling clusters (images), not rows."""
    if n_boot <= 0 or not len(df):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    groups = df.groupby(key).indices
    keys = np.array(list(groups.keys()), dtype=object)
    idx_of = [groups[k] for k in keys]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        rows = np.concatenate([idx_of[i] for i in pick])
        sub = df.iloc[rows]
        vals.append(metric_fn(sub["_g"].values, sub["_p"].values))
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------------------- #

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(here, "results"),
                    help="directory of merged qa_<MODEL_KEY>.csv files")
    ap.add_argument("--out_dir", default=os.path.join(here, "results"))
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="bootstrap replicates for CIs; 0 disables")
    ap.add_argument("--dump_predictions", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "qa_*.csv")))
    files = [f for f in files if not f.endswith(".summary.json")]
    if not files:
        raise SystemExit("no merged qa_*.csv under " + args.results_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    overall_rows, group_rows, long_rows = [], [], []

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

        gold = pd.to_numeric(df["answer"], errors="coerce")
        pred = pd.to_numeric(df[col], errors="coerce")

        # A count cannot be negative. The runner's resolver accepts a leading
        # minus sign, so an output like "-1" or a hyphenated range parsed from
        # the wrong side yields a negative integer, and log((pred+1)/(gold+1))
        # then evaluates to -inf and poisons RMSLE and bias for the whole
        # model. Treat these as unresolved rather than silently clipping them.
        n_negative = int((pred < 0).sum())
        if n_negative:
            pred = pred.where(pred >= 0)

        ok = gold.notna() & pred.notna()

        d = df.loc[ok].copy()
        d["_g"] = gold[ok].astype(float).values
        d["_p"] = pred[ok].astype(float).values
        parse_rate = float(pred.notna().mean())

        if not len(d):
            print("[skip] %s: nothing scoreable" % key)
            continue

        # ---- per tier x dataset ------------------------------------------ #
        gkeys = [k for k in ("tier", "source_dataset") if k in d.columns]
        per_dataset_cas, per_dataset_em = {}, {}
        if gkeys:
            for k, g in d.groupby(gkeys):
                if not isinstance(k, tuple):
                    k = (k, "")
                s = score_block(g["_g"].values, g["_p"].values)
                s.update(model=key, tier=k[0], source_dataset=k[1])
                group_rows.append(s)
                per_dataset_cas.setdefault(k[0], []).append(s["CAS"])
                per_dataset_em.setdefault(k[0], []).append(s["EM"])

        # ---- per tier, macro over datasets ------------------------------- #
        if "tier" in d.columns:
            for tier, g in d.groupby("tier"):
                s = score_block(g["_g"].values, g["_p"].values)
                s.update(model=key, tier=tier, source_dataset="__ALL__",
                         macroCAS=float(np.nanmean(per_dataset_cas.get(tier, [np.nan]))),
                         macroEM=float(np.nanmean(per_dataset_em.get(tier, [np.nan]))))
                group_rows.append(s)

        # ---- overall ------------------------------------------------------ #
        s = score_block(d["_g"].values, d["_p"].values)
        allc = [v for vs in per_dataset_cas.values() for v in vs]
        alle = [v for vs in per_dataset_em.values() for v in vs]
        s.update(model=key, parse_rate=parse_rate, negative_preds=n_negative,
                 macroCAS=float(np.nanmean(allc)) if allc else float("nan"),
                 macroEM=float(np.nanmean(alle)) if alle else float("nan"))

        if args.bootstrap:
            clus = "image" if "image" in d.columns else "row_id"
            lo, hi = bootstrap_ci(d, clus,
                                  lambda g, p: float(cas(g, p).mean()),
                                  args.bootstrap)
            s["CAS_lo"], s["CAS_hi"] = lo, hi

        overall_rows.append(s)

        if args.dump_predictions:
            keep = [c for c in ("row_id", "tier", "source_dataset", "panel",
                                "image", "question", "answer") if c in d.columns]
            L = d[keep].copy()
            L["model"] = key
            L["pred"] = d["_p"].values
            L["raw"] = d.get(key + "_answer_unfiltered", pd.NA)
            long_rows.append(L)

        print("[ok] %-32s n=%6d  CAS %.3f  EM %.3f  rho %+.3f  parse %.3f%s"
              % (key, s["n"], s["CAS"], s["EM"], s["rho"], parse_rate,
                 ("  NEGATIVE=%d" % n_negative) if n_negative else ""))

    ov = pd.DataFrame(overall_rows)
    gr = pd.DataFrame(group_rows)
    ov_path = os.path.join(args.out_dir, "counting_overall.csv")
    gr_path = os.path.join(args.out_dir, "counting_by_group.csv")
    ov.to_csv(ov_path, index=False)
    gr.to_csv(gr_path, index=False)

    if long_rows:
        p = os.path.join(args.out_dir, "counting_predictions.csv")
        pd.concat(long_rows, ignore_index=True).to_csv(p, index=False)
        print("wrote " + p)

    # ---- markdown report -------------------------------------------------- #
    md = [
        "# BioVQA C3 - Counting results",
        "",
        "Generated by `evaluate.py`. `counting_overall.csv` and "
        "`counting_by_group.csv` hold the same numbers in machine-readable "
        "form.",
        "",
        "Headline metric is **macroCAS**: the Count Accuracy Score averaged "
        "with equal weight over the source datasets. `modalCAS` is the same "
        "score for a model that always answers the group's most common count "
        "- the floor any result must clear. `rho` is the Spearman correlation "
        "between predicted and true counts; a model that is not counting at "
        "all scores near zero there whatever its EM.",
        "",
        "## Overall",
        "",
    ]
    cols = ["model", "n", "macroCAS", "CAS", "modalCAS", "macroEM", "EM",
            "modalEM", "acc10", "acc25", "RMSLE", "bias", "rho", "parse_rate"]
    cols = [c for c in cols if c in ov.columns]
    show = ov[cols].sort_values("macroCAS", ascending=False)
    md.append("| " + " | ".join(cols) + " |")
    md.append("|" + "---|" * len(cols))
    for _, r in show.iterrows():
        md.append("| " + " | ".join(
            (r[c] if isinstance(r[c], str) else
             ("%d" % r[c] if c == "n" else "%.3f" % r[c]))
            for c in cols) + " |")

    if "tier" in gr.columns:
        md += ["", "## By tier (macro over source datasets)", ""]
        t = gr[gr["source_dataset"] == "__ALL__"]
        tc = [c for c in ["model", "tier", "n", "macroCAS", "CAS", "modalCAS",
                          "EM", "modalEM", "rho"] if c in t.columns]
        md.append("| " + " | ".join(tc) + " |")
        md.append("|" + "---|" * len(tc))
        for _, r in t.sort_values(["model", "tier"]).iterrows():
            md.append("| " + " | ".join(
                (r[c] if isinstance(r[c], str) else
                 ("%d" % r[c] if c == "n" else "%.3f" % r[c]))
                for c in tc) + " |")

    md += [
        "",
        "## Reading the tier ladder",
        "",
        "`cued` (two different species) and `positional` (the same species "
        "twice) are both two-panel and go through the same resize, so they "
        "differ only in whether the species names the target panel. Read the "
        "observed direction from the table rather than assuming one: across "
        "models this contrast is small and its sign is not consistent, which "
        "is itself the result - locating the panel costs far less than "
        "putting two photographs on one canvas does.",
        "",
        "One caveat on that contrast: in `cued` the two panels come from "
        "different datasets, so their true counts are usually far apart, "
        "while in `positional` both come from the same dataset and the two "
        "counts are close. A model that ignores the panel instruction "
        "entirely therefore lands nearer the right answer in `positional`. "
        "The within-tier panel-confusion rate - how often a prediction is "
        "closer to the OTHER panel's count than to its own - measures the "
        "same thing without that asymmetry, and both panels' gold counts are "
        "already in the merged CSV.",
        "",
        "`single` to the two-panel tiers is the large and consistent effect, "
        "but it confounds acuity loss with distractor interference: two "
        "photographs on one canvas under a fixed pixel budget leaves each "
        "panel roughly 0.71x its linear resolution. Neither component is "
        "isolated without a resolution-matched control.",
        "",
        "`bias` is the mean log ratio of predicted to true count. A negative "
        "value means systematic undercounting; check whether it grows with "
        "scene density, which would indicate an acuity limit rather than a "
        "reasoning one.",
        "",
    ]
    md_path = os.path.join(args.out_dir, "counting_report.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(md) + "\n")

    print()
    print("wrote " + ov_path)
    print("wrote " + gr_path)
    print("wrote " + md_path)
    print()
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()