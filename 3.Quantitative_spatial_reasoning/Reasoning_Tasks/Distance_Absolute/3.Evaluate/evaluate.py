#!/usr/bin/env python3
"""
Authoritative evaluation for the Morphometry task.

  python evaluate.py --results_dir ... --out_dir ...
  python evaluate.py --bootstrap 2000

Outputs:
  morphometry_overall.csv    one row per model
  morphometry_by_group.csv   one row per model x measure_type
  morphometry_report.md      human-readable summary

--------------------------------------------------------------------------- #
WHY rho IS THE HEADLINE AND RELATIVE ERROR IS NOT

The answer is a physical length in centimetres, and beetle elytra occupy a
narrow band. Two strategies reach a respectable relative error without
measuring anything in the image:

  per-type constant   answer one fixed number for every length question and
                another for every width question. The question states which is
                wanted, so this needs no measurement at all - it is the
                prior-driven shortcut in its strongest form. Fitted here from
                the gold data, on the identical rows as the model, and against
                whichever metric it is being compared to.

Rank correlation defeats it, but only when computed WITHIN each measurement
type. See rho_within below.

  relErr    mean |pred - gold| / gold
  acc10/20  share within 10% / 20% of the truth
  rho       Spearman correlation of predicted against true length
  bias      mean log(pred / gold). Negative means systematic under-measurement
  logRMSE   root mean squared log error, scale-free

CONFIDENCE INTERVALS
Bootstrap resamples PLATES. Every beetle on a plate shares one scale bar and
one magnification, and each contributes a length and a width item, so rows
within a plate are close to a single observation. Resampling rows would give
intervals several times too narrow.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean()
                 / (ra.std() * rb.std()))


def best_constant_relerr(gold):
    """The single value minimising mean relative error over these rows."""
    grid = np.linspace(float(np.min(gold)), float(np.max(gold)), 1500)
    err = np.array([np.abs(c / gold - 1.0).mean() for c in grid])
    return float(grid[int(err.argmin())])


def best_constant_acc(gold, tol=0.20):
    """The single value maximising accuracy within tol.

    Not the same constant as the one minimising mean relative error, and the
    gap is large: on this data the relErr-optimal constant scores acc@20 of
    0.30 on length while the acc-optimal constant scores 0.43. A baseline must
    optimise whatever metric it is being compared against, or it flatters the
    models instead of bounding them."""
    grid = np.linspace(float(np.min(gold)), float(np.max(gold)), 1500)
    acc = np.array([(np.abs(c / gold - 1.0) <= tol).mean() for c in grid])
    return float(grid[int(acc.argmax())])


def calibrated(d):
    """Per-type bias removal, then rescore.

    A model can rank the specimens correctly and still miss every tolerance
    band if it systematically over- or under-measures. Dividing each
    prediction by exp(mean log(pred/gold)) WITHIN its measurement type removes
    exactly that offset and leaves only the scatter, so the gap between acc20
    and acc20_calib separates "cannot measure" from "measures but does not
    calibrate". The correction is fitted on the same rows it is applied to, so
    acc20_calib is an optimistic ceiling, not a score a model earned - it is
    reported as a diagnostic and must be labelled as one.

    Also returns the range-compression index: bias(width) - bias(length).
    Positive means the model overestimates the smaller quantity and
    underestimates the larger one, i.e. squeezes every answer toward one
    central value. That is the signature of answering from a size prior rather
    than from the image."""
    if "measure_type" not in d.columns:
        return float("nan"), float("nan"), float("nan")
    accs, ns, bias = [], [], {}
    for mt, g in d.groupby("measure_type"):
        gold, pred = g["_g"].values, g["_p"].values
        b = float(np.log(np.maximum(pred, 1e-9) / gold).mean())
        bias[mt] = b
        adj = pred / np.exp(b)
        accs.append(float((np.abs(adj - gold) / gold <= 0.20).mean()))
        ns.append(len(g))
    acc = float(np.average(accs, weights=ns)) if ns else float("nan")
    spread = (bias.get("width", float("nan"))
              - bias.get("length", float("nan")))
    return acc, spread, float(np.mean(list(bias.values())))


def rho_within(d):
    """Spearman computed WITHIN each measurement type, then averaged.

    Pooled rho is badly inflated here. Elytra length is roughly twice elytra
    width, and the question says which is being asked, so a model that
    measures nothing at all and merely answers a larger number for length than
    for width scores a pooled rho of +0.63. The same predictor scores 0 within
    each type, which is the truth. Always read rho_within."""
    if "measure_type" not in d.columns:
        return spearman(d["_g"].values, d["_p"].values)
    rs = [spearman(g["_g"].values, g["_p"].values)
          for _, g in d.groupby("measure_type")]
    rs = [r for r in rs if r == r]
    return sum(rs) / len(rs) if rs else float("nan")


def block(gold, pred):
    n = len(gold)
    if n == 0:
        return {k: float("nan") for k in
                ("n", "relErr", "acc10", "acc20", "acc25", "rho", "bias",
                 "logRMSE")}
    rel = np.abs(pred - gold) / gold
    lg = np.log(np.maximum(pred, 1e-9) / gold)
    return {
        "n": int(n),
        "relErr": float(rel.mean()),
        "acc10": float((rel <= 0.10).mean()),
        "acc20": float((rel <= 0.20).mean()),
        "acc25": float((rel <= 0.25).mean()),
        "rho": spearman(gold, pred),
        "bias": float(lg.mean()),
        "logRMSE": float(np.sqrt((lg ** 2).mean())),
    }


def baselines(d):
    """Shortcut scores on the same rows the model was scored on.

    The constant is fitted PER MEASUREMENT TYPE, because the question states
    whether length or width is wanted. "Answer 0.99 for every length question
    and 0.53 for every width question" is the actual prior-driven shortcut,
    and it is a far higher floor than a single pooled constant: acc@20 of
    about 0.42 rather than 0.30. Each constant is fitted against the metric it
    is reported against."""
    gold = d["_g"].values
    out = {}

    if "measure_type" in d.columns:
        parts = [(mt, g["_g"].values) for mt, g in d.groupby("measure_type")]
    else:
        parts = [("all", gold)]

    n = sum(len(g) for _, g in parts)
    rel_w, acc_w, shown, cpred = 0.0, 0.0, [], np.empty(0)
    for mt, g in parts:
        c_r = best_constant_relerr(g)
        c_a = best_constant_acc(g, 0.20)
        rel_w += float(np.abs(c_r / g - 1.0).mean()) * len(g)
        acc_w += float((np.abs(c_a / g - 1.0) <= 0.20).mean()) * len(g)
        shown.append("%s:%.2f" % (mt, c_a))
    out.update(const_relErr=rel_w / n, const_acc20=acc_w / n,
               const_values="|".join(shown),
               # a per-type constant is constant within each type, so its
               # within-type rank correlation is undefined - i.e. zero signal
               const_rho_within=0.0)

    return out


def boot(d, key, fn, n_boot, seed=0):
    """fn takes the resampled DATAFRAME, so it can compute within-type
    statistics rather than only pooled ones."""
    if n_boot <= 0 or not len(d):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    groups = d.groupby(key).indices
    ks = list(groups.keys())
    idx = [groups[k] for k in ks]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ks), len(ks))
        s = d.iloc[np.concatenate([idx[i] for i in pick])]
        vals.append(fn(s))
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(here, "results"))
    ap.add_argument("--out_dir", default=os.path.join(here, "results"))
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--max_cm", type=float, default=20.0,
                    help="answers above this are implausible for an elytron "
                         "and are treated as unresolved")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "qa_*.csv")))
    if not files:
        raise SystemExit("no merged qa_*.csv under " + args.results_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    overall, groups = [], []

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

        gold = pd.to_numeric(df["answer_cm"], errors="coerce")
        pred = pd.to_numeric(df[col], errors="coerce")

        # A length must be positive and physically plausible. Values outside
        # that range are parser artefacts, not measurements, and would
        # dominate the log-scale metrics if kept.
        n_bad = int(((pred <= 0) | (pred > args.max_cm)).sum())
        pred = pred.where((pred > 0) & (pred <= args.max_cm))

        ok = gold.notna() & pred.notna() & (gold > 0)
        d = df.loc[ok].copy()
        d["_g"] = gold[ok].astype(float).values
        d["_p"] = pred[ok].astype(float).values
        if not len(d):
            print("[skip] %s: nothing scoreable" % key)
            continue

        for mt, g in d.groupby("measure_type") if "measure_type" in d.columns \
                else []:
            s = block(g["_g"].values, g["_p"].values)
            s.update(model=key, measure_type=mt, **baselines(g))
            groups.append(s)

        s = block(d["_g"].values, d["_p"].values)
        s["rho_within"] = rho_within(d)
        ac, spread, _ = calibrated(d)
        s["acc20_calib"] = ac
        s["compression"] = spread
        s = {("rho_pooled" if k == "rho" else k): v for k, v in s.items()}
        s.update(model=key,
                 parse_rate=float(pred.notna().mean()),
                 implausible=n_bad,
                 plates=int(d["plate_id"].nunique())
                 if "plate_id" in d.columns else -1,
                 **baselines(d))

        if args.bootstrap:
            clus = "plate_id" if "plate_id" in d.columns else "row_id"
            lo, hi = boot(d, clus, rho_within, args.bootstrap)
            s["rho_within_lo"], s["rho_within_hi"] = lo, hi

        overall.append(s)
        print("[ok] %-32s n=%5d  rhoWithin %+.3f  acc20 %.3f "
              "(calib %.3f, floor %.3f)  compression %+.3f  parse %.3f%s"
              % (key, s["n"], s["rho_within"], s["acc20"], s["acc20_calib"],
                 s["const_acc20"], s["compression"], s["parse_rate"],
                 ("  IMPLAUSIBLE=%d" % n_bad) if n_bad else ""))

    ov = pd.DataFrame(overall)
    gr = pd.DataFrame(groups)
    p1 = os.path.join(args.out_dir, "morphometry_overall.csv")
    p2 = os.path.join(args.out_dir, "morphometry_by_group.csv")
    ov.to_csv(p1, index=False)
    gr.to_csv(p2, index=False)

    cols = [c for c in ["model", "n", "plates", "rho_within",
                        "rho_within_lo", "rho_within_hi", "rho_pooled",
                        "relErr", "acc10", "acc20", "acc20_calib",
                        "const_acc20", "compression", "const_relErr",
                        "const_values", "bias", "logRMSE", "parse_rate"]
            if c in ov.columns]
    show = ov[cols].sort_values("rho_within", ascending=False)

    md = [
        "# BioVQA C3 - Morphometry results",
        "",
        "Scale-bar referenced absolute measurement: the model must compare a "
        "red line against a blue 1 cm scale bar and report the red line's "
        "length in centimetres. Generated by `evaluate.py`.",
        "",
        "## Why rho leads",
        "",
        "Beetle elytra occupy a narrow size band, so relative error alone is "
        "satisfiable without measuring anything. The shortcut baseline is "
        "computed on the identical rows as each model:",
        "",
        "- **per-type constant** - answer one fixed number for every length "
        "question and another for every width question. `const_values` lists "
        "the two.",
        "",
        "The constant is fitted **per measurement type**, because the "
        "question states whether length or width is wanted. Answering one "
        "fixed number for every length question and another for every width "
        "question is the real prior-driven shortcut, and it reaches acc@20 of "
        "roughly 0.42 - far above what a single pooled constant would "
        "suggest. Each constant is also fitted against the metric it is "
        "compared to, since the value minimising mean relative error is not "
        "the value maximising acc@20.",
        "",
        "### rho_within, not rho_pooled",
        "",
        "Elytra length is about twice elytra width and the question says "
        "which is being asked, so a model that measures nothing and merely "
        "answers a larger number for length than for width scores a **pooled "
        "rho of +0.63**. The same predictor scores 0 within each type. "
        "`rho_within` is Spearman computed inside each measurement type and "
        "then averaged; `rho_pooled` is reported only so the gap between them "
        "is visible. **Read rho_within.**",
        "",
        "`rho_within` is the column that says whether anything in the image "
        "was measured at all; `relErr` and `acc20` say how precisely.",
        "",
        "### Ranking without calibration",
        "",
        "These two questions come apart sharply in this task, and the split is "
        "the main result. `acc20_calib` rescores each model after removing its "
        "own per-type bias - dividing every prediction by exp(mean "
        "log(pred/gold)) within length and within width. Because the "
        "correction is fitted on the rows it is applied to, it is an "
        "optimistic ceiling and a **diagnostic, not a score**: it says how "
        "well a model would do if someone told it the right scale factor.",
        "",
        "`compression` is bias(width) minus bias(length). Positive means the "
        "model overestimates the smaller quantity and underestimates the "
        "larger one - squeezing every answer toward a single central value, "
        "which is what answering from a remembered beetle size looks like. A "
        "model that reads the scale bar has no reason to be biased in "
        "opposite directions on the two measurements.",
        "",
        "One limitation to state in the paper: without per-plate pixel "
        "coordinates we cannot compute the sharper control - a predictor that "
        "reads the red line's apparent pixel length but never looks at the "
        "blue scale bar. `rho_within` still bounds prior-driven answering, "
        "but it does not by itself prove the scale bar was consulted rather "
        "than apparent size alone.",
        "",
        "## Overall",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for _, r in show.iterrows():
        md.append("| " + " | ".join(
            (r[c] if isinstance(r[c], str) else
             ("%d" % r[c] if c in ("n", "plates") else "%.3f" % r[c]))
            for c in cols) + " |")

    if len(gr):
        gc = [c for c in ["model", "measure_type", "n", "rho", "relErr",
                          "acc20", "bias", "const_acc20"] if c in gr.columns]
        md += ["", "## By measurement", "",
               "Width is roughly half the length, so a model anchored on a "
               "remembered beetle size will show a much larger bias on one "
               "than the other.", "",
               "| " + " | ".join(gc) + " |", "|" + "---|" * len(gc)]
        for _, r in gr.sort_values(["model", "measure_type"]).iterrows():
            md.append("| " + " | ".join(
                (r[c] if isinstance(r[c], str) else
                 ("%d" % r[c] if c == "n" else "%.3f" % r[c]))
                for c in gc) + " |")

    md += [
        "",
        "## Notes",
        "",
        "`bias` is the mean log ratio of predicted to true length. Negative "
        "means systematic under-measurement.",
        "",
        "Confidence intervals bootstrap over PLATES, not rows. Every beetle "
        "on a plate shares one scale bar and one magnification and "
        "contributes both a length and a width item, so rows within a plate "
        "are close to a single observation; resampling rows would give "
        "intervals several times too narrow.",
        "",
    ]
    p3 = os.path.join(args.out_dir, "morphometry_report.md")
    with open(p3, "w") as fh:
        fh.write("\n".join(md) + "\n")

    print()
    for p in (p1, p2, p3):
        print("wrote " + p)
    print()
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()