#!/usr/bin/env python3
"""
Authoritative evaluation for the Relative Size task.

  python evaluate.py --results_dir ... --out_dir ...
  python evaluate.py --bootstrap 2000

Outputs:
  relsize_overall.csv     one row per model
  relsize_by_group.csv    one row per model x measure_type
  relsize_confusion.csv   predicted-letter distribution per model
  relsize_report.md       human-readable summary

--------------------------------------------------------------------------- #
FOUR-WAY CHOICE, CHANCE 0.25

Unlike counting or morphometry there is no skew to correct for: the correct
answer is roughly uniform over A/B/C/D by construction, so plain accuracy is
meaningful and 0.25 is the floor. What can still go wrong is a model that
always answers the same letter, which lands at whatever share that letter
happens to hold. Three columns catch it:

  predTop / predTopShare   the model's own most frequent answer and its share.
                           Near 1.00 means a constant predictor; read nothing
                           else until you have checked this.
  balAcc                   recall averaged over the four gold letters. A
                           constant predictor scores exactly 0.25 here
                           whatever the gold distribution happens to be.
  kappa                    accuracy corrected for both marginals; a constant
                           predictor scores 0.

POSITION BIAS IS A PRE-REGISTERED CONTROL
Llama-3.2-Vision showed a 63% first-option preference elsewhere in this
project, so the predicted-letter distribution is reported for every model and
its distance from uniform is summarised as posBias - the total variation
distance between the predicted and the gold letter distributions. A model can
post a respectable accuracy while still being strongly biased, and the two
numbers mean different things.

CONFIDENCE INTERVALS
Bootstrap resamples PLATES. Up to twenty question sets are drawn from the same
plate, reusing the same specimens in different combinations, so rows within a
plate are far from independent. Resampling rows would give intervals that are
much too narrow.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

LETTERS = ["A", "B", "C", "D"]


def balanced_accuracy(gold, pred):
    """Recall averaged over the gold letters present. A constant predictor
    scores 1/k regardless of how the gold letters are distributed."""
    labs = [l for l in LETTERS if (gold == l).any()]
    if not labs:
        return float("nan")
    return float(np.mean([float((pred[gold == l] == l).mean()) for l in labs]))


def cohen_kappa(gold, pred):
    if len(gold) == 0:
        return float("nan")
    po = float((gold.values == pred.values).mean())
    pe = sum(float((gold == l).mean()) * float((pred == l).mean())
             for l in LETTERS)
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1.0 - pe)


def pos_bias(gold, pred):
    """Total variation distance between predicted and gold letter
    distributions. 0 means the model's answer mix matches the key's; 0.75 is
    the maximum, reached by always answering one letter when the key is
    uniform."""
    if len(gold) == 0:
        return float("nan")
    g = np.array([float((gold == l).mean()) for l in LETTERS])
    p = np.array([float((pred == l).mean()) for l in LETTERS])
    return float(0.5 * np.abs(g - p).sum())


def block(gold, pred):
    n = len(gold)
    if n == 0:
        return {k: float("nan") for k in
                ("n", "acc", "balAcc", "kappa", "posBias", "majority",
                 "predTop", "predTopShare")}
    top = pred.mode()
    top = str(top.iloc[0]) if len(top) else "-"
    return {
        "n": int(n),
        "acc": float((gold.values == pred.values).mean()),
        "balAcc": balanced_accuracy(gold, pred),
        "kappa": cohen_kappa(gold, pred),
        "posBias": pos_bias(gold, pred),
        "majority": float((gold == gold.mode().iloc[0]).mean()),
        "predTop": top,
        "predTopShare": float((pred == top).mean()),
    }


def boot(d, key, fn, n_boot, seed=0):
    """fn takes the resampled dataframe, so it can compute any statistic."""
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
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "qa_*.csv")))
    if not files:
        raise SystemExit("no merged qa_*.csv under " + args.results_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    overall, groups, conf = [], [], []

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

        gold = df["answer_letter"].astype(str).str.upper()
        pred = df[col].astype(str).str.upper()
        ok = gold.isin(LETTERS) & pred.isin(LETTERS)

        d = df.loc[ok].copy()
        d["_g"] = gold[ok].values
        d["_p"] = pred[ok].values
        parse_rate = float(pred.isin(LETTERS).mean())
        if not len(d):
            print("[skip] %s: nothing scoreable" % key)
            continue

        if "measure_type" in d.columns:
            for mt, g in d.groupby("measure_type"):
                s = block(g["_g"], g["_p"])
                s.update(model=key, measure_type=mt)
                groups.append(s)

        row = {"model": key}
        for l in LETTERS:
            row["gold_" + l] = float((d["_g"] == l).mean())
            row["pred_" + l] = float((d["_p"] == l).mean())
        conf.append(row)

        s = block(d["_g"], d["_p"])
        s.update(model=key, parse_rate=parse_rate,
                 plates=int(d["plate_id"].nunique())
                 if "plate_id" in d.columns else -1)

        if args.bootstrap:
            clus = "plate_id" if "plate_id" in d.columns else "row_id"
            lo, hi = boot(d, clus,
                          lambda x: balanced_accuracy(x["_g"], x["_p"]),
                          args.bootstrap)
            s["balAcc_lo"], s["balAcc_hi"] = lo, hi

        overall.append(s)
        print("[ok] %-32s n=%5d  acc %.3f  balAcc %.3f  kappa %+.3f  "
              "posBias %.3f  predTop %s %.2f  parse %.3f"
              % (key, s["n"], s["acc"], s["balAcc"], s["kappa"], s["posBias"],
                 s["predTop"], s["predTopShare"], parse_rate))

    ov = pd.DataFrame(overall)
    gr = pd.DataFrame(groups)
    cf = pd.DataFrame(conf)
    p1 = os.path.join(args.out_dir, "relsize_overall.csv")
    p2 = os.path.join(args.out_dir, "relsize_by_group.csv")
    p3 = os.path.join(args.out_dir, "relsize_confusion.csv")
    ov.to_csv(p1, index=False)
    gr.to_csv(p2, index=False)
    cf.to_csv(p3, index=False)

    cols = [c for c in ["model", "n", "plates", "balAcc", "balAcc_lo",
                        "balAcc_hi", "acc", "kappa", "posBias", "predTop",
                        "predTopShare", "parse_rate"] if c in ov.columns]
    show = ov[cols].sort_values("balAcc", ascending=False)

    md = [
        "# BioVQA C3 - Relative Size results",
        "",
        "Four beetles on one plate, each marked with a coloured line along its "
        "elytra; the model picks the largest. All four share one magnification "
        "and the lines are drawn before any resizing, so on-screen line length "
        "is exactly proportional to real size and no scale bar is needed. "
        "Generated by `evaluate.py`.",
        "",
        "## Reading the table",
        "",
        "Chance is **0.250**. The key is close to uniform over A/B/C/D, so "
        "plain `acc` is meaningful here - unlike the viewpoint task, there is "
        "no gold skew to correct for.",
        "",
        "What can still go wrong is a model that answers the same letter every "
        "time. **Check `predTopShare` first**: near 1.00 means a constant "
        "predictor, and `balAcc` will sit at 0.250 with `kappa` at 0 however "
        "respectable `acc` looks.",
        "",
        "`posBias` is the total variation distance between the model's answer "
        "distribution and the key's. Zero means the answer mix matches; 0.75 "
        "is the maximum. A model can post decent accuracy while still being "
        "strongly biased, and the two numbers say different things.",
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
        gc = [c for c in ["model", "measure_type", "n", "acc", "balAcc",
                          "kappa", "posBias", "predTopShare"]
              if c in gr.columns]
        md += ["", "## By measurement", "",
               "The generator required the winner to exceed the runner-up by "
               "5% on length but 10% on width, so width items have a wider "
               "margin and should be easier. A model that does better on "
               "length than width is doing something other than comparing "
               "line lengths.", "",
               "| " + " | ".join(gc) + " |", "|" + "---|" * len(gc)]
        for _, r in gr.sort_values(["model", "measure_type"]).iterrows():
            md.append("| " + " | ".join(
                (r[c] if isinstance(r[c], str) else
                 ("%d" % r[c] if c == "n" else "%.3f" % r[c]))
                for c in gc) + " |")

    if len(cf):
        cc = list(cf.columns)
        md += ["", "## Answer distribution", "",
               "`gold_X` is the share of items whose answer is X; `pred_X` is "
               "the share of the model's answers that were X. Large gaps are "
               "position bias.", "",
               "| " + " | ".join(cc) + " |", "|" + "---|" * len(cc)]
        for _, r in cf.iterrows():
            md.append("| " + " | ".join(
                (r[c] if isinstance(r[c], str) else "%.3f" % r[c])
                for c in cc) + " |")

    md += [
        "",
        "## Notes",
        "",
        "Selection bias worth stating: the generator only accepted a quartet "
        "when the winner exceeded the runner-up by a clear margin, so the "
        "items are drawn from the easier tail of within-plate spread. "
        "Measured accuracy is therefore on a easier-than-representative "
        "sample, which is what makes the task answerable at all.",
        "",
        "Confidence intervals bootstrap over PLATES, not rows: up to twenty "
        "question sets are drawn from the same plate, reusing the same "
        "specimens in different combinations, so rows within a plate are far "
        "from independent.",
        "",
        "One control worth running once results are in: replace the plate "
        "photograph with flat noise while keeping the coloured lines exactly "
        "where they are. The question is still answerable from the annotation "
        "geometry alone, so the gap between that and the real images is how "
        "much of the score comes from the specimens rather than the drawing.",
        "",
    ]
    p4 = os.path.join(args.out_dir, "relsize_report.md")
    with open(p4, "w") as fh:
        fh.write("\n".join(md) + "\n")

    print()
    for p in (p1, p2, p3, p4):
        print("wrote " + p)
    print()
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
