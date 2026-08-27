# #!/usr/bin/env python3
# """
# Authoritative evaluation for the Counting-Relative task.

#   python evaluate.py --results_dir ... --out_dir ...
#   python evaluate.py --bootstrap 2000
#   python evaluate.py --prior_different_most 0.530 --prior_different_least 0.548

# Outputs:
#   relcount_overall.csv     one row per model
#   relcount_by_group.csv    one row per model x condition x polarity
#   relcount_confusion.csv   predicted-letter distribution per model
#   relcount_report.md       human-readable summary

# --------------------------------------------------------------------------- #
# TWO CONDITIONS WITH DIFFERENT FLOORS

# same       four panels from ONE dataset, and the question names that animal.
#            Nothing but the counts distinguishes the panels, so the floor is
#            chance: 0.250.

# different  four panels from four DIFFERENT datasets, and the question says
#            "animals". Count magnitude is partly predictable from which dataset
#            a panel came from - Bees average 37 per frame while Camel and Kiang
#            average about 8 - so a model that recognises the domain can answer
#            without counting. A Monte-Carlo over the empirical count
#            distributions puts that strategy at roughly 0.53 for most and 0.55
#            for fewest. Those are the --prior_different_* defaults; they are
#            arguments rather than constants because they are a property of the
#            corpus, and a regenerated corpus would change them.

# `lift` is accuracy minus the applicable floor. For the different condition,
# lift - not accuracy - is the number that says whether anything was counted.

# WHAT ELSE CAN GO WRONG

#   predTop / predTopShare  a model that answers one letter every time lands at
#            whatever share that letter holds. Check this column first.
#   balAcc   recall averaged over the four gold letters. A constant predictor
#            scores exactly 0.250 whatever the key looks like.
#   kappa    accuracy corrected for both marginals; 0 for a constant predictor.
#   posBias  total variation distance between the model's answer distribution
#            and the key's. Position bias is a pre-registered control here:
#            Llama-3.2-Vision showed a 63% first-option preference elsewhere in
#            this project, and several models never emit D at all.
#   incoherence  every image carries both a most and a least question, and they
#            cannot share an answer. The share of images where a model gives the
#            same letter to both is a free consistency check that needs no
#            ground truth at all.

# CONFIDENCE INTERVALS
# Bootstrap resamples IMAGES, not rows: each grid contributes two questions.
# """

# import argparse
# import glob
# import os

# import numpy as np
# import pandas as pd

# LETTERS = ["A", "B", "C", "D"]


# def balanced_accuracy(gold, pred):
#     labs = [l for l in LETTERS if (gold == l).any()]
#     if not labs:
#         return float("nan")
#     return float(np.mean([float((pred[gold == l] == l).mean()) for l in labs]))


# def cohen_kappa(gold, pred):
#     if len(gold) == 0:
#         return float("nan")
#     po = float((gold.values == pred.values).mean())
#     pe = sum(float((gold == l).mean()) * float((pred == l).mean())
#              for l in LETTERS)
#     return (po - pe) / (1.0 - pe) if pe < 1.0 else float("nan")


# def pos_bias(gold, pred):
#     if len(gold) == 0:
#         return float("nan")
#     g = np.array([float((gold == l).mean()) for l in LETTERS])
#     p = np.array([float((pred == l).mean()) for l in LETTERS])
#     return float(0.5 * np.abs(g - p).sum())


# def block(gold, pred):
#     n = len(gold)
#     if n == 0:
#         return {k: float("nan") for k in
#                 ("n", "acc", "balAcc", "kappa", "posBias", "predTop",
#                  "predTopShare")}
#     top = pred.mode()
#     top = str(top.iloc[0]) if len(top) else "-"
#     return {
#         "n": int(n),
#         "acc": float((gold.values == pred.values).mean()),
#         "balAcc": balanced_accuracy(gold, pred),
#         "kappa": cohen_kappa(gold, pred),
#         "posBias": pos_bias(gold, pred),
#         "predTop": top,
#         "predTopShare": float((pred == top).mean()),
#     }


# def incoherence(d):
#     """Share of images where the model gave the SAME letter to the most and
#     the least question. Those cannot both be right, and the check needs no
#     ground truth."""
#     if "polarity" not in d.columns:
#         return float("nan")
#     piv = d.pivot_table(index="image", columns="polarity", values="_p",
#                         aggfunc="first")
#     if not {"most", "least"} <= set(piv.columns):
#         return float("nan")
#     both = piv.dropna(subset=["most", "least"])
#     if not len(both):
#         return float("nan")
#     return float((both["most"] == both["least"]).mean())


# def boot(d, key, fn, n_boot, seed=0):
#     if n_boot <= 0 or not len(d):
#         return float("nan"), float("nan")
#     rng = np.random.default_rng(seed)
#     groups = d.groupby(key).indices
#     ks = list(groups.keys())
#     idx = [groups[k] for k in ks]
#     vals = []
#     for _ in range(n_boot):
#         pick = rng.integers(0, len(ks), len(ks))
#         s = d.iloc[np.concatenate([idx[i] for i in pick])]
#         vals.append(fn(s))
#     vals = [v for v in vals if v == v]
#     if not vals:
#         return float("nan"), float("nan")
#     return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# def main():
#     here = os.path.dirname(os.path.abspath(__file__))
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--results_dir", default=os.path.join(here, "results"))
#     ap.add_argument("--out_dir", default=os.path.join(here, "results"))
#     ap.add_argument("--bootstrap", type=int, default=0)
#     ap.add_argument("--prior_different_most", type=float, default=0.530,
#                     help="dataset-prior baseline for different/most")
#     ap.add_argument("--prior_different_least", type=float, default=0.548,
#                     help="dataset-prior baseline for different/least")
#     args = ap.parse_args()

#     floors = {("same", "most"): 0.25, ("same", "least"): 0.25,
#               ("different", "most"): args.prior_different_most,
#               ("different", "least"): args.prior_different_least}

#     files = sorted(glob.glob(os.path.join(args.results_dir, "qa_*.csv")))
#     if not files:
#         raise SystemExit("no merged qa_*.csv under " + args.results_dir)
#     os.makedirs(args.out_dir, exist_ok=True)

#     overall, groups, conf = [], [], []

#     for path in files:
#         key = os.path.basename(path)[len("qa_"):-len(".csv")]
#         df = pd.read_csv(path)
#         col = key + "_answer"
#         if col not in df.columns:
#             cand = [c for c in df.columns
#                     if c.endswith("_answer") and not c.endswith("_unfiltered")]
#             if not cand:
#                 print("[skip] %s: no answer column" % key)
#                 continue
#             col = cand[0]

#         gold = df["answer_letter"].astype(str).str.upper()
#         pred = df[col].astype(str).str.upper()
#         ok = gold.isin(LETTERS) & pred.isin(LETTERS)

#         d = df.loc[ok].copy()
#         d["_g"] = gold[ok].values
#         d["_p"] = pred[ok].values
#         parse_rate = float(pred.isin(LETTERS).mean())
#         if not len(d):
#             print("[skip] %s: nothing scoreable" % key)
#             continue

#         gkeys = [k for k in ("condition", "polarity") if k in d.columns]
#         if gkeys:
#             for k, g in d.groupby(gkeys):
#                 if not isinstance(k, tuple):
#                     k = (k, "")
#                 s = block(g["_g"], g["_p"])
#                 fl = floors.get((k[0], k[1]), 0.25)
#                 s.update(model=key, condition=k[0], polarity=k[1],
#                          floor=fl, lift=s["acc"] - fl)
#                 groups.append(s)

#         row = {"model": key}
#         for l in LETTERS:
#             row["gold_" + l] = float((d["_g"] == l).mean())
#             row["pred_" + l] = float((d["_p"] == l).mean())
#         conf.append(row)

#         s = block(d["_g"], d["_p"])
#         # weighted floor over whichever cells this model actually has
#         if gkeys:
#             w = d.groupby(gkeys).size()
#             tot = float(w.sum())
#             fl = sum(floors.get(k if isinstance(k, tuple) else (k, ""), 0.25)
#                      * v for k, v in w.items()) / tot
#         else:
#             fl = 0.25
#         s.update(model=key, parse_rate=parse_rate, floor=fl,
#                  lift=s["acc"] - fl, incoherence=incoherence(d),
#                  images=int(d["image"].nunique()) if "image" in d.columns else -1)

#         if args.bootstrap:
#             clus = "image" if "image" in d.columns else "row_id"
#             lo, hi = boot(d, clus,
#                           lambda x: balanced_accuracy(x["_g"], x["_p"]),
#                           args.bootstrap)
#             s["balAcc_lo"], s["balAcc_hi"] = lo, hi

#         overall.append(s)
#         print("[ok] %-32s n=%5d  acc %.3f  lift %+.3f  balAcc %.3f  "
#               "kappa %+.3f  posBias %.3f  incoh %.3f  predTop %s %.2f"
#               % (key, s["n"], s["acc"], s["lift"], s["balAcc"], s["kappa"],
#                  s["posBias"], s["incoherence"], s["predTop"],
#                  s["predTopShare"]))

#     ov = pd.DataFrame(overall)
#     gr = pd.DataFrame(groups)
#     cf = pd.DataFrame(conf)
#     p1 = os.path.join(args.out_dir, "relcount_overall.csv")
#     p2 = os.path.join(args.out_dir, "relcount_by_group.csv")
#     p3 = os.path.join(args.out_dir, "relcount_confusion.csv")
#     ov.to_csv(p1, index=False)
#     gr.to_csv(p2, index=False)
#     cf.to_csv(p3, index=False)

#     cols = [c for c in ["model", "n", "images", "balAcc", "balAcc_lo",
#                         "balAcc_hi", "acc", "floor", "lift", "kappa",
#                         "posBias", "incoherence", "predTop", "predTopShare",
#                         "parse_rate"] if c in ov.columns]
#     show = ov[cols].sort_values("balAcc", ascending=False)

#     md = [
#         "# BioVQA C3 - Counting-Relative results",
#         "",
#         "A 2x2 grid of four photographs; the model picks the quadrant holding "
#         "the most, or the fewest, animals. Generated by `evaluate.py`.",
#         "",
#         "## Two conditions, two floors",
#         "",
#         "| condition | panels | floor |",
#         "|---|---|---|",
#         "| `same` | four from ONE dataset, question names the animal | 0.250 |",
#         ("| `different` | four from four DIFFERENT datasets, question says "
#          "'animals' | %.3f most / %.3f least |"
#          % (args.prior_different_most, args.prior_different_least)),
#         "",
#         "In the `different` condition, count magnitude is partly predictable "
#         "from which dataset a panel came from - Bees average about 37 animals "
#         "per frame while Camel and Kiang average about 8 - so a model that "
#         "recognises the domain can answer without counting. The floors above "
#         "come from a Monte-Carlo over the empirical count distributions using "
#         "the generator's own selection rule. **`lift` is accuracy minus the "
#         "applicable floor, and for `different` it is lift, not accuracy, that "
#         "says whether anything was counted.**",
#         "",
#         "The `same` condition has no such shortcut, which makes it the clean "
#         "measurement of relative enumeration; `different` measures whether a "
#         "model falls back on domain recognition when that shortcut exists. "
#         "Comparing a model's lift across the two is the informative contrast.",
#         "",
#         "## Overall",
#         "",
#         "**Check `predTopShare` first.** Near 1.00 means the model answered "
#         "the same letter every time; `balAcc` then sits at 0.250 and `kappa` "
#         "at 0 however respectable `acc` looks.",
#         "",
#         "`incoherence` is the share of grids where the model gave the SAME "
#         "letter to the most and the least question. Those cannot both be "
#         "correct, and the check needs no ground truth - a high value is a "
#         "direct demonstration that the model is not comparing anything.",
#         "",
#         "| " + " | ".join(cols) + " |",
#         "|" + "---|" * len(cols),
#     ]
#     for _, r in show.iterrows():
#         md.append("| " + " | ".join(
#             (r[c] if isinstance(r[c], str) else
#              ("%d" % r[c] if c in ("n", "images") else "%.3f" % r[c]))
#             for c in cols) + " |")

#     if len(gr):
#         gc = [c for c in ["model", "condition", "polarity", "n", "acc",
#                           "floor", "lift", "balAcc", "kappa", "posBias",
#                           "predTopShare"] if c in gr.columns]
#         md += ["", "## By condition and polarity", "",
#                "| " + " | ".join(gc) + " |", "|" + "---|" * len(gc)]
#         for _, r in gr.sort_values(["model", "condition", "polarity"]).iterrows():
#             md.append("| " + " | ".join(
#                 (r[c] if isinstance(r[c], str) else
#                  ("%d" % r[c] if c == "n" else "%.3f" % r[c]))
#                 for c in gc) + " |")

#     if len(cf):
#         cc = list(cf.columns)
#         md += ["", "## Answer distribution", "",
#                "`gold_X` is the share of items whose answer is X; `pred_X` is "
#                "the share of the model's answers that were X. Large gaps are "
#                "position bias; a `pred_D` near zero means the model never "
#                "chose the last option.", "",
#                "| " + " | ".join(cc) + " |", "|" + "---|" * len(cc)]
#         for _, r in cf.iterrows():
#             md.append("| " + " | ".join(
#                 (r[c] if isinstance(r[c], str) else "%.3f" % r[c])
#                 for c in cc) + " |")

#     md += [
#         "",
#         "## Notes",
#         "",
#         "The four panels are pasted at different magnifications, so apparent "
#         "animal size does not track count across panels. The prompt says so "
#         "explicitly, which removes one interpretation of an otherwise "
#         "ambiguous comparison.",
#         "",
#         "Confidence intervals bootstrap over IMAGES, not rows: each grid "
#         "carries both a most and a least question, so rows come in dependent "
#         "pairs.",
#         "",
#         "The `different` floors are command-line arguments, not constants, "
#         "because they are a property of this corpus. Regenerating the grids "
#         "with a different dataset mix would change them, and the value used "
#         "is recorded in `relcount_overall.csv` alongside every score.",
#         "",
#     ]
#     p4 = os.path.join(args.out_dir, "relcount_report.md")
#     with open(p4, "w") as fh:
#         fh.write("\n".join(md) + "\n")

#     print()
#     for p in (p1, p2, p3, p4):
#         print("wrote " + p)
#     print()
#     print(show.to_string(index=False))


# if __name__ == "__main__":
#     main()






#!/usr/bin/env python3
"""
Authoritative evaluation for the Counting-Relative task.

  python evaluate.py --results_dir ... --out_dir ...
  python evaluate.py --bootstrap 2000
  python evaluate.py --prior_different_most 0.530 --prior_different_least 0.548

Outputs:
  relcount_overall.csv     one row per model
  relcount_by_group.csv    one row per model x condition x polarity
  relcount_confusion.csv   predicted-letter distribution per model
  relcount_report.md       human-readable summary

--------------------------------------------------------------------------- #
TWO CONDITIONS WITH DIFFERENT FLOORS

same       four panels from ONE dataset, and the question names that animal.
           Nothing but the counts distinguishes the panels, so the floor is
           chance: 0.250.

different  four panels from four DIFFERENT datasets, and the question says
           "animals". Count magnitude is partly predictable from which dataset
           a panel came from - Bees average 37 per frame while Camel and Kiang
           average about 8 - so a model that recognises the domain can answer
           without counting. A Monte-Carlo over the empirical count
           distributions puts that strategy at roughly 0.53 for most and 0.55
           for fewest. Those are the --prior_different_* defaults; they are
           arguments rather than constants because they are a property of the
           corpus, and a regenerated corpus would change them.

`lift` is accuracy minus the applicable floor. For the different condition,
lift - not accuracy - is the number that says whether anything was counted.

WHAT ELSE CAN GO WRONG

  predTop / predTopShare  a model that answers one letter every time lands at
           whatever share that letter holds. Check this column first.
  balAcc   recall averaged over the four gold letters. A constant predictor
           scores exactly 0.250 whatever the key looks like.
  kappa    accuracy corrected for both marginals; 0 for a constant predictor.
  posBias  total variation distance between the model's answer distribution
           and the key's. Position bias is a pre-registered control here:
           Llama-3.2-Vision showed a 63% first-option preference elsewhere in
           this project, and several models never emit D at all.
  incoherence  every image carries both a most and a least question, and they
           cannot share an answer. The share of images where a model gives the
           same letter to both is a free consistency check that needs no
           ground truth at all.

CONFIDENCE INTERVALS
Bootstrap resamples IMAGES, not rows: each grid contributes two questions.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

LETTERS = ["A", "B", "C", "D"]


def balanced_accuracy(gold, pred):
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
    return (po - pe) / (1.0 - pe) if pe < 1.0 else float("nan")


def pos_bias(gold, pred):
    if len(gold) == 0:
        return float("nan")
    g = np.array([float((gold == l).mean()) for l in LETTERS])
    p = np.array([float((pred == l).mean()) for l in LETTERS])
    return float(0.5 * np.abs(g - p).sum())


def block(gold, pred):
    n = len(gold)
    if n == 0:
        return {k: float("nan") for k in
                ("n", "acc", "balAcc", "kappa", "posBias", "predTop",
                 "predTopShare")}
    top = pred.mode()
    top = str(top.iloc[0]) if len(top) else "-"
    return {
        "n": int(n),
        "acc": float((gold.values == pred.values).mean()),
        "balAcc": balanced_accuracy(gold, pred),
        "kappa": cohen_kappa(gold, pred),
        "posBias": pos_bias(gold, pred),
        "predTop": top,
        "predTopShare": float((pred == top).mean()),
    }


def incoherence(d):
    """Share of question PAIRS where the model gave the same letter to the
    most and the least question. Those cannot both be right, and the check
    needs no ground truth at all.

    Pairing is on (image, animal_term), not on image alone. A GrevysZebra grid
    carries three questions - animals, grevys zebras, zebras - so one image
    yields six rows, and pairing on the image would compare "most animals"
    against "least grevys zebras", which may legitimately share an answer,
    while silently discarding the other two pairs."""
    if "polarity" not in d.columns:
        return float("nan")
    idx = ["image"]
    if "animal_term" in d.columns:
        idx.append("animal_term")
    piv = d.pivot_table(index=idx, columns="polarity", values="_p",
                        aggfunc="first")
    if not {"most", "least"} <= set(piv.columns):
        return float("nan")
    both = piv.dropna(subset=["most", "least"])
    if not len(both):
        return float("nan")
    return float((both["most"] == both["least"]).mean())


def boot(d, key, fn, n_boot, seed=0):
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
    ap.add_argument("--prior_different_most", type=float, default=0.530,
                    help="dataset-prior baseline for different/most")
    ap.add_argument("--prior_different_least", type=float, default=0.548,
                    help="dataset-prior baseline for different/least")
    args = ap.parse_args()

    floors = {("same", "most"): 0.25, ("same", "least"): 0.25,
              ("different", "most"): args.prior_different_most,
              ("different", "least"): args.prior_different_least}

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

        gkeys = [k for k in ("condition", "polarity") if k in d.columns]
        if gkeys:
            for k, g in d.groupby(gkeys):
                if not isinstance(k, tuple):
                    k = (k, "")
                s = block(g["_g"], g["_p"])
                fl = floors.get((k[0], k[1]), 0.25)
                s.update(model=key, condition=k[0], polarity=k[1],
                         floor=fl, lift=s["acc"] - fl)
                groups.append(s)

        row = {"model": key}
        for l in LETTERS:
            row["gold_" + l] = float((d["_g"] == l).mean())
            row["pred_" + l] = float((d["_p"] == l).mean())
        conf.append(row)

        s = block(d["_g"], d["_p"])
        # weighted floor over whichever cells this model actually has
        if gkeys:
            w = d.groupby(gkeys).size()
            tot = float(w.sum())
            fl = sum(floors.get(k if isinstance(k, tuple) else (k, ""), 0.25)
                     * v for k, v in w.items()) / tot
        else:
            fl = 0.25
        # GrevysZebra grids carry three animal terms and therefore 2.2x the
        # rows of any other dataset. A row-weighted same-condition score is
        # dominated by it, so the per-dataset macro is reported alongside.
        macro = float("nan")
        if "source_dataset" in d.columns:
            sub = d[d["source_dataset"].notna()]
            if len(sub):
                per = [float((g["_g"] == g["_p"]).mean())
                       for _, g in sub.groupby("source_dataset")]
                macro = float(np.mean(per)) if per else float("nan")

        # Per-condition accuracy and lift, hoisted into the overall row.
        # The pooled lift below subtracts a WEIGHTED AVERAGE of two very
        # different floors (0.25 for same, ~0.54 for different) from a
        # weighted average of two very different accuracies, so it cannot be
        # read as "beat the baseline": a model that is strong on same and at
        # chance on different produces the same pooled figure as one that is
        # mediocre on both. Only the per-condition columns are interpretable.
        per_cond = {}
        if "condition" in d.columns:
            for cond, g in d.groupby("condition"):
                a = float((g["_g"].values == g["_p"].values).mean())
                cf = floors.get((cond, ""), None)
                if cf is None:
                    sub = {p: floors.get((cond, p), 0.25)
                           for p in g.get("polarity", pd.Series(dtype=str)).unique()}
                    if sub:
                        w = g.groupby("polarity").size()
                        cf = float(sum(sub[p] * w[p] for p in sub) / w.sum())
                    else:
                        cf = 0.25
                per_cond["acc_" + str(cond)] = a
                per_cond["floor_" + str(cond)] = cf
                per_cond["lift_" + str(cond)] = a - cf

        s.update(model=key, parse_rate=parse_rate, floor=fl,
                 lift=s["acc"] - fl, incoherence=incoherence(d), **per_cond,
                 same_macroAcc=macro,
                 images=int(d["image"].nunique()) if "image" in d.columns else -1)

        if args.bootstrap:
            clus = "image" if "image" in d.columns else "row_id"
            lo, hi = boot(d, clus,
                          lambda x: balanced_accuracy(x["_g"], x["_p"]),
                          args.bootstrap)
            s["balAcc_lo"], s["balAcc_hi"] = lo, hi

        overall.append(s)
        print("[ok] %-32s n=%5d  acc %.3f  lift %+.3f  balAcc %.3f  "
              "kappa %+.3f  posBias %.3f  incoh %.3f  predTop %s %.2f"
              % (key, s["n"], s["acc"], s["lift"], s["balAcc"], s["kappa"],
                 s["posBias"], s["incoherence"], s["predTop"],
                 s["predTopShare"]))

    ov = pd.DataFrame(overall)
    gr = pd.DataFrame(groups)
    cf = pd.DataFrame(conf)
    p1 = os.path.join(args.out_dir, "relcount_overall.csv")
    p2 = os.path.join(args.out_dir, "relcount_by_group.csv")
    p3 = os.path.join(args.out_dir, "relcount_confusion.csv")
    ov.to_csv(p1, index=False)
    gr.to_csv(p2, index=False)
    cf.to_csv(p3, index=False)

    # per-condition columns first: they are the only interpretable lift
    cond_cols = [c for c in ov.columns
                 if c.startswith(("acc_", "lift_")) and c not in ("acc_lo",)]
    cond_cols = sorted(cond_cols, key=lambda c: (c.split("_", 1)[1], c[:4]))
    cols = ([c for c in ["model", "n", "images"] if c in ov.columns]
            + cond_cols
            + [c for c in ["incoherence", "balAcc", "kappa", "acc", "floor",
                           "lift", "same_macroAcc", "posBias", "predTop",
                           "predTopShare", "parse_rate"] if c in ov.columns])
    sort_key = ("lift_same" if "lift_same" in ov.columns else "balAcc")
    show = ov[cols].sort_values(sort_key, ascending=False)

    if cond_cols:
        print()
        print("=" * 78)
        print("PER-CONDITION LIFT - the pooled `lift` column is NOT comparable")
        print("=" * 78)
        print("The two conditions have different floors (same 0.250, different")
        print("~0.539), so the pooled figure averages across them and hides")
        print("which one a model is failing. Read the columns below.")
        print()
        pc = [c for c in cond_cols] + ["incoherence"]
        print(ov[["model"] + [c for c in pc if c in ov.columns]]
              .sort_values(sort_key, ascending=False)
              .to_string(index=False, float_format=lambda x: "%+.3f" % x))

    md = [
        "# BioVQA C3 - Counting-Relative results",
        "",
        "A 2x2 grid of four photographs; the model picks the quadrant holding "
        "the most, or the fewest, animals. Generated by `evaluate.py`.",
        "",
        "## Two conditions, two floors",
        "",
        "| condition | panels | floor |",
        "|---|---|---|",
        "| `same` | four from ONE dataset, question names the animal | 0.250 |",
        ("| `different` | four from four DIFFERENT datasets, question says "
         "'animals' | %.3f most / %.3f least |"
         % (args.prior_different_most, args.prior_different_least)),
        "",
        "**The pooled `lift` column is not comparable across models and "
        "should not be quoted.** It subtracts a weighted average of the two "
        "floors from a weighted average of the two accuracies, so a model "
        "that is strong on `same` and at chance on `different` produces the "
        "same figure as one that is mediocre on both. Use `lift_same` and "
        "`lift_different`.",
        "",
        "In the `different` condition, count magnitude is partly predictable "
        "from which dataset a panel came from - Bees average about 37 animals "
        "per frame while Camel and Kiang average about 8 - so a model that "
        "recognises the domain can answer without counting. The floors above "
        "come from a Monte-Carlo over the empirical count distributions using "
        "the generator's own selection rule. **`lift` is accuracy minus the "
        "applicable floor, and for `different` it is lift, not accuracy, that "
        "says whether anything was counted.**",
        "",
        "The `same` condition has no such shortcut, which makes it the clean "
        "measurement of relative enumeration; `different` measures whether a "
        "model falls back on domain recognition when that shortcut exists. "
        "Comparing a model's lift across the two is the informative contrast.",
        "",
        "## Overall",
        "",
        "**Check `predTopShare` first.** Near 1.00 means the model answered "
        "the same letter every time; `balAcc` then sits at 0.250 and `kappa` "
        "at 0 however respectable `acc` looks.",
        "",
        "`same_macroAcc` averages accuracy over the ten source datasets of "
        "the `same` condition with equal weight. GrevysZebra grids carry "
        "three animal terms - animals, grevys zebras, zebras - and therefore "
        "contribute 4,412 rows against 2,000 for every other dataset, so a "
        "row-weighted same-condition number is dominated by one taxon.",
        "",
        "`incoherence` is the share of question pairs where the model gave the SAME "
        "letter to the most and the least question. Those cannot both be "
        "correct, and the check needs no ground truth - a high value is a "
        "direct demonstration that the model is not comparing anything. "
        "Pairs are formed on (image, animal term), not on image alone, "
        "because a GrevysZebra grid carries three separate question pairs.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for _, r in show.iterrows():
        md.append("| " + " | ".join(
            (r[c] if isinstance(r[c], str) else
             ("%d" % r[c] if c in ("n", "images") else "%.3f" % r[c]))
            for c in cols) + " |")

    if len(gr):
        gc = [c for c in ["model", "condition", "polarity", "n", "acc",
                          "floor", "lift", "balAcc", "kappa", "posBias",
                          "predTopShare"] if c in gr.columns]
        md += ["", "## By condition and polarity", "",
               "| " + " | ".join(gc) + " |", "|" + "---|" * len(gc)]
        for _, r in gr.sort_values(["model", "condition", "polarity"]).iterrows():
            md.append("| " + " | ".join(
                (r[c] if isinstance(r[c], str) else
                 ("%d" % r[c] if c == "n" else "%.3f" % r[c]))
                for c in gc) + " |")

    if len(cf):
        cc = list(cf.columns)
        md += ["", "## Answer distribution", "",
               "`gold_X` is the share of items whose answer is X; `pred_X` is "
               "the share of the model's answers that were X. Large gaps are "
               "position bias; a `pred_D` near zero means the model never "
               "chose the last option.", "",
               "| " + " | ".join(cc) + " |", "|" + "---|" * len(cc)]
        for _, r in cf.iterrows():
            md.append("| " + " | ".join(
                (r[c] if isinstance(r[c], str) else "%.3f" % r[c])
                for c in cc) + " |")

    md += [
        "",
        "## Notes",
        "",
        "The four panels are pasted at different magnifications, so apparent "
        "animal size does not track count across panels. The prompt says so "
        "explicitly, which removes one interpretation of an otherwise "
        "ambiguous comparison.",
        "",
        "Confidence intervals bootstrap over IMAGES, not rows: each grid "
        "carries both a most and a least question, so rows come in dependent "
        "pairs.",
        "",
        "The `different` floors are command-line arguments, not constants, "
        "because they are a property of this corpus. Regenerating the grids "
        "with a different dataset mix would change them, and the value used "
        "is recorded in `relcount_overall.csv` alongside every score.",
        "",
    ]
    p4 = os.path.join(args.out_dir, "relcount_report.md")
    with open(p4, "w") as fh:
        fh.write("\n".join(md) + "\n")

    print()
    for p in (p1, p2, p3, p4):
        print("wrote " + p)
    print()
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()