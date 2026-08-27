#!/usr/bin/env python3
"""
Authoritative evaluation for the Relative Position task.

  python evaluate.py --results_dir ... --out_dir ...
  python evaluate.py --bootstrap 2000

Outputs:
  relpos_overall.csv      one row per model
  relpos_by_group.csv     one row per model x condition x dataset
  relpos_errors.csv       error attribution per model
  relpos_report.md        human-readable summary

--------------------------------------------------------------------------- #
THE FLOOR IS NOT 0.250

Four options are offered, so a model choosing at random scores 0.250. But the
gold directions are not uniform: wide photographs produce mostly horizontal
separations, and left plus right account for about half of all items. A model
that always answers the commonest direction therefore does considerably better
than chance. `majority` is that floor, computed on the identical rows as the
model, and `lift` is accuracy minus it.

WHAT THE ERRORS MEAN, WHICH IS THE POINT OF THIS TASK

In the distractor condition the four options are the correct answer plus the
three directions produced by pairing the wrong boxes:

  errRedEmpty    took the EMPTY red box, correct blue     -> ignored the red content
  errBlueEmpty   correct red, took the EMPTY blue box     -> ignored the blue content
  errBothEmpty   took both empty boxes                    -> matched colour only

Those three plus the correct answer exhaust the option set, so every error is
attributable. This is what the distractor condition was built for: with only
two boxes, "find the red rectangle" is enough, and a model never has to check
whether an animal is inside it. The breakdown says which half of that check
each model skips.

  errOffOption   answered a direction that was not offered at all. That is an
                 instruction-following failure, not a localisation one, and it
                 is counted separately so it does not contaminate the above.

OTHER COLUMNS
  balAcc      recall averaged over the gold directions present; a constant
              predictor scores 1/k whatever the skew.
  kappa       accuracy corrected for both marginal distributions.
  angErr      mean angular error in degrees on the 45-degree ring. Answering
              bottom-left when the truth is left is a 45-degree miss, not the
              same error as answering right, and exact match cannot tell those
              apart.
  posBias     total variation distance between the model's answer distribution
              and the key's.
  optPosBias  whether the model prefers a given POSITION in the options list
              rather than a given direction. Llama-3.2-Vision showed a 63%
              first-option preference elsewhere in this project.

CONFIDENCE INTERVALS
Bootstrap resamples SOURCE IMAGES. Several pairs are drawn from one photograph
and, when both conditions are present, the same pair appears twice, so rows are
far from independent.
"""

import argparse
import glob
import math
import os
import re

import numpy as np
import pandas as pd

DIRECTIONS = ["right", "bottom-right", "bottom", "bottom-left", "left",
              "top-left", "top", "top-right"]
ANGLE = {"right": 0, "bottom-right": 45, "bottom": 90, "bottom-left": 135,
         "left": 180, "top-left": 225, "top": 270, "top-right": 315}


def canon(x):
    return re.sub(r"[^a-z]+", "-", str(x).strip().lower()).strip("-")


def ang_err(g, p):
    if g not in ANGLE or p not in ANGLE:
        return float("nan")
    d = abs(ANGLE[g] - ANGLE[p]) % 360
    return min(d, 360 - d)


def balanced_accuracy(gold, pred):
    labs = sorted(set(gold))
    if not labs:
        return float("nan")
    rec = [float((pred[gold == l] == l).mean()) for l in labs
           if (gold == l).sum()]
    return sum(rec) / len(rec) if rec else float("nan")


def cohen_kappa(gold, pred):
    if len(gold) == 0:
        return float("nan")
    po = float((gold.values == pred.values).mean())
    labs = set(gold) | set(pred)
    pe = sum(float((gold == l).mean()) * float((pred == l).mean())
             for l in labs)
    return (po - pe) / (1.0 - pe) if pe < 1.0 else float("nan")


def tv_distance(gold, pred):
    labs = sorted(set(gold) | set(pred))
    if not labs:
        return float("nan")
    g = np.array([float((gold == l).mean()) for l in labs])
    p = np.array([float((pred == l).mean()) for l in labs])
    return float(0.5 * np.abs(g - p).sum())


def block(d):
    gold, pred = d["_g"], d["_p"]
    n = len(gold)
    if n == 0:
        return {k: float("nan") for k in
                ("n", "acc", "majority", "lift", "balAcc", "kappa", "angErr",
                 "within45", "posBias", "predTop", "predTopShare")}
    errs = [ang_err(g, p) for g, p in zip(gold, pred)]
    errs = [e for e in errs if not math.isnan(e)]
    maj = float((gold == gold.mode().iloc[0]).mean())
    acc = float((gold.values == pred.values).mean())
    top = str(pred.mode().iloc[0])
    return {
        "n": int(n),
        "acc": acc,
        "majority": maj,
        "lift": acc - maj,
        "balAcc": balanced_accuracy(gold, pred),
        "kappa": cohen_kappa(gold, pred),
        "angErr": (sum(errs) / len(errs)) if errs else float("nan"),
        "within45": (sum(1 for e in errs if e <= 45) / len(errs))
                    if errs else float("nan"),
        "posBias": tv_distance(gold, pred),
        "predTop": top,
        "predTopShare": float((pred == top).mean()),
    }


def options_shown(d):
    """Were the four options listed in the prompt?

    They were in the original questions and are not in the rewritten ones.
    Two metrics only make sense when they were - the option-position bias, and
    counting answers that fall outside the offered set - so they are skipped
    rather than reported as misleading zeros.
    """
    if not len(d):
        return False
    # merge_results.py records this as a boolean and drops the question text,
    # which is byte-identical on every row and not worth carrying
    if "options_shown" in d.columns:
        return bool(d["options_shown"].iloc[0])
    if "question" not in d.columns:
        return False
    q = str(d["question"].iloc[0])
    return ("Choose exactly one of:" in q) or ("Options:" in q)


def attribute(d):
    """Split the errors by which box pair the model must have used.

    With the options listed, the four choices were exactly the correct answer
    plus the three mis-pairings, so the attribution was exhaustive and
    `errOther` was always zero. Without them the model may answer any of the
    eight directions, so `errOther` becomes informative: it is the share of
    errors that correspond to no box pairing at all, i.e. the model was not
    confusing boxes, it simply did not read the layout.
    """
    out = {"errRedEmpty": float("nan"), "errBlueEmpty": float("nan"),
           "errBothEmpty": float("nan"), "errOther": float("nan"),
           "errOffOption": float("nan")}
    if not len(d):
        return out
    wrong = d[d["_g"] != d["_p"]]
    n = float(len(d))
    if options_shown(d) and "options" in d.columns:
        offered = d["options"].astype(str).str.split("|")
        off = sum(1 for p, o in zip(d["_p"], offered)
                  if p not in [canon(x) for x in o])
        out["errOffOption"] = off / n
    if not len(wrong):
        for k in ("errRedEmpty", "errBlueEmpty", "errBothEmpty", "errOther"):
            out[k] = 0.0
        return out
    hits = {"errRedEmpty": 0, "errBlueEmpty": 0, "errBothEmpty": 0,
            "errOther": 0}
    for _, r in wrong.iterrows():
        p = r["_p"]
        if p == r.get("mis_red_empty"):
            hits["errRedEmpty"] += 1
        elif p == r.get("mis_blue_empty"):
            hits["errBlueEmpty"] += 1
        elif p == r.get("mis_both_empty"):
            hits["errBothEmpty"] += 1
        else:
            hits["errOther"] += 1
    for k, v in hits.items():
        out[k] = v / n

    # ---- chance correction ------------------------------------------- #
    # A model answering at random still lands on a mis-pairing direction
    # roughly one time in eight, so the raw rates above are not evidence of
    # box confusion on their own: a pure guesser scores about 0.125 on each
    # of the three, and reading them as "grabbed the empty red box 16% of
    # the time" would be wrong.
    #
    # The expected rate is the same chance-agreement term as Cohen's kappa:
    # the model's own answer distribution crossed with the distribution of
    # that mis-pairing. Subtracting it leaves the excess actually
    # attributable to confusing the boxes. Negative means the model AVOIDS
    # that pairing more than its own answer habits would predict, which is
    # what a model checking box contents looks like.
    pm = d["_p"].value_counts(normalize=True)
    for col, key in (("mis_red_empty", "errRedEmpty"),
                     ("mis_blue_empty", "errBlueEmpty"),
                     ("mis_both_empty", "errBothEmpty")):
        if col not in d.columns:
            out[key + "X"] = float("nan")
            continue
        mm = d[col].value_counts(normalize=True)
        exp = float(sum(pm.get(k, 0.0) * mm.get(k, 0.0)
                        for k in set(pm.index) | set(mm.index)))
        out[key + "_exp"] = exp
        out[key + "X"] = out[key] - exp
    return out


def opt_pos_bias(d):
    """Does the model prefer a POSITION in the option list rather than a
    direction? Returns the share of answers at each 1-based position, and the
    distance of that distribution from uniform.

    Only meaningful when the options were actually shown in the prompt; with
    the rewritten questions they are not, and there is no position to prefer.
    """
    if not options_shown(d) or "options" not in d.columns or not len(d):
        return float("nan"), {}
    pos = []
    for p, o in zip(d["_p"], d["options"].astype(str).str.split("|")):
        cl = [canon(x) for x in o]
        pos.append(cl.index(p) + 1 if p in cl else 0)
    pos = pd.Series(pos)
    valid = pos[pos > 0]
    if not len(valid):
        return float("nan"), {}
    k = int(valid.max())
    share = {int(i): float((valid == i).mean()) for i in range(1, k + 1)}
    unif = 1.0 / k
    return float(0.5 * sum(abs(v - unif) for v in share.values())), share


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
        vals.append(fn(d.iloc[np.concatenate([idx[i] for i in pick])]))
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

    overall, groups, errors = [], [], []
    opts_were_shown = False

    for path in files:
        key = os.path.basename(path)[len("qa_"):-len(".csv")]
        df = pd.read_csv(path)
        col = key + "_answer"
        if col not in df.columns:
            cand = [c for c in df.columns if c.endswith("_answer")
                    and not c.endswith("_unfiltered")]
            if not cand:
                print("[skip] %s: no answer column" % key)
                continue
            col = cand[0]

        gold = df["answer_dir"].map(canon)
        pred = df[col].map(lambda x: canon(x) if pd.notna(x) else None)
        ok = gold.isin(DIRECTIONS) & pred.isin(DIRECTIONS)

        d = df.loc[ok].copy()
        d["_g"] = gold[ok].values
        d["_p"] = pred[ok].values
        parse_rate = float(pred.isin(DIRECTIONS).mean())
        if not len(d):
            print("[skip] %s: nothing scoreable" % key)
            continue

        gkeys = [k for k in ("condition", "dataset") if k in d.columns]
        if gkeys:
            for k, g in d.groupby(gkeys):
                if not isinstance(k, tuple):
                    k = (k, "")
                s = block(g)
                s.update(model=key, condition=k[0], dataset=k[1],
                         **attribute(g))
                groups.append(s)
        if "condition" in d.columns:
            for cond, g in d.groupby("condition"):
                s = block(g)
                s.update(model=key, condition=cond, dataset="__ALL__",
                         **attribute(g))
                groups.append(s)

        opts_were_shown = opts_were_shown or options_shown(d)
        att = attribute(d)
        opb, share = opt_pos_bias(d)
        errors.append(dict(model=key, **att, optPosBias=opb,
                           **{("optPos%d" % i): v for i, v in share.items()}))

        s = block(d)
        s.update(model=key, parse_rate=parse_rate, optPosBias=opb, **att,
                 images=int(d["source_image"].nunique())
                 if "source_image" in d.columns else -1)

        if args.bootstrap:
            clus = "source_image" if "source_image" in d.columns else "row_id"
            lo, hi = boot(d, clus, lambda x: block(x)["acc"], args.bootstrap)
            s["acc_lo"], s["acc_hi"] = lo, hi

        overall.append(s)
        print("[ok] %-32s n=%6d  acc %.3f  lift %+.3f  angErr %.1f  "
              "within45 %.3f  redX %+.3f  blueX %+.3f  bothX %+.3f  "
              "predTop %.2f  parse %.3f"
              % (key, s["n"], s["acc"], s["lift"], s["angErr"], s["within45"],
                 s.get("errRedEmptyX", float("nan")),
                 s.get("errBlueEmptyX", float("nan")),
                 s.get("errBothEmptyX", float("nan")),
                 s["predTopShare"], parse_rate))

    ov = pd.DataFrame(overall)
    gr = pd.DataFrame(groups)
    er = pd.DataFrame(errors)
    p1 = os.path.join(args.out_dir, "relpos_overall.csv")
    p2 = os.path.join(args.out_dir, "relpos_by_group.csv")
    p3 = os.path.join(args.out_dir, "relpos_errors.csv")
    ov.to_csv(p1, index=False)
    gr.to_csv(p2, index=False)
    er.to_csv(p3, index=False)

    cols = [c for c in ["model", "n", "images", "acc", "acc_lo", "acc_hi",
                        "majority", "lift", "balAcc", "kappa", "angErr",
                        "within45", "posBias", "errRedEmptyX",
                        "errBlueEmptyX", "errBothEmptyX", "errOther",
                        "errRedEmpty", "errBlueEmpty", "errBothEmpty",
                        "optPosBias", "errOffOption", "predTop",
                        "predTopShare", "parse_rate"]
            if c in ov.columns]
    # optPosBias and errOffOption are defined only when the four options are
    # listed in the prompt. They are not, so both are NaN for every model - a
    # column of NaNs in a published table reads as a bug rather than as "not
    # applicable", so it is dropped instead. It returns automatically if the
    # options are ever put back.
    dropped_na = [c for c in cols if c in ov.columns and ov[c].isna().all()]
    if dropped_na:
        cols = [c for c in cols if c not in dropped_na]
        print("not applicable for this prompt, omitted: %s"
              % ", ".join(dropped_na))
    show = ov[cols].sort_values("lift", ascending=False)

    md = [
        "# BioVQA C3 - Relative Position results",
        "",
        "Two animals in one photograph are outlined, one red and one blue; the "
        "model names the direction of the red one relative to the blue one. "
        "Four of the eight compass directions are offered. Generated by "
        "`evaluate.py`.",
        "",
        "## The floor is not chance",
        "",
        ("The prompt %s, so random choice scores %s. Either way the gold "
         "directions are not uniform - wide photographs produce mostly "
         "horizontal separations, and left plus right are about half of all "
         "items - so the real floor is higher than chance. `majority` is the "
         "score of always answering the commonest direction, computed on the "
         "identical rows as the model, and **`lift` is accuracy minus that "
         "floor**. Read lift, not accuracy."
         % (("lists four options" if opts_were_shown else
             "does not list the options, leaving all eight directions open"),
            ("0.250" if opts_were_shown else "0.125"))),
        "",
        "## What the errors mean",
        "",
        ("In the distractor condition the three mis-pairings below are the "
         "directions produced by pairing the wrong boxes. %s"
         % ("Because the four options were exactly those three plus the "
            "correct answer, every error is attributable and `errOther` is "
            "zero by construction."
            if opts_were_shown else
            "The options are not shown, so a model may answer any of the "
            "eight directions and `errOther` is informative: it is the share "
            "of errors matching no box pairing at all, meaning the model was "
            "not confusing boxes but simply not reading the layout.")),
        "",
        "| column | what the model did |",
        "|---|---|",
        "| `errRedEmpty` | took the EMPTY red box with the correct blue - did not check whether the red box contained an animal |",
        "| `errBlueEmpty` | correct red, took the EMPTY blue box |",
        "| `errBothEmpty` | took both empty boxes - matched colour only |",
        "| `errOther` | wrong, but matching no box pairing - not a box confusion |",
        "",
        "**Read the `...X` columns, not the raw rates.** A model answering at "
        "random still lands on a mis-pairing direction about one time in "
        "eight, so a raw `errRedEmpty` of 0.16 is barely above what guessing "
        "produces. `errRedEmptyX` and its siblings subtract the expected rate "
        "- the model's own answer distribution crossed with that "
        "mis-pairing's distribution, the same chance-agreement term as "
        "Cohen's kappa. Positive means genuine box confusion; **negative "
        "means the model avoids that pairing more than its own answer habits "
        "predict, which is what checking box contents looks like**.",
        "| `errOffOption` | answered a direction that was not offered; only defined when the options were shown in the prompt |",
        "",
        "This is what the distractor condition exists for. With only two boxes "
        "a model never has to look at the animals - find the two rectangles, "
        "compare centres. Adding an empty box of each target colour makes "
        "*find the red rectangle* insufficient: it has to be the red rectangle "
        "with a specimen in it. The breakdown says which half of that check "
        "each model skips.",
        "",
        "**The clean-to-distractor drop is the headline contrast.** A model "
        "that was genuinely reading the animals loses little; one that was "
        "matching rectangles falls to near the floor.",
        "",
        "## Overall",
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
        gc = [c for c in ["model", "condition", "dataset", "n", "acc",
                          "majority", "lift", "balAcc", "angErr",
                          "errRedEmpty", "errBlueEmpty", "errBothEmpty"]
              if c in gr.columns]
        md += ["", "## By condition and dataset", "",
               "Rows with `dataset = __ALL__` pool the datasets within a "
               "condition; that is the clean-versus-distractor comparison.",
               "",
               "| " + " | ".join(gc) + " |", "|" + "---|" * len(gc)]
        for _, r in gr.sort_values(["model", "condition", "dataset"]).iterrows():
            md.append("| " + " | ".join(
                (r[c] if isinstance(r[c], str) else
                 ("%d" % r[c] if c == "n" else "%.3f" % r[c]))
                for c in gc) + " |")

    md += [
        "",
        "## Notes",
        "",
        "`angErr` is the mean angular error on the 45-degree ring. Answering "
        "bottom-left when the truth is left is a 45-degree miss, not the same "
        "error as answering right, and exact match cannot tell those apart. "
        "A model with low accuracy but low `angErr` is perceiving the layout "
        "and losing on granularity, which is a different finding from one "
        "that is guessing.",
        "",
        "`optPosBias` measures whether a model prefers a POSITION in the "
        "options list rather than a direction. It is a pre-registered control: "
        "Llama-3.2-Vision showed a 63% first-option preference elsewhere in "
        "this project.",
        "",
        "Confidence intervals bootstrap over SOURCE IMAGES. Several pairs are "
        "drawn from one photograph and, where both conditions exist, the same "
        "pair appears twice, so rows are far from independent.",
        "",
    ]
    p4 = os.path.join(args.out_dir, "relpos_report.md")
    with open(p4, "w") as fh:
        fh.write("\n".join(md) + "\n")

    print()
    for p in (p1, p2, p3, p4):
        print("wrote " + p)
    print()
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()