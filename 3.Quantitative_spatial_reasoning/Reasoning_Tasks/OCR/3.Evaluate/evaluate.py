#!/usr/bin/env python3
"""
Authoritative evaluation for the OCR task.

  python evaluate.py --results_dir ... --out_dir ...
  python evaluate.py --bootstrap 2000

Outputs:
  ocr_overall.csv        one row per model
  ocr_by_position.csv    per-character-position accuracy per model
  ocr_report.md          human-readable summary

--------------------------------------------------------------------------- #
WHY CHARACTER ACCURACY ON THE FULL CODE IS MISLEADING

These accession codes share a long fixed prefix. In a sample of this corpus
eight of twelve characters were identical across every label, so a model that
emits one fixed string and never looks at an image reaches a character accuracy
above 0.8 while reading nothing at all. The exact figure for this corpus is
computed below as constCER / constSuffixAcc, from the per-position modal
character - the optimal constant predictor for character error rate.

Metrics, in the order they should be read:

  EM            exact match on the whole code. Blunt but honest, and the number
                a collection manager actually cares about: a transcription with
                one wrong digit is the wrong specimen.
  suffixEM      exact match on the variable part only. Same quantity as EM
                whenever the prefix is constant, kept for the case where a
                model gets the informative digits right but mangles the prefix.
  suffixCharAcc per-character accuracy over the VARIABLE part. This is the
                headline continuous metric - it is the portion that requires
                reading.
  charAcc       per-character accuracy over the whole code. Reported for
                comparability with OCR literature, and flagged, because the
                constant prefix inflates it.
  CER           normalised edit distance over the whole code, which unlike
                charAcc tolerates insertions and deletions.
  wellformed    share of answers matching A followed by eleven digits. A low
                value with a high charAcc means the model can read but not
                follow the format, which is a different finding.
  parse_rate    share of raw outputs containing a code-like string at all.

CONFIDENCE INTERVALS
Every label is an independent specimen, so unlike the plate-based tasks in this
component there is no clustering here and the bootstrap resamples rows.
"""

import argparse
import glob
import os
import re
from collections import Counter

import numpy as np
import pandas as pd


def levenshtein(a, b):
    """Edit distance. Small strings, so the simple DP is fine."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_acc(gold, pred):
    """Position-wise character accuracy, denominator = gold length.
    A short answer is penalised for the characters it never produced."""
    if not len(gold):
        return float("nan")
    tot = hit = 0
    for g, p in zip(gold, pred):
        tot += len(g)
        hit += sum(1 for i, c in enumerate(g) if i < len(p) and p[i] == c)
    return hit / tot if tot else float("nan")


def cer(gold, pred):
    tot = sum(len(g) for g in gold)
    if not tot:
        return float("nan")
    return sum(levenshtein(g, p) for g, p in zip(gold, pred)) / tot


def modal_string(codes):
    """Per-position modal character: the constant string that minimises
    character error over this corpus."""
    if not codes:
        return ""
    n = max(len(c) for c in codes)
    out = []
    for i in range(n):
        chars = [c[i] for c in codes if len(c) > i]
        out.append(Counter(chars).most_common(1)[0][0] if chars else "")
    return "".join(out)


def block(gold, pred, prefix_len):
    n = len(gold)
    if n == 0:
        return {k: float("nan") for k in
                ("n", "EM", "suffixEM", "suffixCharAcc", "charAcc", "CER",
                 "wellformed")}
    gs = [g[prefix_len:] for g in gold]
    ps = [p[prefix_len:] if len(p) > prefix_len else "" for p in pred]
    return {
        "n": int(n),
        "EM": float(np.mean([g == p for g, p in zip(gold, pred)])),
        "suffixEM": float(np.mean([g == p for g, p in zip(gs, ps)])),
        "suffixCharAcc": char_acc(gs, ps),
        "charAcc": char_acc(gold, pred),
        "CER": cer(gold, pred),
        "wellformed": float(np.mean(
            [bool(re.fullmatch(r"A\d{11}", p)) for p in pred])),
    }


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

    overall, positions = [], []
    prefix, const = "", ""

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

        gold_all = df["answer"].astype(str).str.upper()
        pred_all = df[col].astype(str).str.upper()
        ok = pred_all.notna() & (pred_all != "NAN") & (pred_all != "NONE") \
            & (pred_all.str.len() > 0)

        d = df.loc[ok].copy()
        gold = gold_all[ok].tolist()
        pred = pred_all[ok].tolist()
        parse_rate = float(ok.mean())
        if not gold:
            print("[skip] %s: nothing scoreable" % key)
            continue

        if not prefix:
            prefix = os.path.commonprefix(gold_all.tolist())
            const = modal_string(gold_all.tolist())

        s = block(gold, pred, len(prefix))
        s.update(model=key, parse_rate=parse_rate)

        # constant-string baseline on the identical rows
        cb = block(gold, [const] * len(gold), len(prefix))
        s.update(constEM=cb["EM"], constCharAcc=cb["charAcc"],
                 constSuffixCharAcc=cb["suffixCharAcc"], constCER=cb["CER"])

        if args.bootstrap:
            rng = np.random.default_rng(0)
            vals = []
            gs = [g[len(prefix):] for g in gold]
            ps = [p[len(prefix):] if len(p) > len(prefix) else "" for p in pred]
            for _ in range(args.bootstrap):
                idx = rng.integers(0, len(gs), len(gs))
                vals.append(char_acc([gs[i] for i in idx],
                                     [ps[i] for i in idx]))
            s["suffixCharAcc_lo"] = float(np.percentile(vals, 2.5))
            s["suffixCharAcc_hi"] = float(np.percentile(vals, 97.5))

        overall.append(s)

        # per-position accuracy, to show where reading actually fails
        L = int(np.median([len(g) for g in gold]))
        row = {"model": key}
        for i in range(L):
            hit = [1.0 if (len(p) > i and len(g) > i and p[i] == g[i]) else 0.0
                   for g, p in zip(gold, pred)]
            row["pos%02d" % i] = float(np.mean(hit)) if hit else float("nan")
        positions.append(row)

        print("[ok] %-32s n=%5d  EM %.3f  suffixCharAcc %.3f  "
              "(const %.3f)  charAcc %.3f  wellformed %.3f  parse %.3f"
              % (key, s["n"], s["EM"], s["suffixCharAcc"],
                 s["constSuffixCharAcc"], s["charAcc"], s["wellformed"],
                 parse_rate))

    ov = pd.DataFrame(overall)
    po = pd.DataFrame(positions)
    p1 = os.path.join(args.out_dir, "ocr_overall.csv")
    p2 = os.path.join(args.out_dir, "ocr_by_position.csv")
    ov.to_csv(p1, index=False)
    po.to_csv(p2, index=False)

    cols = [c for c in ["model", "n", "EM", "suffixEM", "suffixCharAcc",
                        "suffixCharAcc_lo", "suffixCharAcc_hi",
                        "constSuffixCharAcc", "charAcc", "constCharAcc",
                        "CER", "constCER", "wellformed", "parse_rate"]
            if c in ov.columns]
    show = ov[cols].sort_values("EM", ascending=False)

    md = [
        "# BioVQA C3 - OCR results",
        "",
        "Transcribe the accession code printed beneath a specimen barcode. "
        "Generated by `evaluate.py`.",
        "",
        "## The constant prefix, and why charAcc is not the headline",
        "",
        "Every code in this corpus begins with the same fixed prefix - "
        "`%s`, %d of %d characters. A model that emits one fixed string and "
        "never looks at an image therefore reaches a character accuracy of "
        "**%.3f** while reading nothing."
        % (prefix, len(prefix),
           int(np.median([len(x) for x in ov.index.astype(str)])) if False
           else len(const),
           float(ov["constCharAcc"].iloc[0]) if len(ov) else float("nan")),
        "",
        "The `const*` columns are that predictor, computed from the "
        "per-position modal character on the identical rows as each model. "
        "Read **EM** first - a transcription with one wrong digit is the wrong "
        "specimen, so exact match is the number that matters operationally - "
        "and **suffixCharAcc** as the continuous companion, since the variable "
        "suffix is the part that requires reading.",
        "",
        "## Overall",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for _, r in show.iterrows():
        md.append("| " + " | ".join(
            (r[c] if isinstance(r[c], str) else
             ("%d" % r[c] if c == "n" else "%.3f" % r[c]))
            for c in cols) + " |")

    if len(po):
        pc = list(po.columns)
        md += ["", "## Per-character-position accuracy", "",
               "Position 0 is the leading letter. Accuracy that is high on the "
               "left and collapses on the right is the signature of a model "
               "reproducing the constant prefix rather than reading the label.",
               "",
               "| " + " | ".join(pc) + " |", "|" + "---|" * len(pc)]
        for _, r in po.iterrows():
            md.append("| " + " | ".join(
                (r[c] if isinstance(r[c], str) else "%.3f" % r[c])
                for c in pc) + " |")

    md += [
        "",
        "## Notes",
        "",
        "`wellformed` is the share of answers matching A followed by eleven "
        "digits. A low value alongside a high `charAcc` means the model can "
        "read but not follow the requested format, which is an "
        "instruction-following finding rather than a reading one.",
        "",
        "The resolver strips whitespace and separators that models insert into "
        "long digit runs, and uppercases the leading letter. It does not pad, "
        "repair the prefix, or snap to a nearby valid code: a wrong character "
        "stays wrong, or the error rate stops meaning anything.",
        "",
        "Every label is an independent specimen, so unlike the plate-based "
        "tasks in this component there is no clustering and the bootstrap "
        "resamples rows.",
        "",
    ]
    p3 = os.path.join(args.out_dir, "ocr_report.md")
    with open(p3, "w") as fh:
        fh.write("\n".join(md) + "\n")

    print()
    print("common prefix %r (%d chars)   constant-string charAcc %.3f"
          % (prefix, len(prefix),
             float(ov["constCharAcc"].iloc[0]) if len(ov) else float("nan")))
    for p in (p1, p2, p3):
        print("wrote " + p)
    print()
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
