#!/usr/bin/env python3
"""
Pre-run inspection of the Viewpoint manifest. Read-only.

  python stats.py                       # reads chunks/ beside this file
  python stats.py --run_dir /path/to/2.Viewpoint

The blocking question this answers: does every gold label fall inside the
accepted-answer list that goes into the prompt? A gold label outside that list
is unanswerable as prompted and would be scored as a model failure when it is a
benchmark defect. If any are found, the exact export line to fix it is printed.

The second question: what does a constant predictor score? Wildbook viewpoints
are heavily skewed toward left and right, and Grevys zebra is photographed from
the right flank by census protocol, so "always answer right" is a strong
strategy. That number, not 1/k, is the floor.
"""

import argparse
import glob
import os
import re

import pandas as pd

DEFAULT_LABELS = ["front", "back", "left", "right", "frontleft", "frontright",
                  "backleft", "backright", "up", "down"]

SEC_PER_Q = {
    "qwenVL2_5_3B": 0.15, "llava_next_mistral_7B": 0.20,
    "llava_next_vicuna_7B": 0.22, "intern3_8B": 0.30, "intern3_5_8B": 0.30,
    "qwenVL2_5_7B": 0.28, "qwen3_8B": 0.30, "molmo_8B": 0.45,
    "llama_3_2_vision_11B": 0.35, "gemma3_12B": 0.38, "gemma4_12B": 0.38,
    "pixtral_12B": 0.40, "phi_3_5_vision": 0.90, "intern3_5_30B": 0.70,
    "qwenVL3_30B": 0.65, "qwenVL3_32B": 0.85, "intern3_38B": 0.90,
    "intern_3_5_38B": 0.90,
}


def canon(x):
    return re.sub(r"[^a-z]", "", str(x).lower())


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=here)
    ap.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    args = ap.parse_args()

    labels = [s.strip().lower() for s in args.labels.split(",") if s.strip()]
    parts = sorted(glob.glob(os.path.join(args.run_dir, "chunks", "chunk*",
                                          "qa.csv")))
    if not parts:
        raise SystemExit("no chunks found. Run split_qa.py first.")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    gold = df["answer"].map(canon)

    print("=" * 76)
    print("SOURCE  " + os.path.join(args.run_dir, "chunks"))
    print("=" * 76)
    print("rows          %d" % len(df))
    print("unique images %d" % df["image"].nunique())
    print("columns       %s" % list(df.columns))
    print()

    print("-" * 76)
    print("ACCEPTED ANSWERS IN PROMPT")
    print("-" * 76)
    print("  " + ", ".join(labels))
    print()
    outside = sorted(set(gold) - set(labels))
    if outside:
        print("*** %d GOLD LABELS ARE NOT IN THE ACCEPTED LIST ***"
              % len(outside))
        for lab in outside:
            print("    %-16s %6d rows" % (lab, int((gold == lab).sum())))
        print()
        print("    Those rows cannot be answered as prompted. Either add them:")
        print('      export VIEWPOINT_LABELS="%s"' % ",".join(labels + outside))
        print("    or drop them from the manifest before running.")
    else:
        print("OK: every gold label is inside the accepted list.")
    labs_present = [l for l in labels if (gold == l).any()]
    unused = [l for l in labels if l not in labs_present]
    if unused:
        print()
        print("NOTE: %s never occur in the gold data. Listing options that "
              % ", ".join(unused))
        print("      never occur adds distractors without measuring anything;")
        print("      consider dropping them from VIEWPOINT_LABELS.")
    print()

    print("-" * 76)
    print("LABEL DISTRIBUTION")
    print("-" * 76)
    print((pd.crosstab(gold, df["tier"]) if "tier" in df.columns
           else gold.value_counts().to_frame("n")).to_string())
    print()

    print("-" * 76)
    print("MAJORITY-CLASS BASELINE (always answer that group's commonest label)")
    print("-" * 76)
    keys = [k for k in ("tier", "dataset") if k in df.columns]
    print("%-12s%-14s%8s%11s%14s" % ("tier", "dataset", "n", "majority", "label"))
    tmp = df.assign(_g=gold)
    if keys:
        for k, g in tmp.groupby(keys):
            if not isinstance(k, tuple):
                k = (k, "")
            top = g["_g"].mode().iloc[0]
            print("%-12s%-14s%8d%11.3f%14s"
                  % (str(k[0]), str(k[1])[:13], len(g),
                     float((g["_g"] == top).mean()), top))
    top = gold.mode().iloc[0]
    print("-" * 76)
    print("%-12s%-14s%8d%11.3f%14s"
          % ("ALL", "", len(gold), float((gold == top).mean()), top))
    print()
    print("Chance with %d options is %.3f. Anything far above that is the real"
          % (len(labels), 1.0 / len(labels)))
    print("floor, and belongs next to the results in the paper.")
    print()

    if "image_abspath" in df.columns:
        uniq = df["image_abspath"].dropna().unique()
        miss = sum(0 if os.path.exists(p) else 1 for p in uniq)
        print("images: %d unique, %d missing" % (len(uniq), miss))
        if miss:
            print("  ATTENTION: fix paths before submitting anything.")
        print()

    n = len(df)
    print("=" * 76)
    print("ESTIMATED RUNTIME  (%d rows, all three tiers in one pass)" % n)
    print("=" * 76)
    total = 0.0
    for m, s in SEC_PER_Q.items():
        h = n * s / 3600.0
        total += h
        print("%-26s%10.2f%10.2f" % (m, s, h))
    print("-" * 46)
    print("%-26s%10s%10.2f" % ("TOTAL (18 models, serial)", "", total))
    print("with 6 chunks and 3 lanes: roughly %.1f h wall clock"
          % (total / 6 / 3))


if __name__ == "__main__":
    main()
