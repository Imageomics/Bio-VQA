#!/usr/bin/env python3
"""
Pre-run inspection of the Relative Size manifest. Read-only.

  python stats.py                     # reads chunks/ beside this file
  python stats.py --run_dir /path/to/4.Distance_Relative

Three things this settles before any GPU time is spent.

1. Is the answer key balanced over A/B/C/D?
   If it is, chance is 0.250 and plain accuracy is meaningful. If one letter
   dominates, that share is the floor instead and evaluate.py's balAcc becomes
   the column to read.

2. How badly are the rows clustered?
   Up to twenty question sets are drawn from the same plate, reusing the same
   specimens in different combinations, so the effective sample size is much
   closer to the plate count than to the row count.

3. Is the length/width split even?
   The generator required a 5% margin on length but 10% on width, so the two
   are not equally hard and should be reported separately.
"""

import argparse
import glob
import os

import pandas as pd

LETTERS = ["A", "B", "C", "D"]

SEC_PER_Q = {
    "qwenVL2_5_3B": 0.30, "llava_next_mistral_7B": 0.35,
    "llava_next_vicuna_7B": 0.38, "intern3_8B": 0.60, "intern3_5_8B": 0.60,
    "qwenVL2_5_7B": 0.55, "qwen3_8B": 0.58, "molmo_8B": 0.85,
    "llama_3_2_vision_11B": 0.65, "gemma3_12B": 0.70, "gemma4_12B": 0.70,
    "pixtral_12B": 0.75, "phi_3_5_vision": 1.60, "intern3_5_30B": 1.30,
    "qwenVL3_30B": 1.20, "qwenVL3_32B": 1.55, "intern3_38B": 1.70,
    "intern_3_5_38B": 1.70,
}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=here)
    args = ap.parse_args()

    parts = sorted(glob.glob(os.path.join(args.run_dir, "chunks", "chunk*",
                                          "qa.csv")))
    if not parts:
        raise SystemExit("no chunks found. Run split_qa.py first.")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

    print("=" * 76)
    print("SOURCE  " + os.path.join(args.run_dir, "chunks"))
    print("=" * 76)
    print("rows          %d" % len(df))
    print("unique images %d" % df["image"].nunique())
    print("plates        %d" % df["plate_id"].nunique())
    print("columns       %s" % list(df.columns))
    print()

    if "image_abspath" in df.columns:
        uniq = df["image_abspath"].dropna().unique()
        miss = sum(0 if os.path.exists(p) else 1 for p in uniq)
        print("images: %d unique, %d missing" % (len(uniq), miss))
        if miss:
            print("  ATTENTION: fix paths before submitting anything.")
        print()

    print("-" * 76)
    print("ANSWER KEY BALANCE  (chance = 0.250 if uniform)")
    print("-" * 76)
    g = df["answer_letter"].astype(str).str.upper()
    print("%-14s%10s%10s" % ("letter", "n", "share"))
    for l in LETTERS:
        print("%-14s%10d%10.3f" % (l, int((g == l).sum()), float((g == l).mean())))
    top = g.mode().iloc[0]
    print("-" * 76)
    print("%-14s%10d%10.3f   <- always-%s baseline"
          % ("majority", int((g == top).sum()), float((g == top).mean()), top))
    print()
    print("If the shares are near 0.250 the key is balanced, chance is 0.250,")
    print("and plain accuracy is meaningful. Any large imbalance becomes the")
    print("floor instead - read balAcc in that case.")
    print()

    if "measure_type" in df.columns:
        print("-" * 76)
        print("MEASUREMENT TYPE")
        print("-" * 76)
        for mt, sub in df.groupby("measure_type", dropna=False):
            gg = sub["answer_letter"].astype(str).str.upper()
            t = gg.mode().iloc[0]
            print("%-10s %6d rows   majority %s at %.3f"
                  % (str(mt), len(sub), t, float((gg == t).mean())))
        print()
        print("The generator required the winner to beat the runner-up by 5%")
        print("on length but 10% on width, so width items have a wider margin")
        print("and should be easier. Report the two separately.")
        print()

    print("-" * 76)
    print("CLUSTERING")
    print("-" * 76)
    per = df.groupby("plate_id").size()
    print("rows per plate: mean %.1f  median %.0f  max %d"
          % (per.mean(), per.median(), per.max()))
    print("Effective sample size is closer to the plate count (%d) than to the"
          % df["plate_id"].nunique())
    print("row count (%d): the same specimens recur across question sets from"
          % len(df))
    print("one plate. evaluate.py bootstraps over plates for this reason.")
    print()

    n = len(df)
    print("=" * 76)
    print("ESTIMATED RUNTIME  (%d rows, near-native resolution)" % n)
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
