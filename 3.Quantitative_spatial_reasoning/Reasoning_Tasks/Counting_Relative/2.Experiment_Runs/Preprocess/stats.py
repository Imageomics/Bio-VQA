#!/usr/bin/env python3
"""
Pre-run inspection of the Counting-Relative manifest. Read-only.

  python stats.py                     # reads chunks/ beside this file
  python stats.py --run_dir /path/to/6.Counting_Relative

Three things this settles before any GPU time is spent.

1. Is the answer key balanced over A/B/C/D?
   If it is, chance is 0.250 for the `same` condition and plain accuracy is
   meaningful there. Any large imbalance becomes the floor instead.

2. Are both polarities present for every grid?
   Each image should carry a most and a least question. A grid with only one
   breaks the incoherence check in evaluate.py.

3. How badly are the rows clustered?
   Two questions per image, so the effective sample size is the image count,
   not the row count.

It does NOT recompute the dataset-prior floor for the `different` condition -
that needs the per-panel source datasets, which the manifest does not carry.
Use the Monte-Carlo figures (about 0.53 most, 0.55 least) as evaluate.py's
--prior_different_* arguments.
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
    print("columns       %s" % list(df.columns))
    print()
    print(pd.crosstab(df["condition"], df["polarity"]).to_string())
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
    tab = pd.crosstab(g, [df["condition"], df["polarity"]], normalize="columns")
    print(tab.round(3).to_string())
    print()
    top = g.mode().iloc[0]
    print("overall majority letter %s at %.3f" % (top, float((g == top).mean())))
    print()
    print("If the shares sit near 0.250 the key is balanced. A single letter")
    print("at 1.000 would mean the gold extraction broke - the word 'Image'")
    print("contains a lower-case a, and a naive [ABCD] search returns A for")
    print("every option label.")
    print()

    print("-" * 76)
    print("POLARITY PAIRING")
    print("-" * 76)
    per = df.groupby("image")["polarity"].nunique()
    print("images with both most and least : %d" % int((per == 2).sum()))
    print("images with only one polarity   : %d" % int((per == 1).sum()))
    if int((per == 1).sum()):
        print("  Those grids cannot contribute to the incoherence check.")
    print()

    print("-" * 76)
    print("CLUSTERING")
    print("-" * 76)
    rows_per = df.groupby("image").size()
    print("rows per image: mean %.2f  max %d" % (rows_per.mean(), rows_per.max()))
    print("Effective sample size is the image count (%d), not the row count"
          % df["image"].nunique())
    print("(%d). evaluate.py bootstraps over images for this reason." % len(df))
    print()

    if df["source_dataset"].notna().any():
        print("-" * 76)
        print("SAME-CONDITION SOURCE DATASETS")
        print("-" * 76)
        print(df["source_dataset"].value_counts().to_string())
        print()

    print("-" * 76)
    print("THE different-CONDITION FLOOR")
    print("-" * 76)
    print("The four panels there come from four different datasets, and count")
    print("magnitude is partly predictable from the domain: Bees average about")
    print("37 animals per frame, Camel and Kiang about 8. A model recognising")
    print("which panel is an insect plate can answer without counting.")
    print()
    print("Monte-Carlo over the empirical count distributions puts that")
    print("strategy at about 0.530 (most) and 0.548 (least). Pass those to")
    print("evaluate.py:")
    print("  --prior_different_most 0.530 --prior_different_least 0.548")
    print("They are defaults there, but state them wherever a number is")
    print("quoted - they are a property of this corpus, not a constant.")
    print()

    n = len(df)
    print("=" * 76)
    print("ESTIMATED RUNTIME  (%d rows)" % n)
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
