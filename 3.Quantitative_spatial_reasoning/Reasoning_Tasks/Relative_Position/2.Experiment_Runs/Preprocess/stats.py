#!/usr/bin/env python3
"""
Pre-run inspection of the Relative Position manifest. Read-only.

  python stats.py                     # reads chunks/ beside this file
  python stats.py --run_dir /path/to/8.Relative_Position

Four things this settles before any GPU time is spent.

1. What is the real floor?
   Four options means random choice scores 0.250, but the gold directions are
   badly skewed - wide photographs produce mostly horizontal separations. The
   majority-direction baseline below is the number a reported accuracy has to
   clear.

2. Is the gold answer always among the offered options, and is its position
   uniform? If the correct answer sits disproportionately first, a
   first-option-preferring model scores well for the wrong reason.

3. Do both conditions cover the same pairs?
   The clean-to-distractor drop is the headline contrast, and it is only clean
   if the two conditions are drawn from the same underlying pairs.

4. How badly are the rows clustered?
   Several pairs come from one source photograph, and each pair may appear in
   both conditions.
"""

import argparse
import glob
import os

import pandas as pd

SEC_PER_Q = {
    "qwenVL2_5_3B": 0.20, "llava_next_mistral_7B": 0.24,
    "llava_next_vicuna_7B": 0.26, "intern3_8B": 0.42, "intern3_5_8B": 0.42,
    "qwenVL2_5_7B": 0.38, "qwen3_8B": 0.40, "molmo_8B": 0.60,
    "llama_3_2_vision_11B": 0.46, "gemma3_12B": 0.50, "gemma4_12B": 0.50,
    "pixtral_12B": 0.53, "phi_3_5_vision": 1.20, "intern3_5_30B": 0.92,
    "qwenVL3_30B": 0.85, "qwenVL3_32B": 1.10, "intern3_38B": 1.20,
    "intern_3_5_38B": 1.20,
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

    print("=" * 78)
    print("SOURCE  " + os.path.join(args.run_dir, "chunks"))
    print("=" * 78)
    print("rows           %d" % len(df))
    print("unique images  %d" % df["image"].nunique())
    if "source_image" in df.columns:
        print("source photos  %d" % df["source_image"].nunique())
    if "pair_id" in df.columns:
        print("unique pairs   %d" % df["pair_id"].nunique())
    print("columns        %s" % list(df.columns))
    print()
    if "condition" in df.columns:
        print(pd.crosstab(df["dataset"], df["condition"]).to_string())
        print()

    if "image_abspath" in df.columns:
        uniq = df["image_abspath"].dropna().unique()
        miss = sum(0 if os.path.exists(p) else 1 for p in uniq)
        print("images: %d unique, %d missing" % (len(uniq), miss))
        if miss:
            print("  ATTENTION: fix --task_root before submitting anything.")
        print()

    print("-" * 78)
    print("THE REAL FLOOR")
    print("-" * 78)
    g = df["answer_dir"].astype(str)
    vc = g.value_counts(normalize=True)
    print("%-16s%10s%10s" % ("direction", "n", "share"))
    for k, v in vc.items():
        print("%-16s%10d%10.3f" % (k, int((g == k).sum()), v))
    print("-" * 78)
    print("%-16s%10d%10.3f   <- always-%s baseline"
          % ("majority", int((g == vc.index[0]).sum()), vc.iloc[0], vc.index[0]))
    print()
    shown = False
    if "question" in df.columns and len(df):
        q = str(df["question"].iloc[0])
        shown = ("Choose exactly one of:" in q) or ("Options:" in q)
    print("The prompt %s, so random choice scores %.3f."
          % ("lists four options" if shown else
             "does not list options - all eight directions are open",
             0.25 if shown else 0.125))
    print("The majority baseline above is higher, and it is the floor a")
    print("reported accuracy must clear. evaluate.py reports")
    print("lift = accuracy - majority.")
    if not shown:
        print()
        print("Because no options are shown, the option-position controls in")
        print("evaluate.py are skipped as inapplicable, and errOther becomes")
        print("meaningful: a wrong answer is no longer forced onto one of the")
        print("three mis-pairings.")
    print()
    if "condition" in df.columns:
        print("majority baseline per condition:")
        for c, sub in df.groupby("condition"):
            gg = sub["answer_dir"].astype(str)
            t = gg.mode().iloc[0]
            print("  %-12s %6d rows   %-14s %.3f"
                  % (c, len(sub), t, float((gg == t).mean())))
        print()

    print("-" * 78)
    print("OPTIONS")
    print("-" * 78)
    if "n_options" in df.columns:
        print("options offered per item: %s"
              % dict(df["n_options"].value_counts()))
    if "answer_in_options" in df.columns:
        bad = int((~df["answer_in_options"].astype(bool)).sum())
        print("rows whose gold answer is NOT offered: %d" % bad)
        if bad:
            print("  Those are unanswerable as posed. Drop or regenerate.")
    if "answer_pos" in df.columns:
        print()
        print("position of the gold answer among the options:")
        pv = df["answer_pos"].value_counts(normalize=True).sort_index()
        for k, v in pv.items():
            print("  position %d   %.3f" % (int(k), v))
        k = int(df["answer_pos"].max())
        print("  uniform would be %.3f each. A skew here means a model that" % (1.0 / k))
        print("  prefers a fixed option POSITION scores well for the wrong")
        print("  reason; evaluate.py reports optPosBias to catch it.")
    print()

    if {"mis_red_empty", "mis_blue_empty", "mis_both_empty"} <= set(df.columns):
        print("-" * 78)
        print("ERROR ATTRIBUTION IS AVAILABLE")
        print("-" * 78)
        have = int(df["mis_red_empty"].notna().sum())
        print("%d of %d rows carry the three recomputed mis-pairings."
              % (have, len(df)))
        print("For those, every wrong option corresponds to a specific box")
        print("confusion, so evaluate.py can say WHICH box a model grabbed")
        print("rather than only that it was wrong.")
        if "gt_recomputed" in df.columns:
            chk = df["gt_recomputed"].notna()
            mism = int((df.loc[chk, "gt_recomputed"].astype(str)
                        != df.loc[chk, "answer_dir"].astype(str)).sum())
            print("geometry check: %d mismatches between the recomputed and"
                  % mism)
            print("stored directions." + ("" if mism == 0 else
                  "  !! Resolve before running."))
        print()

    print("-" * 78)
    print("CLUSTERING")
    print("-" * 78)
    key = "source_image" if "source_image" in df.columns else "image"
    per = df.groupby(key).size()
    print("rows per source photograph: mean %.2f  max %d"
          % (per.mean(), per.max()))
    print("Effective sample size is the photograph count (%d), not the row"
          % df[key].nunique())
    print("count (%d). evaluate.py bootstraps over source images." % len(df))
    print()

    n = len(df)
    print("=" * 78)
    print("ESTIMATED RUNTIME  (%d rows)" % n)
    print("=" * 78)
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