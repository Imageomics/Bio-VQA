#!/usr/bin/env python3
"""
Pre-run inspection of the Counting manifest. Read-only.

  python stats.py                      # reads chunks/ beside this file
  python stats.py --run_dir /path/to/1.Counting

Two numbers decide how the results must be written up:

  modalEM   accuracy of a model that always answers that group's most common
            count. This, not zero, is the floor.
  modalCAS  the same constant predictor scored on the graded Count Accuracy
            Score. On skewed datasets this can be surprisingly high.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

SEC_PER_Q = {
    "qwenVL2_5_3B": 0.15, "llava_next_mistral_7B": 0.20,
    "llava_next_vicuna_7B": 0.22, "intern3_8B": 0.30, "intern3_5_8B": 0.30,
    "qwenVL2_5_7B": 0.28, "qwen3_8B": 0.30, "molmo_8B": 0.45,
    "llama_3_2_vision_11B": 0.35, "gemma3_12B": 0.38, "gemma4_12B": 0.38,
    "pixtral_12B": 0.40, "phi_3_5_vision": 0.90, "intern3_5_30B": 0.70,
    "qwenVL3_30B": 0.65, "qwenVL3_32B": 0.85, "intern3_38B": 0.90,
    "intern_3_5_38B": 0.90,
}


def cas(gold, pred):
    return np.maximum(0.0, 1.0 - np.abs(pred - gold) / np.maximum(np.abs(gold), 1.0))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=here)
    ap.add_argument("--check_images", action="store_true", default=True)
    ap.add_argument("--no_check_images", dest="check_images",
                    action="store_false")
    args = ap.parse_args()

    parts = sorted(glob.glob(os.path.join(args.run_dir, "chunks", "chunk*",
                                          "qa.csv")))
    if not parts:
        raise SystemExit("no chunks found. Run split_qa.py first.")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

    print("=" * 78)
    print("SOURCE  " + os.path.join(args.run_dir, "chunks"))
    print("=" * 78)
    print("rows            %d" % len(df))
    print("unique images   %d" % df["image"].nunique())
    print("columns         %s" % list(df.columns))
    ans = pd.to_numeric(df["answer"], errors="coerce")
    print("non-numeric ans %d" % int(ans.isna().sum()))
    print("count range     %d - %d" % (ans.min(), ans.max()))
    print()

    if args.check_images and "image_abspath" in df.columns:
        uniq = df["image_abspath"].dropna().unique()
        miss = sum(0 if os.path.exists(p) else 1 for p in uniq)
        print("images: %d unique, %d missing" % (len(uniq), miss))
        if miss:
            print("  ATTENTION: fix paths before submitting anything.")
        print()

    print("-" * 78)
    print("%-12s%-16s%8s%8s%7s%7s%7s%10s%10s"
          % ("tier", "dataset", "rows", "mean", "med", "min", "max",
             "modalEM", "modalCAS"))
    print("-" * 78)
    tmp = df.assign(_a=ans)
    keys = [k for k in ("tier", "source_dataset") if k in df.columns]
    for k, g in tmp.groupby(keys):
        if not isinstance(k, tuple):
            k = (k, "")
        a = g["_a"].dropna().values.astype(float)
        if not len(a):
            continue
        modal = float(pd.Series(a).mode().iloc[0])
        mp = np.full_like(a, modal)
        print("%-12s%-16s%8d%8.1f%7.0f%7.0f%7.0f%10.3f%10.3f"
              % (str(k[0]), str(k[1])[:15], len(a), a.mean(), np.median(a),
                 a.min(), a.max(), float((mp == a).mean()),
                 float(cas(a, mp).mean())))
    a = ans.dropna().values.astype(float)
    modal = float(pd.Series(a).mode().iloc[0])
    mp = np.full_like(a, modal)
    print("-" * 78)
    print("%-12s%-16s%8d%8.1f%7.0f%7.0f%7.0f%10.3f%10.3f"
          % ("ALL", "", len(a), a.mean(), np.median(a), a.min(), a.max(),
             float((mp == a).mean()), float(cas(a, mp).mean())))
    print()
    print("modalEM / modalCAS = a model that always answers the group's most")
    print("common count. Any reported result must clear these.")
    print()

    if "panel" in df.columns:
        print("panel balance: %s" % dict(df["panel"].value_counts()))
    if "question" in df.columns:
        print()
        print("question variants: %d" % df["question"].nunique())
    print()

    n = len(df)
    print("=" * 78)
    print("ESTIMATED RUNTIME  (%d rows, all three tiers in one pass)" % n)
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
