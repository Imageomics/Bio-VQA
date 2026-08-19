#!/usr/bin/env python3
"""
Pre-run inspection of the Morphometry manifest. Read-only.

  python stats.py                     # reads chunks/ beside this file
  python stats.py --run_dir /path/to/3.Morphometry

Three questions this settles before any GPU time is spent.

1. Is the answer distribution wide enough for the task to be measurable?
   If a single constant answer scores well, relative error cannot separate
   models and rho must lead. The best-constant baseline is printed below.

2. How badly are the rows clustered?
   Every beetle on a plate shares a scale bar and a magnification, so the
   effective sample size is much closer to the plate count than to the row
   count. The within-plate to total SD ratio quantifies it.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

SEC_PER_Q = {
    "qwenVL2_5_3B": 0.30, "llava_next_mistral_7B": 0.35,
    "llava_next_vicuna_7B": 0.38, "intern3_8B": 0.60, "intern3_5_8B": 0.60,
    "qwenVL2_5_7B": 0.55, "qwen3_8B": 0.58, "molmo_8B": 0.85,
    "llama_3_2_vision_11B": 0.65, "gemma3_12B": 0.70, "gemma4_12B": 0.70,
    "pixtral_12B": 0.75, "phi_3_5_vision": 1.60, "intern3_5_30B": 1.30,
    "qwenVL3_30B": 1.20, "qwenVL3_32B": 1.55, "intern3_38B": 1.70,
    "intern_3_5_38B": 1.70,
}


def best_constant(g):
    """Value minimising mean relative error, and that error."""
    grid = np.linspace(g.min(), g.max(), 1500)
    err = np.array([np.abs(c / g - 1.0).mean() for c in grid])
    i = int(err.argmin())
    return float(grid[i]), float(err[i])


def best_constant_acc(g, tol=0.20):
    """Value maximising accuracy within tol, and that accuracy.

    A baseline has to optimise the metric it is compared against. The
    relErr-optimal constant and the acc-optimal constant are far apart here,
    and using the wrong one understates the floor by more than ten points."""
    grid = np.linspace(g.min(), g.max(), 1500)
    acc = np.array([(np.abs(c / g - 1.0) <= tol).mean() for c in grid])
    i = int(acc.argmax())
    return float(grid[i]), float(acc[i])


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
    print("ANSWER DISTRIBUTION AND THE CONSTANT-PREDICTOR FLOOR")
    print("-" * 76)
    print("%-9s%7s%8s%7s%7s%7s%7s%9s%9s%9s%9s"
          % ("measure", "n", "mean", "sd", "CV", "min", "max",
             "cRelErr", "relErr", "cAcc20", "acc@20%"))
    tot_acc, tot_n = 0.0, 0
    for mt, g in df.groupby("measure_type"):
        a = pd.to_numeric(g["answer_cm"], errors="coerce").dropna().values
        cr, er = best_constant(a)
        ca, aa = best_constant_acc(a)
        tot_acc += aa * len(a); tot_n += len(a)
        print("%-9s%7d%8.3f%7.3f%7.3f%7.2f%7.2f%9.3f%9.3f%9.3f%9.3f"
              % (mt, len(a), a.mean(), a.std(), a.std() / a.mean(), a.min(),
                 a.max(), cr, er, ca, aa))
    print("-" * 76)
    print("%-9s%7d%8s%7s%7s%7s%7s%9s%9s%9s%9.3f"
          % ("PER-TYPE", tot_n, "", "", "", "", "", "", "", "",
             tot_acc / max(tot_n, 1)))
    print()
    print("cRelErr / cAcc20 are the constants that minimise mean relative")
    print("error and maximise acc@20 respectively - they are NOT the same")
    print("value, so a baseline must be fitted against the metric it is")
    print("compared to.")
    print()
    print("The PER-TYPE row is the real floor: the question says whether")
    print("length or width is wanted, so answering one fixed number for every")
    print("length question and another for every width question needs no")
    print("measurement at all. Any reported acc@20 must clear it.")
    print()
    print("That same predictor also inflates POOLED rho to about +0.63,")
    print("because length answers are systematically larger than width ones.")
    print("evaluate.py therefore reports rho computed within each measurement")
    print("type. Do not quote a pooled rho for this task.")
    print()

    # within vs between plate
    w = df.groupby("plate_id")["answer_cm"].std().mean()
    t = df["answer_cm"].std()
    print("within-plate SD %.4f   total SD %.4f   ratio %.2f"
          % (w, t, w / t if t else float("nan")))
    print("A low ratio means beetles on one plate are near-replicates, so the")
    print("effective sample size is closer to the plate count (%d) than to the"
          % df["plate_id"].nunique())
    print("row count (%d). evaluate.py bootstraps over plates for this reason."
          % len(df))
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