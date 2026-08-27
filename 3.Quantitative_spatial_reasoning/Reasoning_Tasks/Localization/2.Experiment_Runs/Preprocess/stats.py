#!/usr/bin/env python3
"""
Pre-run inspection of the Localization manifest. Read-only.

  python stats.py                     # reads chunks/ beside this file
  python stats.py --run_dir /path/to/7.Localization

Three things this settles before any GPU time is spent.

1. What do the trivial baselines score?
   Predicting the whole canvas, or the middle 50%, or the tier's average gold
   box, requires no vision at all. Single-animal photographs often have the
   subject filling much of the frame, so those floors are not always
   negligible - and a reported IoU below them is not a localisation.

2. How large are the targets?
   On the Easy tier the subject is framed on, averaging 37% of the image and
   reaching 99.8%. That is why the baselines below are so high, and why the
   discrimination sits at acc@.75 and acc@.90 rather than at the mean.

3. Do the gold boxes fit inside their own canvases?
   A box that does not is a benchmark defect. split_qa.py flags them and
   evaluate.py excludes them; this is where you see how many there are.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

# Single photographs are far cheaper than the 12,000-pixel multi-panel
# canvases, so these are well below the figures used for the full three-tier
# manifest.
SEC_PER_Q = {
    "qwenVL2_5_3B": 0.18, "llava_next_mistral_7B": 0.22,
    "llava_next_vicuna_7B": 0.24, "intern3_8B": 0.38, "intern3_5_8B": 0.38,
    "qwenVL2_5_7B": 0.34, "qwen3_8B": 0.36, "molmo_8B": 0.55,
    "llama_3_2_vision_11B": 0.42, "gemma3_12B": 0.45, "gemma4_12B": 0.45,
    "pixtral_12B": 0.48, "phi_3_5_vision": 1.10, "intern3_5_30B": 0.85,
    "qwenVL3_30B": 0.78, "qwenVL3_32B": 1.00, "intern3_38B": 1.10,
    "intern_3_5_38B": 1.10,
}


def iou(g, p):
    gx1, gy1, gx2, gy2 = g[0], g[1], g[0] + g[2], g[1] + g[3]
    px1, py1, px2, py2 = p[0], p[1], p[0] + p[2], p[1] + p[3]
    ix1, iy1 = max(gx1, px1), max(gy1, py1)
    ix2, iy2 = min(gx2, px2), min(gy2, py2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    u = (gx2 - gx1) * (gy2 - gy1) + (px2 - px1) * (py2 - py1) - inter
    return inter / u if u > 0 else 0.0


def baseline_row(d, name, boxes):
    g = list(zip(d["gt_x"], d["gt_y"], d["gt_w"], d["gt_h"]))
    v = np.array([iou(a, b) for a, b in zip(g, boxes)])
    return "%-14s%9.3f%9.3f%9.3f%9.3f%9.3f" % (
        name, v.mean(), (v >= .25).mean(), (v >= .5).mean(),
        (v >= .75).mean(), (v >= .9).mean())


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
    print("rows          %d" % len(df))
    print("unique images %d" % df["image"].nunique())
    print("columns       %s" % list(df.columns))
    print()
    print(df.groupby("tier").size().to_string())
    print()

    if "image_abspath" in df.columns:
        uniq = df["image_abspath"].dropna().unique()
        miss = sum(0 if os.path.exists(p) else 1 for p in uniq)
        print("images: %d unique, %d missing" % (len(uniq), miss))
        if miss:
            print("  ATTENTION: fix --task_root / --id_root before submitting.")
        print()

    if "gt_out_of_bounds" in df.columns:
        oob = int(df["gt_out_of_bounds"].fillna(False).astype(bool).sum())
        print("gold boxes outside their own canvas: %d" % oob)
        if oob:
            print("  evaluate.py excludes these; report the count as a")
            print("  benchmark limitation.")
        print()

    print("-" * 78)
    print("CANVAS AND TARGET SIZE")
    print("-" * 78)
    print(df.groupby("tier")[["img_w", "img_h"]].median().to_string())
    print()
    print("gold box area as a fraction of the canvas:")
    print(df.groupby("tier")["gt_area_frac"].describe()
          [["mean", "min", "25%", "50%", "75%", "max"]].round(5).to_string())
    print()
    tiny = float((df["gt_area_frac"] < 0.01).mean())
    print("targets under 1%% of the canvas: %.1f%% of rows" % (100 * tiny))
    print("Those are where localisation collapses. evaluate.py reports size")
    print("quartiles so the failure can be attributed rather than pooled.")
    print()

    print("-" * 78)
    print("TRIVIAL BASELINES  (no vision required)")
    print("-" * 78)
    print("%-14s%9s%9s%9s%9s%9s" % ("baseline", "meanIoU", "acc@.25",
                                     "acc@.50", "acc@.75", "acc@.90"))
    for tier, g in df.groupby("tier"):
        W = g["img_w"].values.astype(float)
        H = g["img_h"].values.astype(float)
        fx = float((g["gt_x"] / g["img_w"]).mean())
        fy = float((g["gt_y"] / g["img_h"]).mean())
        fw = float((g["gt_w"] / g["img_w"]).mean())
        fh = float((g["gt_h"] / g["img_h"]).mean())
        print("  tier=%s  (n=%d)" % (tier, len(g)))
        print("  " + baseline_row(g, "wholeImage",
                                  [(0., 0., w, h) for w, h in zip(W, H)]))
        print("  " + baseline_row(g, "centre50",
                                  [(w * .25, h * .25, w * .5, h * .5)
                                   for w, h in zip(W, H)]))
        print("  " + baseline_row(g, "meanBox",
                                  [(w * fx, h * fy, w * fw, h * fh)
                                   for w, h in zip(W, H)]))
    print()
    print("Any reported IoU must clear all three on its own tier.")
    print()
    print("Where a floor is high, the discrimination has moved to the tighter")
    print("thresholds: read acc@.75 and acc@.90 on that tier instead of the")
    print("mean, and quote every model as a lift over its own baseline.")
    print()

    print("-" * 78)
    print("CLUSTERING")
    print("-" * 78)
    per = df.groupby("image_abspath").size()
    print("rows per image: mean %.2f  max %d" % (per.mean(), per.max()))
    nb, na = df["image"].nunique(), df["image_abspath"].nunique()
    if nb != na:
        print("NOTE: %d distinct files, %d distinct basenames - keyed on the "
              "file." % (na, nb))
    print("Effective sample size is the image count (%d), not the row count"
          % na)
    print("(%d). evaluate.py bootstraps over images for this reason." % len(df))
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