#!/usr/bin/env python3
"""
Pre-run inspection of the OCR manifest. Read-only.

  python stats.py                     # reads chunks/ beside this file
  python stats.py --run_dir /path/to/5.OCR

The question this settles before any GPU time is spent: how much of each code
is actually informative?

These accession numbers share a fixed prefix. If most of every code is
constant, then character accuracy on the full string is close to free and the
only metrics worth reporting are exact match and accuracy on the variable
suffix. The optimal constant predictor is computed below so the floor is a
number rather than an impression.
"""

import argparse
import glob
import os
from collections import Counter

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


def levenshtein(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


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
    codes = df["answer"].astype(str).tolist()

    print("=" * 76)
    print("SOURCE  " + os.path.join(args.run_dir, "chunks"))
    print("=" * 76)
    print("rows          %d" % len(df))
    print("unique images %d" % df["image"].nunique())
    print("unique codes  %d" % df["answer"].nunique())
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
    print("CODE STRUCTURE AND THE CONSTANT-STRING FLOOR")
    print("-" * 76)
    lens = pd.Series([len(c) for c in codes])
    print("code length: %s" % dict(lens.value_counts().sort_index()))

    prefix = os.path.commonprefix(codes)
    L = int(lens.median())
    print("common prefix %r  -> %d of %d characters are identical everywhere"
          % (prefix, len(prefix), L))
    print()

    print("%-6s%12s%10s" % ("pos", "distinct", "modal share"))
    for i in range(L):
        col = [c[i] for c in codes if len(c) > i]
        cnt = Counter(col)
        top = cnt.most_common(1)[0]
        print("%-6d%12d%10.3f" % (i, len(cnt), top[1] / len(col)))
    print()

    modal = "".join(Counter([c[i] for c in codes if len(c) > i])
                    .most_common(1)[0][0] for i in range(L))
    tot = sum(len(c) for c in codes)
    hit = sum(sum(1 for i, ch in enumerate(c) if i < len(modal)
                  and modal[i] == ch) for c in codes)
    ed = sum(levenshtein(modal, c) for c in codes)
    em = sum(1 for c in codes if c == modal) / len(codes)
    suf = [c[len(prefix):] for c in codes]
    stot = sum(len(s) for s in suf)
    ms = modal[len(prefix):]
    shit = sum(sum(1 for i, ch in enumerate(s) if i < len(ms) and ms[i] == ch)
               for s in suf)

    print("OPTIMAL CONSTANT PREDICTOR  %r" % modal)
    print("  exact match                 %.3f" % em)
    print("  character accuracy, full    %.3f   <- inflated by the prefix" % (hit / tot))
    print("  character accuracy, suffix  %.3f   <- the honest floor"
          % (shit / stot if stot else float("nan")))
    print("  CER, full string            %.3f" % (ed / tot))
    print()
    print("Read EM first: one wrong digit is the wrong specimen. Read")
    print("suffixCharAcc as the continuous companion. Character accuracy on")
    print("the full code is reported for comparability but must be quoted")
    print("next to the constant-string figure above.")
    print()

    dup = int(df["answer"].duplicated().sum())
    if dup:
        print("!! %d rows share an answer code with another row" % dup)
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
