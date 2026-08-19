#!/usr/bin/env python3
"""
Parse the Relative Size qa.csv into a flat manifest and shard it for SLURM.

  python split_qa.py --task_root "/path/to/Distance-Relative/Hard"

Nothing is hardcoded: --run_dir defaults to this script's own directory and
--task_root is required.

WHAT THIS TASK IS
Four beetles on ONE plate, each marked with a coloured dotted box and a line
along its elytra. The model picks which has the largest elytral length (or
width). All four share a magnification and the lines are drawn before any
resizing, so on-screen line length is exactly proportional to real size and no
scale bar is needed. That makes it the clean within-plate counterpart to the
absolute morphometry task - and unlike the 2x2 grid variant, its ground truth
and its pixels agree.

INPUT SCHEMA
Two columns, image and qa_pairs, where qa_pairs is a JSON object
{"Q": ..., "A": "Beetle C"}. Flattened here so nothing downstream parses JSON.

COLUMNS ADDED
  row_id         restores original order at merge time
  image_abspath  resolved here, once. Chunk CSVs live in chunks/chunkNN/ while
                 images stay in <task_root>/Images/, so a relative path would
                 be unresolvable from every chunk.
  question       the Q field
  answer         the A field verbatim, e.g. "Beetle C"
  answer_letter  A | B | C | D - what evaluate.py scores against
  measure_type   length | width, parsed from the question
  plate_id       specimen-plate accession from the filename prefix
  set_index      which draw of four beetles this is from that plate

WHY plate_id MATTERS
Up to 20 question sets are drawn from the same plate, reusing the same
specimens in different combinations. Those rows are not independent
observations. evaluate.py bootstraps over plate_id; stats.py prints how many
rows each plate contributes so the effective sample size is visible.

WHY THE SHUFFLE
Rows arrive grouped by plate and, within a plate, in set order. A plain
i::n stride would hand each chunk a systematic slice. A seeded shuffle before
interleaving mixes them; row_id is assigned first, so the original order is
still recoverable at merge time.
"""

import argparse
import ast
import json
import os
import re
import shutil

import pandas as pd


def parse_qa(cell):
    """qa_pairs holds JSON; fall back to a Python literal if it is not valid."""
    s = str(cell)
    try:
        d = json.loads(s)
    except Exception:
        try:
            d = ast.literal_eval(s)
        except Exception:
            return None, None
    if not isinstance(d, dict):
        return None, None
    return d.get("Q"), d.get("A")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_root", required=True,
                    help="directory holding qa.csv and Images/")
    ap.add_argument("--run_dir", default=here)
    ap.add_argument("--n_chunks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42,
                    help="shuffle seed; recorded so the split is reproducible")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    csv_path = os.path.join(args.task_root, "qa.csv")
    img_dir = os.path.join(args.task_root, "Images")
    if not os.path.isfile(csv_path):
        raise SystemExit("qa.csv not found at " + csv_path)

    chunk_root = os.path.join(args.run_dir, "chunks")
    if os.path.isdir(chunk_root):
        if not args.force:
            raise SystemExit(
                "chunks/ already exists at %s\n"
                "  Re-splitting would orphan results already written there.\n"
                "  Use --force to wipe and rebuild." % chunk_root)
        shutil.rmtree(chunk_root)

    df = pd.read_csv(csv_path)
    print("source  %s" % csv_path)
    print("rows    %d" % len(df))

    parsed = df["qa_pairs"].map(parse_qa)
    df["question"] = [p[0] for p in parsed]
    df["answer"] = [p[1] for p in parsed]
    bad = int(df["question"].isna().sum())
    if bad:
        print("!! %d rows had unparseable qa_pairs" % bad)
        df = df[df["question"].notna()].copy()

    # "Beetle C" -> C.
    # NOT a bare [ABCD] search: the word "Beetle" itself contains a B, so a
    # first-match regex resolves every single answer to B. Anchor on the word,
    # then fall back to a standalone letter for any other phrasing.
    letter = df["answer"].astype(str).str.extract(
        r"beetle\s*([ABCD])\b", flags=re.I, expand=False)
    fallback = df["answer"].astype(str).str.extract(
        r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", expand=False)
    df["answer_letter"] = letter.fillna(fallback).str.upper()
    nolabel = int(df["answer_letter"].isna().sum())
    if nolabel:
        print("!! %d answers did not contain an option letter" % nolabel)
        print("   e.g. %r" % df.loc[df["answer_letter"].isna(),
                                    "answer"].iloc[0])
        df = df[df["answer_letter"].notna()].copy()

    df["measure_type"] = df["question"].str.extract(
        r"elytral\s+(length|width)", flags=re.I, expand=False).str.lower()
    unknown = int(df["measure_type"].isna().sum())
    if unknown:
        print("!! %d questions did not name length or width" % unknown)

    df["image"] = df["image"].astype(str).map(os.path.basename)
    df["image_abspath"] = df["image"].map(lambda n: os.path.join(img_dir, n))

    # filename is <plate>_set<N>_<hash>.jpg, e.g. A00000050030_set0_582333.jpg
    df["plate_id"] = df["image"].str.replace(
        r"_set\d+_[0-9a-f]+\.jpe?g$", "", regex=True, flags=re.I)
    df["set_index"] = pd.to_numeric(
        df["image"].str.extract(r"_set(\d+)_", expand=False), errors="coerce")

    missing = int((~df["image_abspath"].map(os.path.exists)).sum())
    print("images  %d unique, %d missing" % (df["image"].nunique(), missing))
    if missing:
        print("        first missing: %s"
              % df.loc[~df["image_abspath"].map(os.path.exists),
                       "image_abspath"].iloc[0])

    keep = ["plate_id", "set_index", "measure_type", "image", "image_abspath",
            "question", "answer", "answer_letter"]
    df = df[keep].reset_index(drop=True)
    df.insert(0, "row_id", range(len(df)))

    # Deterministic shuffle BEFORE interleaving; see the module docstring.
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print()
    print("plates        %d" % df["plate_id"].nunique())
    print("rows / plate  %.1f mean, %d max"
          % (len(df) / max(df["plate_id"].nunique(), 1),
             int(df.groupby("plate_id").size().max())))
    print()
    print("answer letter distribution (chance = 0.250):")
    vc = df["answer_letter"].value_counts(normalize=True).sort_index()
    for k, v in vc.items():
        print("  %s  %5d  %.3f" % (k, int((df["answer_letter"] == k).sum()), v))
    print()
    print("measurement types:")
    print(df["measure_type"].value_counts(dropna=False).to_string())
    print()

    os.makedirs(chunk_root, exist_ok=True)
    for i in range(args.n_chunks):
        sub = df.iloc[i::args.n_chunks].copy()
        d = os.path.join(chunk_root, "chunk%02d" % i)
        os.makedirs(d, exist_ok=True)
        sub.to_csv(os.path.join(d, "qa.csv"), index=False)
        print("chunk%02d  %6d rows  %3d plates  %s"
              % (i, len(sub), sub["plate_id"].nunique(),
                 dict(sub["measure_type"].value_counts())))

    total = sum(len(df.iloc[i::args.n_chunks]) for i in range(args.n_chunks))
    assert total == len(df), "row count mismatch"
    print()
    print("OK: %d rows across %d chunks, no row lost or duplicated."
          % (total, args.n_chunks))
    print("next:  python %s" % os.path.join(args.run_dir, "stats.py"))


if __name__ == "__main__":
    main()
