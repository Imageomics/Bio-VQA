#!/usr/bin/env python3
"""
Parse the Morphometry qa.csv into a flat manifest and shard it for SLURM.

  python split_qa.py --task_root "/path/to/Absolute Distance Measurements"

Nothing is hardcoded: --run_dir defaults to this script's own directory and
--task_root is required. Quote the task root - it contains spaces.

INPUT SCHEMA
The source file has two columns, image and qa_pairs, where qa_pairs is a JSON
object {"Q": ..., "A": "1.117 cm"}. That is flattened here so no downstream
script ever parses JSON or a unit string again.

COLUMNS ADDED
  row_id        restores original order at merge time
  image_abspath resolved here, once. Chunk CSVs live in chunks/chunkNN/ while
                the images stay in <task_root>/Images/, so a relative path
                would be unresolvable from every chunk.
  question      the Q field
  answer        the A field verbatim, e.g. "1.117 cm"
  answer_cm     the same value as a float, in centimetres
  measure_type  length | width, parsed from the question
  plate_id      the specimen-plate accession from the filename prefix

WHY plate_id MATTERS
Every beetle on one plate shares a scale bar and a magnification, and each
contributes both a length and a width item. Within-plate spread is about a
seventh of the total spread, so ~20 rows from one plate are close to a single
observation. Confidence intervals must bootstrap over plate_id, not over rows,
or they come out several times too narrow. evaluate.py does this automatically
when the column is present.
"""

import argparse
import ast
import json
import os
import re
import shutil

import pandas as pd

CM_RE = re.compile(r"([-+]?\d*\.?\d+)\s*(mm|cm)?", re.I)


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


def to_cm(text):
    m = CM_RE.search(str(text).replace(",", ""))
    if not m:
        return None
    v = float(m.group(1))
    if (m.group(2) or "").lower() == "mm":
        v /= 10.0
    return v


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_root", required=True,
                    help='directory holding qa.csv and Images/ '
                         '(quote it - the name contains spaces)')
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

    df["answer_cm"] = df["answer"].map(to_cm)
    df["measure_type"] = df["question"].str.extract(
        r"elytra\s+(length|width)", flags=re.I, expand=False).str.lower()

    # image column may already carry a path fragment; take the basename
    df["image"] = df["image"].astype(str).map(os.path.basename)
    df["image_abspath"] = df["image"].map(lambda n: os.path.join(img_dir, n))

    # filename is <plate>_<hash>.jpg, e.g. A00000001831_753877c6.jpg
    df["plate_id"] = df["image"].str.replace(r"_[0-9a-f]{6,}\.jpe?g$", "",
                                             regex=True, flags=re.I)

    missing = int((~df["image_abspath"].map(os.path.exists)).sum())
    print("images  %d unique, %d missing" % (df["image"].nunique(), missing))
    if missing:
        print("        first missing: %s"
              % df.loc[~df["image_abspath"].map(os.path.exists),
                       "image_abspath"].iloc[0])

    keep = ["plate_id", "measure_type", "image", "image_abspath", "question",
            "answer", "answer_cm"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].reset_index(drop=True)
    df.insert(0, "row_id", range(len(df)))

    # Deterministic shuffle BEFORE interleaving. The source file alternates
    # length, width, length, width..., so with an even --n_chunks a plain
    # i::n stride preserves that parity and every chunk comes out entirely
    # one measurement type - losing a chunk would then lose a whole condition.
    # row_id is assigned first, so the original order is still recoverable at
    # merge time.
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print()
    print("plates        %d" % df["plate_id"].nunique())
    print("rows / plate  %.1f mean" % (len(df) / max(df["plate_id"].nunique(), 1)))
    print(df.groupby("measure_type")["answer_cm"].describe()[
        ["count", "mean", "std", "min", "max"]].to_string())
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