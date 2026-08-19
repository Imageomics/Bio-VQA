#!/usr/bin/env python3
"""
Parse the OCR qa.csv into a flat manifest and shard it for SLURM.

  python split_qa.py --task_root "/path/to/Optical Character Recognition - OCR"

Nothing is hardcoded: --run_dir defaults to this script's own directory and
--task_root is required. Quote the task root - it contains spaces and a dash.

INPUT SCHEMA
Two columns, image_path and qa. Note the differences from every other task in
this component:

  * qa is a PYTHON DICT REPR with single quotes - {'Q': ..., 'A': ...} - not
    JSON. json.loads rejects it; ast.literal_eval is tried first here.
  * image_path already carries the task directory as a prefix, e.g.
    "Optical Character Recognition - OCR/Barcodes/A00000032929.jpg", so it is
    relative to the TASKS root rather than to --task_root. The resolver tries
    the sensible candidates and reports what it could not find.

COLUMNS ADDED
  row_id         restores original order at merge time
  image_abspath  resolved here, once
  question       the Q field
  answer         the A field, uppercased and stripped of separators
  code_len       character length of the answer
  suffix         the part of the answer that actually varies across the corpus
                 (see below)

THE CONSTANT PREFIX
These accession codes share a long fixed prefix - in a sample of the corpus,
eight of twelve characters were identical across every label. That matters
enormously for scoring: a model that emits one fixed string and never looks at
an image can reach a character accuracy above 0.8 while reading nothing. The
common prefix is computed here and stored, so evaluate.py can report accuracy
on the variable suffix separately. stats.py prints how many characters are
actually informative.

A note for anyone extending this task: the answer happens to equal the image
filename stem. That is convenient for verification but means any future
pipeline must never place the filename in the prompt.
"""

import argparse
import ast
import json
import os
import re
import shutil

import pandas as pd


def parse_qa(cell):
    """The qa column is a Python dict repr; JSON is tried as a fallback."""
    s = str(cell)
    try:
        d = ast.literal_eval(s)
    except Exception:
        try:
            d = json.loads(s)
        except Exception:
            return None, None
    if not isinstance(d, dict):
        return None, None
    return d.get("Q"), d.get("A")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_root", required=True,
                    help='directory holding qa.csv and Barcodes/ '
                         '(quote it - the name contains spaces)')
    ap.add_argument("--run_dir", default=here)
    ap.add_argument("--n_chunks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42,
                    help="shuffle seed; recorded so the split is reproducible")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    csv_path = os.path.join(args.task_root, "qa.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit("qa.csv not found at " + csv_path)
    tasks_root = os.path.dirname(os.path.abspath(args.task_root.rstrip("/")))

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

    parsed = df["qa"].map(parse_qa)
    df["question"] = [p[0] for p in parsed]
    df["answer"] = [p[1] for p in parsed]
    bad = int(df["question"].isna().sum())
    if bad:
        print("!! %d rows had an unparseable qa cell" % bad)
        df = df[df["question"].notna()].copy()

    df["answer"] = (df["answer"].astype(str).str.upper()
                    .str.replace(r"[\s\-_.]", "", regex=True))
    df["code_len"] = df["answer"].str.len()

    # image_path is relative to the TASKS root, not to --task_root
    def abspath(rel):
        p = str(rel).strip().lstrip("/")
        base = os.path.basename(p)
        for cand in (os.path.join(tasks_root, p),
                     os.path.join(args.task_root, "Barcodes", base),
                     os.path.join(args.task_root, p),
                     os.path.join(args.task_root, base)):
            if os.path.exists(cand):
                return cand
        return os.path.join(args.task_root, "Barcodes", base)

    df["image"] = df["image_path"].astype(str).map(os.path.basename)
    df["image_abspath"] = df["image_path"].map(abspath)

    missing = int((~df["image_abspath"].map(os.path.exists)).sum())
    print("images  %d unique, %d missing" % (df["image"].nunique(), missing))
    if missing:
        print("        first missing: %s"
              % df.loc[~df["image_abspath"].map(os.path.exists),
                       "image_abspath"].iloc[0])
        print("        tasks root tried: %s" % tasks_root)

    # ---- the constant prefix ------------------------------------------- #
    codes = df["answer"].tolist()
    prefix = os.path.commonprefix(codes) if codes else ""
    df["suffix"] = df["answer"].str[len(prefix):]
    print()
    print("common prefix   %r  (%d of %d characters)"
          % (prefix, len(prefix), int(df["code_len"].median())))
    print("variable suffix %d characters, %d distinct values"
          % (int(df["suffix"].str.len().median()), df["suffix"].nunique()))
    if len(prefix) >= 0.5 * df["code_len"].median():
        print("        NOTE: over half of every code is constant. Character")
        print("        accuracy on the full string is therefore inflated;")
        print("        evaluate.py reports the suffix separately.")

    dup = int(df["answer"].duplicated().sum())
    if dup:
        print("!! %d duplicate answer codes" % dup)

    keep = ["image", "image_abspath", "question", "answer", "code_len",
            "suffix"]
    df = df[keep].reset_index(drop=True)
    df.insert(0, "row_id", range(len(df)))

    # Deterministic shuffle before interleaving: the source is sorted by
    # accession, so a plain i::n stride would hand each chunk a contiguous
    # numeric range. row_id is assigned first, so the original order is still
    # recoverable at merge time.
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print()
    print("code length distribution:")
    print(df["code_len"].value_counts().sort_index().to_string())
    print()

    os.makedirs(chunk_root, exist_ok=True)
    for i in range(args.n_chunks):
        sub = df.iloc[i::args.n_chunks].copy()
        d = os.path.join(chunk_root, "chunk%02d" % i)
        os.makedirs(d, exist_ok=True)
        sub.to_csv(os.path.join(d, "qa.csv"), index=False)
        print("chunk%02d  %6d rows" % (i, len(sub)))

    total = sum(len(df.iloc[i::args.n_chunks]) for i in range(args.n_chunks))
    assert total == len(df), "row count mismatch"
    print()
    print("OK: %d rows across %d chunks, no row lost or duplicated."
          % (total, args.n_chunks))
    print("next:  python %s" % os.path.join(args.run_dir, "stats.py"))


if __name__ == "__main__":
    main()
