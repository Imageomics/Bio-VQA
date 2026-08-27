#!/usr/bin/env python3
"""
Stitch chunked Relative Position results into one CSV per model. Merging only - all
scoring lives in evaluate.py, so the numbers in the paper come from one place.

  python merge_results.py --out_dir /path/to/task/results
  python merge_results.py --run_dir /path/to/runs --out_dir /path/to/results
  python merge_results.py --model gemma3_12B

Reads  <run_dir>/chunks/chunk*/qa_<MODEL_KEY>.csv
Writes <out_dir>/qa_<MODEL_KEY>.csv          sorted by row_id, LEAN columns
       <out_dir>/merge_manifest.json         what was merged, whether anything
                                             is missing, and the prompt text

LEAN OUTPUT
The chunk CSVs carry everything the runner needs at inference time. The merged
results are a different artefact - they are what gets analysed and published -
so run-only and dead columns are stripped here:

  image_abspath   an ABSOLUTE cluster path. It has no meaning outside this
                  filesystem and must not appear in a released file. Any
                  column whose name contains "abspath" or "_path" is dropped
                  unconditionally, in this and every task.
  question        byte-identical on every row after the rewrite, so storing it
                  per row is roughly 0.7 MB of duplicated text. It is recorded
                  ONCE in merge_manifest.json instead, which is also where a
                  reader should look for it.
  answer          a duplicate of answer_dir.
  gt_recomputed   a split-time consistency check, equal to answer_dir by
                  definition once that check passes.
  options, n_options, answer_in_options, answer_pos
                  the option list is no longer shown to the model, so none of
                  these describe anything the model saw.
  n_boxes, red_label, blue_label
                  constant within a condition; `condition` already carries it.

`options_shown` is written as a single boolean so evaluate.py can still tell
whether the option-position controls apply, without keeping the question text.

--keep_all disables all of this and writes every column through.

--run_dir  where the chunks live (defaults to this script's directory)
--out_dir  where the merged CSVs go (defaults to <run_dir>/results)

Keeping the two separate lets the code live in the runs tree while the merged
results sit next to the dataset they describe. Pass the same --out_dir to
evaluate.py as its --results_dir.

The manifest exists so a partially finished sweep is obvious later. A model
whose row count does not match the split must not be reported.
"""

import argparse
import glob
import json
import os
import time

import pandas as pd


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=here)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--model", default="", help="MODEL_KEY substring filter")
    ap.add_argument("--keep_all", action="store_true",
                    help="write every column through instead of the lean set")
    args = ap.parse_args()

    chunk_root = os.path.join(args.run_dir, "chunks")
    out_root = os.path.abspath(args.out_dir or
                               os.path.join(args.run_dir, "results"))
    os.makedirs(out_root, exist_ok=True)

    chunk_dirs = sorted(glob.glob(os.path.join(chunk_root, "chunk*")))
    if not chunk_dirs:
        raise SystemExit("no chunks under " + chunk_root)

    n_expected = sum(len(pd.read_csv(os.path.join(d, "qa.csv")))
                     for d in chunk_dirs)

    keys = set()
    for d in chunk_dirs:
        for f in glob.glob(os.path.join(d, "qa_*.csv")):
            keys.add(os.path.basename(f)[len("qa_"):-len(".csv")])
    keys = sorted(k for k in keys if args.model in k)
    if not keys:
        raise SystemExit("no model outputs found under " + chunk_root)

    print("chunks   %d  (%s)" % (len(chunk_dirs), chunk_root))
    print("out      %s" % out_root)
    print("expected %d rows" % n_expected)
    print("models   %d" % len(keys))
    print()

    # dropped from the merged results; see the module docstring
    DROP = {"question", "answer", "gt_recomputed", "options", "n_options",
            "answer_in_options", "answer_pos", "n_boxes", "red_label",
            "blue_label"}

    manifest = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime()),
                "task": "relative_position",
                "chunk_root": os.path.abspath(chunk_root),
                "out_dir": out_root,
                "chunks": len(chunk_dirs), "expected_rows": n_expected,
                "models": {}}

    for key in keys:
        parts, missing = [], []
        for d in chunk_dirs:
            f = os.path.join(d, "qa_%s.csv" % key)
            if os.path.isfile(f):
                parts.append(pd.read_csv(f))
            else:
                missing.append(os.path.basename(d))

        df = pd.concat(parts, ignore_index=True)
        if "row_id" in df.columns:
            df = df.sort_values("row_id").reset_index(drop=True)
            dupes = int(df["row_id"].duplicated().sum())
        else:
            dupes = -1

        if not args.keep_all:
            # record the prompt once, then stop carrying it per row
            if "question" in df.columns and len(df):
                q = str(df["question"].iloc[0])
                manifest.setdefault("prompt", q)
                manifest.setdefault("prompt_identical_every_row",
                                    bool(df["question"].nunique() == 1))
                df["options_shown"] = ("Choose exactly one of:" in q
                                       or "Options:" in q)
            # any path column is cluster-specific and never shipped
            paths = [c for c in df.columns
                     if "abspath" in c.lower()
                     or c.lower().endswith("_path")]
            drop = [c for c in df.columns if c in DROP] + paths
            dropped_cols = sorted(set(drop))
            df = df.drop(columns=dropped_cols, errors="ignore")
        else:
            dropped_cols = []

        out = os.path.join(out_root, "qa_%s.csv" % key)
        df.to_csv(out, index=False)

        col = key + "_answer"
        parsed = int(df[col].notna().sum()) if col in df.columns else 0
        complete = (len(df) == n_expected and not missing and dupes == 0)

        manifest["models"][key] = {
            "rows": int(len(df)), "chunks_found": len(parts),
            "chunks_missing": missing, "duplicate_row_ids": dupes,
            "resolved_answers": parsed, "complete": bool(complete),
            "columns": list(df.columns), "dropped_columns": dropped_cols,
            "out_csv": out,
        }

        flag = "OK " if complete else "!! "
        print("%s%-32s %6d/%-6d rows  parsed %6d  %s"
              % (flag, key, len(df), n_expected, parsed,
                 ("missing " + ",".join(missing)) if missing else ""))

    mpath = os.path.join(out_root, "merge_manifest.json")
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)

    bad = [k for k, v in manifest["models"].items() if not v["complete"]]
    print()
    if not args.keep_all and manifest["models"]:
        first = next(iter(manifest["models"].values()))
        print("columns written : %d  (%s)"
              % (len(first["columns"]), ", ".join(first["columns"])))
        print("columns dropped : %d  (%s)"
              % (len(first["dropped_columns"]),
                 ", ".join(first["dropped_columns"])))
        print("the prompt text is stored once in merge_manifest.json")
        print()
    print("wrote " + mpath)
    if bad:
        print("INCOMPLETE, do not report until resubmitted: " + ", ".join(bad))
    else:
        print("all %d models complete." % len(keys))
    print()
    print("next:  python %s --results_dir %s"
          % (os.path.join(args.run_dir, "evaluate.py"), out_root))


if __name__ == "__main__":
    main()