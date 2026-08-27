#!/usr/bin/env python3
"""
Stitch chunked Viewpoint results into one CSV per model. Merging only - all
scoring lives in evaluate.py, so the numbers in the paper come from exactly
one place.

  # merged CSVs beside the task data, not beside this script:
  python merge_results.py --run_dir /path/to/2.Viewpoint --out_dir /path/to/anywhere
  python merge_results.py --model gemma3_12B

Reads  <run_dir>/chunks/chunk*/qa_<MODEL_KEY>.csv
Writes <out_dir>/qa_<MODEL_KEY>.csv          sorted by row_id
       <out_dir>/merge_manifest.json         what was merged, and whether
                                             anything is missing

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
    ap.add_argument("--run_dir", default=here,
                    help="directory holding chunks/ (default: this script's)")
    ap.add_argument("--out_dir", default="",
                    help="where merged CSVs go (default: <run_dir>/results)")
    ap.add_argument("--model", default="", help="MODEL_KEY substring filter")
    args = ap.parse_args()

    chunk_root = os.path.join(args.run_dir, "chunks")
    out_root = args.out_dir or os.path.join(args.run_dir, "results")
    out_root = os.path.abspath(out_root)
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

    manifest = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime()),
                "task": "viewpoint",
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
        out = os.path.join(out_root, "qa_%s.csv" % key)
        df.to_csv(out, index=False)

        col = key + "_answer"
        parsed = (int(df[col].notna().sum())
                  if col in df.columns else 0)
        complete = (len(df) == n_expected and not missing and dupes == 0)

        manifest["models"][key] = {
            "rows": int(len(df)), "chunks_found": len(parts),
            "chunks_missing": missing, "duplicate_row_ids": dupes,
            "resolved_labels": parsed, "complete": bool(complete),
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