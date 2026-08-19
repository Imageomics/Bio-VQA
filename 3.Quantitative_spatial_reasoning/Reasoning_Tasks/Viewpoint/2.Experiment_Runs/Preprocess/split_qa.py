#!/usr/bin/env python3
"""
Combine the three Viewpoint tiers into one manifest and shard it for SLURM.

  python split_qa.py --task_root /path/to/.../Viewpoint
  python split_qa.py --task_root ... --run_dir . --n_chunks 6

Nothing is hardcoded: --run_dir defaults to this script's own directory and
--task_root is required.

WHY ONE COMBINED MANIFEST
The model is loaded once per array task, so folding the tiers together turns
three model loads into one. Rows are interleaved, so every chunk holds the same
tier mix and one failed task cannot wipe out a whole tier.

TIER NAMING
The directories are Easy / Medium / Hard, but all three ask for a viewpoint.
What changes is what the model must do to find the target:

  single      one photograph, no panel to locate
  cued        three photographs of DIFFERENT species. The question names the
              species, so the species identifies the panel.
  positional  three photographs of the SAME species. The species is
              uninformative, so the panel must be located from
              left / middle / right.

cued and positional are both three-panel and share the same resize, so
cued -> positional isolates positional indexing with effective resolution held
constant. That contrast is the point of the ladder.

COLUMNS ADDED
  row_id         restores original order at merge time
  tier           single | cued | positional
  n_panels       1 or 3 - drives the panel clause in the prompt, nothing else
  image_abspath  resolved here, once. Chunk CSVs live in chunks/chunkNN/ while
                 images stay under <tier>/Images/, so a relative path would be
                 unresolvable from every chunk.
"""

import argparse
import os
import shutil

import pandas as pd

TIERS = [("single", "Easy", 1), ("cued", "Medium", 3),
         ("positional", "Hard", 3)]


def load_tier(task_root, tier, folder, n_panels):
    d = os.path.join(task_root, folder)
    csv_path = os.path.join(d, "qa.csv")
    if not os.path.isfile(csv_path):
        print("[skip] %s: no qa.csv at %s" % (tier, csv_path))
        return None

    df = pd.read_csv(csv_path)
    img_dir = os.path.join(d, "Images")
    col = "image" if "image" in df.columns else "image_name"
    df["image"] = df[col].astype(str)

    def abspath(name):
        p = str(name).strip()
        if os.path.isabs(p):
            return p
        for cand in (os.path.join(img_dir, os.path.basename(p)),
                     os.path.join(d, p)):
            if os.path.exists(cand):
                return cand
        return os.path.join(img_dir, os.path.basename(p))

    df["image_abspath"] = df["image"].map(abspath)
    df["tier"] = tier
    df["n_panels"] = n_panels

    missing = int((~df["image_abspath"].map(os.path.exists)).sum())
    print("[ok]   %-11s %6d rows  %5d images  missing=%d"
          % (tier, len(df), df["image"].nunique(), missing))
    if missing:
        print("       first missing: %s"
              % df.loc[~df["image_abspath"].map(os.path.exists),
                       "image_abspath"].iloc[0])
    return df


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_root", required=True,
                    help="directory holding Easy/ Medium/ Hard/")
    ap.add_argument("--run_dir", default=here)
    ap.add_argument("--n_chunks", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    chunk_root = os.path.join(args.run_dir, "chunks")
    if os.path.isdir(chunk_root):
        if not args.force:
            raise SystemExit(
                "chunks/ already exists at %s\n"
                "  Re-splitting would orphan results already written there.\n"
                "  Use --force to wipe and rebuild." % chunk_root)
        shutil.rmtree(chunk_root)

    frames = [f for f in (load_tier(args.task_root, t, folder, n)
                          for t, folder, n in TIERS) if f is not None]
    if not frames:
        raise SystemExit("no tiers found under " + args.task_root)

    keep = ["tier", "n_panels", "image", "image_abspath", "question", "answer"]
    for extra in ("dataset", "position"):
        if any(extra in f.columns for f in frames):
            keep.append(extra)
    for f in frames:
        for c in keep:
            if c not in f.columns:
                f[c] = ""

    df = pd.concat([f[keep] for f in frames], ignore_index=True)
    df.insert(0, "row_id", range(len(df)))

    print()
    print("total %d rows" % len(df))
    print(df.groupby("tier").size().to_string())
    print()
    print("gold label distribution:")
    print(pd.crosstab(df["answer"].astype(str).str.lower(),
                      df["tier"]).to_string())
    print()

    os.makedirs(chunk_root, exist_ok=True)
    for i in range(args.n_chunks):
        sub = df.iloc[i::args.n_chunks].copy()
        d = os.path.join(chunk_root, "chunk%02d" % i)
        os.makedirs(d, exist_ok=True)
        sub.to_csv(os.path.join(d, "qa.csv"), index=False)
        print("chunk%02d  %6d rows  %s"
              % (i, len(sub), dict(sub["tier"].value_counts())))

    total = sum(len(df.iloc[i::args.n_chunks]) for i in range(args.n_chunks))
    assert total == len(df), "row count mismatch"
    print()
    print("OK: %d rows across %d chunks, no row lost or duplicated."
          % (total, args.n_chunks))
    print("next:  python %s" % os.path.join(args.run_dir, "stats.py"))


if __name__ == "__main__":
    main()
