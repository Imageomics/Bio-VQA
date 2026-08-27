#!/usr/bin/env python3
"""
Build the Localization manifest from the EASY tier and shard it for SLURM.

  python split_qa.py \\
      --task_root "/path/to/1-Spatial-Visual-Reasoning-tasks/Localization" \\
      --id_root   "/path/to/4-Instance-Level-Identification"

Nothing is hardcoded: --run_dir defaults to this script's own directory,
--task_root is required, and --id_root is needed only because the Easy tier
stores paths like "../4-Instance-Level-Identification/GrevysZebra/Images/x.jpg"
which point outside the Localization directory. Several candidate resolutions
are tried and whatever is missing is reported.

SCOPE
Only the Easy tier - single photographs, one animal per question. The
multi-panel Medium and Hard tiers are deliberately excluded; pass
--tiers easy,medium,hard to include them.

THE COORDINATE FRAME
Gold boxes are in image pixels measured from the top-left corner. img_w and
img_h are read here so the prompt can state the image size and the resolver
can normalise whatever convention a model replies in - Qwen answers in 0-1000,
several models in 0-1 fractions, some in [x1,y1,x2,y2].

A NOTE ON THIS TIER
The subject is large: the gold box averages 37% of the frame and reaches 99.8%.
Trivial baselines are correspondingly high - predicting the average box scores
IoU 0.415 and clears IoU 0.5 on 37% of items - so mean IoU barely discriminates
here and the tight thresholds carry the signal. stats.py prints all of them.

COLUMNS ADDED
  row_id         restores original order at merge time
  tier           easy | medium | hard
  n_panels       1 or 3 - drives the panel clause in the prompt, nothing else
  image_abspath  resolved here, once
  img_w, img_h   canvas dimensions, read from the file header
  gt_x .. gt_h   the gold box as four floats
  gt_area_frac   box area over canvas area. Localization accuracy collapses on
                 small targets, so this is the column to stratify by
  n_targets      how many boxes this image carries, for the panel-confusion
                 diagnostic in evaluate.py
"""

import argparse
import ast
import json
import os
import shutil

import pandas as pd
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

ALL_TIERS = {"easy": ("Easy", 1), "medium": ("Medium", 3), "hard": ("Hard", 3)}


def parse_box(cell):
    """'[137.45, 302.88, 1785.19, 921.81]' -> four floats."""
    try:
        v = ast.literal_eval(str(cell))
    except Exception:
        try:
            v = json.loads(str(cell))
        except Exception:
            return None
    if not isinstance(v, (list, tuple)) or len(v) < 4:
        return None
    try:
        return [float(t) for t in v[:4]]
    except (TypeError, ValueError):
        return None


def load_tier(task_root, id_root, tier, folder, n_panels):
    d = os.path.join(task_root, folder)
    csv_path = os.path.join(d, "qa.csv")
    if not os.path.isfile(csv_path):
        print("[skip] %s: no qa.csv at %s" % (tier, csv_path))
        return None

    df = pd.read_csv(csv_path)

    def abspath(rel):
        p = str(rel).strip()
        cands = [os.path.normpath(os.path.join(d, p)),
                 os.path.normpath(os.path.join(task_root, p)),
                 os.path.join(d, "Images", os.path.basename(p))]
        if id_root:
            # "../4-Instance-Level-Identification/GrevysZebra/Images/x.jpg"
            tail = p.split("4-Instance-Level-Identification", 1)
            if len(tail) == 2:
                cands.insert(0, os.path.join(id_root, tail[1].lstrip("/")))
        for c in cands:
            if os.path.exists(c):
                return c
        return cands[0]

    df["image"] = df["image"].astype(str).map(os.path.basename)
    df["image_abspath"] = df["image_path"].map(abspath)
    df["tier"] = tier
    df["n_panels"] = n_panels

    boxes = df["answer"].map(parse_box)
    bad = int(boxes.isna().sum())
    if bad:
        print("  !! %d gold boxes unparseable, dropped" % bad)
    df = df[boxes.notna()].copy()
    boxes = boxes[boxes.notna()]
    df["gt_x"] = [b[0] for b in boxes]
    df["gt_y"] = [b[1] for b in boxes]
    df["gt_w"] = [b[2] for b in boxes]
    df["gt_h"] = [b[3] for b in boxes]

    # canvas dimensions, read once per unique image from the file header
    dims = {}
    for p in df["image_abspath"].unique():
        try:
            with Image.open(p) as im:
                dims[p] = im.size
        except Exception:
            dims[p] = (0, 0)
    df["img_w"] = df["image_abspath"].map(lambda p: dims[p][0])
    df["img_h"] = df["image_abspath"].map(lambda p: dims[p][1])

    missing = int((~df["image_abspath"].map(os.path.exists)).sum())
    nodim = int((df["img_w"] <= 0).sum())
    print("[ok]   %-7s %6d rows  %5d images  missing=%d  unreadable=%d"
          % (tier, len(df), df["image"].nunique(), missing, nodim))
    if missing:
        print("       first missing: %s"
              % df.loc[~df["image_abspath"].map(os.path.exists),
                       "image_abspath"].iloc[0])

    ok = df["img_w"] > 0
    df = df[ok].copy()
    df["gt_area_frac"] = ((df["gt_w"] * df["gt_h"])
                          / (df["img_w"] * df["img_h"]))

    # a gold box that does not fit inside its own canvas means the stored
    # coordinates and the stored image disagree - a benchmark defect, not a
    # model failure, so it is reported loudly rather than scored
    oob = ((df["gt_x"] < -1) | (df["gt_y"] < -1)
           | (df["gt_x"] + df["gt_w"] > df["img_w"] * 1.02)
           | (df["gt_y"] + df["gt_h"] > df["img_h"] * 1.02))
    if int(oob.sum()):
        print("       !! %d gold boxes fall outside their own canvas"
              % int(oob.sum()))
        r = df[oob].iloc[0]
        print("          e.g. %s  box=[%.0f %.0f %.0f %.0f]  canvas=%dx%d"
              % (r["image"], r["gt_x"], r["gt_y"], r["gt_w"], r["gt_h"],
                 r["img_w"], r["img_h"]))
    df["gt_out_of_bounds"] = oob
    return df


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_root", required=True,
                    help="directory holding Easy/ Medium/ Hard/")
    ap.add_argument("--id_root", default="",
                    help="4-Instance-Level-Identification root, needed for "
                         "the Easy tier's relative paths")
    ap.add_argument("--run_dir", default=here)
    ap.add_argument("--tiers", default="easy",
                    help="comma list of tiers to include; default easy only")
    ap.add_argument("--n_chunks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
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

    want = [t.strip().lower() for t in args.tiers.split(",") if t.strip()]
    bad = [t for t in want if t not in ALL_TIERS]
    if bad:
        raise SystemExit("unknown tier(s): %s. Choose from %s"
                         % (", ".join(bad), ", ".join(ALL_TIERS)))
    print("tiers: %s" % ", ".join(want))
    frames = [f for f in (load_tier(args.task_root, args.id_root, t,
                                    ALL_TIERS[t][0], ALL_TIERS[t][1])
                          for t in want) if f is not None]
    if not frames:
        raise SystemExit("no tiers found under " + args.task_root)

    keep = ["tier", "n_panels", "image", "image_abspath", "img_w", "img_h",
            "question", "answer", "gt_x", "gt_y", "gt_w", "gt_h",
            "gt_area_frac", "gt_out_of_bounds"]
    for extra in ("dataset", "position"):
        if any(extra in f.columns for f in frames):
            keep.append(extra)
    for f in frames:
        for c in keep:
            if c not in f.columns:
                f[c] = pd.NA

    df = pd.concat([f[keep] for f in frames], ignore_index=True)
    # Keyed on the resolved FILE, not the basename: 455 basenames are shared
    # by different canvases across tiers, and grouping on the basename would
    # tell an Easy row it has three targets because a Medium canvas happens to
    # carry the same filename.
    df["n_targets"] = df.groupby("image_abspath")["image_abspath"].transform("size")
    df.insert(0, "row_id", range(len(df)))

    # Deterministic shuffle before interleaving: rows arrive grouped by tier
    # and, inside a multi-panel image, in panel order. row_id is assigned
    # first, so the original order is still recoverable at merge time.
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print()
    print("total %d rows" % len(df))
    nb, na = df["image"].nunique(), df["image_abspath"].nunique()
    if nb != na:
        print("NOTE: %d distinct files but only %d distinct basenames - %d are "
              "shared" % (na, nb, na - nb))
        print("      across tiers. Everything downstream keys on "
              "image_abspath.")
    print(df.groupby("tier").size().to_string())
    print()
    print("canvas sizes by tier:")
    print(df.groupby("tier")[["img_w", "img_h"]].median().to_string())
    print()
    print("gold box area as a fraction of the canvas:")
    print(df.groupby("tier")["gt_area_frac"]
            .describe()[["mean", "min", "25%", "50%", "max"]]
            .round(4).to_string())
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