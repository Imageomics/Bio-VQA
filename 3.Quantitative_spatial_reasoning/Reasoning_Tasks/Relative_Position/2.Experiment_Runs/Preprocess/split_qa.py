#!/usr/bin/env python3
"""
Parse the Relative Position qa.csv into a shardable manifest.

  python split_qa.py --task_root "/path/to/Relative_Position"

Nothing is hardcoded: --run_dir defaults to this script's own directory and
--task_root is required.

WHAT THIS TASK IS
Two animals in one photograph are outlined, one red and one blue, and the model
names the direction of the red one relative to the blue one. Four options are
offered out of the eight compass directions, so chance is 0.250.

Two conditions:

  clean       only the two target boxes are drawn.
  distractor  four extra boxes are added - a second red and a second blue, both
              EMPTY, plus a green and a cyan. The question says which are
              decoys. This exists because with only two boxes a model never has
              to look at the animals: find the two rectangles, compare centres.
              Adding an empty box of each target colour makes "find the red
              rectangle" insufficient - it has to be the red rectangle with a
              specimen in it.

THE MIS-PAIRING STRUCTURE, WHICH IS WHY THIS TASK IS WORTH MORE THAN ACCURACY
In the distractor condition the four offered options are not arbitrary. They
are the correct answer plus the three directions you would get by pairing the
wrong boxes:

  option 1   red_empty  -> blue_target    grabbed the empty RED box
  option 2   red_target -> blue_empty     grabbed the empty BLUE box
  option 3   red_empty  -> blue_empty     grabbed both empty boxes

Verified against the generator's own wrong_pairings field: recomputing each
direction from the stored box coordinates reproduces that list exactly, in that
order. So every wrong answer is attributable to a specific confusion rather
than being generic error, and evaluate.py reports the breakdown. A model that
mostly picks option 1 is failing to check whether the red box contains an
animal; one that spreads evenly across all three is not localising at all.

COLUMNS ADDED
  row_id            restores original order at merge time
  image_abspath     resolved here, once
  answer_dir        the gold direction, normalised
  n_options         how many options were offered (expected 4)
  answer_in_options whether the gold answer was actually offered
  answer_pos        1-based position of the gold answer among the options,
                    for the option-order bias check
  mis_red_empty     direction from red_empty  to blue_target
  mis_blue_empty    direction from red_target to blue_empty
  mis_both_empty    direction from red_empty  to blue_empty
  gt_recomputed     direction recomputed from the stored boxes; must equal
                    answer_dir, and any mismatch is reported loudly
"""

import argparse
import json
import math
import os
import re
import shutil

import pandas as pd

DIRECTIONS = ["right", "bottom-right", "bottom", "bottom-left", "left",
              "top-left", "top", "top-right"]

# The generator's convention, reproduced here so the mis-pairings can be
# recomputed: dx = x_red - x_blue, dy = y_red - y_blue, atan2 in image
# coordinates where y increases downward, then 45-degree bins.
_BINS = [("right", -22.5, 22.5), ("bottom-right", 22.5, 67.5),
         ("bottom", 67.5, 112.5), ("bottom-left", 112.5, 157.5),
         ("left", 157.5, 202.5), ("top-left", -157.5, -112.5),
         ("top", -112.5, -67.5), ("top-right", -67.5, -22.5)]


def canon(text):
    return re.sub(r"[^a-z]+", "-", str(text).strip().lower()).strip("-")


def centre(box):
    """Boxes are stored as [x1, y1, x2, y2]."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def direction(a, b):
    """Direction of box a relative to box b."""
    (x1, y1), (x2, y2) = centre(a), centre(b)
    ang = math.degrees(math.atan2(y1 - y2, x1 - x2))
    ca = ang if ang >= -157.5 else ang + 360.0
    for name, lo, hi in _BINS:
        if lo <= ca < hi:
            return name
    return None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_root", required=True,
                    help="directory holding qa.csv and Images/")
    ap.add_argument("--run_dir", default=here)
    ap.add_argument("--n_chunks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    csv_path = os.path.join(args.task_root, "qa.csv")
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

    def abspath(rel):
        p = str(rel).strip().lstrip("/")
        for cand in (os.path.join(args.task_root, p),
                     os.path.join(args.task_root, "Images",
                                  os.path.basename(p))):
            if os.path.exists(cand):
                return cand
        return os.path.join(args.task_root, "Images", os.path.basename(p))

    df["image"] = df["image"].astype(str).map(os.path.basename)
    df["image_abspath"] = df["image_path"].map(abspath)
    df["answer_dir"] = df["answer"].map(canon)

    opts = df["options"].astype(str).str.split("|")
    df["n_options"] = opts.map(len)
    df["answer_in_options"] = [
        a in [canon(o) for o in os_] for a, os_ in zip(df["answer_dir"], opts)]
    df["answer_pos"] = [
        ([canon(o) for o in os_].index(a) + 1)
        if a in [canon(o) for o in os_] else -1
        for a, os_ in zip(df["answer_dir"], opts)]

    # ---- recompute the mis-pairings from the stored geometry ------------- #
    for c in ("mis_red_empty", "mis_blue_empty", "mis_both_empty",
              "gt_recomputed"):
        df[c] = pd.NA
    bad_geom = 0
    for i in df.index:
        try:
            b = json.loads(df.at[i, "boxes"])
        except Exception:
            continue
        rt, bt = b.get("red_target"), b.get("blue_target")
        if not rt or not bt:
            continue
        df.at[i, "gt_recomputed"] = direction(rt, bt)
        re_, be_ = b.get("red_empty"), b.get("blue_empty")
        if re_:
            df.at[i, "mis_red_empty"] = direction(re_, bt)
        if be_:
            df.at[i, "mis_blue_empty"] = direction(rt, be_)
        if re_ and be_:
            df.at[i, "mis_both_empty"] = direction(re_, be_)

    chk = df["gt_recomputed"].notna()
    mism = int((df.loc[chk, "gt_recomputed"] != df.loc[chk, "answer_dir"]).sum())
    print()
    print("geometry check: %d of %d rows recomputed from the stored boxes"
          % (int(chk.sum()), len(df)))
    if mism:
        print("  !! %d disagree with the stored answer. The direction "
              "convention" % mism)
        print("     in this script does not match the generator's. Do NOT run "
              "until resolved.")
        r = df[chk & (df["gt_recomputed"] != df["answer_dir"])].iloc[0]
        print("     e.g. %s: stored=%s recomputed=%s"
              % (r["image"], r["answer_dir"], r["gt_recomputed"]))
    else:
        print("  all recomputed directions match the stored answers.")

    missing = int((~df["image_abspath"].map(os.path.exists)).sum())
    print()
    print("images  %d unique, %d missing" % (df["image"].nunique(), missing))
    if missing:
        print("        first missing: %s"
              % df.loc[~df["image_abspath"].map(os.path.exists),
                       "image_abspath"].iloc[0])
    n_bad_opt = int((~df["answer_in_options"]).sum())
    if n_bad_opt:
        print("!! %d rows whose gold answer is not among the offered options"
              % n_bad_opt)

    keep = ["dataset", "condition", "pair_id", "source_image", "image",
            "image_abspath", "question", "answer", "answer_dir", "options",
            "n_options", "answer_in_options", "answer_pos", "n_boxes",
            "size_ratio", "red_label", "blue_label", "mis_red_empty",
            "mis_blue_empty", "mis_both_empty", "gt_recomputed"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].reset_index(drop=True)
    df.insert(0, "row_id", range(len(df)))

    # Deterministic shuffle before interleaving: rows arrive grouped by
    # dataset and condition, so a plain stride would hand each chunk a
    # systematic slice. row_id is assigned first, so the original order is
    # still recoverable at merge time.
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print()
    print("total %d rows" % len(df))
    if "condition" in df.columns:
        print(pd.crosstab(df["dataset"], df["condition"]).to_string())
    print()
    print("gold direction distribution (chance with 4 options = 0.250):")
    vc = df["answer_dir"].value_counts(normalize=True).round(3)
    print(vc.to_string())
    print("  majority direction %s at %.3f - that is the floor, not 0.250,"
          % (vc.index[0], vc.iloc[0]))
    print("  for a model that ignores the image and always says the same thing.")
    print()
    print("position of the gold answer among the options (should be uniform):")
    print(df["answer_pos"].value_counts(normalize=True).sort_index()
          .round(3).to_string())
    print()

    os.makedirs(chunk_root, exist_ok=True)
    for i in range(args.n_chunks):
        sub = df.iloc[i::args.n_chunks].copy()
        d = os.path.join(chunk_root, "chunk%02d" % i)
        os.makedirs(d, exist_ok=True)
        sub.to_csv(os.path.join(d, "qa.csv"), index=False)
        cond = (dict(sub["condition"].value_counts())
                if "condition" in sub.columns else {})
        print("chunk%02d  %6d rows  %s" % (i, len(sub), cond))

    total = sum(len(df.iloc[i::args.n_chunks]) for i in range(args.n_chunks))
    assert total == len(df), "row count mismatch"
    print()
    print("OK: %d rows across %d chunks, no row lost or duplicated."
          % (total, args.n_chunks))
    print("next:  python %s" % os.path.join(args.run_dir, "stats.py"))


if __name__ == "__main__":
    main()
