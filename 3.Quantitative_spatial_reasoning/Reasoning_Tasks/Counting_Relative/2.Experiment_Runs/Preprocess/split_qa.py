#!/usr/bin/env python3
"""
Combine the two Counting-Relative conditions into one manifest and shard it.

  python split_qa.py --task_root "/path/to/Counting-Relative"

Nothing is hardcoded: --run_dir defaults to this script's own directory and
--task_root is required.

WHAT THIS TASK IS
A 2x2 grid of four photographs. The model picks which quadrant holds the most
(or the fewest) animals. Two conditions and two polarities, so four cells:

  same       all four panels drawn from ONE source dataset, and the question
             names that animal ("most number of bees"). Chance is 0.250 and
             there is no shortcut: the four panels are the same kind of scene,
             so only the counts distinguish them.
  different  the four panels come from four DIFFERENT datasets and the question
             says "animals". A model that recognises which panel is an insect
             plate rather than an aerial crop can often answer without
             counting: a Monte-Carlo over the source count distributions puts
             that dataset-prior strategy at about 0.53 for most and 0.55 for
             fewest, against a 0.250 chance floor. Pass those to evaluate.py
             with --prior_different so the reported number is the lift over the
             prior, not the raw accuracy.

  most / least   every image carries both questions. They are not independent
             observations, and a model answering the same letter for both is
             wrong on one by construction - evaluate.py reports that as an
             incoherence rate.

COLUMNS ADDED
  row_id         restores original order at merge time
  condition      same | different
  polarity       most | least
  image_abspath  resolved here, once. Chunk CSVs live in chunks/chunkNN/ while
                 images stay in <condition>/Images/, so a relative path would
                 be unresolvable from every chunk.
  answer_letter  A | B | C | D, extracted from "Image C (bottom-left)"
  source_dataset for the same condition only, from the quad_<Dataset>_ filename
  animal_term    the noun the question asks about

THE "Image" TRAP
The gold answers look like "Image B (top-right)". A case-insensitive [ABCD]
search matches the lower-case 'a' inside the word "Image" and returns A for
every option - silently, and the distribution just comes out 100% A. The
extraction below anchors on the word first. stats.py prints the key balance so
a regression is visible immediately.
"""

import argparse
import os
import re
import shutil

import pandas as pd

CONDITIONS = [("same", "Same"), ("different", "Different")]

# "Image B (top-right)" -> B, anchored so the 'a' in "Image" cannot match.
GOLD_RE = re.compile(r"[Ii]mage\s*([ABCD])\b")
POS_RE = re.compile(r"\b(top|bottom)[\s\-_]*(left|right)\b", re.I)
POSITIONS = {"topleft": "A", "topright": "B",
             "bottomleft": "C", "bottomright": "D"}


def gold_letter(text):
    s = str(text)
    m = GOLD_RE.search(s)
    if m:
        return m.group(1)
    m = POS_RE.search(s)
    if m:
        return POSITIONS.get((m.group(1) + m.group(2)).lower())
    m = re.search(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", s)
    return m.group(1) if m else None


def load_condition(task_root, cond, folder):
    d = os.path.join(task_root, folder)
    csv_path = os.path.join(d, "qa.csv")
    if not os.path.isfile(csv_path):
        print("[skip] %s: no qa.csv at %s" % (cond, csv_path))
        return None

    df = pd.read_csv(csv_path)
    img_dir = os.path.join(d, "Images")

    def abspath(rel):
        p = str(rel).strip().lstrip("/")
        for cand in (os.path.join(d, p),
                     os.path.join(img_dir, os.path.basename(p))):
            if os.path.exists(cand):
                return cand
        return os.path.join(img_dir, os.path.basename(p))

    df["image"] = df["image"].astype(str).map(os.path.basename)
    df["image_abspath"] = df["image_path"].map(abspath)
    df["condition"] = cond
    df["answer_letter"] = df["answer"].map(gold_letter)

    q = df["question"].astype(str)
    df["polarity"] = pd.NA
    df.loc[q.str.contains(r"\bmost\b", case=False), "polarity"] = "most"
    df.loc[q.str.contains(r"\b(?:least|fewest)\b", case=False), "polarity"] = "least"
    df["animal_term"] = q.str.extract(
        r"contains\s+(?:most|least|fewest)\s+number\s+of\s+([a-z ]+?)\s*:",
        flags=re.I, expand=False)

    # Same-condition filenames embed the dataset: quad_Bees_17c4ef41.jpg
    if cond == "same":
        df["source_dataset"] = df["image"].str.extract(
            r"^quad_(.+?)_[0-9a-f]+\.jpe?g$", flags=re.I, expand=False)
    else:
        df["source_dataset"] = pd.NA

    bad_letter = int(df["answer_letter"].isna().sum())
    bad_pol = int(df["polarity"].isna().sum())
    missing = int((~df["image_abspath"].map(os.path.exists)).sum())
    print("[ok]   %-10s %6d rows  %5d images  missing=%d  no_letter=%d  "
          "no_polarity=%d"
          % (cond, len(df), df["image"].nunique(), missing, bad_letter, bad_pol))
    if missing:
        print("       first missing: %s"
              % df.loc[~df["image_abspath"].map(os.path.exists),
                       "image_abspath"].iloc[0])
    if bad_letter:
        print("       e.g. unparsed answer: %r"
              % df.loc[df["answer_letter"].isna(), "answer"].iloc[0])
    return df


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_root", required=True,
                    help="directory holding Same/ and Different/")
    ap.add_argument("--run_dir", default=here)
    ap.add_argument("--n_chunks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42,
                    help="shuffle seed; recorded so the split is reproducible")
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

    frames = [f for f in (load_condition(args.task_root, c, folder)
                          for c, folder in CONDITIONS) if f is not None]
    if not frames:
        raise SystemExit("no conditions found under " + args.task_root)

    keep = ["condition", "polarity", "source_dataset", "animal_term",
            "image", "image_abspath", "question", "answer", "answer_letter"]
    df = pd.concat([f[keep] for f in frames], ignore_index=True)
    df = df[df["answer_letter"].notna()].reset_index(drop=True)
    df.insert(0, "row_id", range(len(df)))

    # Deterministic shuffle BEFORE interleaving. Rows alternate most/least and
    # are grouped by condition, so a plain i::n stride with an even --n_chunks
    # would hand each chunk a systematic slice. row_id is assigned first, so
    # the original order is still recoverable at merge time.
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print()
    print("total %d rows" % len(df))
    print(pd.crosstab(df["condition"], df["polarity"]).to_string())
    print()
    print("answer letter distribution (chance = 0.250):")
    print(pd.crosstab(df["answer_letter"], df["condition"],
                      normalize="columns").round(3).to_string())
    print()
    if df["source_dataset"].notna().any():
        print("same-condition source datasets:")
        print(df["source_dataset"].value_counts().to_string())
        print()

    os.makedirs(chunk_root, exist_ok=True)
    for i in range(args.n_chunks):
        sub = df.iloc[i::args.n_chunks].copy()
        d = os.path.join(chunk_root, "chunk%02d" % i)
        os.makedirs(d, exist_ok=True)
        sub.to_csv(os.path.join(d, "qa.csv"), index=False)
        print("chunk%02d  %6d rows  %s  %s"
              % (i, len(sub), dict(sub["condition"].value_counts()),
                 dict(sub["polarity"].value_counts())))

    total = sum(len(df.iloc[i::args.n_chunks]) for i in range(args.n_chunks))
    assert total == len(df), "row count mismatch"
    print()
    print("OK: %d rows across %d chunks, no row lost or duplicated."
          % (total, args.n_chunks))
    print("next:  python %s" % os.path.join(args.run_dir, "stats.py"))


if __name__ == "__main__":
    main()
